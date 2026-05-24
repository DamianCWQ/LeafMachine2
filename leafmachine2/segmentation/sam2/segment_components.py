import glob
import importlib
import json
import os
import sys
import warnings
from contextlib import nullcontext
from time import perf_counter

import cv2
import numpy as np
import torch


PLANT_COMPONENT_CLASS_MAP = {
    0: 'leaf_whole',
    1: 'leaf_partial',
    2: 'leaflet',
    3: 'seed_fruit_one',
    4: 'seed_fruit_many',
    5: 'flower_one',
    6: 'flower_many',
    7: 'bud',
    8: 'specimen',
    9: 'roots',
    10: 'wood',
}

DEFAULT_EXCLUDED_CLASS_IDS = {0, 1}
CLASS_ID_SOURCE_OF_TRUTH = {
    0: 'leaf_whole',
    1: 'leaf_partial',
    2: 'leaflet',
}

# Classes with thin/small morphology where we prefer recall-preserving cleanup.
RECALL_BIASED_CLASS_IDS = {2, 7, 9}

_SEGMENTER_CACHE = {}


def segment_plant_components(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Segment Plant Components]'
    logger.info(f'Segmenting plant components for batch {batch+1} of {n_batches}')

    seg_cfg = cfg['leafmachine'].get('plant_component_segmentation', {})
    if not seg_cfg.get('enable', False):
        return Project, time_report

    _audit_class_id_source_of_truth(logger)

    class_filter, class_policy = _resolve_segment_class_policy(seg_cfg, logger)
    fallback_enabled = bool(seg_cfg.get('specialization_fallback_to_base', True))
    overlay_cfg = cfg['leafmachine'].get('overlay', {})
    overlay_draw_enabled = bool(
        overlay_cfg.get(
            'show_plant_component_segmentations',
            overlay_cfg.get('show_segmentations', True),
        )
    )

    segmenter = _get_or_create_segmenter(seg_cfg, cfg, dir_home, logger)

    logger.info(
        'Plant component segmentation policy | '
        f"active_class_ids={class_policy['active_class_ids']} "
        f"fallback_enabled={fallback_enabled} "
        f"specialization_checkpoint_present={segmenter.specialization_checkpoint_present} "
        f"overlay_draw_enabled={overlay_draw_enabled}"
    )

    minimum_detection_confidence = float(seg_cfg.get('minimum_detection_confidence', 0.0))
    minimum_bbox_size_px = int(seg_cfg.get('minimum_bbox_size_px', 12))
    multimask_output = bool(seg_cfg.get('multimask_output', False))
    enable_contained_box_dedup = bool(seg_cfg.get('enable_contained_box_dedup', True))
    dedup_within_class_only = bool(seg_cfg.get('dedup_within_class_only', True))
    bbox_padding_px = max(0, int(seg_cfg.get('bbox_padding_px', 2)))
    bbox_padding_ratio = max(0.0, float(seg_cfg.get('bbox_padding_ratio', 0.01)))
    max_bbox_area_ratio = min(1.0, max(0.0, float(seg_cfg.get('max_bbox_area_ratio', 0.75))))
    max_bbox_aspect_ratio = max(1.0, float(seg_cfg.get('max_bbox_aspect_ratio', 12.0)))
    mask_postprocess_enable = bool(seg_cfg.get('mask_postprocess_enable', True))
    mask_cleanup_kernel_precision = int(seg_cfg.get('mask_cleanup_kernel_precision', 5))
    mask_cleanup_kernel_recall = int(seg_cfg.get('mask_cleanup_kernel_recall', 3))
    mask_min_component_area_ratio_precision = max(
        0.0, float(seg_cfg.get('mask_min_component_area_ratio_precision', 0.01))
    )
    mask_min_component_area_ratio_recall = max(
        0.0, float(seg_cfg.get('mask_min_component_area_ratio_recall', 0.003))
    )
    mask_fill_ratio_min = max(0.0, min(1.0, float(seg_cfg.get('mask_fill_ratio_min', 0.003))))
    mask_fill_ratio_max = max(
        mask_fill_ratio_min,
        min(1.0, float(seg_cfg.get('mask_fill_ratio_max', 0.98))),
    )
    contour_simplify_epsilon_ratio = max(
        0.0,
        float(seg_cfg.get('contour_simplify_epsilon_ratio', 0.003)),
    )

    save_mask_png = bool(seg_cfg.get('save_mask_png', True))
    save_masked_rgb = bool(seg_cfg.get('save_masked_rgb', True))
    save_polygon_json = bool(seg_cfg.get('save_polygon_json', True))
    save_overlay_images = bool(seg_cfg.get('save_overlay_images', True))

    batch_stats = {
        'detections_total': 0,
        'detections_filtered_out': 0,
        'detections_attempted': 0,
        'segmented_success': 0,
        'segmented_failed': 0,
        'empty_masks': 0,
        'specialization_loaded': bool(segmenter.specialization_loaded),
        'fallback_to_base_used': bool(segmenter.fallback_to_base_used),
        'skipped_classes_0_1': 0,
    }

    for filename, analysis in Project.project_data_list[batch].items():
        analysis['Segmentation_Plant_Components'] = []
        detections = analysis.get('Detections_Plant_Components', [])
        if not detections:
            continue

        image_bgr = _read_image_any_extension(Project.dir_images, filename)
        if image_bgr is None:
            logger.warning(f'Could not read image for segmentation: {filename}')
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_h, image_w = image_rgb.shape[:2]

        candidate_detections = []
        for det_idx, detection in enumerate(detections):
            batch_stats['detections_total'] += 1
            det = _coerce_detection(detection)
            if det is None:
                batch_stats['detections_filtered_out'] += 1
                continue

            class_id = int(det[0])
            if class_id not in class_filter:
                batch_stats['detections_filtered_out'] += 1
                if class_id in DEFAULT_EXCLUDED_CLASS_IDS:
                    batch_stats['skipped_classes_0_1'] += 1
                continue

            det_score = float(det[5]) if len(det) > 5 else None
            if (det_score is not None) and (det_score < minimum_detection_confidence):
                batch_stats['detections_filtered_out'] += 1
                continue

            bbox_xyxy = _normalized_xywh_to_xyxy(
                det[1:5],
                image_w,
                image_h,
                minimum_bbox_size_px,
                bbox_padding_px=bbox_padding_px,
                bbox_padding_ratio=bbox_padding_ratio,
                max_bbox_area_ratio=max_bbox_area_ratio,
                max_bbox_aspect_ratio=max_bbox_aspect_ratio,
            )
            if bbox_xyxy is None:
                batch_stats['detections_filtered_out'] += 1
                continue

            candidate_detections.append(
                {
                    'det_idx': det_idx,
                    'class_id': class_id,
                    'det_score': det_score,
                    'bbox_xyxy': bbox_xyxy,
                }
            )

        if enable_contained_box_dedup and candidate_detections:
            deduped_candidates = _deduplicate_contained_candidates(
                candidate_detections,
                within_class_only=dedup_within_class_only,
            )
            batch_stats['detections_filtered_out'] += max(
                0,
                len(candidate_detections) - len(deduped_candidates),
            )
            candidate_detections = deduped_candidates

        for candidate in candidate_detections:
            det_idx = candidate['det_idx']
            class_id = candidate['class_id']
            det_score = candidate['det_score']
            bbox_xyxy = candidate['bbox_xyxy']

            batch_stats['detections_attempted'] += 1
            try:
                mask, seg_score = segmenter.predict(image_rgb, bbox_xyxy, multimask_output)
            except Exception as exc:
                logger.warning(
                    'runtime_inference_failure | '
                    f'file={filename} class_id={class_id} reason={exc}'
                )
                batch_stats['segmented_failed'] += 1
                continue

            if mask is None:
                batch_stats['empty_masks'] += 1
                continue

            if mask_postprocess_enable:
                mask = _postprocess_mask(
                    mask,
                    class_id,
                    bbox_xyxy,
                    mask_cleanup_kernel_precision,
                    mask_cleanup_kernel_recall,
                    mask_min_component_area_ratio_precision,
                    mask_min_component_area_ratio_recall,
                )

            if _is_invalid_mask_fill_ratio(mask, bbox_xyxy, mask_fill_ratio_min, mask_fill_ratio_max):
                batch_stats['empty_masks'] += 1
                continue

            annotation_dict, contour = _build_annotation_from_mask(
                mask,
                contour_simplify_epsilon_ratio=contour_simplify_epsilon_ratio,
            )
            if annotation_dict is None:
                batch_stats['empty_masks'] += 1
                continue

            class_name = PLANT_COMPONENT_CLASS_MAP.get(class_id, f'class_{class_id}')
            component_name = _build_component_name(filename, class_name, bbox_xyxy)
            annotation_name = f'{class_name}_{det_idx:04d}'
            stem = f'{component_name}__{det_idx:03d}'

            artifact_paths = _save_artifacts(
                stem,
                class_name,
                bbox_xyxy,
                image_bgr,
                mask,
                contour,
                annotation_dict,
                Dirs,
                save_mask_png,
                save_masked_rgb,
                save_polygon_json,
                save_overlay_images,
            )

            annotation_dict['class_id'] = class_id
            annotation_dict['class_name'] = class_name
            annotation_dict['detector_score'] = det_score
            annotation_dict['segmentation_score'] = seg_score
            annotation_dict['coordinate_system'] = 'absolute_image_xy'
            annotation_dict.update(artifact_paths)

            analysis['Segmentation_Plant_Components'].append(
                {component_name: [{annotation_name: annotation_dict}]}
            )
            batch_stats['segmented_success'] += 1

    end_t = perf_counter()
    t_seg_components = (
        f"[Batch {batch+1}/{n_batches}: Plant Component Segmentation elapsed time] "
        f"{round(end_t - start_t)} seconds ({round((end_t - start_t) / 60)} minutes)"
    )
    logger.info(t_seg_components)
    logger.info(f'Plant component segmentation stats | {json.dumps(batch_stats, sort_keys=True)}')

    policy_snapshot = {
        'active_class_ids': class_policy['active_class_ids'],
        'fallback_enabled': fallback_enabled,
        'specialization_checkpoint_present': bool(segmenter.specialization_checkpoint_present),
        'specialization_loaded': bool(segmenter.specialization_loaded),
        'fallback_to_base_used': bool(segmenter.fallback_to_base_used),
        'specialization_failure_reason': segmenter.specialization_failure_reason,
        'overlay_draw_enabled': overlay_draw_enabled,
    }

    time_report['t_seg_components'] = t_seg_components
    time_report[f't_seg_components_stats_batch_{batch+1}'] = json.dumps(batch_stats, sort_keys=True)
    time_report[f't_seg_components_policy_batch_{batch+1}'] = json.dumps(policy_snapshot, sort_keys=True)
    return Project, time_report


class _SAM2Segmenter:
    def __init__(
        self,
        model_cfg_path,
        sam2_checkpoint_path,
        plantsam_checkpoint_path,
        device,
        dir_home,
        logger,
        specialization_fallback_to_base=True,
        specialization_requested=False,
        specialization_checkpoint_present=False,
        cuda_sdpa_mode='auto',
        cuda_enable_cudnn_sdpa=False,
    ):
        build_sam2, SAM2ImagePredictor = _import_sam2_api(dir_home, logger)

        self._device = device
        self._logger = logger
        self._predictor_cls = SAM2ImagePredictor
        self._cpu_fallback_triggered = False
        self._active_image_token = None
        self._active_image_shape = None
        self.specialization_fallback_to_base = bool(specialization_fallback_to_base)
        self.specialization_requested = bool(specialization_requested)
        self.specialization_checkpoint_present = bool(specialization_checkpoint_present)
        self.specialization_checkpoint_path = plantsam_checkpoint_path
        self.specialization_loaded = False
        self.fallback_to_base_used = False
        self.specialization_failure_reason = ''
        self._cuda_sdpa_mode = _normalize_cuda_sdpa_mode(cuda_sdpa_mode)
        self._cuda_enable_cudnn_sdpa = bool(cuda_enable_cudnn_sdpa)
        self._sam2_math_attention_forced = False

        if self._device.startswith('cuda'):
            self._configure_cuda_sdpa_backend()
            if not self._cuda_sdpa_preflight():
                self._device = 'cpu'

        self._model = build_sam2(model_cfg_path, sam2_checkpoint_path, device=self._device)

        if self.specialization_requested and not self.specialization_checkpoint_present:
            if self.specialization_fallback_to_base:
                self.fallback_to_base_used = True
                self.specialization_failure_reason = 'specialization_checkpoint_missing'
                logger.warning(
                    'specialization_checkpoint_missing | '
                    'PlantSAM specialization checkpoint was configured but not found. '
                    'Continuing with SAM2 base weights only.'
                )
            else:
                raise FileNotFoundError(
                    f'PlantSAM specialization checkpoint not found: {plantsam_checkpoint_path}'
                )

        if plantsam_checkpoint_path and os.path.abspath(plantsam_checkpoint_path) != os.path.abspath(sam2_checkpoint_path):
            try:
                self._load_plantsam_weights(plantsam_checkpoint_path, logger)
                self.specialization_loaded = True
            except Exception as exc:
                if self.specialization_fallback_to_base:
                    self.fallback_to_base_used = True
                    self.specialization_failure_reason = f'specialization_load_incompatible: {exc}'
                    logger.warning(
                        'specialization_load_incompatible | '
                        'PlantSAM specialization weights could not be loaded into SAM2. '
                        f'Continuing with SAM2 base weights only. Reason: {exc}'
                    )
                else:
                    raise RuntimeError(
                        'PlantSAM specialization checkpoint was provided but could not be loaded '
                        f'into SAM2: {exc}'
                    ) from exc

        self._predictor = SAM2ImagePredictor(self._model)

        if self._device.startswith('cuda'):
            self._cuda_predictor_preflight()

    def _configure_cuda_sdpa_backend(self):
        if not hasattr(torch.backends, 'cuda'):
            return

        if self._cuda_enable_cudnn_sdpa:
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
        else:
            os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '0'

        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            try:
                torch.backends.cuda.enable_cudnn_sdp(self._cuda_enable_cudnn_sdpa)
            except Exception:
                # Keep running if this backend flag is not supported by the current torch build.
                pass

        if self._cuda_sdpa_mode == 'math_only':
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            self._logger.info(
                f'Configured CUDA SDPA backend to math-only mode (cudnn_sdp={self._cuda_enable_cudnn_sdpa}).'
            )
            return

        # auto mode keeps all available kernels enabled and lets PyTorch choose.
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        self._logger.info(
            f'Configured CUDA SDPA backend to auto mode (flash/mem-efficient/math, '
            f'cudnn_sdp={self._cuda_enable_cudnn_sdpa}).'
        )

    def _cuda_sdpa_preflight(self):
        if not self._device.startswith('cuda'):
            return True

        try:
            with torch.no_grad():
                # Keep sequence length at 64 to avoid cuDNN SDPA frontend constraints
                # on some Windows/Torch/CuDNN combinations during capability probing.
                q = torch.randn((1, 1, 64, 64), device=self._device, dtype=torch.float16)
                k = torch.randn((1, 1, 64, 64), device=self._device, dtype=torch.float16)
                v = torch.randn((1, 1, 64, 64), device=self._device, dtype=torch.float16)

                with warnings.catch_warnings():
                    self._configure_sdpa_warning_filters()

                    with self._make_sdp_context():
                        torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

            return True
        except RuntimeError as exc:
            if self._is_cuda_kernel_error(exc):
                if self._try_relaxed_cuda_backend():
                    return self._cuda_sdpa_preflight()
                self._logger.warning(
                    'SAM2 CUDA kernels were unavailable during preflight. '
                    'Using CPU for component segmentation.'
                )
                return False
            raise

    @staticmethod
    def _configure_sdpa_warning_filters():
        warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')
        warnings.filterwarnings('ignore', message='.*Memory efficient kernel not used because.*')
        warnings.filterwarnings('ignore', message='.*Memory Efficient attention has been runtime disabled.*')
        warnings.filterwarnings('ignore', message='.*Flash attention kernel not used because.*')
        warnings.filterwarnings('ignore', message='.*CuDNN attention kernel not used because.*')
        warnings.filterwarnings('ignore', message='.*TORCH_CUDNN_SDPA_ENABLED.*')
        warnings.filterwarnings('ignore', message='.*USING CUDNN SDPA.*')

    def _cuda_predictor_preflight(self):
        dummy_image = np.zeros((128, 128, 3), dtype=np.uint8)
        dummy_box = np.asarray([[16.0, 16.0, 112.0, 112.0]], dtype=np.float32)

        sdp_context = self._make_sdp_context()

        try:
            with warnings.catch_warnings():
                self._configure_sdpa_warning_filters()

                with sdp_context:
                    self._predictor.set_image(dummy_image)
                    try:
                        self._predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=dummy_box,
                            multimask_output=False,
                        )
                    except TypeError:
                        self._predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=dummy_box[0],
                            multimask_output=False,
                        )
        except RuntimeError as exc:
            if self._is_cuda_kernel_error(exc):
                if self._try_relaxed_cuda_backend():
                    return self._cuda_predictor_preflight()
                self._fallback_to_cpu()
                return
            raise
        finally:
            self._active_image_token = None
            self._active_image_shape = None

    def _fallback_to_cpu(self):
        if self._cpu_fallback_triggered or self._device == 'cpu':
            return False

        self._logger.warning(
            'SAM2 CUDA kernels were unavailable on this system. '
            'Falling back to CPU for component segmentation.'
        )

        self._device = 'cpu'
        target_model = self._model.module if hasattr(self._model, 'module') else self._model
        target_model.to('cpu')
        self._predictor = self._predictor_cls(target_model)
        self._active_image_token = None
        self._active_image_shape = None
        self._cpu_fallback_triggered = True
        return True

    @staticmethod
    def _is_cuda_kernel_error(exc):
        err_str = str(exc).lower()
        return (
            'no available kernel' in err_str
            or 'no available backend' in err_str
            or 'no execution plans support the graph' in err_str
            or 'cudnn frontend error' in err_str
        )

    def _make_sdp_context(self):
        if not self._device.startswith('cuda'):
            return nullcontext()

        if self._cuda_sdpa_mode != 'math_only':
            return nullcontext()

        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel'):
            try:
                return torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_mem_efficient=False,
                    enable_math=True,
                )
            except Exception:
                return nullcontext()

        return nullcontext()

    def _ensure_image_embeddings(self, image_rgb, image_token):
        # Guard against CPython id() reuse: same address after gc can produce same token
        # for a different image. Also compare shape to force re-embed on size change.
        if self._active_image_token == image_token and self._active_image_shape == image_rgb.shape:
            return

        sdp_context = self._make_sdp_context()

        try:
            with warnings.catch_warnings():
                self._configure_sdpa_warning_filters()
                with sdp_context:
                    self._predictor.set_image(image_rgb)
            self._active_image_token = image_token
            self._active_image_shape = image_rgb.shape
            return
        except RuntimeError as exc:
            if not self._is_cuda_kernel_error(exc):
                raise
            if self._try_relaxed_cuda_backend():
                return self._ensure_image_embeddings(image_rgb, image_token)
            if not self._fallback_to_cpu():
                raise

        self._predictor.set_image(image_rgb)
        self._active_image_token = image_token
        self._active_image_shape = image_rgb.shape

    def _load_plantsam_weights(self, plantsam_checkpoint_path, logger):
        checkpoint = torch.load(plantsam_checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            for key in ('model', 'state_dict', 'model_state_dict'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            raise ValueError('PlantSAM checkpoint format was not recognized')

        if all(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}

        target_model = self._model.module if hasattr(self._model, 'module') else self._model
        incompatible = target_model.load_state_dict(checkpoint, strict=False)
        logger.info(
            'Loaded PlantSAM specialization checkpoint with '
            f"{len(incompatible.missing_keys)} missing keys and {len(incompatible.unexpected_keys)} unexpected keys"
        )

    def predict(self, image_rgb, bbox_xyxy, multimask_output):
        image_token = id(image_rgb)
        self._ensure_image_embeddings(image_rgb, image_token)
        box = np.asarray(bbox_xyxy, dtype=np.float32)

        sdp_context = self._make_sdp_context()

        try:
            with warnings.catch_warnings():
                self._configure_sdpa_warning_filters()
                with sdp_context:
                    try:
                        masks, scores, _ = self._predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box[None, :],
                            multimask_output=multimask_output,
                        )
                    except TypeError:
                        masks, scores, _ = self._predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box,
                            multimask_output=multimask_output,
                        )
        except RuntimeError as exc:
            if self._is_cuda_kernel_error(exc) and self._try_relaxed_cuda_backend():
                return self.predict(image_rgb, bbox_xyxy, multimask_output)
            if self._is_cuda_kernel_error(exc) and self._fallback_to_cpu():
                self._ensure_image_embeddings(image_rgb, image_token)
                try:
                    masks, scores, _ = self._predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :],
                        multimask_output=multimask_output,
                    )
                except TypeError:
                    masks, scores, _ = self._predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=multimask_output,
                    )
            else:
                raise

        if masks is None or len(masks) == 0:
            return None, None

        if scores is None or len(scores) == 0:
            best_idx = 0
            best_score = None
        else:
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

        mask = np.asarray(masks[best_idx]).astype(bool)
        return mask, best_score

    def _try_relaxed_cuda_backend(self):
        if not self._device.startswith('cuda'):
            return False

        if self._cuda_enable_cudnn_sdpa:
            self._logger.warning(
                'CUDA kernel error encountered with cuDNN SDPA enabled. '
                'Retrying once with cuDNN SDPA disabled.'
            )
            self._cuda_enable_cudnn_sdpa = False
            self._configure_cuda_sdpa_backend()
            self._active_image_token = None
            self._active_image_shape = None
            return True

        if self._force_sam2_math_attention():
            return True

        if self._cuda_sdpa_mode == 'auto':
            return False

        self._logger.warning(
            f'CUDA kernel error encountered in SDPA mode "{self._cuda_sdpa_mode}". '
            'Retrying once with auto CUDA SDPA backend before CPU fallback.'
        )
        self._cuda_sdpa_mode = 'auto'
        self._configure_cuda_sdpa_backend()
        self._active_image_token = None
        self._active_image_shape = None
        return True

    def _force_sam2_math_attention(self):
        if self._sam2_math_attention_forced:
            return False

        patched_any = False
        for module_name in ('sam2.modeling.sam.transformer', 'sam2.sam2.modeling.sam.transformer'):
            try:
                transformer_module = importlib.import_module(module_name)
            except Exception:
                continue

            if not hasattr(transformer_module, 'MATH_KERNEL_ON') or not hasattr(transformer_module, 'USE_FLASH_ATTN'):
                continue

            transformer_module.MATH_KERNEL_ON = True
            transformer_module.USE_FLASH_ATTN = False
            patched_any = True

        if not patched_any:
            return False

        self._sam2_math_attention_forced = True
        self._logger.warning(
            'CUDA kernel error encountered with SAM2 flash-attention path. '
            'Retrying once with SAM2 math attention enabled.'
        )
        self._active_image_token = None
        self._active_image_shape = None
        return True


def _import_sam2_api(dir_home, logger):
    local_sam2_root = os.path.join(dir_home, 'sam2')
    if os.path.isdir(os.path.join(local_sam2_root, 'sam2')):
        if local_sam2_root not in sys.path:
            sys.path.insert(0, local_sam2_root)

        # If sam2 was previously loaded as a namespace package from the workspace
        # root, clear it so Python can resolve the real package from local_sam2_root.
        existing_sam2 = sys.modules.get('sam2')
        if existing_sam2 is not None and getattr(existing_sam2, '__file__', None) is None:
            for module_name in [name for name in list(sys.modules) if name == 'sam2' or name.startswith('sam2.')]:
                sys.modules.pop(module_name, None)

    try:
        build_module = importlib.import_module('sam2.build_sam')
        predictor_module = importlib.import_module('sam2.sam2_image_predictor')
        return build_module.build_sam2, predictor_module.SAM2ImagePredictor
    except Exception as primary_exc:
        primary_error = primary_exc

    # Some local checkouts expose SAM2 as a namespace package where
    # the real modules are nested under sam2.sam2.*.
    try:
        if importlib.util.find_spec('sam2.sam2.modeling') and importlib.util.find_spec('sam2.modeling') is None:
            sys.modules['sam2.modeling'] = importlib.import_module('sam2.sam2.modeling')
        if importlib.util.find_spec('sam2.sam2.utils') and importlib.util.find_spec('sam2.utils') is None:
            sys.modules['sam2.utils'] = importlib.import_module('sam2.sam2.utils')

        build_module = importlib.import_module('sam2.sam2.build_sam')
        predictor_module = importlib.import_module('sam2.sam2.sam2_image_predictor')
        logger.info('Loaded SAM2 modules using nested package layout (sam2.sam2.*).')
        return build_module.build_sam2, predictor_module.SAM2ImagePredictor
    except Exception as nested_exc:
        nested_error = nested_exc

    # If SAM2 exists as a local checkout at <repo>/sam2, inject it into sys.path.
    if os.path.isdir(os.path.join(local_sam2_root, 'sam2')):
        try:
            build_module = importlib.import_module('sam2.build_sam')
            predictor_module = importlib.import_module('sam2.sam2_image_predictor')
            logger.info(f'Loaded SAM2 modules from local checkout: {local_sam2_root}')
            return build_module.build_sam2, predictor_module.SAM2ImagePredictor
        except Exception as local_exc:
            local_error = local_exc
    else:
        local_error = RuntimeError(f'Local SAM2 checkout was not found at: {local_sam2_root}')

    raise ImportError(
        'Unable to import SAM2 modules. '\
        f'Primary layout error: {primary_error}; '\
        f'Nested layout error: {nested_error}; '\
        f'Local checkout error: {local_error}'
    )


def _get_or_create_segmenter(seg_cfg, cfg, dir_home, logger):
    model_cfg = _resolve_path(dir_home, seg_cfg.get('sam2_model_config', ''))
    sam2_checkpoint = _resolve_path(dir_home, seg_cfg.get('sam2_checkpoint', ''))
    plantsam_checkpoint = _resolve_path(dir_home, seg_cfg.get('plantsam_checkpoint', ''))
    specialization_fallback_to_base = bool(seg_cfg.get('specialization_fallback_to_base', True))
    cuda_sdpa_mode = _normalize_cuda_sdpa_mode(seg_cfg.get('cuda_sdpa_mode', 'auto'))
    cuda_enable_cudnn_sdpa = bool(seg_cfg.get('cuda_enable_cudnn_sdpa', False))
    specialization_requested = bool(seg_cfg.get('plantsam_checkpoint', ''))
    specialization_checkpoint_present = bool(plantsam_checkpoint and os.path.isfile(plantsam_checkpoint))
    specialization_checkpoint_for_model = plantsam_checkpoint if specialization_checkpoint_present else ''

    if not model_cfg:
        raise ValueError('Missing plant_component_segmentation.sam2_model_config')

    if not sam2_checkpoint and plantsam_checkpoint:
        sam2_checkpoint = plantsam_checkpoint

    if not sam2_checkpoint:
        raise ValueError('Missing plant_component_segmentation.sam2_checkpoint')

    if not os.path.isfile(model_cfg):
        raise FileNotFoundError(f'SAM2 model config file not found: {model_cfg}')
    if not os.path.isfile(sam2_checkpoint):
        raise FileNotFoundError(f'SAM2 checkpoint file not found: {sam2_checkpoint}')
    if specialization_requested and not specialization_checkpoint_present and not specialization_fallback_to_base:
        raise FileNotFoundError(f'PlantSAM checkpoint file not found: {plantsam_checkpoint}')
    if specialization_requested and not specialization_checkpoint_present and specialization_fallback_to_base:
        logger.warning(
            'specialization_checkpoint_missing | PlantSAM specialization checkpoint is missing. '
            'specialization_fallback_to_base is enabled, so SAM2 base weights will be used.'
        )

    device = _resolve_device(seg_cfg, cfg, logger)
    cache_key = (
        model_cfg,
        sam2_checkpoint,
        plantsam_checkpoint,
        specialization_checkpoint_for_model,
        specialization_fallback_to_base,
        cuda_sdpa_mode,
        cuda_enable_cudnn_sdpa,
        specialization_requested,
        device,
    )

    if cache_key not in _SEGMENTER_CACHE:
        _SEGMENTER_CACHE[cache_key] = _SAM2Segmenter(
            model_cfg,
            sam2_checkpoint,
            specialization_checkpoint_for_model,
            device,
            dir_home,
            logger,
            specialization_fallback_to_base=specialization_fallback_to_base,
            specialization_requested=specialization_requested,
            specialization_checkpoint_present=specialization_checkpoint_present,
            cuda_sdpa_mode=cuda_sdpa_mode,
            cuda_enable_cudnn_sdpa=cuda_enable_cudnn_sdpa,
        )

    return _SEGMENTER_CACHE[cache_key]


def _resolve_device(seg_cfg, cfg, logger):
    requested_device = str(
        seg_cfg.get(
            'device',
            cfg['leafmachine']['project'].get('device', 'cpu'),
        )
    ).lower()

    if requested_device.startswith('cuda'):
        if torch.cuda.is_available():
            if requested_device == 'cuda':
                return 'cuda:0'
            return requested_device
        logger.warning('CUDA requested for component segmentation, but CUDA is unavailable. Falling back to CPU.')

    return 'cpu'


def _normalize_cuda_sdpa_mode(raw_mode):
    mode = str(raw_mode or 'auto').strip().lower()
    if mode in {'math', 'math-only', 'math_only'}:
        return 'math_only'
    if mode in {'auto', 'default'}:
        return 'auto'
    return 'auto'


def _resolve_path(dir_home, input_path):
    if not input_path:
        return ''

    expanded = os.path.expanduser(str(input_path))
    if os.path.isabs(expanded):
        return expanded

    return os.path.join(dir_home, expanded)


def _parse_segment_classes(raw_classes):
    if raw_classes is None:
        return set()

    if isinstance(raw_classes, str):
        raw_classes = [item.strip() for item in raw_classes.split(',') if item.strip()]

    if not isinstance(raw_classes, (list, tuple, set)):
        return set()

    classes = set()
    for raw_class in raw_classes:
        try:
            classes.add(int(raw_class))
        except (TypeError, ValueError):
            continue

    return classes


def _resolve_segment_class_policy(seg_cfg, logger):
    all_class_ids = set(PLANT_COMPONENT_CLASS_MAP.keys())

    explicit_raw = seg_cfg.get('segment_classes', None)
    explicit_ids = _parse_segment_classes(explicit_raw)
    explicit_configured = explicit_raw is not None

    if explicit_configured and explicit_ids:
        active_ids = explicit_ids & all_class_ids
        policy_source = 'segment_classes'
        excluded_ids = sorted((all_class_ids - active_ids))
    else:
        include_ids = _parse_segment_classes(seg_cfg.get('segment_classes_include', None))
        exclude_raw = seg_cfg.get('segment_classes_exclude', sorted(DEFAULT_EXCLUDED_CLASS_IDS))
        exclude_ids = _parse_segment_classes(exclude_raw)

        if include_ids:
            active_ids = include_ids & all_class_ids
            policy_source = 'segment_classes_include/segment_classes_exclude'
        else:
            active_ids = set(all_class_ids)
            policy_source = 'default_all_classes_minus_segment_classes_exclude'

        active_ids -= (exclude_ids & all_class_ids)
        excluded_ids = sorted((exclude_ids & all_class_ids))

    if not active_ids:
        logger.warning(
            'No valid segment classes resolved from configuration; '
            'falling back to default active classes [2-10].'
        )
        active_ids = set(all_class_ids) - DEFAULT_EXCLUDED_CLASS_IDS
        policy_source = 'fallback_default_active_classes_2_to_10'
        excluded_ids = sorted(DEFAULT_EXCLUDED_CLASS_IDS)

    policy = {
        'active_class_ids': sorted(active_ids),
        'excluded_class_ids': excluded_ids,
        'policy_source': policy_source,
    }

    return active_ids, policy


def _audit_class_id_source_of_truth(logger):
    mismatches = []
    for class_id, expected_name in CLASS_ID_SOURCE_OF_TRUTH.items():
        observed_name = PLANT_COMPONENT_CLASS_MAP.get(class_id)
        if observed_name != expected_name:
            mismatches.append((class_id, expected_name, observed_name))

    if mismatches:
        details = '; '.join(
            [f'id {class_id}: expected {expected}, observed {observed}' for class_id, expected, observed in mismatches]
        )
        raise ValueError(f'Plant component class map audit failed: {details}')

    logger.info(
        'Plant component class audit passed | '
        f"0={PLANT_COMPONENT_CLASS_MAP[0]} 1={PLANT_COMPONENT_CLASS_MAP[1]} 2={PLANT_COMPONENT_CLASS_MAP[2]}"
    )


def _read_image_any_extension(image_dir, filename):
    candidates = glob.glob(os.path.join(image_dir, f'{filename}.*'))
    for candidate in candidates:
        image = cv2.imread(candidate, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return None


def _coerce_detection(detection):
    if not isinstance(detection, (list, tuple)) or len(detection) < 5:
        return None

    try:
        return [float(value) for value in detection]
    except (TypeError, ValueError):
        return None


def _bbox_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def _is_contained_within(inner_box_xyxy, outer_box_xyxy):
    return (
        inner_box_xyxy[0] >= outer_box_xyxy[0]
        and inner_box_xyxy[1] >= outer_box_xyxy[1]
        and inner_box_xyxy[2] <= outer_box_xyxy[2]
        and inner_box_xyxy[3] <= outer_box_xyxy[3]
    )


def _deduplicate_contained_candidates(candidates, within_class_only=True):
    if len(candidates) <= 1:
        return candidates

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -_bbox_area(item['bbox_xyxy']),
            -(item['det_score'] if item['det_score'] is not None else -1.0),
            item['det_idx'],
        ),
    )

    kept = []
    for candidate in sorted_candidates:
        is_contained = any(
            (
                (not within_class_only or candidate['class_id'] == existing['class_id'])
                and _is_contained_within(candidate['bbox_xyxy'], existing['bbox_xyxy'])
            )
            for existing in kept
        )
        if not is_contained:
            kept.append(candidate)

    return sorted(kept, key=lambda item: item['det_idx'])


def _odd_kernel_size(size):
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _remove_small_components(mask_u8, minimum_component_area_px):
    if minimum_component_area_px <= 1:
        return mask_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    cleaned = np.zeros_like(mask_u8)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= minimum_component_area_px:
            cleaned[labels == label_idx] = 1

    if np.any(cleaned):
        return cleaned

    largest_label_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned[labels == largest_label_idx] = 1
    return cleaned


def _postprocess_mask(
    mask,
    class_id,
    bbox_xyxy,
    mask_cleanup_kernel_precision,
    mask_cleanup_kernel_recall,
    mask_min_component_area_ratio_precision,
    mask_min_component_area_ratio_recall,
):
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.size == 0:
        return mask.astype(bool)

    is_recall_biased_class = class_id in RECALL_BIASED_CLASS_IDS
    kernel_size = _odd_kernel_size(
        mask_cleanup_kernel_recall if is_recall_biased_class else mask_cleanup_kernel_precision
    )

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Keep thin structures in recall-biased classes by skipping the opening pass.
        if not is_recall_biased_class and kernel_size >= 3:
            opening_kernel_size = _odd_kernel_size(max(3, kernel_size - 2))
            opening_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (opening_kernel_size, opening_kernel_size),
            )
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, opening_kernel, iterations=1)

    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    minimum_component_area_px = max(
        1,
        int(
            round(
                bbox_area
                * (
                    mask_min_component_area_ratio_recall
                    if is_recall_biased_class
                    else mask_min_component_area_ratio_precision
                )
            )
        ),
    )

    mask_u8 = _remove_small_components(mask_u8, minimum_component_area_px)
    return mask_u8.astype(bool)


def _is_invalid_mask_fill_ratio(mask, bbox_xyxy, min_fill_ratio, max_fill_ratio):
    x1, y1, x2, y2 = bbox_xyxy
    mask_crop = mask[y1:y2, x1:x2]
    if mask_crop.size == 0:
        return True

    fill_ratio = float(np.count_nonzero(mask_crop)) / float(mask_crop.size)
    return fill_ratio < min_fill_ratio or fill_ratio > max_fill_ratio


def _normalized_xywh_to_xyxy(
    det_xywh,
    image_w,
    image_h,
    minimum_bbox_size_px,
    bbox_padding_px=0,
    bbox_padding_ratio=0.0,
    max_bbox_area_ratio=1.0,
    max_bbox_aspect_ratio=12.0,
):
    x_center, y_center, box_w, box_h = det_xywh

    x1 = int(round((x_center - (box_w / 2.0)) * image_w))
    y1 = int(round((y_center - (box_h / 2.0)) * image_h))
    x2 = int(round((x_center + (box_w / 2.0)) * image_w))
    y2 = int(round((y_center + (box_h / 2.0)) * image_h))

    raw_bbox_w = max(1, x2 - x1)
    raw_bbox_h = max(1, y2 - y1)
    padding = int(round(max(float(bbox_padding_px), max(raw_bbox_w, raw_bbox_h) * float(bbox_padding_ratio))))
    if padding > 0:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(1, min(image_w, x2))
    y2 = max(1, min(image_h, y2))

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    if bbox_w < minimum_bbox_size_px or bbox_h < minimum_bbox_size_px:
        return None

    bbox_area = bbox_w * bbox_h
    image_area = max(1, image_w * image_h)
    if max_bbox_area_ratio < 1.0 and bbox_area > (max_bbox_area_ratio * image_area):
        return None

    aspect_ratio = max(float(bbox_w) / float(max(1, bbox_h)), float(bbox_h) / float(max(1, bbox_w)))
    if aspect_ratio > max_bbox_aspect_ratio:
        return None

    return [x1, y1, x2, y2]


def _build_component_name(filename, class_name, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return f'{filename}__{class_name.upper()}__{x1}-{y1}-{x2}-{y2}'


def _build_annotation_from_mask(mask, contour_simplify_epsilon_ratio=0.0):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 3:
        return None, None

    if contour_simplify_epsilon_ratio > 0.0:
        epsilon = contour_simplify_epsilon_ratio * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        if simplified_contour is not None and len(simplified_contour) >= 3:
            contour = simplified_contour

    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None, None

    perimeter = float(cv2.arcLength(contour, True))

    moments = cv2.moments(contour)
    if moments['m00'] > 0:
        centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    else:
        centroid = tuple(np.mean(contour[:, 0, :], axis=0).astype(int).tolist())

    x, y, w, h = cv2.boundingRect(contour)
    bbox = [(int(x), int(y)), (int(x + w), int(y)), (int(x + w), int(y + h)), (int(x), int(y + h))]

    min_rect = cv2.minAreaRect(contour)
    min_rect_box = cv2.boxPoints(min_rect)
    min_rect_box = np.round(min_rect_box).astype(int)

    long_side = float(max(min_rect[1]))
    short_side = float(min(min_rect[1]))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity = float(area / hull_area) if hull_area > 0 else None
    concavity = float(1.0 - convexity) if convexity is not None else None
    circularity = float((4.0 * np.pi * area) / (perimeter * perimeter)) if perimeter > 0 else None

    polygon = contour[:, 0, :].astype(int).tolist()
    polygon_closed = polygon + [polygon[0]]

    annotation_dict = {
        'bbox': bbox,
        'bbox_min': min_rect_box.tolist(),
        'rotate_angle': float(min_rect[2]),
        'long': long_side,
        'short': short_side,
        'area': area,
        'perimeter': perimeter,
        'centroid': centroid,
        'convex_hull': hull_area,
        'convexity': convexity,
        'concavity': concavity,
        'circularity': circularity,
        'degree': int(len(polygon_closed)),
        'aspect_ratio': float(long_side / short_side) if short_side > 0 else None,
        'polygon': polygon,
        'polygon_closed': polygon_closed,
        'polygon_closed_rotated': polygon_closed,
    }

    return annotation_dict, contour


def _save_artifacts(
    stem,
    class_name,
    bbox_xyxy,
    image_bgr,
    mask,
    contour,
    annotation_dict,
    Dirs,
    save_mask_png,
    save_masked_rgb,
    save_polygon_json,
    save_overlay_images,
):
    saved_paths = {}

    if save_mask_png and getattr(Dirs, 'segmentation_plant_components_masks', None):
        mask_path = os.path.join(Dirs.segmentation_plant_components_masks, f'{stem}.png')
        cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))
        saved_paths['mask_path'] = mask_path

    if save_masked_rgb and getattr(Dirs, 'segmentation_plant_components_masked_rgb', None):
        masked_rgb_path = os.path.join(Dirs.segmentation_plant_components_masked_rgb, f'{stem}.png')
        masked_rgb = _create_masked_rgb_crop(image_bgr, mask, bbox_xyxy)
        if masked_rgb is not None:
            cv2.imwrite(masked_rgb_path, masked_rgb)
            saved_paths['masked_rgb_path'] = masked_rgb_path

    if save_polygon_json and getattr(Dirs, 'segmentation_plant_components_polygons', None):
        polygon_path = os.path.join(Dirs.segmentation_plant_components_polygons, f'{stem}.json')
        polygon_payload = {
            'class_name': class_name,
            'bbox_xyxy': bbox_xyxy,
            'polygon_closed': annotation_dict['polygon_closed'],
            'centroid': annotation_dict['centroid'],
            'area': annotation_dict['area'],
            'perimeter': annotation_dict['perimeter'],
            'coordinate_system': 'absolute_image_xy',
        }
        with open(polygon_path, 'w', encoding='utf-8') as fp:
            json.dump(polygon_payload, fp)
        saved_paths['polygon_json_path'] = polygon_path

    if save_overlay_images and getattr(Dirs, 'segmentation_plant_components_overlays', None):
        overlay_path = os.path.join(Dirs.segmentation_plant_components_overlays, f'{stem}.jpg')
        overlay = image_bgr.copy()
        cv2.polylines(overlay, [contour], True, (0, 255, 0), 2)
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            overlay,
            class_name,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(overlay_path, overlay)
        saved_paths['overlay_path'] = overlay_path

    return saved_paths


def _create_masked_rgb_crop(image_bgr, mask, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy

    crop = image_bgr[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    if crop.size == 0 or mask_crop.size == 0:
        return None

    masked = np.zeros_like(crop)
    masked[mask_crop] = crop[mask_crop]
    return masked
