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

_SEGMENTER_CACHE = {}


def segment_plant_components(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Segment Plant Components]'
    logger.info(f'Segmenting plant components for batch {batch+1} of {n_batches}')

    seg_cfg = cfg['leafmachine'].get('plant_component_segmentation', {})
    if not seg_cfg.get('enable', False):
        return Project, time_report

    try:
        segmenter = _get_or_create_segmenter(seg_cfg, cfg, dir_home, logger)
    except Exception as exc:
        logger.warning(f'Plant component segmentation disabled for this run: {exc}')
        for _, analysis in Project.project_data_list[batch].items():
            analysis['Segmentation_Plant_Components'] = []
        end_t = perf_counter()
        t_seg_components = (
            f"[Batch {batch+1}/{n_batches}: Plant Component Segmentation elapsed time] "
            f"{round(end_t - start_t)} seconds ({round((end_t - start_t) / 60)} minutes)"
        )
        logger.info(t_seg_components)
        time_report['t_seg_components'] = t_seg_components
        return Project, time_report

    class_filter = _parse_segment_classes(seg_cfg.get('segment_classes', list(range(11))))
    minimum_detection_confidence = float(seg_cfg.get('minimum_detection_confidence', 0.0))
    minimum_bbox_size_px = int(seg_cfg.get('minimum_bbox_size_px', 12))
    multimask_output = bool(seg_cfg.get('multimask_output', False))

    save_mask_png = bool(seg_cfg.get('save_mask_png', True))
    save_masked_rgb = bool(seg_cfg.get('save_masked_rgb', True))
    save_polygon_json = bool(seg_cfg.get('save_polygon_json', True))
    save_overlay_images = bool(seg_cfg.get('save_overlay_images', True))

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

        for det_idx, detection in enumerate(detections):
            det = _coerce_detection(detection)
            if det is None:
                continue

            class_id = int(det[0])
            if class_id not in class_filter:
                continue

            det_score = float(det[5]) if len(det) > 5 else None
            if (det_score is not None) and (det_score < minimum_detection_confidence):
                continue

            bbox_xyxy = _normalized_xywh_to_xyxy(det[1:5], image_w, image_h, minimum_bbox_size_px)
            if bbox_xyxy is None:
                continue

            try:
                mask, seg_score = segmenter.predict(image_rgb, bbox_xyxy, multimask_output)
            except Exception as exc:
                logger.warning(f'Component segmentation failed on {filename}: {exc}')
                continue

            if mask is None:
                continue

            annotation_dict, contour = _build_annotation_from_mask(mask)
            if annotation_dict is None:
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
            annotation_dict.update(artifact_paths)

            analysis['Segmentation_Plant_Components'].append(
                {component_name: [{annotation_name: annotation_dict}]}
            )

    end_t = perf_counter()
    t_seg_components = (
        f"[Batch {batch+1}/{n_batches}: Plant Component Segmentation elapsed time] "
        f"{round(end_t - start_t)} seconds ({round((end_t - start_t) / 60)} minutes)"
    )
    logger.info(t_seg_components)
    time_report['t_seg_components'] = t_seg_components
    return Project, time_report


class _SAM2Segmenter:
    def __init__(self, model_cfg_path, sam2_checkpoint_path, plantsam_checkpoint_path, device, dir_home, logger):
        build_sam2, SAM2ImagePredictor = _import_sam2_api(dir_home, logger)

        self._device = device
        self._logger = logger
        self._predictor_cls = SAM2ImagePredictor
        self._cpu_fallback_triggered = False
        self._active_image_token = None

        if self._device.startswith('cuda'):
            self._configure_cuda_sdpa_backend()
            if not self._cuda_sdpa_preflight():
                self._device = 'cpu'

        self._model = build_sam2(model_cfg_path, sam2_checkpoint_path, device=self._device)

        if plantsam_checkpoint_path and os.path.abspath(plantsam_checkpoint_path) != os.path.abspath(sam2_checkpoint_path):
            self._load_plantsam_weights(plantsam_checkpoint_path, logger)

        self._predictor = SAM2ImagePredictor(self._model)

        if self._device.startswith('cuda'):
            self._cuda_predictor_preflight()

    def _configure_cuda_sdpa_backend(self):
        if not hasattr(torch.backends, 'cuda'):
            return

        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        self._logger.info('Configured CUDA SDPA backend to math-only for SAM2 compatibility.')

    def _cuda_sdpa_preflight(self):
        if not self._device.startswith('cuda'):
            return True

        try:
            with torch.no_grad():
                q = torch.randn((1, 1, 32, 64), device=self._device, dtype=torch.float16)
                k = torch.randn((1, 1, 32, 64), device=self._device, dtype=torch.float16)
                v = torch.randn((1, 1, 32, 64), device=self._device, dtype=torch.float16)

                with warnings.catch_warnings():
                    self._configure_sdpa_warning_filters()

                    with self._make_sdp_context():
                        torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

            return True
        except RuntimeError as exc:
            if self._is_cuda_kernel_error(exc):
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
                self._fallback_to_cpu()
                return
            raise
        finally:
            self._active_image_token = None

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
        self._cpu_fallback_triggered = True
        return True

    @staticmethod
    def _is_cuda_kernel_error(exc):
        err_str = str(exc).lower()
        return (
            'no available kernel' in err_str
            or 'no available backend' in err_str
            or 'no execution plans support the graph' in err_str
        )

    def _make_sdp_context(self):
        if self._device.startswith('cuda') and hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'sdp_kernel'):
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
        if self._active_image_token == image_token:
            return

        sdp_context = self._make_sdp_context()

        try:
            with sdp_context:
                self._predictor.set_image(image_rgb)
            self._active_image_token = image_token
            return
        except RuntimeError as exc:
            if not self._is_cuda_kernel_error(exc) or not self._fallback_to_cpu():
                raise

        self._predictor.set_image(image_rgb)
        self._active_image_token = image_token

    def _load_plantsam_weights(self, plantsam_checkpoint_path, logger):
        checkpoint = torch.load(plantsam_checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            for key in ('model', 'state_dict', 'model_state_dict'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        if not isinstance(checkpoint, dict):
            logger.warning('PlantSAM checkpoint format was not recognized, using base SAM2 checkpoint only.')
            return

        if all(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}

        target_model = self._model.module if hasattr(self._model, 'module') else self._model
        incompatible = target_model.load_state_dict(checkpoint, strict=False)
        logger.info(
            'Loaded PlantSAM checkpoint with '
            f"{len(incompatible.missing_keys)} missing keys and {len(incompatible.unexpected_keys)} unexpected keys"
        )

    def predict(self, image_rgb, bbox_xyxy, multimask_output):
        image_token = id(image_rgb)
        self._ensure_image_embeddings(image_rgb, image_token)
        box = np.asarray(bbox_xyxy, dtype=np.float32)

        sdp_context = self._make_sdp_context()

        try:
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
    if plantsam_checkpoint and not os.path.isfile(plantsam_checkpoint):
        raise FileNotFoundError(f'PlantSAM checkpoint file not found: {plantsam_checkpoint}')

    device = _resolve_device(seg_cfg, cfg, logger)
    cache_key = (model_cfg, sam2_checkpoint, plantsam_checkpoint, device)

    if cache_key not in _SEGMENTER_CACHE:
        _SEGMENTER_CACHE[cache_key] = _SAM2Segmenter(
            model_cfg,
            sam2_checkpoint,
            plantsam_checkpoint,
            device,
            dir_home,
            logger,
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


def _resolve_path(dir_home, input_path):
    if not input_path:
        return ''

    expanded = os.path.expanduser(str(input_path))
    if os.path.isabs(expanded):
        return expanded

    return os.path.join(dir_home, expanded)


def _parse_segment_classes(raw_classes):
    if isinstance(raw_classes, str):
        raw_classes = [item.strip() for item in raw_classes.split(',') if item.strip()]

    classes = set()
    for raw_class in raw_classes:
        try:
            classes.add(int(raw_class))
        except (TypeError, ValueError):
            continue

    if not classes:
        return set(range(11))

    return classes


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


def _normalized_xywh_to_xyxy(det_xywh, image_w, image_h, minimum_bbox_size_px):
    x_center, y_center, box_w, box_h = det_xywh

    x1 = int(round((x_center - (box_w / 2.0)) * image_w))
    y1 = int(round((y_center - (box_h / 2.0)) * image_h))
    x2 = int(round((x_center + (box_w / 2.0)) * image_w))
    y2 = int(round((y_center + (box_h / 2.0)) * image_h))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(1, min(image_w, x2))
    y2 = max(1, min(image_h, y2))

    if (x2 - x1) < minimum_bbox_size_px or (y2 - y1) < minimum_bbox_size_px:
        return None

    return [x1, y1, x2, y2]


def _build_component_name(filename, class_name, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return f'{filename}__{class_name.upper()}__{x1}-{y1}-{x2}-{y2}'


def _build_annotation_from_mask(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 3:
        return None, None

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
