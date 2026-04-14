from __future__ import annotations

import os

# Ensure settings can initialize during import in isolated test environments.
os.environ.setdefault("LM2_UPLOAD_DIR", "./uploads")
os.environ.setdefault("LM2_OUTPUT_DIR", "./outputs")

from api import runner


def test_collect_results_includes_jpg_and_normalizes_paths(tmp_path):
    out = tmp_path / "job-out"
    nested = out / "run" / "Plant_Components" / "Segmentation_Whole_Leaves"
    nested.mkdir(parents=True)

    (nested / "leaf_a.jpg").write_bytes(b"jpg")
    (nested / "leaf_b.png").write_bytes(b"png")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")

    files = runner._collect_results(str(out))

    assert "run/Plant_Components/Segmentation_Whole_Leaves/leaf_a.jpg" in files
    assert "run/Plant_Components/Segmentation_Whole_Leaves/leaf_b.png" in files
    assert all(not f.endswith(".txt") for f in files)


def test_collect_result_artifacts_classifies_measurements_and_segmentation(tmp_path):
    out = tmp_path / "job-out"

    meas_dir = out / "run" / "Data" / "Measurements"
    seg_dir = out / "run" / "Plant_Components" / "Segmentation_Whole_Leaves"
    meas_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)

    (meas_dir / "run_MEASUREMENTS.csv").write_text("filename,component_name\nimg,leaf\n", encoding="utf-8")
    (seg_dir / "leaf_overlay.jpg").write_bytes(b"jpg")

    artifacts = runner._collect_result_artifacts(str(out))
    by_path = {item["path"]: item for item in artifacts}

    measurements_key = "run/Data/Measurements/run_MEASUREMENTS.csv"
    segmentation_key = "run/Plant_Components/Segmentation_Whole_Leaves/leaf_overlay.jpg"

    assert by_path[measurements_key]["kind"] == "measurements_csv"
    assert by_path[measurements_key]["media_type"] == "table"
    assert by_path[segmentation_key]["kind"] == "segmentation_image"
    assert by_path[segmentation_key]["media_type"] == "image"


def test_parse_results_csv_and_build_measurement_records(tmp_path):
    out = tmp_path / "job-out"
    run_name = "demo-run"
    meas_dir = out / run_name / "Data" / "Measurements"
    meas_dir.mkdir(parents=True)

    csv_content = (
        "filename,component_name,area,perimeter,bbox_min_long_side,bbox_min_short_side,"
        "units,conversion_factor_applied,aspect_ratio,annotation_name\n"
        "img_1,leaf,120.5,55.2,24.1,11.7,cm,2.0,2.06,whole_leaf\n"
    )
    (meas_dir / f"{run_name}_MEASUREMENTS.csv").write_text(csv_content, encoding="utf-8")

    results = runner._parse_results_csv(str(out), run_name)
    assert len(results) == 1

    record = results[0]
    assert record["filename"] == "img_1"
    assert record["component_type"] == "whole_leaf"
    assert record["area"] == 120.5
    assert record["aspect_ratio"] == 2.06

    measurement_records = runner._build_measurement_records(results)
    assert measurement_records == [
        {
            "filename": "img_1",
            "component_name": "leaf",
            "component_type": "whole_leaf",
            "area": 120.5,
            "perimeter": 55.2,
            "bbox_min_long_side": 24.1,
            "bbox_min_short_side": 11.7,
            "units": "cm",
            "conversion_factor_applied": 2.0,
            "aspect_ratio": 2.06,
        }
    ]


def test_truncate_payload_limits_items():
    full = [1, 2, 3, 4]

    unlimited, dropped_unlimited = runner._truncate_payload(full, 0)
    limited, dropped_limited = runner._truncate_payload(full, 2)

    assert unlimited == full
    assert dropped_unlimited == 0
    assert limited == [1, 2]
    assert dropped_limited == 2
