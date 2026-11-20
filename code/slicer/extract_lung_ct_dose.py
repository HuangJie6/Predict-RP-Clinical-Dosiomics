"""Batch-export planning CT, lung masks, and cumulative RT dose with 3D Slicer.

Run this file with Slicer.exe, not with a system Python interpreter.  The
SlicerRT extension is required.  Source DICOM files are read-only.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import slicer
import vtk
from DICOMLib import DICOMUtils


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
LOG_PATH = PROJECT_DIR.parent / "outputs" / "slicer" / "extraction.log"
MANIFEST_FIELDS = [
    "case",
    "status",
    "message",
    "ct_series_uid",
    "ct_series_description",
    "ct_slices",
    "rtstruct_file",
    "selected_lung_segments",
    "lung_mask_voxels",
    "lung_volume_ml",
    "dose_file",
    "dose_units",
    "ct_file",
    "lung_mask_file",
    "dose_original_file",
]


class IncompleteCaseError(RuntimeError):
    """The case is intentionally excluded because required research data are absent."""


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("input_root", "output_root", "case_name_filter_csv"):
        if config.get(key):
            path = Path(config[key])
            config[key] = path if path.is_absolute() else (PROJECT_DIR / path).resolve()
    extension = str(config.get("output_extension", ".nrrd")).lower()
    if extension not in {".nrrd", ".nii", ".nii.gz"}:
        raise ValueError("output_extension must be .nrrd, .nii, or .nii.gz")
    config["output_extension"] = extension
    return config


def require_slicer_modules() -> None:
    missing = []
    if not hasattr(slicer.modules, "dicomrtimportexport"):
        missing.append("SlicerRT (DICOM RT Import/Export)")
    if not hasattr(slicer.modules, "segmentations"):
        missing.append("Segmentations")
    if missing:
        raise RuntimeError(
            "Missing Slicer module(s): "
            + ", ".join(missing)
            + ". Install the SlicerRT extension using Extension Manager, restart Slicer, and run again."
        )


def output_path(case_output: Path, base_name: str, extension: str) -> Path:
    return case_output / f"{base_name}{extension}"


def canonical_case_name(case_dir: Path, config: dict) -> str:
    """Use the clinical-table patient_id for output folders without renaming source DICOM data."""
    aliases = {
        str(source).casefold(): str(target).strip()
        for source, target in config.get("case_name_aliases", {}).items()
    }
    return aliases.get(case_dir.name.casefold(), case_dir.name)


def normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def configured_case_filter(config: dict) -> set[str]:
    """Load explicit names and/or a cohort CSV produced before Slicer starts."""
    names = {str(name).casefold() for name in config.get("case_name_filter", [])}
    csv_value = config.get("case_name_filter_csv")
    if not csv_value:
        return names
    path = Path(csv_value)
    if not path.is_file():
        raise FileNotFoundError(f"Cohort patient list does not exist: {path}")
    column = str(config.get("case_name_filter_column", "patient_id"))
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"Cohort list must contain column {column!r}: {path}")
        names.update(str(row.get(column, "")).strip().casefold() for row in reader if row.get(column, "").strip())
    if not names:
        raise ValueError(f"Cohort patient list is empty: {path}")
    return names


def lung_side(name: str) -> str | None:
    """Classify common clinical RTSTRUCT lung names without accepting target-subtracted ROIs."""
    value = normalized_name(name)
    if any(token in value for token in ("ptv", "gtv", "ctv", "itv", "minus", "avoid")):
        return None
    whole = {"lung", "lungs", "lungall", "alllung", "totallung", "bothlungs", "pulmonary"}
    left = {"lungl", "llung", "leftlung", "lungleft", "pulml", "lpulm"}
    right = {"lungr", "rlung", "rightlung", "lungright", "pulmr", "rpulm"}
    if value in whole:
        return "whole"
    if value in left:
        return "left"
    if value in right:
        return "right"
    return None


def segment_table(segmentation_node) -> list[dict]:
    segmentation = segmentation_node.GetSegmentation()
    ids = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(ids)
    result = []
    for index in range(ids.GetNumberOfValues()):
        segment_id = ids.GetValue(index)
        name = segmentation.GetSegment(segment_id).GetName()
        result.append({"id": segment_id, "name": name, "side": lung_side(name)})
    return result


def choose_lung_segments(segmentation_node) -> tuple[list[dict], dict[str, dict], list[str]]:
    table = segment_table(segmentation_node)
    warnings = []

    def best(side: str):
        matches = [row for row in table if row["side"] == side]
        return min(matches, key=lambda row: (len(normalized_name(row["name"])), row["name"])) if matches else None

    left = best("left")
    right = best("right")
    whole = best("whole")
    by_side = {key: value for key, value in (("left", left), ("right", right), ("whole", whole)) if value}
    if left and right:
        selected = [left, right]
    elif whole:
        selected = [whole]
        warnings.append("Bilateral pair not found; used whole-lung segment.")
    elif left or right:
        selected = [left or right]
        warnings.append("Only one lung side was found; combined mask is unilateral.")
    else:
        available = ", ".join(row["name"] for row in table)
        raise IncompleteCaseError(f"No usable lung ROI in RTSTRUCT. Available ROIs: {available}")
    return selected, by_side, warnings


def all_series_info(database) -> list[dict]:
    result = []
    for uid in DICOMUtils.allSeriesUIDsInDatabase(database):
        files = list(database.filesForSeries(uid))
        if not files:
            continue
        first = files[0]
        result.append(
            {
                "uid": uid,
                "files": files,
                "modality": database.fileValue(first, "0008,0060").upper(),
                "description": database.fileValue(first, "0008,103E"),
            }
        )
    return result


def select_input_series(case_dir: Path, database) -> tuple[dict, dict, dict | None, list[str]]:
    series = all_series_info(database)
    warnings = []
    ct_candidates = [item for item in series if item["modality"] == "CT"]
    if not ct_candidates:
        raise IncompleteCaseError("No CT series found.")
    ct_info = max(ct_candidates, key=lambda item: len(item["files"]))
    if len(ct_candidates) > 1:
        warnings.append(
            f"Found {len(ct_candidates)} CT series; selected the series with most instances "
            f"({len(ct_info['files'])} files, {ct_info['description']!r})."
        )

    rtstruct_candidates = [item for item in series if item["modality"] == "RTSTRUCT"]
    if not rtstruct_candidates:
        raise IncompleteCaseError("No RTSTRUCT series found.")
    named_rtstruct = [
        item
        for item in rtstruct_candidates
        if any("strctrsets" in Path(file_name).name.casefold() for file_name in item["files"])
    ]
    rtstruct_info = (named_rtstruct or rtstruct_candidates)[0]
    if len(rtstruct_candidates) > 1:
        warnings.append(f"Found {len(rtstruct_candidates)} RTSTRUCT series; selected {Path(rtstruct_info['files'][0]).name}.")

    exact_dose_files = [path for path in case_dir.iterdir() if path.is_file() and re.search(r"_Dose\.dcm$", path.name, re.IGNORECASE)]
    dose_info = None
    if len(exact_dose_files) == 1:
        exact_path = exact_dose_files[0].resolve()
        for item in series:
            if item["modality"] != "RTDOSE":
                continue
            if any(Path(file_name).resolve() == exact_path for file_name in item["files"]):
                dose_info = item
                dose_info = dict(dose_info)
                dose_info["selected_file"] = str(exact_path)
                break
        if dose_info is None:
            warnings.append(f"Exact cumulative dose exists but was not indexed as RTDOSE: {exact_path.name}")
    elif not exact_dose_files:
        warnings.append("No cumulative *_Dose.dcm file; beam files were intentionally not substituted.")
    else:
        names = ", ".join(path.name for path in exact_dose_files)
        warnings.append(f"Multiple cumulative *_Dose.dcm files found; dose export skipped: {names}")
    return ct_info, rtstruct_info, dose_info, warnings


def node_from_ids(node_ids, class_name: str, predicate=None):
    for node_id in node_ids:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node and node.IsA(class_name) and (predicate is None or predicate(node)):
            return node
    return None


def load_series(series_uid: str) -> list[str]:
    messages = []
    node_ids = DICOMUtils.loadSeriesByUID([series_uid], messages=messages)
    if messages:
        logging.warning("DICOM load messages for %s: %s", series_uid, " | ".join(messages))
    return list(node_ids)


def save_node(node, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not slicer.util.saveNode(node, str(path)):
        raise RuntimeError(f"Could not save node {node.GetName()!r} to {path}")
    return str(path)


def export_binary_mask(segmentation_node, segment_ids: list[str], reference_volume, output_file: Path) -> tuple[str, int, float]:
    id_array = vtk.vtkStringArray()
    for segment_id in segment_ids:
        id_array.InsertNextValue(segment_id)
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", output_file.stem)
    success = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
        segmentation_node, id_array, labelmap, reference_volume
    )
    if not success:
        raise RuntimeError(f"Failed to rasterize segment(s): {segment_ids}")
    array = slicer.util.arrayFromVolume(labelmap)
    array[:] = (array > 0).astype(array.dtype)
    slicer.util.arrayFromVolumeModified(labelmap)
    voxel_count = int(np.count_nonzero(array))
    if voxel_count == 0:
        raise RuntimeError(f"Rasterized mask is empty: {segment_ids}")
    spacing = reference_volume.GetSpacing()
    volume_ml = voxel_count * float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0
    saved = save_node(labelmap, output_file)
    slicer.mrmlScene.RemoveNode(labelmap)
    return saved, voxel_count, volume_ml


def write_case_metadata(case_output: Path, record: dict) -> None:
    path = case_output / "metadata.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)


def write_manifest(output_root: Path, records: list[dict]) -> None:
    path = output_root / "manifest.csv"
    temporary = output_root / "manifest.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def expected_core_outputs(case_output: Path, config: dict) -> list[Path]:
    """Outputs required before a case may be safely skipped on a restart."""
    extension = config["output_extension"]
    return [
        output_path(case_output, "CT", extension),
        output_path(case_output, "LungMask", extension),
        output_path(case_output, "Dose_original", extension),
    ]


def process_case(case_dir: Path, config: dict) -> dict:
    extension = config["output_extension"]
    case_name = canonical_case_name(case_dir, config)
    case_output = config["output_root"] / case_name
    record = {field: "" for field in MANIFEST_FIELDS}
    record.update({"case": case_name, "status": "processing", "message": ""})
    warnings = []

    if not config.get("overwrite", False) and all(path.exists() for path in expected_core_outputs(case_output, config)):
        existing_metadata = case_output / "metadata.json"
        if existing_metadata.exists():
            try:
                with existing_metadata.open("r", encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("status") == "ok" and str(previous.get("dose_units", "")).upper() == "GY":
                    logging.info("[%s] Complete export exists; keeping verified metadata.", case_name)
                    return previous
            except Exception:
                pass
        raise RuntimeError(
            f"Existing outputs for {case_name} lack verified metadata. "
            "Inspect this case before rerunning with overwrite=true."
        )

    slicer.mrmlScene.Clear(0)
    with DICOMUtils.TemporaryDICOMDatabase() as database:
        logging.info("[%s] Indexing DICOM files", case_dir.name)
        if not DICOMUtils.importDicom(str(case_dir), database):
            raise RuntimeError("DICOM import failed.")
        ct_info, rtstruct_info, dose_info, selection_warnings = select_input_series(case_dir, database)
        warnings.extend(selection_warnings)
        if config.get("require_complete_case", True) and dose_info is None:
            detail = " | ".join(selection_warnings) or "No unique cumulative *_Dose.dcm RTDOSE was selected."
            raise IncompleteCaseError(detail)

        ct_ids = load_series(ct_info["uid"])
        ct_node = node_from_ids(ct_ids, "vtkMRMLScalarVolumeNode")
        if ct_node is None:
            raise RuntimeError("Selected CT series did not create a scalar volume node.")
        record.update(
            ct_series_uid=ct_info["uid"],
            ct_series_description=ct_info["description"],
            ct_slices=len(ct_info["files"]),
        )

        struct_ids = load_series(rtstruct_info["uid"])
        segmentation_node = node_from_ids(struct_ids, "vtkMRMLSegmentationNode")
        if segmentation_node is None:
            segmentation_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if segmentation_node is None:
            raise RuntimeError("RTSTRUCT did not create a segmentation node. Check SlicerRT installation and Error Log.")
        record["rtstruct_file"] = str(rtstruct_info["files"][0])

        selected, by_side, lung_warnings = choose_lung_segments(segmentation_node)
        warnings.extend(lung_warnings)
        if config.get("require_complete_case", True) and not (
            ("left" in by_side and "right" in by_side) or "whole" in by_side
        ):
            found = ", ".join(item["name"] for item in selected)
            raise IncompleteCaseError(f"Complete bilateral/whole-lung contour not available; found: {found}")
        record["selected_lung_segments"] = "|".join(item["name"] for item in selected)

        case_output.mkdir(parents=True, exist_ok=True)
        record["ct_file"] = save_node(ct_node, output_path(case_output, "CT", extension))
        mask_file, voxel_count, volume_ml = export_binary_mask(
            segmentation_node,
            [item["id"] for item in selected],
            ct_node,
            output_path(case_output, "LungMask", extension),
        )
        record.update(
            lung_mask_file=mask_file,
            lung_mask_voxels=voxel_count,
            lung_volume_ml=round(volume_ml, 3),
        )

        if dose_info:
            dose_ids = load_series(dose_info["uid"])
            dose_node = node_from_ids(
                dose_ids,
                "vtkMRMLScalarVolumeNode",
                lambda node: node.GetAttribute("DicomRtImport.DoseVolume") == "1",
            )
            if dose_node is None:
                dose_node = node_from_ids(dose_ids, "vtkMRMLScalarVolumeNode")
            if dose_node is None:
                raise RuntimeError("Selected cumulative RTDOSE did not create a scalar volume node.")
            record["dose_file"] = dose_info["selected_file"]
            record["dose_units"] = database.fileValue(dose_info["selected_file"], "3004,0002")
            record["dose_original_file"] = save_node(
                dose_node, output_path(case_output, "Dose_original", extension)
            )

    record["status"] = "warning" if warnings else "ok"
    record["message"] = " | ".join(warnings)
    record["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_case_metadata(case_output, record)
    return record


def main() -> int:
    configure_logging()
    config = load_config()
    require_slicer_modules()
    input_root = config["input_root"]
    output_root = config["output_root"]
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    case_filter = configured_case_filter(config)
    cases = sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir()
            and (
                not case_filter
                or path.name.casefold() in case_filter
                or canonical_case_name(path, config).casefold() in case_filter
            )
        ),
        key=lambda path: path.name.casefold(),
    )
    if not cases:
        raise RuntimeError("No case directories matched the configuration.")

    logging.info("Starting %d case(s). Source=%s Output=%s", len(cases), input_root, output_root)
    records = []
    failed = 0
    for index, case_dir in enumerate(cases, start=1):
        logging.info("[%d/%d] %s", index, len(cases), case_dir.name)
        try:
            record = process_case(case_dir, config)
            logging.info("[%s] %s: %s", case_dir.name, record["status"], record.get("message", ""))
        except IncompleteCaseError as exc:
            logging.warning("[%s] EXCLUDED: %s", case_dir.name, exc)
            record = {field: "" for field in MANIFEST_FIELDS}
            record.update(
                case=case_dir.name,
                status="excluded",
                message=str(exc),
                completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            failed += 1
            logging.exception("[%s] FAILED", case_dir.name)
            case_name = canonical_case_name(case_dir, config)
            case_output = output_root / case_name
            case_output.mkdir(parents=True, exist_ok=True)
            record = {field: "" for field in MANIFEST_FIELDS}
            record.update(
                case=case_name,
                status="error",
                message=str(exc),
                completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                traceback=traceback.format_exc(),
            )
            write_case_metadata(case_output, record)
            if not config.get("continue_on_error", True):
                records.append(record)
                write_manifest(output_root, records)
                raise
        finally:
            slicer.mrmlScene.Clear(0)
        records.append(record)
        write_manifest(output_root, records)

    summary = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": len(records),
        "ok": sum(record["status"] == "ok" for record in records),
        "warning": sum(record["status"] == "warning" for record in records),
        "skipped": sum(record["status"] == "skipped" for record in records),
        "excluded": sum(record["status"] == "excluded" for record in records),
        "error": sum(record["status"] == "error" for record in records),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    logging.info("Finished: %s", summary)
    return 1 if failed else 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        configure_logging()
        logging.exception("Fatal error before/during batch processing")
    if slicer.util.mainWindow() is None:
        slicer.util.exit(exit_code)
