import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SegmentMeta:
    seg_id: int
    mileage_start_mm: float
    mileage_end_mm: float
    timestamp: str
    trigger_mode: str
    scan_line_count: int
    pointcloud_path: str
    meta_file: Optional[str] = None
    corrected_pointcloud_path: Optional[str] = None


def save_meta(meta: SegmentMeta, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(meta)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_meta(path: str) -> SegmentMeta:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Meta file not found: {source}")
    with source.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SegmentMeta(**payload)


def move_to_anomaly(meta: SegmentMeta, anomaly_dir: str) -> SegmentMeta:
    directory = Path(anomaly_dir)
    directory.mkdir(parents=True, exist_ok=True)

    source_pc = Path(meta.pointcloud_path)
    if source_pc.exists():
        target_pc = directory / source_pc.name
        source_pc.replace(target_pc)
        meta.pointcloud_path = str(target_pc)

    if meta.meta_file:
        source_meta = Path(meta.meta_file)
        if source_meta.exists():
            target_meta = directory / source_meta.name
            source_meta.replace(target_meta)
            meta.meta_file = str(target_meta)

    if meta.corrected_pointcloud_path:
        source_corrected = Path(meta.corrected_pointcloud_path)
        if source_corrected.exists():
            target_corrected = directory / source_corrected.name
            source_corrected.replace(target_corrected)
            meta.corrected_pointcloud_path = str(target_corrected)

    return meta


def delete_segment(meta: SegmentMeta) -> None:
    for path in (meta.pointcloud_path, meta.meta_file, meta.corrected_pointcloud_path):
        if not path:
            continue
        candidate = Path(path)
        if candidate.exists():
            candidate.unlink()

