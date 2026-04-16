import argparse
import logging
import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import compute_segment_length, load_config, setup_logging
from core.coverage import CoverageMap, SegmentState
from core.odometry import create_odometry
from core.detector import Detector
from core.detection_adapter import create_pointcloud_detection_fn
from utils.file_manager import SegmentMeta, save_meta
from utils.pointcloud import extract_roi, remove_nan_points, save_pointcloud

logger = logging.getLogger(__name__)


def ensure_paths(cfg: Dict[str, Any]) -> None:
    """Create all configured data directories before starting."""
    paths = cfg["paths"]
    for key in ("data_dir", "pending_dir", "anomaly_dir", "corrected_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    Path(paths["coverage_map"]).parent.mkdir(parents=True, exist_ok=True)


def _process_points_cpu(points, roi_x, roi_z):
    """CPU-bound processing: remove NaN and extract ROI. Safe for ProcessPoolExecutor."""
    from utils.pointcloud import remove_nan_points, extract_roi
    import numpy as _np

    pts = remove_nan_points(points)
    pts = extract_roi(pts, roi_x, roi_z)
    # Ensure contiguous float32 array for efficient IO
    return _np.ascontiguousarray(pts, dtype=_np.float32)


def _save_and_update(seg_id, seg_info, points, pending_dir, meta_path, coverage, cfg):
    """I/O-bound: save pointcloud, write meta and update coverage map."""
    from utils.pointcloud import save_pointcloud
    from utils.file_manager import SegmentMeta, save_meta
    from datetime import datetime as _dt
    pc_path = Path(pending_dir) / f"segment_{int(seg_id):05d}.ply"
    # Save point cloud (fast ASCII or binary writer)
    save_pointcloud(points, str(pc_path))

    meta = SegmentMeta(
        seg_id=int(seg_id),
        mileage_start_mm=seg_info["start_mm"],
        mileage_end_mm=seg_info["end_mm"],
        timestamp=_dt.utcnow().isoformat() + "Z",
        trigger_mode=cfg["scanner"]["trigger_mode"],
        scan_line_count=get_scan_line_count(cfg),
        pointcloud_path=str(pc_path),
        meta_file=str(meta_path),
    )
    save_meta(meta, str(meta_path))

    coverage.update(
        seg_id,
        state=SegmentState.PENDING,
        scan_time=meta.timestamp,
        point_cloud_file=str(pc_path),
        meta_file=str(meta_path),
    )


def create_coverage_map(cfg: Dict[str, Any]) -> CoverageMap:
    coverage_file = cfg["paths"]["coverage_map"]
    segment_length_mm = compute_segment_length(cfg)
    return CoverageMap(coverage_file, cfg["track"]["total_length_mm"], segment_length_mm)


def create_scanner(cfg: Dict[str, Any]):
    try:
        from core.scanner import Scanner as ProfilerScanner
    except Exception as exc:
        raise RuntimeError(
            "Failed to import scanner module. Please ensure the Mech-Eye SDK is installed "
            "and scanner.py can be imported from the current project." 
        ) from exc

    return ProfilerScanner(cfg)


def get_scan_line_count(cfg: Dict[str, Any]) -> int:
    mode = cfg["scanner"]["trigger_mode"]
    if mode == "encoder":
        return cfg["scanner"]["encoder"]["scan_line_count"]
    return cfg["scanner"]["fixed_rate"]["scan_line_count"]


def wait_for_mileage(odometry, target_mm: float) -> None:
    logger.info("Waiting for target mileage: %.2f mm", target_mm)
    while True:
        current = odometry.get_mileage_mm()
        if current >= target_mm:
            logger.info("Reached target mileage: %.2f mm", current)
            return
        time.sleep(0.1)


def scan_trip(cfg: Dict[str, Any]) -> List[str]:
    ensure_paths(cfg)
    coverage = create_coverage_map(cfg)
    targets = coverage.get_unscanned(cfg["mission"]["segments_per_trip"])
    if not targets:
        logger.info("没有未扫描的段，扫描任务已完成。")
        return []

    odometry = create_odometry(cfg)
    odometry.reset()
    odometry.start()

    scanner = create_scanner(cfg)
    connected = scanner.connect()
    if not connected:
        odometry.stop()
        raise RuntimeError("Scanner connection failed.")
    scanner.configure()
    # Producer-consumer pipeline
    scanned_segments: List[str] = []
    task_q = queue.Queue(maxsize=cfg.get("performance", {}).get("queue_size", 8))
    cpu_workers = cfg.get("performance", {}).get("cpu_workers", 2)
    io_workers = cfg.get("performance", {}).get("io_workers", 2)

    cpu_pool = ProcessPoolExecutor(max_workers=cpu_workers)
    io_pool = ThreadPoolExecutor(max_workers=io_workers)

    def consumer_loop():
        while True:
            item = task_q.get()
            if item is None:
                break
            seg_id, seg_info, raw_points = item

            fut = cpu_pool.submit(
                _process_points_cpu,
                raw_points,
                (cfg["roi"]["x_min"], cfg["roi"]["x_max"]),
                (cfg["roi"]["z_min"], cfg["roi"]["z_max"]),
            )

            def _on_cpu_done(f, sid=seg_id, sinfo=seg_info):
                try:
                    processed = f.result()
                    pending_dir = Path(cfg["paths"]["pending_dir"])
                    meta_path = pending_dir / f"segment_{int(sid):05d}.json"
                    io_pool.submit(_save_and_update, sid, sinfo, processed, pending_dir, meta_path, coverage, cfg)
                except Exception as exc:
                    logger.exception("处理段 %s 失败：%s", sid, exc)

            fut.add_done_callback(_on_cpu_done)
            task_q.task_done()

    consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
    consumer_thread.start()

    try:
        for seg_id, seg_info in targets:
            start_mm = seg_info["start_mm"]
            wait_for_mileage(odometry, start_mm)

            raw = scanner.acquire_point_cloud()
            if raw is None or getattr(raw, "size", 0) == 0:
                logger.warning("段 %s 未获取到有效点云，跳过。", seg_id)
                continue

            # 入队；若队列满则进行同步处理以避免丢失
            try:
                task_q.put((seg_id, seg_info, raw), timeout=5)
            except queue.Full:
                logger.warning("队列已满，直接同步处理段 %s。", seg_id)
                processed = _process_points_cpu(raw, (cfg["roi"]["x_min"], cfg["roi"]["x_max"]), (cfg["roi"]["z_min"], cfg["roi"]["z_max"]))
                pending_dir = Path(cfg["paths"]["pending_dir"])
                meta_path = pending_dir / f"segment_{int(seg_id):05d}.json"
                _save_and_update(seg_id, seg_info, processed, pending_dir, meta_path, coverage, cfg)

            scanned_segments.append(seg_id)

    finally:
        # Shutdown pipeline
        task_q.put(None)
        consumer_thread.join()
        cpu_pool.shutdown(wait=True)
        io_pool.shutdown(wait=True)
        try:
            scanner.disconnect()
        except Exception:
            logger.exception("关闭扫描器时发生异常。")
        odometry.stop()

    return scanned_segments


def detect_trip(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    ensure_paths(cfg)
    coverage = create_coverage_map(cfg)
    detect_fn = create_pointcloud_detection_fn(cfg)
    detector = Detector(cfg, coverage, detect_fn)
    result = detector.run_trip()
    logger.info(
        "检测完成。异常段数=%d，正常段数=%d。",
        len(result["anomaly"]),
        len(result["normal"]),
    )
    return result


def show_status(cfg: Dict[str, Any]) -> None:
    coverage = create_coverage_map(cfg)
    summary = coverage.summary()
    total = coverage.total_segments()
    scanned = summary.get(SegmentState.NORMAL.value, 0) + summary.get(SegmentState.ANOMALY.value, 0)
    pending = summary.get(SegmentState.PENDING.value, 0)
    unscanned = summary.get(SegmentState.UNSCANNED.value, 0)

    print("Coverage status:")
    print(f"  total segments: {total}")
    print(f"  unscanned: {unscanned}")
    print(f"  pending: {pending}")
    print(f"  normal: {summary.get(SegmentState.NORMAL.value, 0)}")
    print(f"  anomaly: {summary.get(SegmentState.ANOMALY.value, 0)}")
    print(f"  scanned complete: {scanned}/{total} ({scanned / total * 100 if total else 0:.1f}%)")


def reset_coverage(cfg: Dict[str, Any]) -> None:
    coverage_map_path = Path(cfg["paths"]["coverage_map"])
    if coverage_map_path.exists():
        coverage_map_path.unlink()
        logger.info("已删除旧覆盖图：%s", coverage_map_path)
    create_coverage_map(cfg)
    logger.info("已重新初始化覆盖图。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轨道检测系统主入口")
    parser.add_argument("command", choices=["scan", "detect", "status", "reset"])
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.exception("配置加载失败。")
        return 1

    setup_logging(cfg)

    try:
        if args.command == "scan":
            scanned = scan_trip(cfg)
            print(f"已完成扫描段: {scanned}")
        elif args.command == "detect":
            result = detect_trip(cfg)
            print(f"异常段: {result['anomaly']}")
            print(f"正常段: {result['normal']}")
        elif args.command == "status":
            show_status(cfg)
        elif args.command == "reset":
            reset_coverage(cfg)
        return 0
    except Exception as exc:
        logger.exception("命令执行失败。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
