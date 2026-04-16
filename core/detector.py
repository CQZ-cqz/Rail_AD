import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from utils.file_manager import (
    SegmentMeta,
    delete_segment,
    load_meta,
    move_to_anomaly,
    save_meta,
)
from utils.pointcloud import load_pointcloud, save_pointcloud
import csv
from datetime import datetime, timezone

from core.coverage import CoverageMap, SegmentState
from algorithms.corrector_discriminator import AssessResult, CorrectionDiscriminator
from algorithms.RailMotionCorrector import RailMotionCorrector


logger = logging.getLogger(__name__)


class Detector:
    """返程检测任务，处理 pending 段并分发正常/异常结果。"""

    def __init__(
        self,
        cfg: Dict,
        coverage: CoverageMap,
        detect_fn: Callable[[object], bool],
    ):
        self._cfg = cfg
        self._coverage = coverage
        self._detect_fn = detect_fn
        self._anomaly_dir = Path(cfg["paths"]["anomaly_dir"])
        self._anomaly_dir.mkdir(parents=True, exist_ok=True)
        correction_cfg = cfg.get("correction", {})
        self._correction_enabled = correction_cfg.get("enabled", False)

        if self._correction_enabled:
            self._discriminator  = CorrectionDiscriminator(cfg)
            corr_cfg             = correction_cfg.get("corrector", {})
            self._rail_corrector = RailMotionCorrector(
                frame_thickness_mm   = float(corr_cfg.get("frame_thickness_mm", 1.0)),
                smooth_window_frames = int(corr_cfg.get("smooth_window_frames", 301)),
                smooth_polyorder     = int(corr_cfg.get("smooth_polyorder", 3)),
                rail_head_percentile = float(corr_cfg.get("rail_head_percentile", 92.0)),
                min_points_per_frame = int(corr_cfg.get("min_points_per_frame", 10)),
                dual_rail_mode       = bool(corr_cfg.get("dual_rail_mode", False)),
                gauge_nominal_mm     = float(corr_cfg.get("gauge_nominal_mm", 1435.0)),
            )
            self._correct_lateral = bool(corr_cfg.get("correct_lateral", True))
            self._correct_height  = bool(corr_cfg.get("correct_height", False))
            self._correct_roll    = bool(corr_cfg.get("correct_roll", False))
        else:
            self._discriminator  = None
            self._rail_corrector = None

        self._corrected_dir = Path(cfg["paths"].get("corrected_dir", "data/corrected"))
        self._corrected_dir.mkdir(parents=True, exist_ok=True)
        self._alert_log_path = Path(cfg["paths"].get("alert_log", "log/alert_log.csv"))
        self._alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self, seg_info: Dict) -> Optional[SegmentMeta]:
        meta_file = seg_info.get("meta_file")
        if not meta_file:
            return None
        try:
            return load_meta(meta_file)
        except Exception as exc:
            logger.warning("加载元数据失败：%s (%s)", meta_file, exc)
            return None

    def run_trip(self) -> Dict[str, List[str]]:
        pending = self._coverage.get_pending()
        anomalies: List[str] = []
        normals: List[str] = []

        if not pending:
            logger.info("当前没有待检测的 pending 段。")
            return {"anomaly": anomalies, "normal": normals}

        for seg_id, seg_info in pending:
            point_cloud_path = seg_info.get("point_cloud_file")
            if not point_cloud_path:
                logger.warning("段 %s 缺少 point_cloud_file，已跳过。", seg_id)
                continue

            try:
                points = load_pointcloud(point_cloud_path)
            except Exception as exc:
                logger.warning("加载点云文件失败：%s (%s)", point_cloud_path, exc)
                continue

            if points.size == 0:
                logger.warning("段 %s 的点云为空，已跳过。", seg_id)
                continue

            # 矫正判别器
            if self._correction_enabled:
                assess_result, diag = self._discriminator.assess(points)

                if assess_result == AssessResult.RESCAN_REQUIRED:
                    logger.warning(
                        "段 %s 扫描质量不合格 (valid_frame_ratio=%.3f)，回退为 UNSCANNED。",
                        seg_id, diag.get("valid_frame_ratio", 0.0),
                    )
                    self._write_alert_log(seg_id, "low_quality", diag)
                    # 删除临时文件
                    segment_meta = self._load_metadata(seg_info)
                    if segment_meta is not None:
                        try:
                            delete_segment(segment_meta)
                        except Exception as exc:
                            logger.warning("删除段 %s 文件失败：%s", seg_id, exc)
                    else:
                        p = Path(point_cloud_path)
                        if p.exists():
                            p.unlink()
                    # 回退覆盖图状态
                    self._coverage.update(
                        seg_id,
                        state=SegmentState.UNSCANNED,
                        scan_time=None,
                        point_cloud_file=None,
                        meta_file=None,
                    )
                    continue

                elif assess_result == AssessResult.APPLY_CORRECTION:
                    logger.info(
                        "段 %s 执行矫正 (dx_std=%.3f mm)。", seg_id, diag.get("dx_std", 0.0)
                    )
                    corrected_points = self._rail_corrector.correct(
                        points,
                        correct_lateral=self._correct_lateral,
                        correct_height=self._correct_height,
                        correct_roll=self._correct_roll,
                    )
                    corrected_path = self._corrected_dir / f"segment_{int(seg_id):05d}_corrected.ply"
                    save_pointcloud(corrected_points, str(corrected_path))
                    # 更新 meta 文件中的 corrected_pointcloud_path
                    segment_meta = self._load_metadata(seg_info)
                    if segment_meta is not None:
                        segment_meta.corrected_pointcloud_path = str(corrected_path)
                        save_meta(segment_meta, segment_meta.meta_file)
                        self._coverage.update(seg_id, corrected_pointcloud_file=str(corrected_path))
                    points = corrected_points

                else:
                    logger.info(
                        "段 %s 跳过矫正 (dx_std=%.3f mm)。", seg_id, diag.get("dx_std", 0.0)
                    )
            # 矫正判别结束

            is_anomaly = bool(self._detect_fn(points))
            detect_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            segment_meta = self._load_metadata(seg_info)

            if is_anomaly:
                if segment_meta is not None:
                    segment_meta = move_to_anomaly(segment_meta, self._anomaly_dir)
                    save_meta(segment_meta, segment_meta.meta_file)
                    self._coverage.update(
                        seg_id,
                        state=SegmentState.ANOMALY,
                        detect_time=detect_time,
                        point_cloud_file=segment_meta.pointcloud_path,
                        meta_file=segment_meta.meta_file,
                    )
                else:
                    source_path = Path(point_cloud_path)
                    destination = self._anomaly_dir / source_path.name
                    if source_path.exists():
                        source_path.replace(destination)
                    self._coverage.update(
                        seg_id,
                        state=SegmentState.ANOMALY,
                        detect_time=detect_time,
                        point_cloud_file=str(destination),
                        meta_file=None,
                    )
                anomalies.append(seg_id)
                logger.info("段 %s 检测为异常，已移动到 anomaly 目录。", seg_id)
            else:
                if segment_meta is not None:
                    try:
                        delete_segment(segment_meta)
                    except Exception as exc:
                        logger.warning("删除段 %s 文件失败：%s", seg_id, exc)
                else:
                    source_path = Path(point_cloud_path)
                    if source_path.exists():
                        source_path.unlink()
                self._coverage.update(
                    seg_id,
                    state=SegmentState.NORMAL,
                    detect_time=detect_time,
                    point_cloud_file=None,
                    meta_file=None,
                )
                normals.append(seg_id)
                logger.info("段 %s 检测为正常，已删除临时点云。", seg_id)

        return {"anomaly": anomalies, "normal": normals}
    
    def _write_alert_log(self, seg_id: str, reason: str, diag: dict) -> None:
        """Append one row to the alert log CSV."""
        fieldnames = ["timestamp", "seg_id", "reason", "valid_frame_ratio", "dx_std"]
        row = {
            "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seg_id":             seg_id,
            "reason":             reason,
            "valid_frame_ratio":  diag.get("valid_frame_ratio", ""),
            "dx_std":             diag.get("dx_std", ""),
        }
        write_header = not self._alert_log_path.exists()
        with self._alert_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    print("Detector module does not execute standalone. Please call it from main.py.")
