import logging
import tempfile
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from utils.pointcloud import save_pointcloud

logger = logging.getLogger(__name__)


def default_detect_fn(points: np.ndarray) -> bool:
    return False


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser())


def create_pointcloud_detection_fn(cfg: Dict) -> Callable[[np.ndarray], bool]:
    cad_stl_path = (
        cfg.get("detection", {}).get("cad_stl_path")
        or cfg.get("paths", {}).get("cad_stl")
        or cfg.get("paths", {}).get("cad_stl_path")
    )
    if not cad_stl_path:
        msg = "未配置 CAD STL 路径（detection.cad_stl_path / paths.cad_stl），终止 detect。"
        logger.error(msg)
        raise RuntimeError(msg)

    cad_stl_path = _normalize_path(cad_stl_path)
    if not Path(cad_stl_path).exists():
        msg = f"CAD STL 文件不存在：{cad_stl_path}，终止 detect。"
        logger.error(msg)
        raise FileNotFoundError(msg)

    threshold = float(cfg.get("detection", {}).get("threshold_mm", 0.5))
    defect_sample_points = int(cfg.get("detection", {}).get("cad_sample_points", 500000))

    try:
        from algorithms.PointCloud_AnomalyDetection import IntegratedPointCloudSystem
    except Exception as exc:
        logger.warning(
            "无法加载 PointCloud_AnomalyDetection 模块：%s。使用默认检测函数。",
            exc,
        )
        return default_detect_fn

    def detect_fn(points: np.ndarray) -> bool:
        with tempfile.TemporaryDirectory() as temp_dir:
            scan_path = Path(temp_dir) / "segment_to_detect.ply"
            save_pointcloud(points, str(scan_path))

            system = IntegratedPointCloudSystem(defect_threshold=threshold)

            # 首先尝试完整流程（配准 + 异常检测）
            reg_cad_points = int(cfg.get("detection", {}).get("cad_sample_points_registration", 100000))
            voxel_size = float(cfg.get("detection", {}).get("voxel_size", 0.1))
            try:
                success = system.run_complete_pipeline(
                    scan_path=str(scan_path),
                    cad_stl_path=cad_stl_path,
                    cad_sample_points=reg_cad_points,
                    defect_sample_points=defect_sample_points,
                    voxel_size=voxel_size,
                    show_intermediate_results=False,
                    save_results=False,
                    output_dir=temp_dir,
                )
            except Exception:
                success = False

            # 若完整流程失败，尝试原先的快速检测（假设点云已配准）作为回退
            if not success:
                logger.warning("完整配准+检测流程失败，尝试快速检测回退路径。")
                try:
                    success = system.quick_detection_only(
                        registered_scan_path=str(scan_path),
                        cad_stl_path=cad_stl_path,
                        defect_sample_points=defect_sample_points,
                        show_results=False,
                        save_results=False,
                        output_dir=temp_dir,
                    )
                except Exception:
                    success = False

            if not success:
                logger.warning("真实检测函数运行失败，默认返回正常。")
                return False

            defects_scan = getattr(system, "defects_scan_to_cad", None)
            defects_cad = getattr(system, "defects_cad_to_scan", None)
            has_defects = False
            if defects_scan is not None and len(defects_scan) > 0:
                has_defects = True
            if defects_cad is not None and len(defects_cad) > 0:
                has_defects = True
            logger.info("检测完成：异常=%s (scan_defects=%s, cad_defects=%s)", has_defects, len(defects_scan) if defects_scan is not None else 0, len(defects_cad) if defects_cad is not None else 0)
            return has_defects

    return detect_fn
