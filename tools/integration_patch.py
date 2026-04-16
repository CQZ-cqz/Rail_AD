"""
将 RailMotionCorrector 集成到 IntegratedPointCloudSystem 的方法示例。

修改位置：run_complete_pipeline() 中，load_point_cloud 之后、
          center_point_cloud 之前，插入以下代码段。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import open3d as o3d
from algorithms.RailMotionCorrector import RailMotionCorrector
from algorithms.PointCloud_AnomalyDetection import IntegratedPointCloudSystem

# ──────────────────────────────────────────────
# 1. 参数配置（根据实际数据调整）
# ──────────────────────────────────────────────

CORRECTOR_CONFIG = dict(
    # 切帧厚度：建议 = 扫描线间距 × 1~2
    # 例如 1 mm 间距 → 取 1.0~2.0
    frame_thickness_mm=1.0,

    # 平滑窗口（帧数）：
    #   - 目标：窗口覆盖的 Y 距离 >> 振动波长，同时 << 轨道真实曲率变化距离
    #   - 典型检测车振动：~0.5~3 Hz，车速 ~10 km/h = 2.8 m/s
    #     → 振动波长 ~0.9~5.6 m；取窗口覆盖 ~200~500 mm 通常合适
    #   - 帧数 = 覆盖距离 / frame_thickness_mm，例如 300 mm / 1 mm = 301（奇数）
    smooth_window_frames=301,

    smooth_polyorder=3,

    # 轨头识别百分位：
    #   - 纯钢轨截面（已分割好）：90~93
    #   - 含道碴/地面点的原始截面：95~98
    rail_head_percentile=92.0,

    min_points_per_frame=10,

    # 双轨模式：点云同时包含左右两条钢轨时设为 True
    dual_rail_mode=False,
    gauge_nominal_mm=1435.0,
)

# ──────────────────────────────────────────────
# 2. 在 run_complete_pipeline 中的插入位置
# ──────────────────────────────────────────────

def apply_motion_correction(system, show_diagnosis=True):
    """
    在 system.source_cloud 加载完毕后、中心化之前调用此函数。

    Args:
        system: IntegratedPointCloudSystem 实例（source_cloud 已加载）
        show_diagnosis: 是否弹出诊断图（首次调试时建议开启）
    """
    corrector = RailMotionCorrector(**CORRECTOR_CONFIG)

    print("\n🔧 运动畸变补偿（方案B）...")
    system.source_cloud = corrector.correct_o3d(
        system.source_cloud,
        correct_lateral=True,   # ✅ 必开：消除 X 方向蛇形
        correct_height=False,   # 视情况：高度漂移不显著时关闭
        correct_roll=False,     # 可选：横滚补偿，建议确认 lateral 稳定后再开
    )

    corrector.print_summary()

    if show_diagnosis:
        corrector.plot_diagnosis(show_roll=False)

    return corrector  # 返回供后续分析


# ──────────────────────────────────────────────
# 3. 修改后的 run_complete_pipeline 调用示例
# ──────────────────────────────────────────────

def run_pipeline_with_correction(scan_path, cad_stl_path):
    """展示完整调用顺序（仅展示修改部分，其余不变）"""

    system = IntegratedPointCloudSystem(defect_threshold=0.5)

    # ── 第一阶段：加载（不变）──
    system.source_cloud = system.load_point_cloud(scan_path, downsample_ratio=1.0)
    system.target_cloud = system.load_stl_and_sample_points(cad_stl_path, num_points=1_000_000)

    # ── ★ 新增：运动畸变补偿（在中心化之前）★ ──
    corrector = apply_motion_correction(system, show_diagnosis=True)

    # ── 第二阶段及后续：与原始代码完全一致 ──
    success = system.run_complete_pipeline(
        scan_path=scan_path,
        cad_stl_path=cad_stl_path,
        show_intermediate_results=True,
        save_results=False,
    )
    return success


# ──────────────────────────────────────────────
# 4. 参数调优指引
# ──────────────────────────────────────────────

TUNING_GUIDE = """
调优步骤（建议顺序）：

① 先看诊断图的 "有效帧率"：
   - < 70%：降低 rail_head_percentile（例如 92→88），
             或增大 min_points_per_frame 反向检验点云密度
   - > 95%：正常

② 看 X 修正量曲线是否合理：
   - 修正量 std 应在 0.5~5 mm 量级（车体振动典型范围）
   - 如果 std > 10 mm，可能轨头检测出错，调整 rail_head_percentile
   - 如果修正量几乎为 0，说明点云可能已经很好或检测失败

③ 调整 smooth_window_frames：
   - 诊断图中 "原始轨头X" 与 "S-G平滑" 的曲线应：
       蓝线（原始）有明显高频抖动
       深蓝线（平滑）跟随低频趋势
   - 若平滑线也有高频抖动 → 增大窗口
   - 若平滑线偏离真实轨道走向（在曲线段过度平滑）→ 减小窗口

④ 确认 lateral 补偿效果后，可尝试开启 correct_height=True
   （但高度漂移通常比横向小，有时不必开）

⑤ 横滚补偿（correct_roll=True）精度依赖轨头点数，
   建议 rail_head_percentile ≤ 90 且截面点数充足时才开启
"""

if __name__ == "__main__":
    print(TUNING_GUIDE)
