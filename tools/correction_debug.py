"""
correction_debug.py
====================
矫正调试脚本 —— 仅用于 RailMotionCorrector 的参数调优与效果验证。

依赖：
  - PointCloud_AnomalyDetection.py   （复用 load_point_cloud / preprocess_point_cloud）
  - RailMotionCorrector.py           （核心矫正模块）

用法：
  1. 修改底部 CONFIG 中的文件路径与参数
  2. 直接运行：python correction_debug.py
  3. 根据输出的诊断图调整参数，直到满意为止
  4. 将确认好的参数填入主流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from pathlib import Path

# ── 复用已有模块 ─────────────────────────────────────────────
from algorithms.PointCloud_AnomalyDetection import IntegratedPointCloudSystem
from algorithms.RailMotionCorrector import RailMotionCorrector
from core.config import load_config as _load_config

_cfg = _load_config("config.yaml")
_corr = _cfg.get("correction", {}).get("corrector", {})

# 按优先级自动选择可用的中文字体
def set_chinese_font():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
            print(f"[字体] 使用: {font}")
            return
    print("[字体] ⚠️ 未找到中文字体，文字可能显示为方块")

set_chinese_font()
# ════════════════════════════════════════════════════════════
#  CONFIG — 调试时只需修改这里
# ════════════════════════════════════════════════════════════

CONFIG = dict(
    scan_path               = _corr.get("debug_scan_path", 
                                        "data/pending/segment_00000.ply"),       # 调试用点云文件路径
    downsample_ratio        = 1.0,                                               # 加载时的下采样率（调试时建议保持原始分辨率）
    preprocess_voxel_size   = None,                                              # 预处理阶段的体素滤波大小（mm），None=不使用
    frame_thickness_mm      = _corr.get("frame_thickness_mm", 1.0),              # 切帧厚度，建议 = 扫描线间距 × 1~2
    smooth_window_frames    = _corr.get("smooth_window_frames", 5001.0),         # S-G 平滑窗口（帧数），建议覆盖至少一个振动周期，且为奇数
    smooth_polyorder        = _corr.get("smooth_polyorder", 3),                  # S-G 平滑多项式阶数
    rail_head_percentile    = _corr.get("rail_head_percentile", 0.0),            # 轨头高度识别百分位
    min_points_per_frame    = _corr.get("min_points_per_frame", 10),             # 每帧最小点数
    dual_rail_mode          = _corr.get("dual_rail_mode", False),                # 双轨模式
    gauge_nominal_mm        = _corr.get("gauge_nominal_mm", 1435.0),             # 标准轨距（mm）
    correct_lateral         = _corr.get("correct_lateral", True),                # 是否矫正横向（X）畸变，建议必开
    correct_height          = _corr.get("correct_height", False),                # 是否矫正高度（Z）畸变，视情况开启
    correct_roll            = _corr.get("correct_roll", False),                  # 是否矫正横滚（点云绕 Y 轴的旋转），建议确认 lateral 稳定后再开
    show_o3d_before_after       = True,                                          # 是否显示 Open3D 前后对比窗口
    show_cross_section_compare  = True,                                          # 是否显示截面轮廓对比图
    cross_section_y_positions   = None,                                          # 截面对比的 Y 位置列表，None=自动选取头/中/尾三个位置
    save_corrected_cloud        = False,        
    output_path             = "data/clouds/corrected_debug.ply",
)

# ════════════════════════════════════════════════════════════


def load_and_preprocess(cfg: dict) -> o3d.geometry.PointCloud:
    """复用 IntegratedPointCloudSystem 的加载和预处理方法。"""
    helper = IntegratedPointCloudSystem()

    pcd = helper.load_point_cloud(cfg["scan_path"], downsample_ratio=cfg["downsample_ratio"])
    if pcd is None:
        sys.exit("❌ 点云加载失败，请检查路径")

    if cfg["preprocess_voxel_size"] is not None:
        print(f"[预处理] 体素滤波 voxel_size={cfg['preprocess_voxel_size']} mm ...")
        pcd = helper.preprocess_point_cloud(pcd, voxel_size=cfg["preprocess_voxel_size"])

    return pcd


def build_corrector(cfg: dict) -> RailMotionCorrector:
    return RailMotionCorrector(
        frame_thickness_mm=cfg["frame_thickness_mm"],
        smooth_window_frames=cfg["smooth_window_frames"],
        smooth_polyorder=cfg["smooth_polyorder"],
        rail_head_percentile=cfg["rail_head_percentile"],
        min_points_per_frame=cfg["min_points_per_frame"],
        dual_rail_mode=cfg["dual_rail_mode"],
        gauge_nominal_mm=cfg["gauge_nominal_mm"],
    )


# ────────────────────────────────────────────────────────────
#  可视化工具
# ────────────────────────────────────────────────────────────

def show_o3d_before_after(pcd_before: o3d.geometry.PointCloud,
                          pcd_after: o3d.geometry.PointCloud):
    """
    在两个独立的 Open3D 窗口中分别显示矫正前后的点云。
    颜色：矫正前=蓝色，矫正后=绿色。
    """
    before = copy.deepcopy(pcd_before)
    after = copy.deepcopy(pcd_after)
    before.paint_uniform_color([0.3, 0.5, 1.0])   # 蓝
    after.paint_uniform_color([0.2, 0.8, 0.3])    # 绿

    print("\n[Open3D] 显示矫正前点云（蓝色）→ 关闭窗口后显示矫正后 ...")
    o3d.visualization.draw_geometries(
        [before, o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        window_name="矫正前（蓝色）", width=1280, height=720
    )
    print("[Open3D] 显示矫正后点云（绿色）→ 关闭窗口继续 ...")
    o3d.visualization.draw_geometries(
        [after, o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)],
        window_name="矫正后（绿色）", width=1280, height=720
    )


def extract_cross_section(points: np.ndarray, y_pos: float,
                           half_width: float = 2.0) -> np.ndarray:
    """提取 Y ≈ y_pos 附近 ±half_width mm 的 XZ 截面点。"""
    mask = np.abs(points[:, 1] - y_pos) < half_width
    return points[mask][:, [0, 2]]   # 返回 (N, 2) XZ


def plot_cross_section_compare(pts_before: np.ndarray, pts_after: np.ndarray,
                                y_positions: list, half_width: float = 2.0):
    """
    在同一画布上对比若干 Y 位置处矫正前后的截面轮廓（XZ 平面）。
    蓝点=矫正前，绿点=矫正后。
    """
    n = len(y_positions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("截面轮廓对比（XZ 平面）  蓝=矫正前  绿=矫正后",
                 fontsize=13, fontweight='bold')

    for ax, y in zip(axes, y_positions):
        xz_b = extract_cross_section(pts_before, y, half_width)
        xz_a = extract_cross_section(pts_after, y, half_width)

        if len(xz_b) < 3:
            ax.set_title(f"Y={y:.0f} mm\n(点数不足)")
            continue

        ax.scatter(xz_b[:, 0], xz_b[:, 1], s=4, alpha=0.5,
                   color='steelblue', label='矫正前')
        ax.scatter(xz_a[:, 0], xz_a[:, 1], s=4, alpha=0.5,
                   color='limegreen', label='矫正后')

        # 标注轨头中心偏移量
        x_shift = np.mean(xz_a[:, 0]) - np.mean(xz_b[:, 0])
        z_shift = np.mean(xz_a[:, 1]) - np.mean(xz_b[:, 1])
        ax.set_title(f"Y = {y:.0f} mm\nΔX={x_shift:+.3f} mm  ΔZ={z_shift:+.3f} mm",
                     fontsize=9)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z (mm)")
        ax.set_aspect('equal')
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_point_density_along_y(points: np.ndarray, frame_thickness: float):
    """检查 Y 方向点密度分布，辅助判断 frame_thickness 是否合适。"""
    y = points[:, 1]
    y_min, y_max = y.min(), y.max()
    bins = int((y_max - y_min) / frame_thickness) + 1
    counts, edges = np.histogram(y, bins=bins)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(edges[:-1], counts, width=frame_thickness * 0.9,
           align='edge', color='steelblue', alpha=0.7)
    ax.axhline(np.mean(counts), color='red', linestyle='--',
               linewidth=1, label=f'均值 {np.mean(counts):.0f} pts/帧')
    ax.set_xlabel("Y 方向位置 (mm)")
    ax.set_ylabel("每帧点数")
    ax.set_title(f"Y 方向点密度分布  |  frame_thickness={frame_thickness} mm  |  "
                 f"总帧数估计={bins}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_extended_diagnosis(corrector: RailMotionCorrector):
    """
    扩展诊断图：在标准诊断图基础上增加修正量频谱分析（FFT），
    帮助判断 smooth_window 是否把振动频率覆盖住了。
    """
    info = corrector.correction_info
    if not info:
        return

    dx = info['dx_correction']
    y = info['frame_y']
    dt = np.mean(np.diff(y)) if len(y) > 1 else 1.0

    # FFT
    n = len(dx)
    freqs = np.fft.rfftfreq(n, d=dt)          # 单位：1/mm
    amplitudes = np.abs(np.fft.rfft(dx)) * 2 / n

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    fig.suptitle("扩展诊断：修正量分析", fontsize=13, fontweight='bold')

    # 1. X 修正量时域
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y, info['x_raw'], alpha=0.35, color='steelblue',
             linewidth=0.8, label='原始轨头 X')
    ax1.plot(y, info['x_smooth'], linewidth=1.8, color='navy',
             label=f"S-G 平滑（窗口={corrector.smooth_window} 帧）")
    ax1.fill_between(y, info['x_raw'], info['x_smooth'],
                     alpha=0.2, color='crimson', label='修正量')
    invalid_y = y[~info['valid_mask']]
    if len(invalid_y):
        for iy in invalid_y:
            ax1.axvline(iy, color='orange', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel("X 位置 (mm)")
    ax1.set_xlabel("行进方向 Y (mm)")
    ax1.set_title("横向轨头中心轨迹  （橙色竖线 = 插值帧）")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. X 修正量频谱
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(freqs[1:] * 1000, amplitudes[1:], color='crimson', linewidth=0.8)
    # 标注平滑窗口对应的截止频率
    cutoff_freq_per_mm = 1.0 / (corrector.smooth_window * dt)
    ax2.axvline(cutoff_freq_per_mm * 1000, color='navy', linestyle='--',
                linewidth=1.5,
                label=f'S-G 截止频率 ≈ {cutoff_freq_per_mm*1000:.2f} /m')
    ax2.set_xlabel("空间频率 (1/m)")
    ax2.set_ylabel("幅值 (mm)")
    ax2.set_title("X 修正量频谱（FFT）")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    # 3. 修正量累积分布
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_dx = np.sort(np.abs(dx))
    cdf = np.arange(1, len(sorted_dx) + 1) / len(sorted_dx)
    ax3.plot(sorted_dx, cdf * 100, color='steelblue', linewidth=1.5)
    for pct in [50, 90, 99]:
        val = np.percentile(np.abs(dx), pct)
        ax3.axvline(val, linestyle=':', linewidth=1, color='gray')
        ax3.text(val, pct, f" P{pct}={val:.2f}mm", fontsize=7, color='gray',
                 va='bottom')
    ax3.set_xlabel("|修正量| (mm)")
    ax3.set_ylabel("累积百分比 (%)")
    ax3.set_title("X 修正量绝对值 CDF")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
#  主调试流程
# ────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    print("=" * 60)
    print("  钢轨点云运动畸变矫正调试脚本")
    print("=" * 60)
    print(f"  数据文件  : {cfg['scan_path']}")
    print(f"  帧厚度    : {cfg['frame_thickness_mm']} mm")
    print(f"  平滑窗口  : {cfg['smooth_window_frames']} 帧")
    print(f"  补偿项    : lateral={cfg['correct_lateral']}, "
          f"height={cfg['correct_height']}, roll={cfg['correct_roll']}")
    print("=" * 60)

    # ── 1. 加载点云（复用 IntegratedPointCloudSystem）──────
    print("\n[1/5] 加载点云 ...")
    pcd_original = load_and_preprocess(cfg)
    pts_original = np.asarray(pcd_original.points)
    print(f"  点云范围 X: [{pts_original[:,0].min():.1f}, {pts_original[:,0].max():.1f}] mm")
    print(f"  点云范围 Y: [{pts_original[:,1].min():.1f}, {pts_original[:,1].max():.1f}] mm")
    print(f"  点云范围 Z: [{pts_original[:,2].min():.1f}, {pts_original[:,2].max():.1f}] mm")

    # ── 2. 点密度预检（帮助选择合适的 frame_thickness）────
    print("\n[2/5] Y 方向点密度预检 ...")
    plot_point_density_along_y(pts_original, cfg["frame_thickness_mm"])

    # ── 3. 运行矫正 ────────────────────────────────────────
    print("\n[3/5] 运行矫正 ...")
    corrector = build_corrector(cfg)
    pcd_corrected = corrector.correct_o3d(
        pcd_original,
        correct_lateral=cfg["correct_lateral"],
        correct_height=cfg["correct_height"],
        correct_roll=cfg["correct_roll"],
    )
    corrector.print_summary()

    # ── 4. 诊断图 ──────────────────────────────────────────
    print("\n[4/5] 绘制诊断图 ...")

    # 标准诊断图（S-G 平滑效果）
    corrector.plot_diagnosis(show_roll=cfg["correct_roll"])

    # 扩展诊断图（频谱 + CDF）
    plot_extended_diagnosis(corrector)

    # ── 5. 矫正效果对比 ────────────────────────────────────
    print("\n[5/5] 矫正效果可视化 ...")
    pts_corrected = np.asarray(pcd_corrected.points)

    # 截面轮廓对比
    if cfg["show_cross_section_compare"]:
        y_range = pts_original[:, 1]
        if cfg["cross_section_y_positions"] is None:
            # 自动选头、中、尾三个位置
            y_lo, y_hi = y_range.min(), y_range.max()
            y_positions = [
                y_lo + (y_hi - y_lo) * 0.15,
                y_lo + (y_hi - y_lo) * 0.50,
                y_lo + (y_hi - y_lo) * 0.85,
            ]
        else:
            y_positions = cfg["cross_section_y_positions"]
        plot_cross_section_compare(pts_original, pts_corrected, y_positions)

    # Open3D 前后对比窗口
    if cfg["show_o3d_before_after"]:
        show_o3d_before_after(pcd_original, pcd_corrected)

    # ── 保存矫正后点云（可选）──────────────────────────────
    if cfg["save_corrected_cloud"]:
        out_path = Path(cfg["output_path"])
        o3d.io.write_point_cloud(str(out_path), pcd_corrected)
        print(f"\n✅ 矫正后点云已保存: {out_path}")

    print("\n✅ 调试完成。")
    print("   → 满意后将 CONFIG 中的参数复制到主流程的 CORRECTOR_CONFIG 中。")


if __name__ == "__main__":
    main()
