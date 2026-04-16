import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
#  坐标系约定
#  Y 轴 —— 行进方向（扫描叠加方向）
#  X 轴 —— 横向（蛇形畸变主要体现在此）
#  Z 轴 —— 高度（轨头顶部在 Z 方向最高）
# ============================================================


class RailProfileExtractor:
    """
    从单帧 XZ 截面点云中提取钢轨几何特征。

    支持单轨 / 双轨模式，以及基于高度阈值的简单轨头检测。
    """

    def __init__(self,
                 rail_head_percentile: float = 92.0,
                 min_points_per_frame: int = 10):
        """
        Args:
            rail_head_percentile: Z 高度百分位，高于此值的点视为轨头区域。
                                  单轨截面建议 90~95；含道碴/轨枕时可适当提高到 95~98。
            min_points_per_frame: 帧内有效点数下限，低于此值跳过该帧。
        """
        self.rail_head_percentile = rail_head_percentile
        self.min_points_per_frame = min_points_per_frame

    def extract_single_rail(self, xz_points: np.ndarray):
        """
        提取单条钢轨的轨头中心（X 重心, Z 高度均值）以及横滚角估计。

        Args:
            xz_points: shape (N, 2)，列顺序 [X, Z]

        Returns:
            dict with keys: x_center, z_center, roll_rad
            若点数不足返回 None
        """
        if len(xz_points) < self.min_points_per_frame:
            return None

        z_vals = xz_points[:, 1]
        z_thresh = np.percentile(z_vals, self.rail_head_percentile)
        head_mask = z_vals >= z_thresh

        if head_mask.sum() < 3:
            return None

        head_pts = xz_points[head_mask]
        x_center = np.mean(head_pts[:, 0])
        z_center = np.mean(head_pts[:, 1])

        # ---- 横滚角：轨头顶部区域的 PCA 主轴相对水平的偏角 ----
        if head_mask.sum() >= 5:
            centered = head_pts - head_pts.mean(axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_dir = eigvecs[:, np.argmax(eigvals)]   # [dx, dz]
            # 理想水平时 main_dir ≈ (1, 0)，偏角即横滚
            roll_rad = np.arctan2(main_dir[1], main_dir[0])
        else:
            roll_rad = 0.0

        return dict(x_center=x_center, z_center=z_center, roll_rad=roll_rad)

    def extract_dual_rail(self, xz_points: np.ndarray, gauge_nominal_mm: float = 1435.0):
        """
        双轨模式：将截面按 X 方向分成左右两半，分别提取轨头，
        返回两条轨道的中心及轨距偏差。

        Args:
            xz_points: shape (N, 2)，列顺序 [X, Z]
            gauge_nominal_mm: 标准轨距（mm），用于合理性校验

        Returns:
            dict with left_rail, right_rail (各含 x_center/z_center/roll_rad),
                  gauge_measured
            若检测失败返回 None
        """
        if len(xz_points) < self.min_points_per_frame * 2:
            return None

        x_median = np.median(xz_points[:, 0])
        left_mask = xz_points[:, 0] < x_median
        right_mask = ~left_mask

        left = self.extract_single_rail(xz_points[left_mask])
        right = self.extract_single_rail(xz_points[right_mask])

        if left is None or right is None:
            return None

        gauge = abs(right['x_center'] - left['x_center'])

        # 合理性检查：超出标准轨距 ±100 mm 则认为分割失败
        if abs(gauge - gauge_nominal_mm) > 100:
            return None

        return dict(left_rail=left, right_rail=right, gauge_measured=gauge)


# ============================================================


class RailMotionCorrector:
    """
    基于钢轨截面特征的逐帧运动畸变补偿。

    处理的畸变类型：
      1. 横向位移（X 方向蛇形波动）   —— correct_lateral=True
      2. 高度漂移（Z 方向系统偏移）    —— correct_height=True
      3. 横滚角（截面绕 Y 轴旋转）     —— correct_roll=True

    坐标系：Y = 行进方向，XZ = 截面平面
    """

    def __init__(self,
                 frame_thickness_mm: float = 1.0,
                 smooth_window_frames: int = 101,
                 smooth_polyorder: int = 3,
                 rail_head_percentile: float = 92.0,
                 min_points_per_frame: int = 10,
                 dual_rail_mode: bool = False,
                 gauge_nominal_mm: float = 1435.0):
        """
        Args:
            frame_thickness_mm:    切帧厚度（Y 方向，mm），建议与扫描线间距一致或稍大。
            smooth_window_frames:  Savitzky-Golay 平滑窗口（帧数，必须为奇数）。
                                   经验值：覆盖约 50~200 mm 的行进距离。
                                   窗口越大越平滑，但会抹掉真实的轨道曲线变化，需根据
                                   振动频率与真实轨道曲率权衡。
            smooth_polyorder:      S-G 多项式阶数，3 阶适合大多数场景。
            rail_head_percentile:  轨头高度识别百分位（见 RailProfileExtractor）。
            min_points_per_frame:  最小有效点数。
            dual_rail_mode:        True = 同时处理左右两条钢轨（点云包含双轨时使用）。
            gauge_nominal_mm:      标准轨距，仅双轨模式使用。
        """
        # 确保平滑窗口为奇数
        if smooth_window_frames % 2 == 0:
            smooth_window_frames += 1

        self.frame_thickness = frame_thickness_mm
        self.smooth_window = smooth_window_frames
        self.smooth_polyorder = smooth_polyorder
        self.dual_rail = dual_rail_mode
        self.gauge_nominal = gauge_nominal_mm

        self.extractor = RailProfileExtractor(
            rail_head_percentile=rail_head_percentile,
            min_points_per_frame=min_points_per_frame
        )

        # 供外部访问的中间结果
        self.correction_info = {}

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------

    def _slice_frames(self, points: np.ndarray):
        """
        沿 Y 轴切帧。
        Returns: list of (y_center, indices_in_points)
        """
        y = points[:, 1]
        y_min, y_max = y.min(), y.max()
        # 传统切帧方法：每隔 frame_thickness 切一帧，查找在中心附近的点
        # frames = []
        # y_cur = y_min + self.frame_thickness / 2.0
        # while y_cur <= y_max:
        #     half = self.frame_thickness / 2.0
        #     idx = np.where(np.abs(y - y_cur) < half)[0]
        #     if len(idx) >= self.extractor.min_points_per_frame:
        #         frames.append((y_cur, idx))
        #     y_cur += self.frame_thickness

        # 向量化切帧：先生成所有帧中心，然后用 np.digitize 一次性分配点到帧
        half = self.frame_thickness / 2.0
        # 一次性生成所有帧中心
        frame_centers = np.arange(y_min + half, y_max, self.frame_thickness)

        # 用 digitize 把每个点分配到最近的帧 bin，只扫一遍
        bin_edges = np.arange(y_min, y_max + self.frame_thickness, self.frame_thickness)
        bin_ids = np.digitize(y, bin_edges) - 1      # 每个点属于第几个 bin
        bin_ids = np.clip(bin_ids, 0, len(frame_centers) - 1)

        frames = []
        for i, y_center in enumerate(frame_centers):
            idx = np.where(bin_ids == i)[0]           # 只是在小数组上查找
            if len(idx) >= self.extractor.min_points_per_frame:
                frames.append((y_center, idx))
        return frames

    @staticmethod
    def _interpolate_gaps(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """对无效帧（检测失败）用线性插值填充，保证 S-G 滤波连续。"""
        if valid_mask.all():
            return values
        x_all = np.arange(len(values))
        x_valid = x_all[valid_mask]
        v_valid = values[valid_mask]
        return np.interp(x_all, x_valid, v_valid)

    def _savgol(self, signal: np.ndarray) -> np.ndarray:
        """自适应窗口的 Savitzky-Golay 平滑（防止帧数不足时报错）。"""
        n = len(signal)
        window = min(self.smooth_window, n if n % 2 == 1 else n - 1)
        window = max(window, self.smooth_polyorder + 2)
        if window % 2 == 0:
            window -= 1
        return savgol_filter(signal, window, self.smooth_polyorder)

    # ----------------------------------------------------------
    # 主接口
    # ----------------------------------------------------------

    def correct(self,
                points: np.ndarray,
                correct_lateral: bool = True,
                correct_height: bool = False,
                correct_roll: bool = False) -> np.ndarray:
        """
        对整段点云执行运动畸变补偿。

        Args:
            points:           输入点云，shape (N, 3)，列顺序 [X, Y, Z]
            correct_lateral:  补偿 X 方向横向位移（蛇形）
            correct_height:   补偿 Z 方向高度系统漂移
            correct_roll:     补偿截面横滚角（绕 Y 轴旋转）

        Returns:
            corrected_points: shape (N, 3)，与输入等大
        """
        print(f"\n{'='*60}")
        print(f"[RailMotionCorrector] 开始畸变补偿")
        print(f"  总点数       : {len(points)}")
        print(f"  帧厚度       : {self.frame_thickness} mm")
        print(f"  平滑窗口     : {self.smooth_window} 帧")
        print(f"  补偿项       : lateral={correct_lateral}, "
              f"height={correct_height}, roll={correct_roll}")
        print(f"{'='*60}")

        # ---------- Step 1: 切帧 ----------
        frames = self._slice_frames(points)
        n_frames = len(frames)
        print(f"[Step 1] 切帧完成，共 {n_frames} 帧")
        if n_frames < 10:
            print("  ⚠️  帧数过少，建议减小 frame_thickness 或检查点云 Y 范围")

        # ---------- Step 2: 逐帧提取轨头特征 ----------
        frame_y = np.zeros(n_frames)
        x_raw = np.zeros(n_frames)
        z_raw = np.zeros(n_frames)
        roll_raw = np.zeros(n_frames)
        valid = np.zeros(n_frames, dtype=bool)

        for i, (y_center, idx) in enumerate(frames):
            frame_y[i] = y_center
            xz = points[idx][:, [0, 2]]   # 取 X 和 Z 列

            if self.dual_rail:
                result = self.extractor.extract_dual_rail(xz, self.gauge_nominal)
                if result is not None:
                    # 取左右轨道中心的均值作为"车体横向位置"
                    x_raw[i] = (result['left_rail']['x_center'] +
                                result['right_rail']['x_center']) / 2.0
                    z_raw[i] = (result['left_rail']['z_center'] +
                                result['right_rail']['z_center']) / 2.0
                    roll_raw[i] = (result['left_rail']['roll_rad'] +
                                   result['right_rail']['roll_rad']) / 2.0
                    valid[i] = True
            else:
                result = self.extractor.extract_single_rail(xz)
                if result is not None:
                    x_raw[i] = result['x_center']
                    z_raw[i] = result['z_center']
                    roll_raw[i] = result['roll_rad']
                    valid[i] = True

        valid_ratio = valid.mean() * 100
        print(f"[Step 2] 特征提取完成，有效帧率 {valid_ratio:.1f}%")
        if valid_ratio < 50:
            print("  ⚠️  有效帧率过低，请检查 rail_head_percentile 参数或点云质量")

        # 插值填充无效帧
        x_raw = self._interpolate_gaps(x_raw, valid)
        z_raw = self._interpolate_gaps(z_raw, valid)
        roll_raw = self._interpolate_gaps(roll_raw, valid)

        # ---------- Step 3: Savitzky-Golay 平滑 → 分离真实轨道与扫描偏差 ----------
        x_smooth = self._savgol(x_raw)
        z_smooth = self._savgol(z_raw)
        roll_smooth = self._savgol(roll_raw)

        # 偏差 = 原始 - 平滑（即由车体振动引入的误差）
        dx_correction = x_raw - x_smooth    # 需要减掉的横向偏差
        dz_correction = z_raw - z_smooth    # 需要减掉的高度偏差
        droll_correction = roll_raw - roll_smooth

        print(f"[Step 3] 平滑完成")
        print(f"  横向偏差 (X): max={np.max(np.abs(dx_correction)):.3f} mm, "
              f"std={np.std(dx_correction):.3f} mm")
        print(f"  高度偏差 (Z): max={np.max(np.abs(dz_correction)):.3f} mm, "
              f"std={np.std(dz_correction):.3f} mm")
        print(f"  横滚偏差    : max={np.degrees(np.max(np.abs(droll_correction))):.3f}°, "
              f"std={np.degrees(np.std(droll_correction)):.3f}°")

        # 保存中间结果（必须在 Step 4 之前，Step 4 的插值要读这里）
        self.correction_info = dict(
            frame_y=frame_y,
            x_raw=x_raw, x_smooth=x_smooth, dx_correction=dx_correction,
            z_raw=z_raw, z_smooth=z_smooth, dz_correction=dz_correction,
            roll_raw=roll_raw, roll_smooth=roll_smooth, droll_correction=droll_correction,
            valid_mask=valid
        )

        # ---------- Step 4: 按 Y 插值修正量，全量施加（无盲区）----------
        # 用 np.interp 把帧级修正量插值到每一个点的 Y 坐标，
        # 超出 frame_y 范围的点自动用边界值填充（clamp），不产生 NaN。
        corrected = points.copy()
        all_y = points[:, 1]

        if correct_lateral:
            dx_per_point = np.interp(all_y, frame_y, dx_correction)
            corrected[:, 0] -= dx_per_point

        if correct_height:
            dz_per_point = np.interp(all_y, frame_y, dz_correction)
            corrected[:, 2] -= dz_per_point

        if correct_roll:
            droll_per_point = np.interp(all_y, frame_y, droll_correction)
            cx_per_point    = np.interp(all_y, frame_y, x_smooth)
            cz_per_point    = np.interp(all_y, frame_y, z_smooth)
            angles = -droll_per_point
            cos_a  = np.cos(angles)
            sin_a  = np.sin(angles)
            x_rel  = corrected[:, 0] - cx_per_point
            z_rel  = corrected[:, 2] - cz_per_point
            corrected[:, 0] = cos_a * x_rel - sin_a * z_rel + cx_per_point
            corrected[:, 2] = sin_a * x_rel + cos_a * z_rel + cz_per_point

        print(f"[Step 4] 全量插值修正完成 ✅")
        print(f"{'='*60}\n")

        return corrected

    def correct_o3d(self,
                    pcd: o3d.geometry.PointCloud,
                    correct_lateral: bool = True,
                    correct_height: bool = False,
                    correct_roll: bool = False) -> o3d.geometry.PointCloud:
        """
        Open3D 点云版本，直接返回修正后的 PointCloud 对象（不修改原对象）。
        """
        pts = np.asarray(pcd.points)
        corrected_pts = self.correct(pts, correct_lateral, correct_height, correct_roll)

        corrected_pcd = copy.deepcopy(pcd)
        corrected_pcd.points = o3d.utility.Vector3dVector(corrected_pts)
        return corrected_pcd

    # ----------------------------------------------------------
    # 可视化与诊断
    # ----------------------------------------------------------

    def plot_diagnosis(self, show_roll: bool = False):
        """
        绘制修正量诊断图，用于参数调优。

        建议在第一次处理数据时调用，确认平滑效果合理后再批量处理。
        """
        if not self.correction_info:
            print("请先调用 correct() 后再绘图")
            return

        info = self.correction_info
        y = info['frame_y']

        n_plots = 3 if show_roll else 2
        fig, axes = plt.subplots(n_plots, 2, figsize=(16, 4 * n_plots))
        fig.suptitle("钢轨扫描运动畸变诊断", fontsize=14, fontweight='bold')

        # ---- 横向 X ----
        ax = axes[0, 0]
        ax.plot(y, info['x_raw'], alpha=0.4, color='steelblue', label='原始轨头 X')
        ax.plot(y, info['x_smooth'], linewidth=2, color='navy', label='S-G 平滑（真实轨道）')
        # 标注无效帧
        invalid_y = y[~info['valid_mask']]
        if len(invalid_y):
            ax.scatter(invalid_y, info['x_raw'][~info['valid_mask']],
                       color='red', s=10, zorder=5, label='插值帧')
        ax.set_ylabel('X 位置 (mm)')
        ax.set_title('横向轨头中心轨迹')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(y, info['dx_correction'], color='crimson', linewidth=0.8)
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        ax.fill_between(y, info['dx_correction'], 0,
                        alpha=0.15, color='crimson')
        ax.set_ylabel('修正量 (mm)')
        ax.set_title(f"X 修正量  std={np.std(info['dx_correction']):.3f} mm  "
                     f"max={np.max(np.abs(info['dx_correction'])):.3f} mm")
        ax.grid(True, alpha=0.3)

        # ---- 高度 Z ----
        ax = axes[1, 0]
        ax.plot(y, info['z_raw'], alpha=0.4, color='seagreen', label='原始轨头 Z')
        ax.plot(y, info['z_smooth'], linewidth=2, color='darkgreen', label='S-G 平滑')
        ax.set_ylabel('Z 位置 (mm)')
        ax.set_xlabel('行进方向 Y (mm)')
        ax.set_title('高度轨头中心轨迹')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(y, info['dz_correction'], color='darkorange', linewidth=0.8)
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        ax.fill_between(y, info['dz_correction'], 0, alpha=0.15, color='darkorange')
        ax.set_ylabel('修正量 (mm)')
        ax.set_xlabel('行进方向 Y (mm)')
        ax.set_title(f"Z 修正量  std={np.std(info['dz_correction']):.3f} mm  "
                     f"max={np.max(np.abs(info['dz_correction'])):.3f} mm")
        ax.grid(True, alpha=0.3)

        # ---- 横滚角（可选）----
        if show_roll:
            ax = axes[2, 0]
            ax.plot(y, np.degrees(info['roll_raw']),
                    alpha=0.4, color='mediumpurple', label='原始横滚角')
            ax.plot(y, np.degrees(info['roll_smooth']),
                    linewidth=2, color='indigo', label='S-G 平滑')
            ax.set_ylabel('横滚角 (°)')
            ax.set_xlabel('行进方向 Y (mm)')
            ax.set_title('横滚角轨迹')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[2, 1]
            roll_deg = np.degrees(info['droll_correction'])
            ax.plot(y, roll_deg, color='purple', linewidth=0.8)
            ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
            ax.fill_between(y, roll_deg, 0, alpha=0.15, color='purple')
            ax.set_ylabel('修正量 (°)')
            ax.set_xlabel('行进方向 Y (mm)')
            ax.set_title(f"横滚修正量  std={np.std(roll_deg):.3f}°  "
                         f"max={np.max(np.abs(roll_deg)):.3f}°")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def print_summary(self):
        """打印修正统计摘要。"""
        if not self.correction_info:
            print("请先调用 correct() 后再查看摘要")
            return
        info = self.correction_info
        valid_pct = info['valid_mask'].mean() * 100
        print("\n========== 修正摘要 ==========")
        print(f"  有效帧率          : {valid_pct:.1f}%")
        print(f"  X 修正量 std/max  : "
              f"{np.std(info['dx_correction']):.3f} / "
              f"{np.max(np.abs(info['dx_correction'])):.3f} mm")
        print(f"  Z 修正量 std/max  : "
              f"{np.std(info['dz_correction']):.3f} / "
              f"{np.max(np.abs(info['dz_correction'])):.3f} mm")
        print(f"  横滚修正量 std/max: "
              f"{np.degrees(np.std(info['droll_correction'])):.3f} / "
              f"{np.degrees(np.max(np.abs(info['droll_correction']))):.3f}°")
        print("================================\n")