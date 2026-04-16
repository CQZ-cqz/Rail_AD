import open3d as o3d
import numpy as np
import trimesh
import math
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import time


class IntegratedPointCloudSystem:
    """
    集成的点云配准与异常检测系统
    """

    def __init__(self, defect_threshold=0.3):
        """
        初始化系统

        Args:
            defect_threshold: 异常检测的距离阈值，单位与点云一致 (mm)
        """
        # 配准相关属性
        self.source_cloud = None  # 扫描点云
        self.target_cloud = None  # CAD模型点云
        self.registration_result = None

        # 异常检测相关属性
        self.defect_threshold = defect_threshold
        self.registered_scan_cloud = None  # 配准后的扫描点云
        self.cad_sample_cloud = None  # CAD采样点云（用于异常检测）
        self.distances_scan_to_cad = None
        self.distances_cad_to_scan = None
        self.defects_scan_to_cad = None
        self.defects_cad_to_scan = None

    # ==================== 点云加载和预处理 ====================

    def load_stl_and_sample_points(self, stl_path, num_points=100000):
        """
        加载STL文件并采样点云
        """
        try:
            mesh = trimesh.load(stl_path)
            print(f"STL文件加载成功: {stl_path}")
            print(f"网格顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")

            # 从网格表面采样点
            points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

            # 创建Open3D点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # 估计法向量
            point_cloud.estimate_normals()

            return point_cloud

        except Exception as e:
            print(f"加载STL文件失败: {e}")
            return None

    def load_point_cloud(self, file_path, downsample_ratio=0.1):
        """
        加载点云文件（支持PCD和PLY格式）
        """
        try:
            file_path = Path(file_path)

            if file_path.suffix.lower() == '.pcd':
                point_cloud = o3d.io.read_point_cloud(str(file_path))
            elif file_path.suffix.lower() == '.ply':
                point_cloud = o3d.io.read_point_cloud(str(file_path))
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")

            # 下采样
            if downsample_ratio < 1.0:
                point_cloud = point_cloud.random_down_sample(sampling_ratio=downsample_ratio)

            print(f"点云文件加载成功: {file_path}")
            print(f"点数: {len(point_cloud.points)}")

            # 确保有法向量
            if not point_cloud.has_normals():
                point_cloud.estimate_normals()

            return point_cloud

        except Exception as e:
            print(f"加载点云文件失败: {e}")
            return None

    def preprocess_point_cloud(self, point_cloud, voxel_size=0.05):
        """
        点云预处理：下采样和去噪
        """
        print(f"预处理前点数: {len(point_cloud.points)}")

        # 体素下采样
        point_cloud = point_cloud.voxel_down_sample(voxel_size)

        # 统计异常值移除
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 重新估计法向量
        point_cloud.estimate_normals()

        print(f"预处理后点数: {len(point_cloud.points)}")
        return point_cloud

    # ==================== 点云配准相关方法 ====================

    def get_point_cloud_info(self, point_cloud, name="点云"):
        """
        获取点云的基本信息：中心点、尺寸等
        """
        points = np.asarray(point_cloud.points)

        # 计算边界框
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound

        print(f"\n{name}信息:")
        print(f"  中心点: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  尺寸: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
        print(f"  边界: min({min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f})")
        print(f"        max({max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f})")

        return {
            'center': center,
            'size': size,
            'min_bound': min_bound,
            'max_bound': max_bound
        }

    def center_point_cloud(self, point_cloud):
        """
        将点云中心对齐到原点
        """
        points = np.asarray(point_cloud.points)
        center = np.mean(points, axis=0)

        # 创建平移矩阵
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -center

        # 应用变换
        centered_cloud = copy.deepcopy(point_cloud)
        centered_cloud.transform(translation_matrix)

        print(f"点云已中心化，原中心点: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        return centered_cloud, translation_matrix

    def auto_pose_alignment(self, source_cloud, target_cloud):
        """
        根据点云的长宽高自动进行位姿调整
        """
        print("\n=== 开始自动位姿调整 ===")

        # 获取两个点云的信息
        source_info = self.get_point_cloud_info(source_cloud, "源点云")
        target_info = self.get_point_cloud_info(target_cloud, "目标点云")

        # 计算尺寸比例
        size_ratio = target_info['size'] / (source_info['size'] + 1e-6)
        print(f"尺寸比例 (目标/源): ({size_ratio[0]:.2f}, {size_ratio[1]:.2f}, {size_ratio[2]:.2f})")

        # 基于主轴对齐的简单策略
        source_size = source_info['size']
        target_size = target_info['size']

        # 找到最大轴
        source_main_axis = np.argmax(source_size)
        target_main_axis = np.argmax(target_size)

        print(f"源点云主轴: {['X', 'Y', 'Z'][source_main_axis]} (长度: {source_size[source_main_axis]:.2f})")
        print(f"目标点云主轴: {['X', 'Y', 'Z'][target_main_axis]} (长度: {target_size[target_main_axis]:.2f})")

        # 构造初始旋转矩阵（简单的轴对齐）
        rotation = np.eye(3)
        if source_main_axis != target_main_axis:
            # 需要旋转对齐主轴
            if source_main_axis == 0 and target_main_axis == 1:  # X -> Y
                rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            elif source_main_axis == 0 and target_main_axis == 2:  # X -> Z
                rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            elif source_main_axis == 1 and target_main_axis == 0:  # Y -> X
                rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            elif source_main_axis == 1 and target_main_axis == 2:  # Y -> Z
                rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            elif source_main_axis == 2 and target_main_axis == 0:  # Z -> X
                rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            elif source_main_axis == 2 and target_main_axis == 1:  # Z -> Y
                rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        # 构造4x4变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = rotation

        # 应用变换
        aligned_source = copy.deepcopy(source_cloud)
        aligned_source.transform(transformation)

        print("自动位姿调整完成")
        return aligned_source, transformation

    def compute_fpfh_features(self, point_cloud, voxel_size=0.05):
        """
        计算FPFH特征
        """
        radius_normal = voxel_size
        radius_feature = voxel_size * 2

        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return fpfh

    def global_registration(self, source, target, voxel_size=0.05):
        """
        全局配准（粗配准）
        """
        print("开始全局配准...")

        # 计算FPFH特征
        source_fpfh = self.compute_fpfh_features(source, voxel_size)
        target_fpfh = self.compute_fpfh_features(target, voxel_size)

        # 距离阈值
        distance_threshold = voxel_size * 10

        # RANSAC配准
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(5000, 0.7))

        return result

    def local_registration_cosine_decay(
            self,
            source,
            target,
            init_transformation,
            voxel_size=0.05,
            max_iterations=10,
            rmse_delta_stop=1e-3,
            threshold_start_factor=1.0,
            threshold_end_factor=0.3
    ):
        """
        ICP 局部配准：余弦下降距离阈值 + 精度阈值停止
        """
        print("👉 开始余弦递减 ICP 多阶段配准...")

        current_transformation = init_transformation
        prev_rmse = None

        # 循环
        for i in range(max_iterations):
            # 计算余弦退火后的阈值
            progress = i / (max_iterations - 1) if max_iterations > 1 else 1
            cos_decay = 0.5 * (1 + math.cos(progress * math.pi))  # 从1到0平滑下降
            threshold_factor = threshold_end_factor + (threshold_start_factor - threshold_end_factor) * cos_decay
            distance_threshold = voxel_size * threshold_factor

            # ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, distance_threshold, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
                # o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            current_transformation = icp_result.transformation

            # 打印
            print(
                f"[迭代 {i + 1}/{max_iterations}] 阈值={distance_threshold:.4f}, RMSE={icp_result.inlier_rmse:.6f}, fitness={icp_result.fitness:.6f}")

            # 判断收敛
            if prev_rmse is not None:
                rmse_improve = prev_rmse - icp_result.inlier_rmse
                if rmse_improve >= 0 and rmse_improve < rmse_delta_stop:
                    print(f"✅ 提前停止：rmse改善 {rmse_improve:.6f} < 阈值 {rmse_delta_stop}")
                    break
            prev_rmse = icp_result.inlier_rmse

        # 封装结果
        final_result = o3d.pipelines.registration.RegistrationResult()
        final_result.transformation = current_transformation
        final_result.fitness = icp_result.fitness
        final_result.inlier_rmse = icp_result.inlier_rmse
        final_result.correspondence_set = icp_result.correspondence_set

        return final_result

    # ==================== 异常检测相关方法 ====================

    # def compute_distances(self):
    #     """
    #     计算双向最近距离
    #     """
    #     if self.registered_scan_cloud is None or self.cad_sample_cloud is None:
    #         raise RuntimeError("请先完成配准流程")

    #     print("🔎 正在计算距离场 (scan->cad & cad->scan)...")
    #     scan_pts = np.asarray(self.registered_scan_cloud.points)
    #     cad_pts = np.asarray(self.cad_sample_cloud.points)

    #     # KD-Tree 加速
    #     cad_kdtree = o3d.geometry.KDTreeFlann(self.cad_sample_cloud)
    #     scan_kdtree = o3d.geometry.KDTreeFlann(self.registered_scan_cloud)

    #     # 计算 scan->cad 距离
    #     dist_scan_to_cad = np.zeros(len(scan_pts))
    #     for i, pt in enumerate(scan_pts):
    #         [_, idx, d2] = cad_kdtree.search_knn_vector_3d(pt, 1)
    #         dist_scan_to_cad[i] = np.sqrt(d2[0])

    #     # 计算 cad->scan 距离
    #     dist_cad_to_scan = np.zeros(len(cad_pts))
    #     for i, pt in enumerate(cad_pts):
    #         [_, idx, d2] = scan_kdtree.search_knn_vector_3d(pt, 1)
    #         dist_cad_to_scan[i] = np.sqrt(d2[0])

    #     self.distances_scan_to_cad = dist_scan_to_cad
    #     self.distances_cad_to_scan = dist_cad_to_scan

    #     # 提取缺陷点
    #     self.defects_scan_to_cad = scan_pts[dist_scan_to_cad > self.defect_threshold]
    #     self.defects_cad_to_scan = cad_pts[dist_cad_to_scan > self.defect_threshold]

    #     print(f"✅ 计算完成: 阈值={self.defect_threshold} mm")
    #     print(f"  Scan->CAD 异常点数: {len(self.defects_scan_to_cad)}")
    #     print(f"  CAD->Scan 异常点数: {len(self.defects_cad_to_scan)}")
    def compute_distances(self, use_bidirectional=True):
        if self.registered_scan_cloud is None or self.cad_sample_cloud is None:
            raise RuntimeError("请先完成配准流程")
        
        # 可选：先降采样（临时副本）
        scan_down = self.registered_scan_cloud.voxel_down_sample(0.5)   # 0.5mm 体素
        cad_down = self.cad_sample_cloud.voxel_down_sample(0.5)
        
        # 单向距离（扫描 -> CAD）
        dist = np.asarray(scan_down.compute_point_cloud_distance(cad_down))
        self.distances_scan_to_cad = dist
        self.defects_scan_to_cad = np.asarray(scan_down.points)[dist > self.defect_threshold]
        
        if use_bidirectional:
            # 反向距离（可选）
            dist_rev = np.asarray(cad_down.compute_point_cloud_distance(scan_down))
            self.distances_cad_to_scan = dist_rev
            self.defects_cad_to_scan = np.asarray(cad_down.points)[dist_rev > self.defect_threshold]
        else:
            self.distances_cad_to_scan = None
            self.defects_cad_to_scan = None


        # print(f"✅ 计算完成: 阈值={self.defect_threshold} mm")
        # print(f"  Scan->CAD 异常点数: {len(self.defects_scan_to_cad)}")
        # print(f"  CAD->Scan 异常点数: {len(self.defects_cad_to_scan)}")

    def visualize_distance_field(self, mode="scan_to_cad"):
        """
        可视化距离场 (伪彩色)
        mode: 'scan_to_cad' or 'cad_to_scan'
        """
        if mode == "scan_to_cad":
            pts = np.asarray(self.registered_scan_cloud.points)
            dists = self.distances_scan_to_cad
            pcd = self.registered_scan_cloud
        else:
            pts = np.asarray(self.cad_sample_cloud.points)
            dists = self.distances_cad_to_scan
            pcd = self.cad_sample_cloud

        # 归一化映射到 [0,1]
        d_min, d_max = dists.min(), dists.max()
        colors = plt.cm.jet((dists - d_min) / (d_max - d_min))[:, :3]
        pcd_vis = copy.deepcopy(pcd)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors)

        print(f"🎨 可视化模式: {mode}, 距离范围: {d_min:.4f} ~ {d_max:.4f}")
        o3d.visualization.draw_geometries([pcd_vis], width=1280, height=720)

    def visualize_defects(self):
        """
        可视化缺陷点（红色为扫描多余/突起，蓝色为CAD缺失部分）
        """
        cad_vis = copy.deepcopy(self.cad_sample_cloud)
        cad_vis.paint_uniform_color([0.8, 0.8, 0.8])

        # 扫描侧缺陷（红色）
        defects_scan = o3d.geometry.PointCloud()
        if self.defects_scan_to_cad is not None and len(self.defects_scan_to_cad) > 0:
            defects_scan.points = o3d.utility.Vector3dVector(self.defects_scan_to_cad)
            defects_scan.paint_uniform_color([1, 0, 0])
        else:
            defects_scan = None

        # CAD侧缺陷（蓝色）
        defects_cad = o3d.geometry.PointCloud()
        if self.defects_cad_to_scan is not None and len(self.defects_cad_to_scan) > 0:
            defects_cad.points = o3d.utility.Vector3dVector(self.defects_cad_to_scan)
            defects_cad.paint_uniform_color([0, 0, 1])
        else:
            defects_cad = None

        print("🔴 红色 = 扫描中多余或突起")
        print("🔵 蓝色 = CAD 中存在但扫描缺失（需要计算双向距离才能显示）")
        
        geometries = [cad_vis]
        if defects_scan is not None:
            geometries.append(defects_scan)
        if defects_cad is not None:
            geometries.append(defects_cad)
        
        o3d.visualization.draw_geometries(geometries, width=1280, height=720)

    # ==================== 可视化相关方法 ====================

    def visualize_geometries(self, source_pc, target_pc, title="点云可视化", axis_size=100.0):
        """
        可视化两个点云 + 坐标轴
        """
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        src = copy.deepcopy(source_pc)
        tgt = copy.deepcopy(target_pc)
        src.paint_uniform_color([1, 0, 0])  # 红色 - 扫描点云
        tgt.paint_uniform_color([0, 1, 0])  # 绿色 - CAD点云
        o3d.visualization.draw_geometries([src, tgt, axis], window_name=title, width=1000, height=800)

    # ==================== 保存结果相关方法 ====================

    def save_defect_points(self, output_dir="defect_output"):
        """
        保存缺陷点云
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        defect_scan_path = out / "defects_scan_to_cad.pcd"
        defect_cad_path = out / "defects_cad_to_scan.pcd"

        # 保存
        if len(self.defects_scan_to_cad) > 0:
            pcd_scan = o3d.geometry.PointCloud()
            pcd_scan.points = o3d.utility.Vector3dVector(self.defects_scan_to_cad)
            o3d.io.write_point_cloud(str(defect_scan_path), pcd_scan)
            print(f"💾 已保存: {defect_scan_path}")

        if len(self.defects_cad_to_scan) > 0:
            pcd_cad = o3d.geometry.PointCloud()
            pcd_cad.points = o3d.utility.Vector3dVector(self.defects_cad_to_scan)
            o3d.io.write_point_cloud(str(defect_cad_path), pcd_cad)
            print(f"💾 已保存: {defect_cad_path}")

    def save_results(self, output_dir="./integrated_results"):
        """
        保存配准和异常检测结果
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存配准结果
        if self.registration_result is not None:
            # 保存变换矩阵
            transformation_file = output_path / "transformation_matrix.txt"
            np.savetxt(transformation_file, self.registration_result.transformation, fmt='%.6f')
            print(f"变换矩阵已保存到: {transformation_file}")

            # 保存配准后的点云
            if self.registered_scan_cloud is not None:
                registered_cloud_file = output_path / "registered_scan_cloud.pcd"
                o3d.io.write_point_cloud(str(registered_cloud_file), self.registered_scan_cloud)
                print(f"配准后的扫描点云已保存到: {registered_cloud_file}")

        # 保存异常检测结果
        self.save_defect_points(output_path / "defects")

        # 保存综合报告
        report_file = output_path / "integrated_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("集成点云配准与异常检测报告\n")
            f.write("=" * 60 + "\n\n")

            if self.registration_result is not None:
                f.write("配准结果:\n")
                f.write(f"  适应度 (Fitness): {self.registration_result.fitness:.6f}\n")
                f.write(f"  均方根误差 (RMSE): {self.registration_result.inlier_rmse:.6f}\n")
                f.write(f"  内点数量: {len(self.registration_result.correspondence_set)}\n\n")

            if self.defects_scan_to_cad is not None:
                f.write("异常检测结果:\n")
                f.write(f"  检测阈值: {self.defect_threshold} mm\n")
                f.write(f"  扫描点云异常点数: {len(self.defects_scan_to_cad)}\n")
                f.write(f"  CAD点云异常点数: {len(self.defects_cad_to_scan)}\n")

                if hasattr(self, 'distances_scan_to_cad'):
                    f.write(f"  扫描到CAD平均距离: {np.mean(self.distances_scan_to_cad):.4f} mm\n")
                    f.write(f"  扫描到CAD最大距离: {np.max(self.distances_scan_to_cad):.4f} mm\n")
                if hasattr(self, 'distances_cad_to_scan'):
                    f.write(f"  CAD到扫描平均距离: {np.mean(self.distances_cad_to_scan):.4f} mm\n")
                    f.write(f"  CAD到扫描最大距离: {np.max(self.distances_cad_to_scan):.4f} mm\n")

        print(f"综合报告已保存到: {report_file}")

    # ==================== 主要集成流程 ====================

    def run_complete_pipeline(self,
                              scan_path,
                              cad_stl_path,
                              cad_sample_points=300000,
                              defect_sample_points=500000,
                              voxel_size=0.1,
                              voxel_size_registration=0.5,
                              show_intermediate_results=True,
                              save_results=True,
                              output_dir="./integrated_results"):
        """
        完整的集成流程：配准 + 异常检测

        Args:
            scan_path: 扫描点云文件路径
            cad_stl_path: CAD模型STL文件路径
            cad_sample_points: CAD模型采样点数（用于配准）
            defect_sample_points: CAD模型采样点数（用于异常检测）
            voxel_size: 体素下采样大小
            show_intermediate_results: 是否显示中间结果
            save_results: 是否保存结果
            output_dir: 输出目录
        """
        print("=" * 80)
        print("🚀 开始集成点云配准与异常检测流程")
        print("=" * 80)

        # ==================== 第一阶段：数据加载 ====================
        print("\n📁 第一阶段：数据加载")
        print("-" * 40)

        # 加载扫描点云
        self.source_cloud = self.load_point_cloud(scan_path, downsample_ratio=1.0)
        if self.source_cloud is None:
            print("❌ 扫描点云加载失败")
            return False

        # 加载CAD模型（配准用）
        self.target_cloud = self.load_stl_and_sample_points(cad_stl_path, num_points=cad_sample_points)
        if self.target_cloud is None:
            print("❌ CAD模型加载失败")
            return False

        if show_intermediate_results:
            self.visualize_geometries(
                self.source_cloud, self.target_cloud,
                title="原始点云"
            )
        start_time = time.time()

        # ==================== 第二阶段：点云配准 ====================
        print("\n🔧 第二阶段：点云配准")
        print("-" * 40)

        # 中心化点云
        print("🔹 中心化点云...")
        source_centered, _ = self.center_point_cloud(self.source_cloud)
        target_centered, _ = self.center_point_cloud(self.target_cloud)

        # 自动位姿调整
        print("🔹 自动位姿调整...")
        source_aligned, _ = self.auto_pose_alignment(source_centered, target_centered)

        if show_intermediate_results:
            self.visualize_geometries(
                source_aligned, target_centered,
                title="位姿调整后"
            )

        # 点云预处理
        print("🔹 点云预处理...")
        source_processed = self.preprocess_point_cloud(source_aligned, voxel_size_registration)
        target_processed = self.preprocess_point_cloud(target_centered, voxel_size_registration)

        # 配准（全局 + 局部）
        print("🔹 开始配准...")
        registered_time = time.time()
        try:
            # 全局配准
            # global_result = self.global_registration(source_processed, target_processed, voxel_size)
            # print(f"全局配准 - 适应度: {global_result.fitness:.6f}, RMSE: {global_result.inlier_rmse:.6f}")

            # 局部配准
            local_result = self.local_registration_cosine_decay(
                source_processed, target_processed,
                np.eye(4),
                voxel_size=voxel_size,
                max_iterations=10,
                rmse_delta_stop=5e-3,
                threshold_start_factor=20.0,
                threshold_end_factor=1.0
            )

            self.registration_result = local_result
            print(f"✅ 配准完成 - 适应度: {local_result.fitness:.6f}, RMSE: {local_result.inlier_rmse:.6f}")

        except Exception as e:
            print(f"❌ 配准失败: {e}")
            return False
        registered_time = time.time() - registered_time
        print(f"配准时间: {registered_time:.6f}秒")

        # 生成配准后的扫描点云（用于异常检测）
        self.registered_scan_cloud = copy.deepcopy(source_aligned)
        self.registered_scan_cloud.transform(self.registration_result.transformation)

        if show_intermediate_results:
            self.visualize_geometries(
                self.registered_scan_cloud, target_centered,
                title="配准完成"
            )

        # ==================== 第三阶段：异常检测 ====================
        print("\n🔍 第三阶段：异常检测")
        print("-" * 40)

        # 重新采样CAD模型（用于异常检测，更高密度）
        # print("🔹 重新采样CAD模型用于异常检测...")
        # self.cad_sample_cloud = self.load_stl_and_sample_points(cad_stl_path, num_points=defect_sample_points)
        # if self.cad_sample_cloud is None:
        #     print("❌ CAD模型重新采样失败")
        #     return False
        #
        # # 对CAD采样点云进行相同的中心化处理
        # cad_sample_centered, _ = self.center_point_cloud(self.cad_sample_cloud)
        # self.cad_sample_cloud = cad_sample_centered
        self.cad_sample_cloud = target_centered

        # 计算距离场
        print("🔹 计算距离场...")
        defect_time = time.time()
        try:
            self.compute_distances()
        except Exception as e:
            print(f"❌ 距离计算失败: {e}")
            return False
        defect_time = time.time() - defect_time
        print(f"异常检测时间: {defect_time:.6f}秒")
        end_time = time.time() - start_time
        print(f"检测时间: {end_time:.6f}秒")

        # 可视化结果
        if show_intermediate_results:
            print("🔹 可视化距离场...")
            self.visualize_distance_field(mode="scan_to_cad")

            print("🔹 可视化异常点...")
            self.visualize_defects()

        # ==================== 第四阶段：结果保存 ====================
        if save_results:
            print("\n💾 第四阶段：保存结果")
            print("-" * 40)
            self.save_results(output_dir)

        # ==================== 流程完成 ====================
        print("\n" + "=" * 80)
        print("✅ 集成流程完成！")
        print("=" * 80)

        # 打印总结信息
        print(f"\n📊 流程总结:")
        print(f"  配准适应度: {self.registration_result.fitness:.6f}")
        print(f"  配准RMSE: {self.registration_result.inlier_rmse:.6f}")
        print(f"  异常检测阈值: {self.defect_threshold} mm")
        print(f"  扫描点云异常点数: {len(self.defects_scan_to_cad)}")
        print(f"  CAD点云异常点数: {len(self.defects_cad_to_scan)}")
        if hasattr(self, 'distances_scan_to_cad'):
            print(f"  平均距离偏差: {np.mean(self.distances_scan_to_cad):.4f} mm")
            print(f"  最大距离偏差: {np.max(self.distances_scan_to_cad):.4f} mm")

        return True

    def quick_detection_only(self,
                             registered_scan_path,
                             cad_stl_path,
                             defect_sample_points=500000,
                             show_results=True,
                             save_results=True,
                             output_dir="./detection_results"):
        """
        快速异常检测流程（假设已有配准好的点云）

        Args:
            registered_scan_path: 已配准的扫描点云路径
            cad_stl_path: CAD模型STL文件路径
            defect_sample_points: CAD模型采样点数
            show_results: 是否显示结果
            save_results: 是否保存结果
            output_dir: 输出目录
        """
        print("=" * 60)
        print("🔍 快速异常检测流程")
        print("=" * 60)

        # 加载已配准的扫描点云
        self.registered_scan_cloud = self.load_point_cloud(registered_scan_path, downsample_ratio=1.0)
        if self.registered_scan_cloud is None:
            print("❌ 配准点云加载失败")
            return False

        # 加载和采样CAD模型
        self.cad_sample_cloud = self.load_stl_and_sample_points(cad_stl_path, num_points=defect_sample_points)
        if self.cad_sample_cloud is None:
            print("❌ CAD模型加载失败")
            return False

        # 计算距离场和异常检测
        try:
            self.compute_distances()
        except Exception as e:
            print(f"❌ 距离计算失败: {e}")
            return False

        # 可视化结果
        if show_results:
            self.visualize_distance_field(mode="scan_to_cad")
            self.visualize_defects()

        # 保存结果
        if save_results:
            self.save_defect_points(output_dir)

        print("✅ 快速异常检测完成！")
        return True


def main():
    """
    主函数 - 使用示例
    """
    # 创建集成系统
    system = IntegratedPointCloudSystem(defect_threshold=0.5)  # 设置异常检测阈值为1.0mm

    # 文件路径（请根据实际情况修改）
    scan_path = "data/clouds/corrected_scan_00003_01.ply"
    cad_stl_path = "data/cad/QU70_SingleSide_05mm.STL"

    # 运行完整流程
    success = system.run_complete_pipeline(
        scan_path=scan_path,
        cad_stl_path=cad_stl_path,
        cad_sample_points=1000000,  # 配准用采样点数
        defect_sample_points=500000,  # 异常检测用采样点数
        voxel_size=0.1,  # 体素下采样大小
        voxel_size_registration=0.5,  # 配准用体素大小
        show_intermediate_results=True,  # 显示中间结果
        save_results=False,  # 保存结果
        output_dir="./integrated_output"  # 输出目录
    )

    if success:
        print("🎉 所有流程执行成功！")

        # 可以进行额外的分析
        print("\n📈 详细分析:")
        if system.distances_scan_to_cad is not None:
            distances = system.distances_scan_to_cad
            print(f"距离统计信息:")
            print(f"  最小距离: {np.min(distances):.4f} mm")
            print(f"  最大距离: {np.max(distances):.4f} mm")
            print(f"  平均距离: {np.mean(distances):.4f} mm")
            print(f"  标准差: {np.std(distances):.4f} mm")
            print(f"  中位数: {np.median(distances):.4f} mm")

            # 统计不同阈值下的异常点数
            thresholds = [0.5, 1.0, 2.0, 3.0]
            print(f"\n不同阈值下的异常点统计:")
            for thresh in thresholds:
                anomaly_count = np.sum(distances > thresh)
                anomaly_ratio = anomaly_count / len(distances) * 100
                print(f"  阈值 {thresh} mm: {anomaly_count} 点 ({anomaly_ratio:.2f}%)")
    else:
        print("❌ 流程执行失败！")


def example_detection_only():
    """
    仅异常检测的示例（假设已有配准好的点云）
    """
    system = IntegratedPointCloudSystem(defect_threshold=0.3)

    # 假设已有配准好的点云文件
    registered_scan_path = "data/pointcloud/registration_output/registered_point_cloud.pcd"
    cad_stl_path = "data/pointcloud/QU70_front.STL"

    success = system.quick_detection_only(
        registered_scan_path=registered_scan_path,
        cad_stl_path=cad_stl_path,
        defect_sample_points=500000,
        show_results=True,
        save_results=True,
        output_dir="./detection_only_output"
    )

    if success:
        print("🎉 异常检测完成！")
    else:
        print("❌ 异常检测失败！")


if __name__ == "__main__":
    # 运行完整流程
    main()

    # 如果只需要异常检测，可以取消下面的注释
    # example_detection_only()