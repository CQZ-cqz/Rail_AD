import torch
import numpy as np
import cv2
import os
import time
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from model.de_resnet import de_resnet18, de_wide_resnet50_2, de_resnet34
from model.resnet import resnet18, wide_resnet50_2, resnet34
from utils.utils_test import *
from torch.utils.data import Dataset, DataLoader


# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RD4ADDetector:
    def __init__(self, checkpoint_path, anomaly_threshold=0.006, min_defect_area=10):
        """
        初始化RD4AD检测器
        Args:
            checkpoint_path: 模型权重路径
        """
        self.device = device
        self.encoder, self.bn, self.decoder, self.backbone_name = self.load_model(checkpoint_path)
        self.max_batch_size = 1 if self.backbone_name == "wres50" else 2
        self.anomaly_threshold = float(anomaly_threshold)
        self.min_defect_area = int(min_defect_area)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_postprocess_params(self, anomaly_threshold=None, min_defect_area=None):
        if anomaly_threshold is not None:
            self.anomaly_threshold = float(anomaly_threshold)
        if min_defect_area is not None:
            self.min_defect_area = int(min_defect_area)
        
    def _build_backbone(self, backbone_name):
        if backbone_name == "res18":
            encoder, bn = resnet18(pretrained=True)
            decoder = de_resnet18(pretrained=False)
            return encoder, bn, decoder
        if backbone_name == "res34":
            encoder, bn = resnet34(pretrained=True)
            decoder = de_resnet34(pretrained=False)
            return encoder, bn, decoder
        if backbone_name == "wres50":
            encoder, bn = wide_resnet50_2(pretrained=True)
            decoder = de_wide_resnet50_2(pretrained=False)
            return encoder, bn, decoder
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    def _candidate_backbones(self, checkpoint_path):
        name = os.path.basename(checkpoint_path).lower()
        if "res18" in name:
            return ["res18", "res34", "wres50"]
        if "wres50" in name or "res50" in name:
            return ["wres50", "res34", "res18"]
        if "res34" in name:
            return ["res34", "res18", "wres50"]
        return ["res34", "res18", "wres50"]

    def load_model(self, checkpoint_path):
        """加载模型，按权重自动匹配骨干网络。"""
        ckp = torch.load(checkpoint_path, map_location=self.device)

        last_error = None
        for backbone_name in self._candidate_backbones(checkpoint_path):
            try:
                encoder, bn, decoder = self._build_backbone(backbone_name)
                encoder = encoder.to(self.device)
                bn = bn.to(self.device)
                decoder = decoder.to(self.device)

                decoder.load_state_dict(ckp['decoder'])
                bn.load_state_dict(ckp['bn'])

                encoder.eval()
                bn.eval()
                decoder.eval()
                print(f"RD4AD loaded backbone: {backbone_name}")
                return encoder, bn, decoder, backbone_name
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"无法匹配checkpoint与网络骨干: {checkpoint_path}. "
            f"请确认权重与res18/res34/wres50之一对应。最后错误: {last_error}"
        )
    
    def preprocess_images(self, images):
        """
        预处理图像列表
        Args:
            images: 图像列表，每个元素为numpy数组 (H, W, 3) BGR格式
        Returns:
            tensor_images: 预处理后的张量 [B, 3, 256, 256]
            original_sizes: 原始图像尺寸列表
        """
        tensor_images = []
        original_sizes = []
        
        for img in images:
            # 记录原始尺寸
            original_h, original_w = img.shape[:2]
            original_sizes.append((original_h, original_w))
            
            # 转换为RGB并归一化
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb / 255.0
            img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
            
            # 应用变换
            img_tensor = self.transform(img_tensor)
            tensor_images.append(img_tensor)
        
        # 堆叠成批次
        tensor_images = torch.stack(tensor_images, 0)
        return tensor_images, original_sizes
    
    def gpu_quick_segmentation(self, anomaly_maps, threshold=0.006, min_area=10):
        """
        GPU加速的快速分割
        Args:
            anomaly_maps: [B, H, W] 异常分数图
            threshold: 分割阈值
            min_area: 最小区域面积
        Returns:
            binary_masks: 二值掩码
            defect_bboxes: 缺陷边界框列表
            has_defect: 是否有缺陷的布尔列表
        """
        # 阈值分割
        binary_masks = (anomaly_maps > threshold).float()
        
        batch_bboxes = []
        has_defect_list = []
        
        for i in range(binary_masks.shape[0]):
            mask = binary_masks[i]
            
            # 计算非零像素数量作为面积估计
            area = torch.sum(mask).item()
            
            if area >= min_area:
                # 简单的边界框计算
                nonzero_indices = torch.nonzero(mask)
                if len(nonzero_indices) > 0:
                    min_coords = torch.min(nonzero_indices, dim=0)[0]
                    max_coords = torch.max(nonzero_indices, dim=0)[0]
                    bbox = [min_coords[1].item(), min_coords[0].item(), 
                           max_coords[1].item() - min_coords[1].item(),
                           max_coords[0].item() - min_coords[0].item()]
                    batch_bboxes.append([bbox])
                    has_defect_list.append(True)
                else:
                    batch_bboxes.append([])
                    has_defect_list.append(False)
            else:
                batch_bboxes.append([])
                has_defect_list.append(False)
        
        return binary_masks, batch_bboxes, has_defect_list
    
    def detect_batch(self, images, save_dir=None, camera_ids=None):
        """
        批量检测多张图像
        Args:
            images: 图像列表，每个元素为numpy数组 (H, W, 3) BGR格式
            save_dir: 结果保存目录，如果为None则不保存
            camera_ids: 相机ID列表，用于标识不同相机的图像
        Returns:
            results: 检测结果列表，每个元素为字典，包含:
                     - anomaly_score: 异常分数
                     - has_defect: 是否有缺陷
                     - bboxes: 缺陷边界框列表
                     - anomaly_map: 异常热力图 [256, 256]
                     - binary_mask: 二值掩码 [256, 256]
        """
        if len(images) == 0:
            return []
            
        results = []
        total = len(images)
        step = max(1, int(self.max_batch_size))

        for start in range(0, total, step):
            end = min(start + step, total)
            batch_images = images[start:end]

            tensor_images, original_sizes = self.preprocess_images(batch_images)
            tensor_images = tensor_images.to(self.device)

            with torch.no_grad():
                features = self.encoder(tensor_images)
                outputs = self.decoder(self.bn(features))
                anomaly_maps_batch = cal_anomaly_map_batch(features, outputs, out_size=256, amap_mode='add', device=self.device)
                anomaly_maps_batch = batch_gaussian_filter(anomaly_maps_batch, sigma=4)
                anomaly_scores_batch = anomaly_maps_batch.amax(dim=(1, 2))

                binary_masks, bbox_list, has_defect_list = self.gpu_quick_segmentation(
                    anomaly_maps_batch,
                    threshold=self.anomaly_threshold,
                    min_area=self.min_defect_area,
                )

                anomaly_maps_batch = anomaly_maps_batch * binary_masks
                anomaly_maps_batch = torch.where(
                    anomaly_maps_batch < self.anomaly_threshold,
                    0,
                    anomaly_maps_batch - self.anomaly_threshold,
                )
                anomaly_maps_batch = min_max_norm_batch(anomaly_maps_batch)

                anomaly_maps_np = anomaly_maps_batch.cpu().numpy()
                anomaly_scores_np = anomaly_scores_batch.cpu().numpy()
                binary_masks_np = binary_masks.cpu().numpy()

                for i in range(end - start):
                    global_idx = start + i
                    result = {
                        'anomaly_score': anomaly_scores_np[i],
                        'has_defect': has_defect_list[i],
                        'bboxes': bbox_list[i],
                        'anomaly_map': anomaly_maps_np[i],
                        'binary_mask': binary_masks_np[i],
                        'original_size': original_sizes[i]
                    }
                    results.append(result)

                    if save_dir and has_defect_list[i]:
                        self.save_result(
                            batch_images[i],
                            result,
                            save_dir,
                            camera_ids[global_idx] if camera_ids else global_idx,
                        )

            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return results
    
    def save_result(self, original_image, result, save_dir, camera_id):
        """
        保存检测结果
        Args:
            original_image: 原始图像
            result: 检测结果字典
            save_dir: 保存目录
            camera_id: 相机ID
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 调整图像尺寸到256x256用于显示
        display_image = cv2.resize(original_image, (256, 256))
        heatmap = cvt2heatmap(result['anomaly_map'] * 255)
        
        # 叠加热力图
        result_image = show_cam_on_image(display_image, heatmap)
        
        # 绘制边界框
        bboxes = result['bboxes']
        for bbox in bboxes:
            x, y, w, h = bbox
            # 绘制矩形框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 添加标签
            cv2.putText(result_image, f'Defect', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 保存图像
        timestamp = int(time.time() * 1000)
        output_dir = os.path.join(save_dir, f"camera_{camera_id}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{timestamp}.jpg")
        cv2.imwrite(output_path, result_image)
        
        # 保存二值掩码
        mask_path = os.path.join(output_dir, f"{timestamp}_mask.jpg")
        binary_mask_vis = (result['binary_mask'] * 255).astype(np.uint8)
        cv2.imwrite(mask_path, binary_mask_vis)
        
        # 保存检测信息
        info_path = os.path.join(output_dir, f"{timestamp}.txt")
        with open(info_path, 'w') as f:
            f.write(f"Camera ID: {camera_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Anomaly Score: {result['anomaly_score']:.4f}\n")
            f.write(f"Defect BBoxes: {bboxes}\n")
            f.write(f"Original Size: {result['original_size'][1]}x{result['original_size'][0]}\n")
            f.write(f"Defect Area: {np.sum(result['binary_mask'])} pixels\n")
        
        print(f"Saved defect result: {output_path}, anomaly score: {result['anomaly_score']:.4f}")

# 使用示例
def main():
    # 初始化检测器
    checkpoint_path = "weights/RD4AD_wres50_rail_mix.pth"
    detector = RD4ADDetector(checkpoint_path)
    
    # 模拟从多台相机获取图像
    # 这里假设您有自己的相机取流代码
    # camera_images = [img1, img2, img3, ...]  # 每台相机的图像
    # camera_ids = [0, 1, 2, ...]  # 相机ID
    
    # 示例使用（需要替换为实际的相机图像）
    camera_images = [
        cv2.imread("save_bmp/cam0/20620251027_223328.png"),
        cv2.imread("save_bmp/cam1/57820251030_165423.png"),
        # 更多图像...
    ]
    camera_ids = [0, 1]  # 相机ID
    
    # 批量检测
    results = detector.detect_batch(
        images=camera_images,
        save_dir="detection_results",
        camera_ids=camera_ids
    )
    
    # 处理结果
    for i, result in enumerate(results):
        print(f"Camera {camera_ids[i]}: "
              f"Anomaly Score: {result['anomaly_score']:.4f}, "
              f"Has Defect: {result['has_defect']}")

if __name__ == "__main__":
    main()