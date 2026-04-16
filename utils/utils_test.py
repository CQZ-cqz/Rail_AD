import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings



warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_anomaly_map_batch(fs_list, ft_list, out_size=224, amap_mode='mul', device='cuda'):
    """
    批量计算异常图的优化版本
    
    Args:
        fs_list: list of tensors, 学生网络特征列表 [batch_size, C, H, W]
        ft_list: list of tensors, 教师网络特征列表 [batch_size, C, H, W]
        out_size: 输出异常图尺寸
        amap_mode: 融合模式 'mul' 或 'add'
        device: 计算设备
    
    Returns:
        anomaly_map_batch: 批量异常图 [batch_size, out_size, out_size]
        a_map_list_batch: 各层异常图列表 [layers, batch_size, out_size, out_size]
    """
    batch_size = fs_list[0].size(0)
    
    if amap_mode == 'mul':
        anomaly_map_batch = torch.ones([batch_size, out_size, out_size], device=device)
    else:
        anomaly_map_batch = torch.zeros([batch_size, out_size, out_size], device=device)
    
    # a_map_list_batch = []
    
    for i in range(len(ft_list)):
        fs = fs_list[i]  # [batch_size, C, H, W]
        ft = ft_list[i]  # [batch_size, C, H, W]
        
        # 批量计算余弦相似度
        # 归一化特征
        # fs_norm = F.normalize(fs, p=2, dim=1)
        # ft_norm = F.normalize(ft, p=2, dim=1)
        
        # 计算余弦距离 (1 - 余弦相似度)
        # 使用向量化计算整个batch
        a_map = 1 - F.cosine_similarity(fs, ft, dim=1)  # [batch_size, H, W]
        # a_map = torch.sum((fs_norm - ft_norm) ** 2, dim=1) / 2.0  # 近似余弦距离
        a_map = a_map.unsqueeze(1)  # [batch_size, 1, H, W]
        
        # 批量上采样
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', 
                             align_corners=True)  # [batch_size, 1, out_size, out_size]
        a_map = a_map.squeeze(1)  # [batch_size, out_size, out_size]
        
        # a_map_list_batch.append(a_map)
        
        # 批量融合
        if amap_mode == 'mul':
            anomaly_map_batch *= a_map
        else:
            anomaly_map_batch += a_map
    
    return anomaly_map_batch    # , a_map_list_batch

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation_multi_proj(encoder,proj,bn, decoder, dataloader,device):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for (img, gt, label, _, _) in dataloader:

            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),4)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def batch_gaussian_filter(tensor, sigma=2, kernel_size=5):
    """在GPU上批量进行高斯滤波"""
    # 创建高斯核
    x = torch.arange(kernel_size, device=tensor.device) - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma).pow(2))
    gauss = gauss / gauss.sum()
    
    # 创建2D高斯核
    gauss_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
    
    # 应用卷积进行滤波
    tensor = tensor.unsqueeze(1)  # [batch, 1, H, W]
    padded = F.pad(tensor, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    filtered = F.conv2d(padded, gauss_2d, groups=1)
    return filtered.squeeze(1)  # [batch, H, W]

def apply_background_masking_batch(anomaly_maps_batch, mask_regions, device):
    """批量应用背景掩码"""
    batch_size, h, w = anomaly_maps_batch.shape
    mask = create_background_mask_batch((h, w), mask_regions, device)
    return anomaly_maps_batch * mask.unsqueeze(0)  # 广播到整个batch

def create_background_mask_batch(anomaly_map_shape, mask_regions, device):
    """创建批量背景掩码"""
    h, w = anomaly_map_shape
    mask = torch.ones((h, w), device=device, dtype=torch.float32)
    
    for region in mask_regions:
        x1, y1, x2, y2 = region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 0
    
    return mask

def min_max_norm_batch(tensor):
    """批量归一化"""
    batch_size = tensor.size(0)
    tensor_flat = tensor.view(batch_size, -1)
    min_vals = tensor_flat.min(dim=1)[0].view(batch_size, 1, 1)
    max_vals = tensor_flat.max(dim=1)[0].view(batch_size, 1, 1)
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)

def gpu_quick_segmentation(anomaly_maps, threshold=0.5, min_area=50, device='cuda'):
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