"""
PE-MMNet v4 统一训练脚本

功能：
1. --mode train：正常训练（默认150 epoch，结束后自动评估）
2. --mode eval：仅加载已有模型进行推理和评估
3. --mode ablation：运行消融实验（6个变体）

使用方法：
  # 训练完整模型（检测模式）
  python run_train.py --mode train --variant resnet18 --epochs 150

  # 训练分割模型
  python run_train.py --mode train --task segmentation --epochs 100

  # 评估检查点（检测模式）
  python run_train.py --mode eval --checkpoint ./checkpoints/resnet18_cnn_attn_cross_attn_taskdetection_offset0_best.pt

  # 评估检查点（分割模式）
  python run_train.py --mode eval --checkpoint ./checkpoints/resnet18_cnn_attn_cross_attn_tasksegmentation_offset0_best.pt

  # 运行消融实验
  python run_train.py --mode ablation
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from wcwidth import wcswidth
import re

# 盒子固定宽度
BOX_WIDTH = 100

# ANSI 颜色代码正则
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')


def display_width(text):
    """返回文本在终端中的实际显示宽度（忽略ANSI代码）"""
    # 去除ANSI颜色代码
    clean = ANSI_PATTERN.sub('', text)
    width = wcswidth(clean)
    if width < 0:
        width = len(clean)
    return width


def pad_to_width(text, target_width):
    """在右侧补空格，使总显示宽度等于 target_width"""
    cur = display_width(text)
    if cur < target_width:
        return text + ' ' * (target_width - cur)
    return text


def print_box_line(content, width=BOX_WIDTH):
    """打印盒子的单行，自动左右加空格并补齐宽度"""
    inner = ' ' + content + ' '
    padded = pad_to_width(inner, width - 2)  # 减去左右边框
    return '║' + padded + '║'


def print_box_divider(width=BOX_WIDTH):
    """打印盒子分隔线"""
    return '╠' + '═' * (width - 2) + '╣'

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 禁用输出缓冲（用于批量训练时的实时日志）
import functools
print = functools.partial(print, flush=True)

from utils.config import ensure_config, get_data_root, get_checkpoints_dir, get_results_dir, get_data_batches
from data.dataset_multimodal import create_multibatch_dataloaders
from models.pe_tsnet_multimodal import (
    PETSNetMultimodal,
    ResNet18Backbone2D,
    TemporalFeatureExtractor,
    CrossAttentionFusion,
    MultiTaskHead,
    create_model,
    get_arch_specific_config
)
from models.pe_tsnet_yolo import SwinYOLOFPN, ViTYOLOFPN
from models.pe_tsnet_detr import DETRStyle
from training.mono_loss import (
    MultimodalCrackLoss,
    SegmentationLoss,
    MultimodalSegmentationLoss,
    DiceLoss,
    box_iou,
    YOLOLoss,
    YOLOTargetAssigner,
    DETRLoss,
    HungarianMatcher
)
from utils.console import (
    print_title, print_section, print_result, print_results_table,
    print_info, print_warning, print_error, print_success,
    print_progress, print_header, print_divider
)

# 从配置读取路径
CHECKPOINT_DIR = get_checkpoints_dir()
RESULTS_DIR = get_results_dir()
DATA_BATCHES = ["单次扫描", "参数化扫描1", "参数化扫描2"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 显存自适应配置
# =============================================================================

def get_free_gpu_memory_gb():
    """获取空闲显存（GB）"""
    if not torch.cuda.is_available():
        return 0
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    return total - allocated


def auto_select_config(args):
    """
    根据可用显存自动选择配置

    逻辑：
    1. 用户指定了任一参数 → 该参数使用用户值，未指定的继续自动
    2. 完全自动模式 → 根据显存选择
    3. CPU模式 → 固定低配置

    Args:
        args: argparse 参数对象，应包含 _user_specified 属性

    Returns:
        dict: {'image_size', 'batch_size', 'fp16'}
    """
    import argparse

    # CPU 模式
    if not torch.cuda.is_available():
        print_info("未检测到GPU，将使用CPU训练（速度较慢）")
        return {'image_size': 256, 'batch_size': 4, 'fp16': False}

    free_gb = get_free_gpu_memory_gb()

    # 获取用户显式指定的标志（通过 argparse 解析时设置）
    has_user_image_size = getattr(args, '_user_specified', {}).get('image_size', False)
    has_user_batch_size = getattr(args, '_user_specified', {}).get('batch_size', False)

    # FP16 逻辑：--no_fp16 时禁用
    fp16_enabled = not args.no_fp16

    # 构建配置
    config = {'fp16': fp16_enabled}

    # 如果用户没有指定 batch_size，根据显存自动选择
    if not has_user_batch_size and args.batch_size is None:
        if free_gb >= 10.0:
            config['batch_size'] = 16
        elif free_gb >= 6.0:
            config['batch_size'] = 8
        elif free_gb >= 4.0:
            config['batch_size'] = 8
        else:
            config['batch_size'] = 4
    else:
        config['batch_size'] = args.batch_size if args.batch_size is not None else 16

    # 如果用户没有指定 image_size，根据显存自动选择
    if not has_user_image_size:
        if free_gb >= 10.0:
            config['image_size'] = 512
        elif free_gb >= 6.0:
            config['image_size'] = 512
        elif free_gb >= 4.0:
            config['image_size'] = 384
        else:
            config['image_size'] = 256
    else:
        config['image_size'] = args.image_size

    return config


def print_auto_config_info(config, free_gb):
    """打印自动配置信息"""
    fp16_str = "启用" if config['fp16'] else "禁用"
    print_info(f"检测到可用显存: {free_gb:.1f} GB")
    print_info(f"自动选择配置: 图像 {config['image_size']}x{config['image_size']}, batch={config['batch_size']}, FP16={fp16_str}")
    if config['image_size'] != 512 or config['batch_size'] != 16:
        print_info("如需强制指定配置，请使用 --image_size 或 --batch_size 参数")


# =============================================================================
# 模型变体定义
# =============================================================================

class Model1DOnly(nn.Module):
    """仅时序分支"""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.branch_1d = TemporalFeatureExtractor(
            input_dim=1, hidden_dim=32, num_heads=4, dropout=dropout
        )
        self.output_head = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 6)
        )

    def forward(self, x_1d, x_2d):
        feat_1d = self.branch_1d(x_1d)
        output = self.output_head(feat_1d)
        output[:, :4] = F.relu(output[:, :4])
        output[:, 4:] = torch.sigmoid(output[:, 4:])
        return output


class Model2DOnly(nn.Module):
    """仅空间分支"""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.branch_2d = ResNet18Backbone2D(in_channels=2, pretrained=True)
        self.output_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 6)
        )

    def forward(self, x_1d, x_2d):
        feat_2d = self.branch_2d(x_2d)
        output = self.output_head(feat_2d)
        output[:, :4] = F.relu(output[:, :4])
        output[:, 4:] = torch.sigmoid(output[:, 4:])
        return output


class ModelConcat(nn.Module):
    """双分支 + 拼接"""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.branch_2d = ResNet18Backbone2D(in_channels=2, pretrained=True)
        self.branch_1d = TemporalFeatureExtractor(
            input_dim=1, hidden_dim=32, num_heads=4, dropout=dropout
        )
        self.output_head = nn.Sequential(
            nn.Linear(576, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 6)
        )

    def forward(self, x_1d, x_2d):
        feat_2d = self.branch_2d(x_2d)
        feat_1d = self.branch_1d(x_1d)
        fused = torch.cat([feat_2d, feat_1d], dim=-1)
        output = self.output_head(fused)
        output[:, :4] = F.relu(output[:, :4])
        output[:, 4:] = torch.sigmoid(output[:, 4:])
        return output


class ModelAdd(nn.Module):
    """双分支 + 加法"""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.branch_2d = ResNet18Backbone2D(in_channels=2, pretrained=True)
        self.branch_1d = TemporalFeatureExtractor(
            input_dim=1, hidden_dim=32, num_heads=4, dropout=dropout
        )
        self.proj_1d = nn.Linear(64, 512)
        self.output_head = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 6)
        )

    def forward(self, x_1d, x_2d):
        feat_2d = self.branch_2d(x_2d)
        feat_1d = self.proj_1d(self.branch_1d(x_1d))
        fused = feat_2d + feat_1d
        output = self.output_head(fused)
        output[:, :4] = F.relu(output[:, :4])
        output[:, 4:] = torch.sigmoid(output[:, 4:])
        return output


class ModelCrossAttn(nn.Module):
    """双分支 + Cross-Attention"""
    def __init__(self, dropout=0.2):
        super().__init__()
        self.branch_2d = ResNet18Backbone2D(in_channels=2, pretrained=True)
        self.branch_1d = TemporalFeatureExtractor(
            input_dim=1, hidden_dim=32, num_heads=4, dropout=dropout
        )
        self.cross_attention = CrossAttentionFusion(
            dim_2d=512, dim_1d=64, num_heads=4, dropout=dropout
        )
        self.output_head = MultiTaskHead(
            input_dim=576, hidden_dim=256, dropout=dropout
        )

    def forward(self, x_1d, x_2d):
        feat_2d = self.branch_2d(x_2d)
        feat_1d = self.branch_1d(x_1d)
        fused = self.cross_attention(feat_2d, feat_1d)
        return self.output_head(fused)


# 模型变体映射
VARIANT_MODELS = {
    '1d_only': Model1DOnly,
    '2d_only': Model2DOnly,
    'concat': ModelConcat,
    'add': ModelAdd,
    'cross_attn': ModelCrossAttn,
    'resnet18': PETSNetMultimodal,
    # YOLO-FPN 变体
    'swin_yolo': SwinYOLOFPN,
    'vit_yolo': ViTYOLOFPN,
    # DETR 变体
    'detr': DETRStyle,
    # PatchTST 变体（需额外实现）
    # 'swin_patchtst': SwinYOLOFPN + PatchTST,
}

VARIANT_DISPLAY_NAMES = {
    '1d_only': '仅时序分支',
    '2d_only': '仅空间分支',
    'concat': '双分支+拼接',
    'add': '双分支+加法',
    'cross_attn': 'Cross-Attention',
    'resnet18': 'ResNet18基线',
    # YOLO-FPN 变体
    'swin_yolo': 'Swin-YOLO',
    'vit_yolo': 'ViT-YOLO',
    # DETR 变体
    'detr': 'DETR',
    # PatchTST 变体
    # 'swin_patchtst': 'Swin-PatchTST',
}


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model(model, device, data_roots=None, predict_offset=0,
                  seq_len=300, seq_interp_mode='interpolate',
                  remove_contours=False, disabled_batches=None,
                  task='detection', image_size=256, variant_key=None,
                  triple_channel=False):
    """评估模型性能

    Args:
        variant_key: 模型变体标识，用于判断是否需要解包 (output, density) 元组
    """
    if data_roots is None:
        data_roots = get_data_batches()

    _, test_loader = create_multibatch_dataloaders(
        data_roots=data_roots,
        batch_size=16, augment=False,
        image_size=image_size,
        predict_offset=predict_offset,
        seq_len=seq_len,
        seq_interp_mode=seq_interp_mode,
        remove_contours=remove_contours,
        disabled_batches=disabled_batches or [],
        task=task,
        triple_channel=triple_channel
    )

    model = model.to(device)
    model.eval()

    # 判断是否为新变体（返回元组）
    is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']

    all_preds, all_targets = [], []

    # 分割任务专用：收集 Dice 系数
    all_dice_scores = []

    with torch.no_grad():
        for (seq_1d, img_2d), labels in test_loader:
            seq_1d, img_2d = seq_1d.to(device), img_2d.to(device)
            raw_outputs = model(seq_1d, img_2d)

            # 解包新变体的输出（train/eval模式下返回元组）
            if is_new_variant:
                outputs, global_density = raw_outputs  # (B, 6) 或 (B, N, 6), global_density
                # 如果是网格预测，取第一个（推理模式下模型已处理）
                if outputs.dim() == 3:
                    outputs = outputs[:, 0, :]  # 取第一个预测
            else:
                outputs = raw_outputs
                global_density = None

            if task == 'detection':
                # 检测任务：使用 global_density（如果有）或 outputs 的 density 部分
                if global_density is not None:
                    all_preds.append(global_density.cpu())
                else:
                    all_preds.append(outputs[:, 5:6].cpu())
                all_targets.append(labels[:, 5:6].cpu())
            elif task == 'segmentation':
                # 分割任务：outputs 是 (batch, 1, 256, 256)
                # 计算 Dice Score
                preds_binary = (outputs > 0.5).float()
                intersection = (preds_binary * labels.to(device)).sum()
                dice = (2. * intersection + 1e-7) / (preds_binary.sum() + labels.sum() + 1e-7)
                all_dice_scores.append(dice.item())
            else:
                # multitask: outputs 是 (mask, detection) 元组（仅限旧变体）
                if isinstance(outputs, tuple):
                    mask, detection = outputs
                    all_preds.append(detection.cpu().numpy())

                    # 修复：初始化 target_mask 以避免 UnboundLocalError
                    target_mask = None
                    target_det = None

                    # labels 是 (mask, detection) 元组
                    if isinstance(labels, tuple):
                        target_mask, target_det = labels
                        target_det = target_det.to(device)
                        target_mask = target_mask.to(device)
                    else:
                        target_det = labels.to(device)

                    all_targets.append(target_det.cpu().numpy())

                    # 同时计算 Dice Score
                    preds_binary = (mask > 0.5).float()
                    if target_mask is not None:
                        intersection = (preds_binary * target_mask).sum()
                        dice = (2. * intersection + 1e-7) / (preds_binary.sum() + target_mask.sum() + 1e-7)
                        all_dice_scores.append(dice.item())
                else:
                    # 新变体的 multitask（如果有）
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(labels.numpy())

    # 计算评估指标
    if task == 'detection':
        preds, targets = np.vstack(all_preds), np.vstack(all_targets)

        # 调试：检查数据
        print(f"  [DEBUG] Eval preds shape: {preds.shape}, targets shape: {targets.shape}")

        # 提取密度值（可能是 (batch, 1) 或 (batch, 6)）
        if preds.shape[1] == 1:
            pred_d = preds[:, 0]
            target_d = targets[:, 0]
        else:
            pred_d = preds[:, 5]
            target_d = targets[:, 5]

        print(f"  [DEBUG] pred_d: min={np.nanmin(pred_d):.4f}, max={np.nanmax(pred_d):.4f}, std={np.nanstd(pred_d):.4f}")
        print(f"  [DEBUG] target_d: min={np.nanmin(target_d):.4f}, max={np.nanmax(target_d):.4f}, std={np.nanstd(target_d):.4f}")

        # 安全的计算
        pred_d = np.nan_to_num(pred_d, nan=0.0)
        target_d = np.nan_to_num(target_d, nan=0.0)

        mse = np.mean((pred_d - target_d) ** 2)
        target_var = np.var(target_d)
        if target_var > 1e-8:
            r2 = 1 - mse / target_var
        else:
            r2 = 0.0
            print(f"  [DEBUG] R2 计算跳过：target 方差过小")

        # 定位指标（仅当 preds 是完整 6 维时计算）
        if preds.shape[1] >= 6 and targets.shape[1] >= 6:
            ious = box_iou(
                torch.from_numpy(np.clip(preds[:, :4], 0, 1)),
                torch.from_numpy(targets[:, :4])
            ).numpy()
            mIoU = np.nanmean(ious)
        else:
            mIoU = 0.0

        # 物理自洽性
        diffs = pred_d[1:] - pred_d[:-1]
        violation_rate = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0.0

        return {
            'r2': float(r2) if not np.isnan(r2) else 0.0,
            'rmse': float(np.sqrt(mse)) if not np.isnan(mse) else 0.0,
            'mae': float(np.mean(np.abs(pred_d - target_d))) if not np.isnan(np.mean(np.abs(pred_d - target_d))) else 0.0,
            'mIoU': float(mIoU) if not np.isnan(mIoU) else 0.0,
            'violation_rate': float(violation_rate),
            'dice': 0.0,  # 检测任务不计算 Dice
        }
    else:
        # 分割任务：基于 Dice Score 计算
        avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0

        return {
            'r2': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'mIoU': 0.0,
            'violation_rate': 0.0,
            'dice': float(avg_dice),
        }


def eval_checkpoint(checkpoint_path, device, image_size=None):
    """评估检查点

    Args:
        checkpoint_path: 检查点路径
        device: 计算设备
        image_size: 图像尺寸（如果为 None，从检查点读取）
    """
    print_info(f"加载模型: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 处理检查点格式，获取配置
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # 尝试从检查点获取配置
        saved_config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        saved_config = {}

    # 从检查点或参数获取任务类型
    task = saved_config.get('task', 'detection')
    if 'tasksegmentation' in checkpoint_path:
        task = 'segmentation'
    elif 'taskmultitask' in checkpoint_path:
        task = 'multitask'

    # 从检查点或参数获取图像尺寸
    if image_size is None:
        image_size = saved_config.get('image_size', 512)

    # 尝试确定模型类型
    variant_key = None
    for key in VARIANT_MODELS.keys():
        if key in os.path.basename(checkpoint_path):
            variant_key = key
            break
    if variant_key is None:
        variant_key = saved_config.get('variant', 'resnet18')

    # 创建模型
    if variant_key == 'resnet18':
        model = PETSNetMultimodal(
            seq_len=300,
            image_channels=2,
            image_size=image_size,
            pretrained_2d=True,
            task=task
        )
    else:
        ModelClass = VARIANT_MODELS.get(variant_key, PETSNetMultimodal)
        model = ModelClass()

    # 加载权重
    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict}
    model.load_state_dict(matched, strict=False)

    # 评估（传递任务类型、图像尺寸和变体标识）
    return evaluate_model(model, device, task=task, image_size=image_size, variant_key=variant_key)


# =============================================================================
# 训练函数
# =============================================================================

def freeze_model_backbone(model, freeze_2d=True, freeze_1d=False, freeze_names=None):
    """冻结模型骨干网络

    Args:
        model: 模型实例
        freeze_2d: 是否冻结2D骨干
        freeze_1d: 是否冻结1D骨干
        freeze_names: 其他需要冻结的参数名列表
    """
    frozen_count = 0
    trainable_count = 0

    # 定义需要冻结的参数名前缀（兼容多种模型命名）
    # PETSNetMultimodal: branch_2d, branch_1d
    # SwinYOLOFPN/ViTYOLOFPN/DETRStyle: backbone_2d, backbone_1d
    def is_2d_backbone(name):
        return 'branch_2d' in name or 'backbone_2d' in name

    def is_1d_backbone(name):
        return 'branch_1d' in name or 'backbone_1d' in name

    for name, param in model.named_parameters():
        should_freeze = False
        if freeze_2d and is_2d_backbone(name):
            should_freeze = True
        if freeze_1d and is_1d_backbone(name):
            should_freeze = True
        if freeze_names and any(fn in name for fn in freeze_names):
            should_freeze = True
        # 默认冻结骨干，保留检测头和融合层可训练
        if 'output_head' not in name and 'fusion' not in name and 'head' not in name:
            if freeze_2d and is_2d_backbone(name):
                param.requires_grad = False
                frozen_count += 1
            elif freeze_1d and is_1d_backbone(name):
                param.requires_grad = False
                frozen_count += 1
            elif freeze_names and any(fn in name for fn in freeze_names):
                param.requires_grad = False
                frozen_count += 1
            else:
                trainable_count += 1
        else:
            trainable_count += 1
    print_info(f"冻结策略: {frozen_count} 个参数冻结, {trainable_count} 个参数可训练")
    return model


def staged_training(variant_key, config, device, data_roots=None, task_id=None):
    """分阶段训练：先短序列预训练，再长序列微调

    Args:
        variant_key: 模型变体标识
        config: 配置字典
        device: 设备
        data_roots: 数据根目录列表
        task_id: 任务ID（用于检查点标记）

    Returns:
        model: 训练好的模型
    """
    from data.dataset_multimodal import create_multibatch_dataloaders

    # 将 task_id 添加到 config
    if task_id:
        config['task_id'] = task_id

    if data_roots is None:
        data_roots = get_data_batches()

    # 获取短序列批次列表（从配置或默认）
    short_batches = config.get('short_batches', ['单次扫描'])
    phase1_roots = [r for r in data_roots if any(b in os.path.basename(r) for b in short_batches)]

    if not phase1_roots:
        print_warning("未找到短序列批次，将使用全部数据训练")
        phase1_roots = data_roots

    print_title("阶段1: 短序列预训练")
    print_info(f"使用批次: {[os.path.basename(r) for r in phase1_roots]}")

    # 阶段1配置
    phase1_config = config.copy()
    phase1_config['epochs'] = min(50, config['epochs'])  # 阶段1最多50个epoch
    phase1_config['patience'] = min(10, config['patience'])
    phase1_checkpoint = None  # 阶段1不保存单独检查点

    # 创建阶段1数据加载器
    train_loader_1, test_loader_1 = create_multibatch_dataloaders(
        data_roots=phase1_roots,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        predict_offset=config.get('predict_offset', 0),
        seq_len=config.get('feature_len', 300),
        seq_interp_mode=config.get('seq_interp_mode', 'interpolate'),
        remove_contours=config.get('remove_contours', False),
        disabled_batches=config.get('disabled_batches', []),
        task=config.get('task', 'detection'),
        triple_channel=config.get('triple_channel', False)
    )

    # 创建模型
    ModelClass = VARIANT_MODELS.get(variant_key, PETSNetMultimodal)
    model = ModelClass(
        seq_len=config.get('feature_len', 300),
        image_channels=2,
        image_size=config['image_size'],
        pretrained_2d=True,
        dropout=config['dropout'],
        task=config.get('task', 'detection')
    ) if variant_key == 'resnet18' else ModelClass(dropout=config['dropout'])
    model = model.to(device)

    # 阶段1训练
    model, _ = train_model(model, train_loader_1, test_loader_1, phase1_config, device, task_id=task_id)

    # 阶段2: 长序列微调
    print_title("阶段2: 长序列微调")
    print_info(f"使用批次: {[os.path.basename(r) for r in data_roots]}")

    # 冻结骨干网络
    if config.get('freeze_backbone', True):
        freeze_2d = config.get('freeze_2d', True)
        freeze_1d = config.get('freeze_1d', False)
        print_info(f"冻结策略: freeze_2d={freeze_2d}, freeze_1d={freeze_1d}")
        model = freeze_model_backbone(model, freeze_2d=freeze_2d, freeze_1d=freeze_1d)

    # 创建阶段2数据加载器（全部数据）
    train_loader_2, test_loader_2 = create_multibatch_dataloaders(
        data_roots=data_roots,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        predict_offset=config.get('predict_offset', 0),
        seq_len=config.get('feature_len', 300),
        seq_interp_mode=config.get('seq_interp_mode', 'interpolate'),
        remove_contours=config.get('remove_contours', False),
        disabled_batches=config.get('disabled_batches', []),
        task=config.get('task', 'detection'),
        triple_channel=config.get('triple_channel', False)
    )

    # 阶段2配置（使用剩余的epoch）
    phase2_epochs = config['epochs'] - phase1_config['epochs']
    if phase2_epochs > 0:
        phase2_config = config.copy()
        phase2_config['epochs'] = phase2_epochs
        phase2_config['patience'] = config['patience']
        model, _ = train_model(model, train_loader_2, test_loader_2, phase2_config, device, task_id=task_id)
    else:
        print_info("阶段1已完成全部训练，跳过阶段2")

    return model


# =============================================================================
# ETA 估算器：指数移动平均 + 置信区间
# =============================================================================

class ETAEstimator:
    """指数移动平均 ETA 估算器"""

    def __init__(self, total_epochs, alpha=0.3):
        self.total = total_epochs
        self.alpha = alpha  # 近期权重
        self.epoch_times = []
        self.ema = None
        self._epoch_start = None

    def start_epoch(self):
        """开始一个 epoch，计时"""
        import time
        self._epoch_start = time.time()

    def end_epoch(self):
        """结束一个 epoch，返回耗时"""
        import time
        if self._epoch_start is None:
            return 0
        epoch_time = time.time() - self._epoch_start
        self._epoch_start = None
        self.update(epoch_time)
        return epoch_time

    def update(self, epoch_time):
        """更新一个 epoch 的耗时"""
        self.epoch_times.append(epoch_time)
        if self.ema is None:
            self.ema = epoch_time
        else:
            self.ema = self.alpha * epoch_time + (1 - self.alpha) * self.ema

    def get_eta(self, current_epoch):
        """获取 ETA 信息"""
        remaining = self.total - current_epoch
        eta_seconds = self.ema * remaining

        # 预计完成时间
        finish = datetime.now() + __import__('datetime').timedelta(seconds=eta_seconds)

        # 置信区间（至少5个epoch后才计算）
        if len(self.epoch_times) >= 5:
            import statistics
            recent = self.epoch_times[-min(10, len(self.epoch_times)):]
            std = statistics.stdev(recent) if len(recent) > 1 else 0
            confidence = f"±{std:.1f}s"
        else:
            confidence = ""

        # 格式化 ETA
        if eta_seconds >= 3600:
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds >= 60:
            eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        return {
            'ema': self.ema,
            'eta_seconds': eta_seconds,
            'eta_str': eta_str,
            'finish_time': finish.strftime("%H:%M"),
            'confidence': confidence,
            'remaining_epochs': remaining,
        }


def estimate_training_time(model, train_loader, test_loader, criterion, optimizer, device, config):
    """
    通过运行 2 个 epoch 测量实际训练速度（状态安全版）

    估算前保存完整训练状态，估算后恢复。

    返回: (avg_epoch_time, estimated_total_minutes)
    """
    import copy
    from utils.console import print_info, print_warning

    print_info("执行训练速度估算...")

    # 保存训练状态
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer_state = {k: v.clone() for k, v in optimizer.state_dict().items()}
    was_training = model.training

    # 保存 scheduler 状态（如果有）
    scheduler_state = None
    if hasattr(model, '_scheduler'):
        scheduler_state = copy.deepcopy(model._scheduler.state_dict())

    times = []
    model.train()

    # FP16 支持
    fp16_enabled = config.get('fp16', True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if fp16_enabled else None
    variant_key = config.get('variant', 'resnet18')
    task = config.get('task', 'detection')

    # 仅估算 2 个 epoch（足够估算速度）
    num_warmup_epochs = 2

    for i in range(num_warmup_epochs):
        epoch_start = time.time()

        # 训练一个 epoch
        for (seq_1d, img_2d), labels in train_loader:
            seq_1d = seq_1d.to(device)
            img_2d = img_2d.to(device)
            if isinstance(labels, tuple):
                labels = tuple(l.to(device) for l in labels)
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            if fp16_enabled:
                with torch.amp.autocast('cuda'):
                    is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                    if is_new_variant:
                        output, _ = model(seq_1d, img_2d)  # 返回 (outputs, global_density)
                        # 新变体跳过（估算阶段不计算精确损失）
                        loss = torch.tensor(1.0, device=device, requires_grad=True)
                    else:
                        output = model(seq_1d, img_2d)
                        if task == 'segmentation':
                            loss = criterion(output, labels)
                        elif task == 'multitask':
                            loss = criterion(output, labels)
                        else:
                            loss, _ = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                    if is_new_variant:
                        output, _ = model(seq_1d, img_2d)
                        loss = torch.tensor(1.0, device=device, requires_grad=True)
                    else:
                        output = model(seq_1d, img_2d)
                        if task == 'segmentation':
                            loss = criterion(output, labels)
                        elif task == 'multitask':
                            loss = criterion(output, labels)
                        else:
                            loss, _ = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        times.append(time.time() - epoch_start)

    # 取后1个的平均（忽略第一个，可能较慢）
    avg_time = times[-1] if times else 60.0

    # 系数：验证开销 1.2x，早停缓冲 0.85x
    overhead_factor = 1.2
    early_stop_factor = 0.85

    estimated_total = avg_time * config['epochs'] * overhead_factor * early_stop_factor

    # 恢复训练状态
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    if was_training:
        model.train()

    print_info(f"估算完成: {avg_time:.1f}s/epoch, 预估总时间: ~{estimated_total/60:.1f}小时")

    return avg_time, estimated_total / 60  # 返回(单轮时间, 总时间分钟)


def train_model(model, train_loader, test_loader, config, device, checkpoint_path=None, task_id=None):
    """训练模型

    Args:
        task_id: 任务ID（用于检查点标记，团队协作时使用）
    """
    # 将 task_id 添加到 config
    if task_id:
        config['task_id'] = task_id

    # 根据任务类型选择损失函数
    task = config.get('task', 'detection')
    fp16_enabled = config.get('fp16', True)  # 默认启用 FP16
    variant_key = config.get('variant', 'resnet18')

    # 新变体专用损失
    assigner = None
    matcher = None

    if task == 'segmentation':
        criterion = SegmentationLoss()
    elif task == 'multitask':
        criterion = MultimodalSegmentationLoss(
            lambda_seg=1.0,
            lambda_det=0.5
        )
    else:
        # detection 模式
        # 检查是否为新变体
        if variant_key in ['swin_yolo', 'vit_yolo']:
            criterion = YOLOLoss(
                lambda_box=1.0,
                lambda_conf=1.0,
                lambda_mono=0.1
            )
            assigner = YOLOTargetAssigner(grid_size=16, nearby_range=2)
        elif variant_key == 'detr':
            matcher = HungarianMatcher()
            criterion = DETRLoss(matcher=matcher)
        else:
            criterion = MultimodalCrackLoss(
                lambda_mse_density=config['lambda_mse'],
                lambda_mono=config['lambda_mono'],
                lambda_loc=config['lambda_loc'],
                lambda_conf=config['lambda_conf']
            )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'], weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    # FP16 混合精度
    scaler = torch.amp.GradScaler('cuda') if fp16_enabled else None

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    best_epoch = 0  # 最佳Epoch追溯

    # 上一轮指标（用于颜色对比）
    prev_metrics = {'r2': 0.0, 'rmse': float('inf'), 'mae': float('inf'), 'violations': float('inf')}

    # 记录第一轮的基准值（用于趋势计算）
    first_epoch_metrics = None

    # 初始化 ETA 估算器
    eta_estimator = ETAEstimator(config['epochs'], alpha=0.3)

    # 动态训练时间估算（仅在首次训练时）
    if config.get('do_estimate', True) and config['epochs'] >= 10:
        try:
            _, est_total_minutes = estimate_training_time(
                model, train_loader, test_loader, criterion, optimizer, device, config
            )
            hours = est_total_minutes / 60
            print_info(f"[预估] 总训练时间: ~{hours:.1f}小时 (基于前2轮测量)")
        except Exception as e:
            print_info(f"[估算] 估算失败，继续训练: {e}")

    # ============================================================
    # 干跑验证（提前触发 OOM，避免训练中途崩溃）
    # ============================================================
    print_info("执行干跑验证...")
    model.train()
    # 保存模型状态以便 NaN 回滚
    dry_run_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    dry_run_fp16_disabled = False
    try:
        (seq_1d, img_2d), labels = next(iter(train_loader))
        # 修复：多任务标签可能是元组，需要解包后再移动到设备
        seq_1d = seq_1d.to(device)
        img_2d = img_2d.to(device)
        if isinstance(labels, tuple):
            labels = tuple(l.to(device) for l in labels)
        else:
            labels = labels.to(device)
        optimizer.zero_grad()

        # 前向传播
        if fp16_enabled and not dry_run_fp16_disabled:
            with torch.amp.autocast('cuda'):
                # 检查是否为新变体
                is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                if is_new_variant:
                    outputs, global_density = model(seq_1d, img_2d)
                    if variant_key in ['swin_yolo', 'vit_yolo']:
                        assigned_target, pos_mask = assigner(labels)
                        loss = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                    else:  # detr
                        indices = matcher(outputs, {'labels': labels})
                        loss = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                else:
                    outputs = model(seq_1d, img_2d)
                    if task == 'segmentation':
                        loss = criterion(outputs, labels)
                    elif task == 'multitask':
                        loss = criterion(outputs, labels)
                    else:
                        loss, _ = criterion(outputs, labels)
        else:
            with torch.amp.autocast('cuda', enabled=False):
                # 检查是否为新变体
                is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                if is_new_variant:
                    outputs, global_density = model(seq_1d, img_2d)
                    if variant_key in ['swin_yolo', 'vit_yolo']:
                        assigned_target, pos_mask = assigner(labels)
                        loss = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                    else:  # detr
                        indices = matcher(outputs, {'labels': labels})
                        loss = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                else:
                    outputs = model(seq_1d, img_2d)
                    if task == 'segmentation':
                        loss = criterion(outputs, labels)
                    elif task == 'multitask':
                        loss = criterion(outputs, labels)
                    else:
                        loss, _ = criterion(outputs, labels)

        # NaN 检查
        is_nan = torch.isnan(loss).item()
        debug_str = f"loss={loss.item():.6f}, is_nan={is_nan}"
        if is_nan:
            debug_str += " [NaN!]"
            print(f"  [DEBUG] Dry run: {debug_str}")
            print(f"    outputs[:3]={outputs[:3].detach().cpu().numpy()}")
            print(f"    labels[:3]={labels[:3].cpu().numpy()}")
        else:
            print(f"  [DEBUG] Dry run: {debug_str}")
        if is_nan:
            torch.cuda.empty_cache()
            # 重试不使用 FP16
            model.load_state_dict(dry_run_model_state)
            optimizer.zero_grad()
            dry_run_fp16_disabled = True

            with torch.amp.autocast('cuda', enabled=False):
                is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                if is_new_variant:
                    outputs, global_density = model(seq_1d, img_2d)
                    if variant_key in ['swin_yolo', 'vit_yolo']:
                        assigned_target, pos_mask = assigner(labels)
                        loss = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                    else:  # detr
                        indices = matcher(outputs, {'labels': labels})
                        loss = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                else:
                    outputs = model(seq_1d, img_2d)
                    if task == 'segmentation':
                        loss = criterion(outputs, labels)
                    elif task == 'multitask':
                        loss = criterion(outputs, labels)
                    else:
                        loss, _ = criterion(outputs, labels)

                if torch.isnan(loss):
                    raise RuntimeError("NaN loss 持续存在，可能是数据或模型问题")
                print_warning("FP16 已禁用用于此训练")

        # 反向传播
        if fp16_enabled and not dry_run_fp16_disabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print_success("干跑验证通过 ✅")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            print_error(f"显存不足！{e}")
            print_info("建议：python run_train.py --image_size 256 --batch_size 4")
            raise RuntimeError("OOM_ERROR_SUGGEST_RESTART") from e
        raise
    finally:
        torch.cuda.empty_cache()
        model.train()

    # 如果干跑时禁用了 FP16，主训练也禁用
    if dry_run_fp16_disabled:
        fp16_enabled = False
        if scaler is not None:
            scaler = None
        print_warning("主训练也禁用 FP16（与干跑一致）")

    # 初始化 epoch_elapsed（防止异常路径下未定义）
    epoch_elapsed = 0.0

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()  # 记录 epoch 开始时间

        try:
            for train_batch_idx, ((seq_1d, img_2d), labels) in enumerate(train_loader):
                # 修复：多任务标签可能是元组，需要解包后再移动到设备
                seq_1d = seq_1d.to(device)
                img_2d = img_2d.to(device)
                if isinstance(labels, tuple):
                    labels = tuple(l.to(device) for l in labels)
                else:
                    labels = labels.to(device)
                optimizer.zero_grad()

                if fp16_enabled:
                    with torch.amp.autocast('cuda'):
                        # 检查是否为新变体
                        is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                        if is_new_variant:
                            outputs, global_density = model(seq_1d, img_2d)
                            if variant_key in ['swin_yolo', 'vit_yolo']:
                                assigned_target, pos_mask = assigner(labels)
                                loss_total = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                            else:  # detr
                                indices = matcher(outputs, {'labels': labels})
                                loss_total = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                        else:
                            # resnet18 等标准变体
                            outputs = model(seq_1d, img_2d)
                            if task == 'segmentation':
                                loss_total = criterion(outputs, labels)
                            elif task == 'multitask':
                                loss_total = criterion(outputs, labels)
                            else:
                                loss_total, _ = criterion(outputs, labels)
                    # 调试：在 autocast 外部打印
                    if epoch == 0 and train_batch_idx < 3:
                        print(f"  [DEBUG] FP16 outputs[:2]={outputs[:2].detach().cpu().numpy()}")
                        print(f"  [DEBUG] FP16 outputs is_nan={torch.isnan(outputs).any().item()}")
                    scaler.scale(loss_total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 检查是否为新变体
                    is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                    if is_new_variant:
                        outputs, global_density = model(seq_1d, img_2d)
                        if variant_key in ['swin_yolo', 'vit_yolo']:
                            assigned_target, pos_mask = assigner(labels)
                            loss_total = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                        else:  # detr
                            indices = matcher(outputs, {'labels': labels})
                            loss_total = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                    else:
                        outputs = model(seq_1d, img_2d)
                        if epoch == 0 and train_batch_idx < 3:
                            print(f"  [DEBUG] outputs[:2]={outputs[:2].detach().cpu().numpy()}")
                            print(f"  [DEBUG] outputs is_nan={torch.isnan(outputs).any().item()}")
                        if task == 'segmentation':
                            loss_total = criterion(outputs, labels)
                        elif task == 'multitask':
                            loss_total = criterion(outputs, labels)
                        else:
                            loss_total, _ = criterion(outputs, labels)
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                train_loss += loss_total.item()

                # 调试：检查前几个 batch 的 loss
                if epoch == 0 and train_batch_idx < 3:
                    loss_str = f"loss={loss_total.item():.6f}"
                    if torch.isnan(loss_total).item():
                        loss_str += " [NaN!]"
                        # 额外调试：检查模型输出
                        print(f"  [DEBUG] Train Batch {train_batch_idx}: {loss_str}")
                        print(f"    outputs[:3]={outputs[:3].detach().cpu().numpy()}")
                        print(f"    labels[:3]={labels[:3].cpu().numpy()}")
                    else:
                        print(f"  [DEBUG] Train Batch {train_batch_idx}: {loss_str}")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print_error(f"训练中发生 OOM！")
                print_info(f"建议手动运行：python run_train.py --image_size 256 --batch_size 4")
                raise RuntimeError("OOM_ERROR_SUGGEST_RESTART") from e
            raise

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for (seq_1d, img_2d), labels in test_loader:
                # 修复：多任务标签可能是元组，需要解包后再移动到设备
                seq_1d = seq_1d.to(device)
                img_2d = img_2d.to(device)
                if isinstance(labels, tuple):
                    labels = tuple(l.to(device) for l in labels)
                else:
                    labels = labels.to(device)
                # 检查是否为新变体
                is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr']
                if is_new_variant:
                    outputs, global_density = model(seq_1d, img_2d)
                    if epoch == 0 and len(all_preds) < 3:
                        print(f"  [DEBUG] Val {variant_key} outputs[:2]={outputs[:2].detach().cpu().numpy()}")
                        print(f"  [DEBUG] Val global_density[:2]={global_density[:2].detach().cpu().numpy()}")
                    if variant_key in ['swin_yolo', 'vit_yolo']:
                        assigned_target, pos_mask = assigner(labels)
                        loss_total = criterion(outputs, assigned_target, global_density, labels[:, 5:6], pos_mask)
                        # 收集预测用于指标计算（取global_density）
                        all_preds.append(global_density.cpu())
                        all_targets.append(labels[:, 5:6].cpu())
                    else:  # detr
                        indices = matcher(outputs, {'labels': labels})
                        loss_total = criterion(outputs, {'labels': labels}, global_density, labels[:, 5:6], indices)
                        all_preds.append(global_density.cpu())
                        all_targets.append(labels[:, 5:6].cpu())
                else:
                    outputs = model(seq_1d, img_2d)
                    if epoch == 0 and len(all_preds) < 3:
                        print(f"  [DEBUG] Val outputs[:2]={outputs[:2].detach().cpu().numpy()}")
                        print(f"  [DEBUG] Val outputs is_nan={torch.isnan(outputs).any().item()}")
                    # 根据任务类型计算损失
                    if task == 'segmentation':
                        loss_total = criterion(outputs, labels)
                    elif task == 'multitask':
                        loss_total = criterion(outputs, labels)
                    else:
                        # detection: outputs (batch, 6), labels (batch, 6)
                        loss_total, _ = criterion(outputs, labels)
                        # 收集预测用于指标计算
                        all_preds.append(outputs[:, 5:6].cpu())
                        all_targets.append(labels[:, 5:6].cpu())

                val_loss += loss_total.item()

        val_loss /= len(test_loader)
        scheduler.step()

        # 计算评估指标
        if task == 'detection' and all_preds:
            import numpy as np
            from scipy import stats
            preds = torch.cat(all_preds).numpy().flatten()
            targets = torch.cat(all_targets).numpy().flatten()

            # 调试：检查数据
            print(f"  [DEBUG] Val: preds.shape={preds.shape}, targets.shape={targets.shape}")
            print(f"  [DEBUG] preds: min={np.nanmin(preds):.4f}, max={np.nanmax(preds):.4f}, std={np.nanstd(preds):.4f}")
            print(f"  [DEBUG] targets: min={np.nanmin(targets):.4f}, max={np.nanmax(targets):.4f}, std={np.nanstd(targets):.4f}")

            # 安全的 R2 计算
            if len(preds) > 1 and np.nanstd(preds) > 1e-8 and np.nanstd(targets) > 1e-8:
                r2 = stats.pearsonr(preds, targets)[0] ** 2
                if np.isnan(r2):
                    r2 = 0.0
            else:
                r2 = 0.0
                print(f"  [DEBUG] R2 计算跳过：方差过小或样本不足")

            rmse = np.sqrt(np.nanmean((preds - targets) ** 2))
            mae = np.nanmean(np.abs(preds - targets))
            # 计算违反率
            diffs = np.diff(targets)
            violations = np.sum(diffs > 0) / max(len(diffs), 1)

            # 颜色标记：↑变好(绿) ↓变差(红) -持平(灰)
            GREEN = "\033[92m"
            RED = "\033[91m"
            GRAY = "\033[90m"
            RESET = "\033[0m"

            def _color_val(val, prev, lower_is_better=False, fmt=".4f"):
                """根据变化返回带颜色的箭头和值"""
                if prev == 0.0 or prev == float('inf'):
                    return f"{GRAY}{val:{fmt}}{RESET}"
                if lower_is_better:
                    if val < prev - 0.0001:
                        return f"{GREEN}↓{val:{fmt}}{RESET}"
                    elif val > prev + 0.0001:
                        return f"{RED}↑{val:{fmt}}{RESET}"
                    else:
                        return f"{GRAY}-{val:{fmt}}{RESET}"
                else:
                    if val > prev + 0.0001:
                        return f"{GREEN}↑{val:{fmt}}{RESET}"
                    elif val < prev - 0.0001:
                        return f"{RED}↓{val:{fmt}}{RESET}"
                    else:
                        return f"{GRAY}-{val:{fmt}}{RESET}"

            r2_str = _color_val(r2, prev_metrics['r2'], lower_is_better=False)
            rmse_str = _color_val(rmse, prev_metrics['rmse'], lower_is_better=True)
            mae_str = _color_val(mae, prev_metrics['mae'], lower_is_better=True)
            vio_str = _color_val(violations*100, prev_metrics['violations']*100, lower_is_better=True, fmt=".1f")

            # 分两行显示：R2, RMSE | MAE, 违反率
            metrics_line1 = f"  R2: {r2_str} | RMSE: {rmse_str}"
            metrics_line2 = f"  MAE: {mae_str} | 违反率: {vio_str}%"

            # 更新上一轮指标
            prev_metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'violations': violations}

            # 记录第一轮的基准值（用于趋势计算）
            if first_epoch_metrics is None:
                first_epoch_metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'violations': violations,
                                      'loss': val_loss, 'time': epoch_elapsed}
        else:
            metrics_line1 = ""
            metrics_line2 = ""


        # 保存最佳模型
        if val_loss < best_loss - config['min_delta']:
            best_loss = val_loss
            best_epoch = epoch + 1  # 记录最佳Epoch
            patience_counter = 0
            if checkpoint_path:
                # 保存模型权重和元数据（task、config、task_id）
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'task': config.get('task', 'detection'),
                    'config': config,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'timestamp': datetime.now().isoformat(),
                }
                # 添加 task_id（如果有，团队协作时使用）
                if config.get('task_id'):
                    checkpoint_data['task_id'] = config['task_id']
                torch.save(checkpoint_data, checkpoint_path)
        else:
            patience_counter += 1

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 获取GPU显存使用（如果有GPU）
        gpu_mem_str = ""
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            gpu_mem_str = f"GPU: {gpu_mem_allocated:.1f}GB"

        # 计算本轮耗时并更新 ETA
        epoch_elapsed = time.time() - epoch_start
        eta_estimator.update(epoch_elapsed)
        eta_info = eta_estimator.get_eta(epoch + 1)

        # 总耗时
        total_elapsed = time.time() - start_time

        # 普通输出：两行对齐
        # 第一行：Epoch + 训练信息
        epoch_prefix = f"Epoch {epoch+1}/{config['epochs']} | "
        line1 = (f"{epoch_prefix}"
                 f"Loss: {train_loss:.4f} (Best: {best_loss:.4f}) | "
                 f"LR: {current_lr:.2e} | Time: {epoch_elapsed:.1f}s | {gpu_mem_str}")

        # 第二行：验证指标（缩进与 "Epoch x/x | " 对齐）
        if metrics_line1 and metrics_line2:
            indent = " " * len(epoch_prefix)
            line2 = indent + metrics_line1[2:]  # 去掉 metrics_line1 前缀的 "  "
        else:
            line2 = ""

        print_info(line1)
        if line2:
            print_info(line2)

        # 每10轮输出盒子汇总（升级版）
        if (epoch + 1) % 10 == 0:
            # 计算趋势（与第一轮相比）
            if first_epoch_metrics:
                loss_delta = val_loss - first_epoch_metrics['loss']
                r2_delta = r2 - first_epoch_metrics['r2']
                rmse_delta = rmse - first_epoch_metrics['rmse']
                mae_delta = mae - first_epoch_metrics['mae']
                vio_delta = (violations - first_epoch_metrics['violations']) * 100
            else:
                loss_delta = r2_delta = rmse_delta = mae_delta = vio_delta = 0.0

            # 进度条
            progress = (epoch + 1) / config['epochs']
            progress_pct = f"{progress * 100:.0f}%"
            progress_bar = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))

            # 边框
            content_width = BOX_WIDTH - 2
            top_border = "╔" + "═" * content_width + "╗"
            bot_border = "╚" + "═" * content_width + "╝"

            # Epoch标题行
            epoch_title = f"Epoch {epoch+1}/{config['epochs']} | Best: {best_epoch} | Progress: [{progress_bar}] {progress_pct}"

            # 箭头
            loss_arrow = "↓" if loss_delta < 0 else ("↑" if loss_delta > 0 else "-")
            r2_arrow = "↑" if r2_delta > 0 else ("↓" if r2_delta < 0 else "-")
            rmse_arrow = "↓" if rmse_delta < 0 else ("↑" if rmse_delta > 0 else "-")
            mae_arrow = "↓" if mae_delta < 0 else ("↑" if mae_delta > 0 else "-")
            vio_arrow = "↓" if vio_delta < 0 else ("↑" if vio_delta > 0 else "-")

            # GPU显存
            gpu_part = gpu_mem_str.split()[-1] if gpu_mem_str else "N/A"

            # 生成盒子
            print_info(top_border)
            print_info(print_box_line(epoch_title))
            print_info(print_box_divider())

            # Train行
            print_info(print_box_line(f"Train | Loss: {train_loss:.4f}/{val_loss:.4f} (Best: {best_loss:.4f}) | Δ {loss_arrow}{loss_delta:+.4f}"))
            print_info(print_box_line(f"       | LR: {current_lr:.2e} | Time: {epoch_elapsed:.1f}s | Patience: {patience_counter}/{config['patience']}"))
            print_info(print_box_divider())

            # Valid行
            print_info(print_box_line(f"Valid | R2: {r2_arrow}{r2:.4f} ({r2_delta:+.4f}) | RMSE: {rmse_arrow}{rmse:.4f} ({rmse_delta:+.4f})"))
            print_info(print_box_line(f"       | MAE: {mae_arrow}{mae:.4f} ({mae_delta:+.4f}) | 违反率: {vio_arrow}{violations*100:.1f}% ({vio_delta:+.1f}%)"))
            print_info(print_box_divider())

            # System行 - 使用 EMA + 置信区间 ETA
            eta_display = f"ETA: {eta_info['eta_str']}" if eta_info['eta_str'] != '0.0s' else "ETA: 计算中"
            confidence_display = f"{eta_info['confidence']}" if eta_info['confidence'] else ""
            finish_display = f"| 完成: {eta_info['finish_time']}" if eta_info['finish_time'] else ""
            print_info(print_box_line(f"System| EMA: {eta_info['ema']:.1f}s/epoch | {eta_display} {confidence_display} {finish_display}"))
            print_info(print_box_line(f"       | GPU: {gpu_part} | 进度: {progress_pct} | 剩余: {eta_info['remaining_epochs']}轮"))

            print_info(bot_border)

        if patience_counter >= config['patience']:
            print_warning(f"早停: 连续 {config['patience']} 个epoch未改善")
            break

    total_time = time.time() - start_time
    print_info(f"训练完成! 总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    return model, {'val_loss': best_loss}


def train_variant(variant_key, config, device, data_roots=None, task_id=None):
    """训练单个变体（消融实验用）"""
    if data_roots is None:
        data_roots = get_data_batches()

    # 根据 variant_key 选择模型类
    ModelClass = VARIANT_MODELS.get(variant_key, PETSNetMultimodal)
    display_name = VARIANT_DISPLAY_NAMES.get(variant_key, variant_key)
    print_info(f"训练变体: {display_name} ({variant_key})")

    # 检查点命名包含变体、架构信息和 predict_offset
    predict_offset = config.get('predict_offset', 0)
    backbone_2d = config.get('backbone_2d', 'resnet18')
    backbone_1d = config.get('backbone_1d', 'cnn_attn')
    fusion = config.get('fusion', 'cross_attn')
    task = config.get('task', 'detection')
    force_retrain = config.get('force_retrain', False)
    resume_path = config.get('resume', None)

    # 将 task_id 添加到 config（用于检查点元数据）
    if task_id:
        config['task_id'] = task_id

    # 修复：检查点文件名包含 variant_key，避免不同变体互相覆盖
    # 格式：{variant}_{backbone_2d}_{backbone_1d}_{fusion}_task{task}_offset{offset}_best.pt
    save_path = os.path.join(
        CHECKPOINT_DIR,
        f"{variant_key}_{backbone_2d}_{backbone_1d}_{fusion}_task{task}_offset{predict_offset}_best.pt"
    )

    # ========== 模型实例化（根据变体类型）==========

    def create_variant_model():
        """根据变体类型创建模型"""
        if variant_key == 'resnet18':
            # 完整模型：使用 PETSNetMultimodal
            return PETSNetMultimodal(
                seq_len=config.get('feature_len', 300),
                image_channels=2,
                image_size=config['image_size'],
                pretrained_2d=True,
                dropout=config['dropout'],
                task=task
            )
        else:
            # 消融变体：使用对应的简化模型
            return ModelClass(dropout=config['dropout'])

    # ========== 强化检查点加载逻辑 ==========

    # Case 1: predict_offset > 0 → 强制从头训练
    if predict_offset > 0:
        if os.path.exists(save_path):
            print_warning(f"[WARN] 检测到 offset={predict_offset} 的检查点存在: {save_path}")
            print_warning(f"   策略：强制重新训练，不加载旧权重")
            print_warning(f"   原因：不同 offset 的模型权重不兼容")
        model = create_variant_model()
        start_epoch = 0
        best_metric = float('inf')

    # Case 2: force_retrain=True → 强制从头训练
    elif force_retrain:
        if os.path.exists(save_path):
            print_warning(f"[WARN] 检测到检查点存在，但 force_retrain=True")
            print_warning(f"   策略：强制重新训练，不加载旧权重")
        model = create_variant_model()
        start_epoch = 0
        best_metric = float('inf')

    # Case 3: 指定 resume 路径 → 加载指定检查点并继续训练
    elif resume_path:
        if os.path.exists(resume_path):
            print_info(f"[OK] 从指定路径加载检查点: {resume_path}")
            print_info(f"   继续训练...")
            model = create_variant_model()
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_metric = checkpoint.get('best_metric', float('inf'))
            else:
                model.load_state_dict(checkpoint)
                start_epoch = 0
                best_metric = float('inf')
        else:
            print_error(f"[ERROR] 指定的 resume 路径不存在: {resume_path}")
            print_info(f"   策略：从头开始训练")
            model = create_variant_model()
            start_epoch = 0
            best_metric = float('inf')

    # Case 4: predict_offset == 0 且检查点存在 → 加载并返回
    elif os.path.exists(save_path):
        print_info(f"[OK] 加载检查点: {save_path}")
        model = create_variant_model()
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print_info(f"   继续训练...")
        model.eval()
        return model

    # Case 3: 无检查点 → 从头训练
    else:
        print_info(f"[NEW] 未找到检查点，从头训练")
        model = create_variant_model()

    # =========================================

    model = model.to(device)

    # 加载数据
    print_info("加载数据集...")
    train_loader, test_loader = create_multibatch_dataloaders(
        data_roots=data_roots,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        augment=True,
        predict_offset=predict_offset,
        seq_len=config.get('feature_len', 300),
        seq_interp_mode=config.get('seq_interp_mode', 'interpolate'),
        remove_contours=config.get('remove_contours', False),
        disabled_batches=config.get('disabled_batches', []),
        task=task,
        triple_channel=config.get('triple_channel', False)
    )

    # 验证数据集非空
    if len(train_loader.dataset) == 0:
        print_error("错误: 训练数据为空!")
        raise ValueError("训练数据为空，请检查数据路径配置")
    if len(test_loader.dataset) == 0:
        print_error("错误: 测试数据为空!")
        raise ValueError("测试数据为空，请检查数据路径配置")

    print_info(f"训练样本: {len(train_loader.dataset)}, 测试样本: {len(test_loader.dataset)}")
    if predict_offset > 0:
        print_info(f"时间偏移: 预测 {predict_offset} 步后的标签 ({predict_offset * 0.05}s)")

    # 训练
    model, metrics = train_model(
        model, train_loader, test_loader, config, device, save_path, task_id=task_id
    )
    model.eval()
    return model


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PE-MMNet v4 统一训练脚本')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'ablation', 'eval'],
                        help='模式: train(训练), ablation(消融), eval(评估)')

    # 数据
    parser.add_argument('--batches', nargs='+', default=None,
                        help='数据批次列表（默认全部）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小，不指定则自动根据显存选择')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--lambda_mono', type=float, default=0.1, help='单调性损失权重')
    parser.add_argument('--predict_offset', type=int, default=0,
                        help='时间偏移量（预测未来多少步，如1=0.05s后，2=0.1s后）')

    # 模型
    parser.add_argument('--variant', type=str, default='resnet18',
                        choices=['1d_only', '2d_only', 'concat', 'add', 'cross_attn', 'resnet18',
                                'swin_yolo', 'vit_yolo', 'detr'],
                        help='模型变体（train模式）')

    # 架构配置（train模式，可选）
    parser.add_argument('--backbone_2d', type=str, default='resnet18',
                        choices=['resnet18', 'vit_small'],
                        help='2D空间骨干网络（默认resnet18）')
    parser.add_argument('--backbone_1d', type=str, default='cnn_attn',
                        choices=['cnn_attn', 'transformer', 'dlinear'],
                        help='1D时序骨干网络（默认cnn_attn）')
    parser.add_argument('--fusion', type=str, default='cross_attn',
                        choices=['cross_attn', 'concat', 'adaptive'],
                        help='多模态融合策略（默认cross_attn）')

    # 数据处理配置
    parser.add_argument('--feature_len', type=int, default=300,
                        help='1D序列统一特征长度（默认300）')
    parser.add_argument('--seq_interp_mode', type=str, default='interpolate',
                        choices=['interpolate', 'pool'],
                        help='长序列处理模式：interpolate（插值）或 pool（池化），默认interpolate')

    # 图像尺寸（默认None，由auto_select_config根据显存自动选择）
    parser.add_argument('--image_size', type=int, default=None,
                        choices=[256, 384, 512, 768, 1024],
                        help='图像尺寸：256 / 384 / 512(默认) / 768 / 1024，更高分辨率需要更多显存')

    # FP16 混合精度（默认启用）
    parser.add_argument('--no_fp16', action='store_true', default=False,
                        help='禁用 FP16 混合精度（默认启用，省显存 50%%）')

    # 任务模式
    parser.add_argument('--task', type=str, default='detection',
                        choices=['detection', 'segmentation', 'multitask'],
                        help='任务模式: detection(矩形框), segmentation(掩膜), multitask(两者)')

    # 等值线预处理（仅用于参数化扫描4）
    parser.add_argument('--remove_contours', action='store_true', default=False,
                        help='对参数化扫描4启用等值线去除预处理')
    parser.add_argument('--disable_batch', type=str, nargs='+', default=[],
                        choices=['单次扫描', '参数化扫描1', '参数化扫描2', '参数化扫描3', '参数化扫描4'],
                        help='禁用指定的批次（可多个），如 --disable_batch 参数化扫描4')

    # 架构自适应学习率
    parser.add_argument('--arch_lr_scale', type=float, default=None,
                        help='架构学习率缩放因子（如 0.5, 2.0），覆盖自动配置')

    # 检查点
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--resume', type=str, default=None, help='从检查点继续训练')
    parser.add_argument('--force_retrain', action='store_true',
                        help='强制重新训练，即使检查点已存在')
    parser.add_argument('--task_id', type=str, default=None,
                        help='任务ID（用于检查点标记，团队协作时使用）')

    # 分阶段训练
    parser.add_argument('--staged_train', action='store_true', default=False,
                        help='启用分阶段训练：先短序列预训练，再长序列微调')
    parser.add_argument('--short_batches', nargs='+', default=None,
                        help='短序列批次列表，用于阶段1训练（默认使用单次扫描）')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='阶段2是否冻结骨干网络（默认True）')
    parser.add_argument('--freeze_2d', action='store_true', default=True,
                        help='阶段2是否冻结2D骨干（默认True）')
    parser.add_argument('--freeze_1d', action='store_true', default=False,
                        help='阶段2是否冻结1D骨干（默认False）')

    # 三通道时序输入
    parser.add_argument('--triple_channel', action='store_true', default=False,
                        help='启用三通道时序输入：初始温度 + 当前温度 + 温度变化率')

    # 评估
    parser.add_argument('--eval_output', type=str, default=None, help='评估结果保存路径')

    args = parser.parse_args()

    # 跟踪用户显式指定的参数（用于 auto_select_config 决策）
    args._user_specified = {
        'image_size': args.image_size is not None,  # 非默认值时为 True
        'batch_size': args.batch_size is not None,
    }

    set_seed(42)
    device = get_device()

    # 架构自适应学习率
    arch_config = get_arch_specific_config(
        args.backbone_2d, args.backbone_1d,
        args_lr=args.lr if args.lr != 3e-4 else None,  # 只有非默认值才覆盖
        args_dropout=args.dropout if args.dropout != 0.2 else None
    )

    # 应用架构缩放因子
    if args.arch_lr_scale is not None:
        arch_config['lr'] *= args.arch_lr_scale

    # 显存自适应配置
    auto_config = auto_select_config(args)
    free_gb = get_free_gpu_memory_gb()
    print_auto_config_info(auto_config, free_gb)

    # 配置
    config = {
        'epochs': args.epochs,
        'batch_size': auto_config['batch_size'],
        'learning_rate': arch_config['lr'],
        'dropout': arch_config['dropout'],
        'patience': args.patience,
        'min_delta': 1e-4,
        'lambda_mse': 1.0,
        'lambda_mono': args.lambda_mono,
        'lambda_loc': 1.0,
        'lambda_conf': 1.0,
        'predict_offset': args.predict_offset,
        # 架构配置
        'backbone_2d': args.backbone_2d,
        'backbone_1d': args.backbone_1d,
        'fusion': args.fusion,
        'feature_len': args.feature_len,
        'seq_interp_mode': args.seq_interp_mode,
        'image_size': auto_config['image_size'],  # 使用自适应配置
        'fp16': auto_config['fp16'],  # FP16 配置
        'remove_contours': args.remove_contours,
        'disabled_batches': args.disable_batch,
        'task': args.task,
        'triple_channel': args.triple_channel,
        # 检查点控制
        'force_retrain': args.force_retrain,
        'resume': args.resume,
        'variant': args.variant,  # 用于 train_model 选择损失函数
        # 记录原始值用于追溯
        'final_image_size': auto_config['image_size'],
        'final_batch_size': auto_config['batch_size'],
        'fp16_enabled': auto_config['fp16'],
    }

    # 打印架构配置
    print_info(f"架构配置: 2D={args.backbone_2d}, 1D={args.backbone_1d}, Fusion={args.fusion}")
    print_info(f"序列长度: {args.feature_len}, 处理模式: {args.seq_interp_mode}")
    print_info(f"实际训练配置: 图像 {config['final_image_size']}x{config['final_image_size']}, batch={config['final_batch_size']}, FP16={'启用' if config['fp16_enabled'] else '禁用'}")
    if args.remove_contours:
        print_info(f"等值线预处理: 启用")
    if args.disable_batch:
        print_info(f"禁用批次: {args.disable_batch}")

    # 打印标题
    print_header("PE-MMNet v4 统一训练")
    print_info(f"设备: {device}")
    print_info(f"模式: {args.mode}")

    # 确保配置有效
    if not ensure_config(interactive=True):
        print_error("配置无效，请检查 config.json")
        return

    # 获取数据批次
    data_roots = get_data_batches()
    if not data_roots:
        print_error("未找到数据批次，请检查数据根目录配置")
        return

    print_info(f"数据批次: {[os.path.basename(r) for r in data_roots]}")

    # ============================================================
    # 评估模式
    # ============================================================
    if args.mode == 'eval':
        if not args.checkpoint:
            print_error("评估模式需要 --checkpoint 参数")
            return

        metrics = eval_checkpoint(args.checkpoint, device, image_size=args.image_size)
        display_name = VARIANT_DISPLAY_NAMES.get(
            next((k for k in VARIANT_MODELS if k in args.checkpoint), 'resnet18'),
            os.path.basename(args.checkpoint)
        )

        # 根据任务类型打印不同指标
        task = 'detection'
        if 'tasksegmentation' in args.checkpoint:
            task = 'segmentation'
        elif 'taskmultitask' in args.checkpoint:
            task = 'multitask'

        if task == 'segmentation':
            print_results_table({
                '模型': display_name,
                'Dice': (metrics['dice'] * 100, ".2f", "%"),
            })
        elif task == 'multitask':
            print_results_table({
                '模型': display_name,
                'R2': metrics['r2'],
                'mIoU': metrics['mIoU'],
                'Dice': (metrics['dice'] * 100, ".2f", "%"),
            })
        else:
            print_results_table({
                '模型': display_name,
                'R2': metrics['r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'mIoU': metrics['mIoU'],
                '违反率': (metrics['violation_rate'] * 100, ".1f", "%"),
            })

        # 保存结果
        if args.eval_output:
            with open(args.eval_output, 'w', encoding='utf-8') as f:
                json.dump({
                    'checkpoint': args.checkpoint,
                    'variant': display_name,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            print_success(f"结果已保存: {args.eval_output}")

        # 输出纯文本 JSON 行（供批量工具解析）
        print()
        print("__EVAL_JSON__" + json.dumps(metrics) + "__EVAL_JSON__")

    # ============================================================
    # 训练模式
    # ============================================================
    elif args.mode == 'train':
        print_section(f"开始训练: {VARIANT_DISPLAY_NAMES.get(args.variant, args.variant)}")
        print_info(f"任务模式: {args.task}")

        # 分阶段训练模式
        if args.staged_train:
            print_info("启用分阶段训练模式")
            # 更新配置中的分阶段参数
            config['staged_train'] = True
            config['short_batches'] = args.short_batches or ['单次扫描']
            config['freeze_backbone'] = args.freeze_backbone
            config['freeze_2d'] = args.freeze_2d
            config['freeze_1d'] = args.freeze_1d
            model = staged_training(args.variant, config, device, data_roots)
        else:
            model = train_variant(args.variant, config, device, data_roots, task_id=args.task_id)

        if model is not None:
            # 训练完成后自动评估
            print_title("训练完成，开始评估...")
            metrics = evaluate_model(
                model, device, data_roots,
                predict_offset=config.get('predict_offset', 0),
                seq_len=config.get('feature_len', 300),
                seq_interp_mode=config.get('seq_interp_mode', 'interpolate'),
                remove_contours=config.get('remove_contours', False),
                disabled_batches=config.get('disabled_batches', []),
                task=args.task,
                image_size=config.get('image_size', 256),
                variant_key=args.variant,
                triple_channel=config.get('triple_channel', False)
            )

            if args.task == 'segmentation':
                print_results_table({
                    'Dice': (metrics['dice'] * 100, ".2f", "%"),
                })
            elif args.task == 'multitask':
                print_results_table({
                    'R2': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'mIoU': metrics['mIoU'],
                    'Dice': (metrics['dice'] * 100, ".2f", "%"),
                })
            else:
                print_results_table({
                    'R2': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'mIoU': metrics['mIoU'],
                    '违反率': (metrics['violation_rate'] * 100, ".1f", "%"),
                })

            # 输出纯文本 JSON 行（供批量工具解析）
            print()
            print("__EVAL_JSON__" + json.dumps(metrics) + "__EVAL_JSON__")

    # ============================================================
    # 消融实验模式
    # ============================================================
    elif args.mode == 'ablation':
        print_title("消融实验")
        print_info("将训练6个模型变体，预计需要较长时间...")

        results = {}
        total_start = time.time()

        for i, (variant_key, ModelClass) in enumerate(VARIANT_MODELS.items()):
            print_progress(i + 1, len(VARIANT_MODELS), f"训练变体: {VARIANT_DISPLAY_NAMES[variant_key]}")

            model = train_variant(variant_key, config, device, data_roots)

            if model is not None:
                metrics = evaluate_model(
                    model, device, data_roots,
                    predict_offset=config.get('predict_offset', 0),
                    seq_len=config.get('feature_len', 300),
                    seq_interp_mode=config.get('seq_interp_mode', 'interpolate'),
                    remove_contours=config.get('remove_contours', False),
                    disabled_batches=config.get('disabled_batches', []),
                    task=args.task,
                    image_size=config.get('image_size', 256),
                    variant_key=variant_key,
                    triple_channel=config.get('triple_channel', False)
                )
                results[variant_key] = {**metrics, 'display_name': VARIANT_DISPLAY_NAMES[variant_key]}

                # 根据任务类型打印不同指标
                if args.task == 'segmentation':
                    print_info(f"  结果: Dice={metrics['dice']*100:.2f}%")
                elif args.task == 'multitask':
                    print_info(f"  结果: R2={metrics['r2']:.4f}, mIoU={metrics['mIoU']:.4f}, Dice={metrics['dice']*100:.2f}%")
                else:
                    print_info(f"  结果: R2={metrics['r2']:.4f}, mIoU={metrics['mIoU']:.4f}, 违反率={metrics['violation_rate']*100:.1f}%")

        # 保存报告
        total_time = time.time() - total_start

        print_title("消融实验完成")

        # 打印汇总结果表格（展平数据）
        print()
        print("  " + "=" * 80)
        print("  " + "消融实验结果汇总")
        print("  " + "=" * 80)
        print()
        print(f"  {'变体':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'mIoU':>8} {'违反率':>8}")
        print("  " + "-" * 80)
        for k, v in results.items():
            print(f"  {VARIANT_DISPLAY_NAMES.get(k, k):<20} {v['r2']:>8.4f} {v['rmse']:>8.4f} {v['mae']:>8.4f} {v['mIoU']:>8.4f} {v['violation_rate']*100:>7.1f}%")
        print("  " + "=" * 80)

        print_info(f"总耗时: {total_time/60:.1f}分钟")

        # 保存报告
        report_path = os.path.join(RESULTS_DIR, "ablation_results.md")
        json_path = os.path.join(RESULTS_DIR, "ablation_results.json")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        report = ["# 消融实验结果\n", f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                  f"总耗时: {total_time/60:.1f}分钟\n\n",
                  "| 变体 | R2 | RMSE | MAE | mIoU | 违反率 |\n",
                  "|------|-----|------|-----|------|--------|\n"]
        for k, v in results.items():
            report.append(f"| {v['display_name']} | {v['r2']:.4f} | {v['rmse']:.4f} | {v['mae']:.4f} | {v['mIoU']:.4f} | {v['violation_rate']*100:.1f}% |\n")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(report))

        print_success(f"报告: {report_path}")
        print_success(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
