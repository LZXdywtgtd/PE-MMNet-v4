"""
PE-MMNet v4: YOLO-FPN 变体模型

包含 Swin-YOLO-FPN 和 ViT-YOLO-FPN 两种架构变体

架构特点：
- 2D骨干：Swin-Tiny 或 ViT-Small
- 特征金字塔：FPN 多尺度融合
- 检测头：YOLO式网格回归 (16x16网格)
- 融合：CrossAttentionFusion
- 输出：[x, y, l, w, confidence, density]

关键设计：
- 使用 self.training 属性区分训练/推理模式
- 训练模式返回完整网格预测
- 推理模式返回最高conf的预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pe_tsnet_multimodal import TemporalFeatureExtractor, MultiTaskHead, CrossAttentionFusion


# =============================================================================
# 2D 骨干网络：Swin Transformer
# =============================================================================

class SwinBackbone2D(nn.Module):
    """
    Swin Transformer 骨干网络

    使用 timm 加载预训练的 Swin-Tiny，输出多尺度特征用于 FPN

    Args:
        pretrained: 是否使用 ImageNet 预训练权重
    """

    def __init__(self, pretrained=True):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("timm 未安装，请运行: pip install timm>=0.9.0")

        # 创建 Swin-Tiny 模型，features_only=True 返回多尺度特征
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True
        )

        # 输出通道数: [96, 192, 384, 768] (C3, C4, C5, C6)
        self.out_channels = self.swin.feature_info.channels()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像

        Returns:
            features: 多尺度特征列表 [C3, C4, C5, C6]
        """
        return self.swin(x)


# =============================================================================
# 2D 骨干网络：Vision Transformer
# =============================================================================

class ViTYOLOBackbone2D(nn.Module):
    """
    Vision Transformer YOLO 骨干网络

    使用 timm 加载预训练的 ViT-Small，将输出转换为 2D 特征图供 FPN 使用

    Args:
        pretrained: 是否使用 ImageNet 预训练权重
        output_size: 输出特征图尺寸（默认 16x16）
    """

    def __init__(self, pretrained=True, output_size=16):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("timm 未安装，请运行: pip install timm>=0.9.0")

        self.output_size = output_size

        # 创建 ViT-Small，features_only=True 返回特征图
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            features_only=True
        )

        # 获取 ViT 输出通道数
        self.out_channels = self.vit.feature_info.channels()[-1]  # 最后一层 [384]

        # 添加一个投影层将 ViT 特征转换为固定尺寸
        self.proj = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.out_channels = 256

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像

        Returns:
            feat: (B, 256, output_size, output_size) 2D特征图
        """
        # 获取 ViT 特征
        features = self.vit(x)  # 返回列表 [B, 384, H', W']

        # 取最后一层特征
        feat = features[-1]

        # 上/下采样到目标尺寸
        feat = F.interpolate(
            feat,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False
        )

        # 投影到统一通道数
        feat = self.proj(feat)

        return feat


# =============================================================================
# 特征金字塔网络：FPN
# =============================================================================

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) 颈部网络

    自顶向下融合多尺度特征，输出统一通道数的特征图

    Args:
        in_channels_list: 各层输入通道数列表
        out_channels: 输出通道数
    """

    def __init__(self, in_channels_list=[96, 192, 384, 768], out_channels=256):
        super().__init__()

        # 侧向连接：1x1 卷积降维
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 输出卷积：3x3 卷积减少混叠效应
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表，从高分辨率到低分辨率

        Returns:
            fpn_features: 融合后的多尺度特征列表
        """
        # 自顶向下融合
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]

        # 从最高层开始，逐步融合低层特征
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )

        # 应用输出卷积
        fpn_features = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]

        return fpn_features


# =============================================================================
# YOLO 检测头
# =============================================================================

class YOLOFPNHead(nn.Module):
    """
    YOLO FPN 检测头

    从 FPN 最后一层特征预测网格位置

    Args:
        in_channels: 输入通道数
        grid_size: 网格尺寸（默认 16 -> 16x16=256 网格）
        num_predictions: 每个网格的预测数
    """

    def __init__(self, in_channels=256, grid_size=16, num_predictions=1):
        super().__init__()

        self.grid_size = grid_size
        self.num_predictions = num_predictions

        # 预测卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 6 = [x, y, l, w, conf, density]
        self.pred = nn.Conv2d(128, 6 * num_predictions, kernel_size=1)

    def forward(self, fpn_features):
        """
        Args:
            fpn_features: FPN 特征列表，我们取最后一层

        Returns:
            grid_pred: (B, num_grids, 6) 网格预测
                num_grids = grid_size * grid_size * num_predictions
        """
        # 取 FPN 最后一层（最高语义层）
        feat = fpn_features[-1]  # (B, 256, grid_size, grid_size)

        # 预测
        feat = self.conv(feat)
        pred = self.pred(feat)  # (B, 6, grid_size, grid_size)

        # 重排维度: (B, 6, H, W) -> (B, H, W, 6)
        pred = pred.permute(0, 2, 3, 1)

        # 展平为网格预测: (B, H*W, 6)
        B, H, W, C = pred.shape
        grid_pred = pred.reshape(B, H * W, C)

        return grid_pred


# =============================================================================
# Swin-YOLO-FPN 模型
# =============================================================================

class SwinYOLOFPN(nn.Module):
    """
    Swin-YOLO-FPN 模型

    架构：Swin-Tiny -> FPN -> YOLO Head -> CrossAttentionFusion -> MultiTaskHead

    输出：
    - 训练模式 (model.train()): 返回完整网格预测
    - 推理模式 (model.eval()): 返回最高 conf 的预测

    Args:
        seq_len: 1D 序列长度（默认 300）
        image_channels: 2D 图像通道数（默认 2：温度场+应力场）
        image_size: 输入图像尺寸（默认 256）
        pretrained_2d: 2D 骨干是否使用预训练权重
        dropout: Dropout 比例
        grid_size: YOLO 网格尺寸（默认 16 -> 256 个网格）
    """

    def __init__(self, seq_len=300, image_channels=2, image_size=256,
                 pretrained_2d=True, dropout=0.2, grid_size=16):
        super().__init__()

        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.grid_size = grid_size

        # 2D 分支：Swin 骨干 + FPN + YOLO Head
        self.backbone_2d = SwinBackbone2D(pretrained=pretrained_2d)
        self.fpn = FPNNeck(in_channels_list=self.backbone_2d.out_channels, out_channels=256)
        self.yolo_head = YOLOFPNHead(in_channels=256, grid_size=grid_size)

        # 1D 分支：时序特征提取
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout
        )
        feat_dim_1d = 64

        # 融合层：CrossAttentionFusion（自动适配维度）
        self.fusion = CrossAttentionFusion(dim_2d=6, dim_1d=feat_dim_1d)

        # 输出头
        fused_dim = 6 + feat_dim_1d  # YOLO 预测维度 + 1D 特征维度
        self.output_head = MultiTaskHead(
            input_dim=fused_dim,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, x_1d, x_2d):
        """
        前向传播

        Args:
            x_1d: (B, seq_len) 温度时序
            x_2d: (B, 2, H, W) 温度场+应力场图像

        Returns:
            output: 检测结果
                - 训练模式: (B, num_grids, 6) 完整网格预测
                - 推理模式: (B, 6) 最高 conf 的预测
            global_density: (B, 1) 全局密度（用于损失计算）
        """
        # 2D 分支
        multi_scale_features = self.backbone_2d(x_2d)
        fpn_features = self.fpn(multi_scale_features)
        grid_pred = self.yolo_head(fpn_features)  # (B, 256, 6)

        # ✅ 使用 self.training 判断模式（关键设计）
        if self.training:
            # 训练模式：返回完整网格预测用于损失计算
            grid_feat = grid_pred  # (B, 256, 6)
        else:
            # 推理模式：取最高 conf 的预测
            conf = grid_pred[..., 4:5]  # (B, 256, 1)
            best_idx = conf.argmax(dim=1, keepdim=True)  # (B, 1, 1)

            # gather 需要扩展索引
            best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, 6)  # (B, 1, 6)
            grid_feat = grid_pred.gather(1, best_idx_expanded).squeeze(1)  # (B, 6)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)  # (B, 64)

        # 融合（CrossAttentionFusion 自动适配 grid_feat 的维度）
        fused = self.fusion(grid_feat, feat_1d)

        # 全局密度：取所有网格预测密度的最大值
        global_density = grid_pred[..., 5:6].max(dim=1, keepdim=True)[0]  # (B, 1)

        # 输出
        output = self.output_head(fused)

        return output, global_density

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# ViT-YOLO-FPN 模型
# =============================================================================

class ViTYOLOFPN(nn.Module):
    """
    ViT-YOLO-FPN 模型

    与 Swin-YOLO-FPN 类似，但使用 ViT-Small 作为 2D 骨干

    Args:
        seq_len: 1D 序列长度（默认 300）
        image_channels: 2D 图像通道数（默认 2：温度场+应力场）
        image_size: 输入图像尺寸（默认 256）
        pretrained_2d: 2D 骨干是否使用预训练权重
        dropout: Dropout 比例
        grid_size: YOLO 网格尺寸（默认 16 -> 256 个网格）
    """

    def __init__(self, seq_len=300, image_channels=2, image_size=256,
                 pretrained_2d=True, dropout=0.2, grid_size=16):
        super().__init__()

        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.grid_size = grid_size

        # 2D 分支：ViT 骨干 + 简化 FPN + YOLO Head
        self.backbone_2d = ViTYOLOBackbone2D(pretrained=pretrained_2d, output_size=grid_size)

        # 简化的 FPN（单层）
        self.fpn = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.yolo_head = YOLOFPNHead(in_channels=256, grid_size=grid_size)

        # 1D 分支
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout
        )
        feat_dim_1d = 64

        # 融合层
        self.fusion = CrossAttentionFusion(dim_2d=6, dim_1d=feat_dim_1d)

        # 输出头
        fused_dim = 6 + feat_dim_1d
        self.output_head = MultiTaskHead(
            input_dim=fused_dim,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, x_1d, x_2d):
        """
        前向传播

        Args:
            x_1d: (B, seq_len) 温度时序
            x_2d: (B, 2, H, W) 温度场+应力场图像

        Returns:
            output: 检测结果
            global_density: (B, 1) 全局密度
        """
        # 2D 分支
        feat_2d = self.backbone_2d(x_2d)  # (B, 256, grid_size, grid_size)
        fpn_feat = self.fpn(feat_2d)  # (B, 256, grid_size, grid_size)
        grid_pred = self.yolo_head([fpn_feat])  # (B, 256, 6) - 需要列表格式

        # ✅ 使用 self.training 判断模式
        if self.training:
            grid_feat = grid_pred
        else:
            conf = grid_pred[..., 4:5]
            best_idx = conf.argmax(dim=1, keepdim=True)
            best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, 6)
            grid_feat = grid_pred.gather(1, best_idx_expanded).squeeze(1)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)

        # 融合
        fused = self.fusion(grid_feat, feat_1d)

        # 全局密度
        global_density = grid_pred[..., 5:6].max(dim=1, keepdim=True)[0]

        output = self.output_head(fused)

        return output, global_density

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
