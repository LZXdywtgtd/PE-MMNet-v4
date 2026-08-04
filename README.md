# PE-MMNet v4 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v4**
>
> 基于深度学习的日用陶瓷热震裂纹实时预测系统

---

## 项目简介

本项目实现了一个多模态融合的裂纹检测/分割模型，同时处理温度时序（1D）和温度场+应力场图像（2D）：

| 任务模式 | 输入 | 输出 |
|----------|------|------|
| **Detection** | 温度时序 + 图像 | [x, y, l, w, conf, density] |
| **Segmentation** | 温度时序 + 图像 | 256×256 二值掩膜 |
| **Multitask** | 温度时序 + 图像 | 检测向量 + 分割掩膜 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：PyTorch、timm（ViT/Swin预训练）、scipy、numpy、pandas 等。

### 2. 配置数据路径

首次运行会提示配置数据路径，或手动创建 `config.json`：

```json
{
  "data_root": "D:\\你的路径\\参考输入"
}
```

### 3. 开始训练

**交互式菜单（推荐新手）：**
```bash
python train_launcher.py
```

**团队协作训练：**
```bash
python team_train.py              # 交互式菜单
python team_train.py --auto       # 自动执行所有可执行任务
python team_train.py --auto --force  # 强制执行（含硬件警告任务）
```

**直接训练：**
```bash
python run_train.py --mode train --variant resnet18 --epochs 150
```

---

## 模型变体

| 变体 | 架构 | 显存占用 | 说明 |
|------|------|----------|------|
| `resnet18` | ResNet-18 + Cross-Attention | ~2GB | 轻量基线，稳定快速 |
| `swin_yolo` | Swin-Tiny + YOLO-FPN + Cross-Attention | ~4GB | 空间网格定位，图像注意力 |
| `vit_yolo` | ViT-Small + YOLO-FPN + Cross-Attention | ~3GB | 全局自注意力，轻量 |
| `detr` | ResNet-18 + Transformer + Cross-Attention | ~5GB | 全局上下文，100个Object Queries |

### 训练示例

```bash
# 基线模型
python run_train.py --variant resnet18 --epochs 150

# YOLO 系列（推荐配合较低学习率）
python run_train.py --variant swin_yolo --epochs 150 --lr 1e-4 --batch_size 4
python run_train.py --variant vit_yolo --epochs 150 --lr 1e-4 --batch_size 4

# DETR 系列
python run_train.py --variant detr --epochs 150 --lr 1e-4 --batch_size 4
```

### 高级优化参数

```bash
# 门控融合（温度/应力分治策略）
python run_train.py --variant resnet18 --fusion gated --epochs 150

# 坐标注意力（保留位置信息）
python run_train.py --variant swin_yolo --use_coord_attn --epochs 150

# 分阶段训练（先短序列后长序列）
python run_train.py --variant resnet18 --staged_train --epochs 150

# ThermalCutMix 数据增强（物理安全版，仅作用于温度通道）
python run_train.py --variant resnet18 --aug_cutmix_prob 0.3 --epochs 150
```

---

## 目录结构

```
project_v4/
├── run_train.py              # 统一训练入口
├── train_launcher.py          # 交互式启动器
├── team_train.py              # 团队协作训练脚本
├── config.json                # 数据路径配置
│
├── tasks/                    # 团队任务配置目录（协调者分发）
│   ├── team_baseline.json
│   ├── team_optimization.json
│   └── batch_ablation.json
│
├── checkpoints/               # 模型检查点
├── logs/                     # 训练日志
│
├── data/                     # 数据加载
│   └── dataset_multimodal.py
│
├── models/                   # 模型定义
│   ├── pe_tsnet_multimodal.py   # 基础模型 + SE/CoordAtt/CrossAttn
│   ├── pe_tsnet_yolo.py         # SwinYOLOFPN + ViTYOLOFPN
│   └── pe_tsnet_detr.py         # DETRStyle
│
├── training/                 # 训练相关
│   ├── mono_loss.py          # 损失函数（YOLO/DETR/检测/分割）
│   └── augmentation.py       # 数据增强
│
├── utils/                    # 工具
│   ├── config.py            # 配置管理
│   └── console.py           # 控制台输出
│
└── docs/                    # 文档
    ├── 快速配置指南.md
    ├── 架构设计文档.md
    ├── 团队协作训练指南.md
    ├── 开发人员文档.md
    └── 控制台输出样式.md
```

---

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| R² | 决定系数（密度预测相关性） | 越接近1越好 |
| RMSE | 均方根误差 | 越小越好 |
| MAE | 平均绝对误差 | 越小越好 |
| 违反率 | 单调性违反比例 | 越小越好 |
| mIoU | 分割交并比 | 越大越好 |

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速配置指南](docs/快速配置指南.md) | 环境配置、首次使用、数据准备 |
| [架构设计文档](docs/架构设计文档.md) | 模型架构、数据流、损失函数 |
| [团队协作训练指南](docs/团队协作训练指南.md) | team_train.py 完整使用手册 |
| [开发人员文档](开发人员文档.md) | 代码规范、模块详解、API |
| [控制台输出样式](docs/控制台输出样式.md) | 颜色代码、盒子样式 |

---

## 版本信息

**当前版本：v4.6.1 (2026-08-05)**

### v4.6.1 修复内容

- **swin_yolo**：推理模式 gather 维度不匹配 → 改用 `torch.arange` 直接索引
- **vit_yolo**：ViT 不支持 512×512 输入 → 增加 `input_resize` 层（512→224）
- **detr**：ResNet18Backbone 返回向量而非特征图 → 增加 `set_spatial_output()` 方法
- **detr**：推理/训练模式形状不匹配 → 训练模式直接返回 `detr_pred` 供损失计算
- **UserWarning**：`global_density` 维度 `(B,1,1)` vs `(B,1)` → squeeze 消除
- **UnicodeEncodeError**：Emoji → 替换为 `[OK]`

### v4.6.0 新增功能

- 动态训练时间估算（训练前测量2轮）
- 实时 ETA 显示（EMA + 置信区间 + 预计完成时间）
- 检查点元数据保存 `task_id`
- 团队任务执行日志
- `--force` 参数强制执行硬件警告任务
