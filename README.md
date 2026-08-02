# PE-MMNet v4 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v4**
>
> 基于深度学习的日用陶瓷热震裂纹实时预测系统

---

## 项目简介

本项目实现了一个多模态融合的裂纹检测/分割模型，支持三种任务模式：

| 任务模式 | 输入 | 输出 | 适用场景 |
|----------|------|------|----------|
| **Detection** | 温度时序 + 图像 | 6维向量 [x, y, l, w, conf, density] | 标准检测任务 |
| **Segmentation** | 温度时序 + 图像 | 256x256 二值掩膜 | 像素级裂纹分割 |
| **Multitask** | 温度时序 + 图像 | 检测向量 + 分割掩膜 | 同时需要检测和分割 |

**输入模态**：
- **模态 A（1D）**：300 点高频温度时序（10Hz × 30s）
- **模态 B（2D）**：256x256 温度场 + 应力场图像

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

首次运行会自动提示配置数据路径，或手动创建 `config.json`：

```json
{
  "data_root": "D:\\你的路径\\参考输入"
}
```

### 3. 开始训练

**推荐方式：统一启动器**

```bash
python train_launcher.py
```

菜单选项：
- `[1]` 快速训练（预设模板）
- `[2]` 自定义配置（详细参数说明）
- `[3]` 消融实验
- `[4]` 队列配置向导（批量训练）
- `[5]` 添加到现有队列

**快捷批量训练**

```bash
python train_launcher.py --quick "resnet18,100,0,detection" "vit_small,100,0,detection"
```

**直接训练（高级用户）**

```bash
# 检测模式（默认）
python run_train.py --mode train --epochs 100

# 分割模式
python run_train.py --mode train --task segmentation --epochs 100

# 评估模型
python run_train.py --mode eval --checkpoint ./checkpoints/xxx.pt
```

---

## 核心功能

### 显存自适应配置

自动检测 GPU 显存，选择最佳配置：

| 空闲显存 | 图像尺寸 | 批次大小 | FP16 |
|----------|----------|----------|------|
| ≥10 GB | 512 | 16 | ✅ |
| 6-10 GB | 512 | 8 | ✅ |
| 4-6 GB | 384 | 8 | ✅ |
| <4 GB | 256 | 4 | ❌ |

### 模型变体

| 变体 | 说明 | 2D骨干 | 检测方式 |
|------|------|---------|----------|
| `full` | 完整 MM-DBFNet（默认） | ResNet-18 | MLP回归 |
| `1d_only` | 仅时序分支 | - | MLP回归 |
| `2d_only` | 仅空间分支 | ResNet-18 | MLP回归 |
| `concat` | 双分支拼接融合 | ResNet-18 | MLP回归 |
| `add` | 双分支加法融合 | ResNet-18 | MLP回归 |
| `cross_attn` | Cross-Attention 融合 | ResNet-18 | MLP回归 |
| `swin_yolo_fpn` | Swin-YOLO-FPN | Swin-Tiny | YOLO网格回归 |
| `vit_yolo_fpn` | ViT-YOLO-FPN | ViT-Small | YOLO网格回归 |
| `detr_style` | DETR风格 | ResNet-18 | Transformer |

### 新增变体训练示例

```bash
# Swin-YOLO-FPN（推荐）
python run_train.py --mode train --variant swin_yolo_fpn --epochs 100

# ViT-YOLO-FPN
python run_train.py --mode train --variant vit_yolo_fpn --epochs 100

# DETR风格（需要更小学习率）
python run_train.py --mode train --variant detr_style --epochs 100 --lr 1e-4
```

### 时间偏移预测

支持预测未来时刻的裂纹状态：

```bash
python run_train.py --predict_offset 0  # 当前时刻（默认）
python run_train.py --predict_offset 1  # 0.05秒后
python run_train.py --predict_offset 2  # 0.1秒后
```

---

## 目录结构

```
project_v4/
├── run_train.py              # 统一训练入口
├── train_launcher.py         # 统一启动器（推荐）
├── launcher.py               # 交互式配置
├── config.json               # 数据路径配置
├── requirements.txt          # 依赖
│
├── data/                     # 数据加载
│   └── dataset_multimodal.py
│
├── models/                   # 模型定义
│   ├── pe_tsnet_multimodal.py  # 原有模型
│   ├── pe_tsnet_yolo.py         # YOLO-FPN变体
│   └── pe_tsnet_detr.py         # DETR风格变体
│
├── training/                 # 损失函数
│   └── mono_loss.py             # 包含YOLO/DETR损失
│
├── tools/                    # 工具
│   ├── batch_train_gui.py    # 批量训练工具
│   ├── streamlit_app.py      # 可视化工具
│   └── visualization.py
│
├── utils/                    # 工具函数
│   ├── config.py
│   └── console.py
│
├── docs/                     # 文档
│   ├── 快速配置指南.md
│   ├── 架构设计文档.md
│   ├── 调参与算法工程指导文档.md
│   └── 项目算法与训练实验设计报告.md
│
├── checkpoints/              # 模型检查点
└── benchmark_results/        # 实验结果
```

---

## 常用命令参考

### 训练命令

```bash
# 基本训练
python run_train.py --mode train --epochs 150

# 指定架构
python run_train.py --mode train --backbone_2d vit_small --backbone_1d transformer

# 高分辨率
python run_train.py --mode train --image_size 768

# 禁用 FP16
python run_train.py --mode train --no_fp16

# 强制重新训练
python run_train.py --mode train --force_retrain
```

### 评估命令

```bash
# 评估检查点
python run_train.py --mode eval --checkpoint ./checkpoints/xxx.pt

# 指定图像尺寸
python run_train.py --mode eval --checkpoint ./checkpoints/xxx.pt --image_size 512
```

### 消融实验

```bash
python run_train.py --mode ablation --epochs 50
```

---

## 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| Detection | R² | 决定系数，越接近1越好 |
| Detection | RMSE | 均方根误差，越小越好 |
| Detection | mIoU | 平均 IoU，定位精度 |
| Detection | 违反率 | 单调性违反百分比，越小越好 |
| Segmentation | Dice | Dice 系数，越接近1越好 |

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速配置指南](docs/快速配置指南.md) | 环境配置、首次使用 |
| [架构设计文档](docs/架构设计文档.md) | 代码架构、调用关系 |
| [调参指南](docs/调参与算法工程指导文档.md) | 损失函数权重、调参建议 |
| [算法报告](docs/项目算法与训练实验设计报告.md) | 算法原理、实验设计 |

---

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)
