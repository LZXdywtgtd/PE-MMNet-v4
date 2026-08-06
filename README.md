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

### 2. 配置数据路径

首次运行会提示配置数据路径，或手动创建 `config.json`：

```json
{
  "data_root": "D:\\你的路径\\参考输入"
}
```

### 3. 开始训练

```bash
# 交互式菜单（推荐新手）
python train_launcher.py

# 直接训练
python run_train.py --mode train --variant resnet18 --epochs 150

# 团队协作
python team_train.py
```

---

## 模型变体

| 变体 | 架构 | 显存占用 | 说明 |
|------|------|----------|------|
| `resnet18` | ResNet-18 + Cross-Attention | ~2GB | 轻量基线，稳定快速 |
| `swin_yolo` | Swin-Tiny + YOLO-FPN + Cross-Attention | ~4GB | 空间网格定位 |
| `vit_yolo` | ViT-Small + YOLO-FPN + Cross-Attention | ~3GB | 全局自注意力 |
| `swin_yolo_patchtst` | Swin-Tiny + YOLO-FPN + PatchTST | ~4GB | PatchTST时序骨干 |
| `detr` | ResNet-18 + Transformer + Cross-Attention | ~5GB | 全局上下文 |

### 训练示例

```bash
# 基线模型
python run_train.py --variant resnet18 --epochs 150

# YOLO 系列（推荐较低学习率）
python run_train.py --variant swin_yolo --epochs 150 --lr 1e-4

# DETR 系列
python run_train.py --variant detr --epochs 150 --lr 1e-4

# 高级优化
python run_train.py --variant resnet18 --fusion gated --epochs 150
python run_train.py --variant resnet18 --staged_train --epochs 150
python run_train.py --variant resnet18 --triple_channel --epochs 150
```

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速配置指南](docs/user_guides/快速配置指南.md) | 环境配置、首次使用、数据准备 |
| [架构设计文档](docs/architecture/架构设计文档.md) | 模型架构、数据流、损失函数 |
| [开发人员文档](docs/dev_reference/开发人员文档.md) | 代码规范、模块详解、API |
| [团队协作训练指南](docs/collaboration/团队协作训练指南.md) | team_train.py 使用手册 |
| [算法与实验报告](docs/experiment_reports/项目算法与训练实验设计报告.md) | 算法原理、实验设计 |
| [调参指南](docs/experiment_reports/调参与算法工程指导文档.md) | 损失函数配置、超参数调优 |

详细文档结构见 [开发人员文档](docs/dev_reference/开发人员文档.md#二目录结构)。
