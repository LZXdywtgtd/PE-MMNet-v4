# PE-MMNet v4 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v4**
>
> 基于深度学习的日用陶瓷热震裂纹实时预测系统

---

## 项目简介

本项目实现了一个多模态融合的裂纹检测/分割模型：

| 任务模式 | 输入 | 输出 |
|----------|------|------|
| **Detection** | 温度时序 + 图像 | [x, y, l, w, conf, density] |
| **Segmentation** | 温度时序 + 图像 | 256x256 二值掩膜 |
| **Multitask** | 温度时序 + 图像 | 检测向量 + 分割掩膜 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据路径

首次运行会自动提示配置，或手动创建 `config.json`：

```json
{
  "data_root": "D:\\你的路径\\参考输入"
}
```

### 3. 开始训练

**团队协作（推荐）**：
```bash
python team_train.py
```
- 自动检测 GPU 显存，显示硬件等级
- 任务状态可视化：已完成/可执行/警告/锁定
- 支持依赖管理：前置任务未完成自动锁定
- 可导入队友的检查点文件
- 支持自动批量执行

**自动执行模式**：
```bash
# 自动执行所有可执行任务
python team_train.py --auto

# 强制执行（包括硬件警告任务）
python team_train.py --auto --force
```

**交互式菜单**：
```bash
python train_launcher.py
```

**直接训练**：
```bash
python run_train.py --mode train --epochs 100
```

---

## 模型变体

| 变体 | 说明 | 适用场景 | 显存 |
|------|------|----------|------|
| `resnet18` | ResNet18 + CrossAttn（默认） | 标准检测 | ~2GB |
| `swin_yolo` | Swin-Tiny + YOLO-FPN | 空间定位 | ~4GB |
| `vit_yolo` | ViT-Small + YOLO-FPN | 全局注意力 | ~3GB |
| `detr` | ResNet-18 + Transformer | 全局感知 | ~5GB |

### 训练示例

```bash
# 基础模型
python run_train.py --variant resnet18 --epochs 150

# YOLO系列
python run_train.py --variant swin_yolo --epochs 150 --lr 1e-4
python run_train.py --variant vit_yolo --epochs 150 --lr 1e-4
python run_train.py --variant detr --epochs 150 --lr 1e-4

# 优化组合
python run_train.py --variant swin_yolo --fusion gated --use_coord_attn --epochs 150

# 分阶段训练
python run_train.py --variant resnet18 --staged_train --epochs 150
```

---

## 高级功能

| 功能 | 参数 | 说明 |
|------|------|------|
| 分阶段训练 | `--staged_train` | 先预训练后微调 |
| 坐标注意力 | `--use_coord_attn` | 保留位置信息 |
| 门控融合 | `--fusion gated` | 温度/应力分治 |
| 三通道输入 | `--triple_channel` | 初始+当前温度+变化率 |
| ThermalCutMix | `--aug_cutmix_prob 0.5` | 物理安全增强 |

---

## 实时 ETA 显示

训练过程中每10轮显示训练进度盒子和预估完成时间：

```
╔════════════════════════════════════════════════════════════════╗
║ Epoch 15/150 | Best: 12 | Progress: [████████░░░░░░░░] 10%  ║
╠════════════════════════════════════════════════════════════════╣
║ Train | Loss: 0.4562 | LR: 3.00e-04 | Time: 1.3s          ║
╠════════════════════════════════════════════════════════════════╣
║ System| EMA: 1.3s/epoch | ETA: 2m55s | 完成: 14:32 ±0.2s   ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 目录结构

```
project_v4/
├── run_train.py           # 统一训练入口
├── train_launcher.py      # 交互式启动器
├── team_train.py          # 团队协作脚本
├── config.json            # 数据路径配置
├── tasks/                # 任务配置目录
│   ├── team_baseline.json
│   ├── team_optimization.json
│   └── batch_ablation.json
├── checkpoints/           # 检查点目录
├── logs/                  # 训练日志目录
│   └── team_training.log # 团队任务执行日志
│
├── data/                  # 数据加载
│   └── dataset_multimodal.py
│
├── models/                # 模型定义
│   ├── pe_tsnet_multimodal.py  # 基础模型
│   ├── pe_tsnet_yolo.py        # YOLO-FPN变体
│   ├── pe_tsnet_detr.py        # DETR风格变体
│   ├── pe_tsnet_patchtst.py    # PatchTST骨干
│   └── pe_tsnet_fusion.py      # 门控融合
│
├── training/              # 训练相关
│   ├── mono_loss.py       # 损失函数
│   └── augmentation.py    # 数据增强
│
└── docs/                 # 文档
    ├── 快速配置指南.md
    ├── 团队协作训练指南.md
    └── 开发人员文档.md
```

---

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| R² | 决定系数 | 越接近1越好 |
| RMSE | 均方根误差 | 越小越好 |
| mIoU | 定位精度 | 越大越好 |
| 违反率 | 单调性违反 | 越小越好 |

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [快速配置指南](docs/快速配置指南.md) | 环境配置、首次使用 |
| [团队协作训练指南](docs/团队协作训练指南.md) | team_train.py 使用 |
| [开发人员文档](开发人员文档.md) | 代码规范、模块详解 |
| [API文档](api.md) | 接口参考 |

---

## 版本信息

当前版本：v4.6.0 (2026-08-03)

主要更新：
- 动态训练时间估算（训练前测量2轮）
- 实时 ETA 显示（EMA + 置信区间 + 预计完成时间）
- 检查点元数据保存 task_id
- 团队任务执行日志
- `--force` 参数强制执行硬件警告任务
