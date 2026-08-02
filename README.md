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

**团队协作（推荐新手）**：
```bash
python team_train.py
```
选择数字即可训练！

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

| 变体 | 说明 | 适用场景 |
|------|------|----------|
| `full` | 完整 MM-DBFNet（默认） | 标准检测 |
| `swin_yolo_fpn` | Swin-Tiny + YOLO-FPN | 空间定位 |
| `vit_yolo_fpn` | ViT-Small + YOLO-FPN | 全局注意力 |
| `detr_style` | ResNet-18 + Transformer | 全局感知 |
| `swin_yolo_patchtst` | Swin + YOLO + PatchTST | 时序增强 |

### 训练示例

```bash
# YOLO系列（推荐）
python run_train.py --variant swin_yolo_fpn --epochs 100 --lr 1e-4

# 优化组合
python run_train.py --variant full --fusion gated --use_coord_attn --epochs 100

# 分阶段训练
python run_train.py --variant full --staged_train --epochs 100
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

## 目录结构

```
project_v4/
├── run_train.py           # 统一训练入口
├── train_launcher.py      # 交互式启动器
├── team_train.py          # 团队协作脚本
├── config.json            # 数据路径配置
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
    ├── 架构设计文档.md
    ├── 调参与算法工程指导文档.md
    ├── 项目算法与训练实验设计报告.md
    └── 控制台输出样式.md
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
| [控制台输出样式](docs/控制台输出样式.md) | 训练输出颜色说明 |
| [架构设计文档](docs/架构设计文档.md) | 模块关系、数据流 |
| [调参指南](docs/调参与算法工程指导文档.md) | 损失权重、调参建议 |
| [算法报告](docs/项目算法与训练实验设计报告.md) | 算法原理 |
| [开发人员文档](开发人员文档.md) | 代码规范、模块详解 |
| [API文档](api.md) | 接口参考 |

---

## 版本信息

当前版本：v4.5.0 (2026-08-02)

主要更新：导师反馈8项优化建议 + 团队协作功能
