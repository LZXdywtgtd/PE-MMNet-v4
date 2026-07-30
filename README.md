# PE-MMNet v4 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v4**
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
- **模态 A（1D）**：300 点高频温度时序（10Hz x 30s）
- **模态 B（2D）**：256x256 温度场 + 应力场图像

---

## 快速开始

### 首次使用必读

**请先阅读 [docs/快速配置指南.md](docs/快速配置指南.md)，包含：**
1. 环境配置（Python、CUDA、PyTorch）
2. 数据路径配置
3. 常见问题排查

### 快速命令

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置数据路径（首次运行会自动提示）
python run_train.py --mode train

# 3. 评估预训练模型
python run_train.py --mode eval --checkpoint ./checkpoints/resnet18_cnn_attn_cross_attn_taskdetection_offset0_best.pt
```

---

## 目录结构

```
project_v4/
├── run_train.py              # 统一训练入口（唯一标准入口）
├── launcher.py               # 交互式启动器（可选）
├── config.json               # 路径配置文件
├── requirements.txt          # Python 依赖
├── data/                     # 数据加载模块
│   └── dataset_multimodal.py # 多批次数据集、图像预处理、分割标签
├── models/                   # 模型定义
│   └── pe_tsnet_multimodal.py # PETSNetMultimodal、MaskDecoder
├── training/                 # 损失函数
│   └── mono_loss.py          # 检测损失、Dice Loss、多任务损失
├── tools/                    # 可视化与分析工具
│   ├── streamlit_app.py      # Streamlit 交互式分析工具
│   ├── visualization.py       # 命令行可视化工具
│   └── structure_analyzer.py # 裂纹/气孔分离模块
├── utils/                    # 工具模块
│   ├── config.py             # 配置中心
│   └── console.py            # 打印工具
├── docs/                     # 文档目录
│   ├── 快速配置指南.md        # 首次使用必读
│   ├── CUDA安装指南.md        # GPU环境配置
│   ├── 项目算法与训练实验设计报告.md
│   └── 调参与算法工程指导文档.md
├── checkpoints/              # 模型检查点
└── benchmark_results/        # 实验结果
```

---

## 统一训练脚本 (run_train.py)

### 三种模式

| 模式 | 命令 | 说明 |
|------|------|------|
| **训练** | `--mode train` | 训练模型，结束后自动评估 |
| **评估** | `--mode eval` | 评估已有检查点 |
| **消融** | `--mode ablation` | 运行6个变体消融实验 |

### 任务模式

```bash
# 检测模式（默认）- 输出矩形边界框
python run_train.py --mode train --task detection

# 分割模式 - 输出像素级裂纹掩膜
python run_train.py --mode train --task segmentation

# 多任务模式 - 同时输出检测框和分割掩膜
python run_train.py --mode train --task multitask
```

### 常用命令

```bash
# 训练完整模型（检测模式，默认150 epoch）
python run_train.py --mode train --variant full --task detection

# 训练分割模型
python run_train.py --mode train --task segmentation --epochs 100

# 评估分割模型
python run_train.py --mode eval --checkpoint ./checkpoints/xxx_tasksegmentation_offset0_best.pt

# 消融实验（分割模式）
python run_train.py --mode ablation --task segmentation
```

### 模型变体

| 变体 | 说明 |
|------|------|
| `full` | 完整MM-DBFNet（默认） |
| `1d_only` | 仅时序分支 |
| `2d_only` | 仅空间分支 |
| `concat` | 双分支+拼接 |
| `add` | 双分支+加法 |
| `cross_attn` | Cross-Attention |

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `detection` | 任务模式：detection / segmentation / multitask |
| `--epochs` | `150` | 训练轮数 |
| `--batch_size` | `16` | 批次大小 |
| `--lr` | `3e-4` | 学习率 |
| `--feature_len` | `300` | 1D序列长度 |
| `--predict_offset` | `0` | 预测偏移（0=当前，1=0.05s后，2=0.1s后...）|

---

## 模型架构

### 非对称双分支融合网络

```
输入
├── 1D分支：温度时序 (batch, 300)
│   └── Micro + Macro 1D-CNN + Self-Attention → 64维特征
└── 2D分支：温度场+应力场 (batch, 2, 256, 256)
    └── ResNet-18 → 512维特征

融合：Cross-Attention → 576维融合特征

输出（根据 task 参数）：
├── detection: (batch, 6) → [x, y, l, w, confidence, density]
├── segmentation: (batch, 1, 256, 256) → 二值掩膜
└── multitask: (mask, detection) 元组
```

### MaskDecoder（分割头）

```
输入: (batch, 576) 融合特征 → reshape → (batch, 576, 1, 1)
│   ConvTranspose2d 576→256 → 2×2
│   ConvTranspose2d 256→128 → 4×4
│   ConvTranspose2d 128→64 → 8×8
│   ConvTranspose2d 64→32 → 16×16
│   ConvTranspose2d 32→16 → 32×32
│   ConvTranspose2d 16→8 → 64×64
│   ConvTranspose2d 8→4 → 128×128
│   ConvTranspose2d 4→1 → 256×256
└── Sigmoid → (batch, 1, 256, 256) 二值掩膜
```

共 8 次转置卷积，逐步从 1×1 上采样到 256×256。

---

## 损失函数

| 任务 | 损失函数 | 组成 |
|------|----------|------|
| Detection | `MultimodalCrackLoss` | MSE + 单调性 + DIoU + BCE |
| Segmentation | `SegmentationLoss` | Dice Loss + BCE |
| Multitask | `MultimodalSegmentationLoss` | Dice + BCE + 检测损失 |

---

## 评估指标

### 检测任务

| 指标 | 说明 |
|------|------|
| R2 | 决定系数，越接近1越好 |
| RMSE | 均方根误差，越小越好 |
| MAE | 平均绝对误差，越小越好 |
| mIoU | 平均IoU，定位精度 |
| 违反率 | 单调性违反百分比，越小越好 |

### 分割任务

| 指标 | 说明 |
|------|------|
| Dice | Dice系数，越接近1越好 |
| IoU | 交并比，越接近1越好 |

---

## 交互式启动器 (launcher.py)

```bash
# 启动交互式菜单
python launcher.py

# 菜单选项：
# [1-4] 预设模板快速启动
# [s] 分割任务预设
# [m] 多任务预设
# [c] 自定义配置
# [a] 消融实验预设
# [l] 加载上次配置
# [d] 数据诊断
# [q] 退出
```

---

## 批量训练与对比工具 (tools/batch_train_gui.py)

批量训练多个模型变体，自动评估并生成对比表格。

```bash
# 启动交互式界面
python tools/batch_train_gui.py

# 快捷命令
python tools/batch_train_gui.py --quick "resnet18,100,0" "vit_small,100,0"
```

**快捷命令格式**：`backbone_2d,epochs,offset,task`

**交互式菜单**：
- [1] 查看命令列表
- [2] 添加训练命令
- [3] 编辑命令
- [4] 删除命令
- [5] 清空所有命令
- [6] 批量导入命令
- [7] ▶ 开始批量训练
- [8] 评估已有检查点
- [9] 📊 生成对比表格
- [a] ⚡ 添加消融实验预设

**预设模板**：
- CNN vs Transformer 对比 (6条)
  - resnet18_offset0, vit_small_offset0, resnet18_offset1, vit_small_offset1, resnet18_offset2, vit_small_offset2
- 简化对比 (2条)

---

## 可视化工具 (tools/)

### Streamlit 交互式分析工具

```bash
# 启动交互式可视化工具
streamlit run tools/streamlit_app.py
```

**功能**：
- **实时参数调整**：裂纹阈值、气孔阈值、合并距离等滑块
- **三种视图模式**：二值掩膜 / 结构分解 / 对比视图
- **气孔与裂纹分离**：自动分离并可视化两种结构
- **像素级诊断**：输入坐标查看该点是否被识别为裂纹/气孔
- **裁剪区域可视化**：自动检测基板边界，显示裁剪框
- **自动调参**：网格搜索最佳参数组合

**参数说明**：

| 参数 | 说明 |
|------|------|
| 裂纹阈值 | 低于此值为裂纹候选（d_hist 中暗色区域） |
| 气孔阈值 | 高于此值为气孔候选（d_hist 中亮色区域） |
| 合并距离 | 相邻裂纹碎片合并的距离阈值 |
| 最小面积 | 裂纹最小面积（像素），小于此值被过滤 |
| 边界排除 | 边缘区域的宽度，该区域像素不参与检测 |

### 命令行可视化工具

```bash
# 预处理检查
python tools/visualization.py --check-preprocess \
    --d_hist "d_hist/d001.png" --threshold 0.1

# 多图对比
python tools/visualization.py --compare \
    --batch "参数化扫描1" --index 5

# 批量对比
python tools/visualization.py --batch-compare \
    --batch "参数化扫描1" --start 0 --end 20
```

### 结构分析模块

```python
from tools.structure_analyzer import separate_crack_pore, auto_optimize_params

# 分离裂纹和气孔
crack_mask, pore_mask, overlay, stats = separate_crack_pore(
    img,
    crack_threshold=0.2,
    pore_threshold=0.7
)

# 自动寻优最佳参数
result = auto_optimize_params(img)
```

---

## 相关文档

| 文档 | 目标读者 | 内容 |
|------|----------|------|
| [快速配置指南.md](docs/快速配置指南.md) | **所有用户** | 环境配置、首次使用 |
| [CUDA安装指南.md](docs/CUDA安装指南.md) | GPU用户 | CUDA/PyTorch安装 |
| [项目算法与训练实验设计报告.md](docs/项目算法与训练实验设计报告.md) | 导师/组长 | 算法原理、实验设计 |
| [调参与算法工程指导文档.md](docs/调参与算法工程指导文档.md) | 算法组员 | 调参指南、故障排查 |

---

## 常见问题

**Q: 提示"未找到数据批次"？**
A: 首次运行会提示输入数据路径，按提示配置即可。详见 [快速配置指南.md](docs/快速配置指南.md)

**Q: 如何使用GPU训练？**
A: 确保已安装CUDA和GPU版PyTorch。详见 [CUDA安装指南.md](docs/CUDA安装指南.md)

**Q: 训练loss是NaN怎么办？**
A: 降低学习率：`--lr 1e-4`，或减小单调性权重：`--lambda_mono 0.01`

**Q: 分割模式和检测模式有什么区别？**
A: 检测模式输出6维向量（边界框），分割模式输出256×256二值掩膜（像素级）

**Q: 想要增加新的数据批次？**
A: 将数据放入数据目录，首次运行会提示配置路径。系统会自动识别新批次。

---

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)
