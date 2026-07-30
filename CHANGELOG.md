# 更新日志 (CHANGELOG)

所有重大更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v4.2.1] - 2026-07-30

### 新增：批量训练与对比工具

#### 交互式批量训练工具 (tools/batch_train_gui.py)

- **命令管理**：添加/编辑/删除/清空训练命令
- **批量导入**：支持快捷格式批量添加命令
- **预设模板**：一键添加 CNN vs Transformer 对比实验
- **命令去重**：自动检测并跳过重复命令
- **实时日志**：禁用缓冲，实时显示训练进度
- **自动评估**：训练完成后自动评估所有模型
- **对比表格**：按 R² 排序，一目了然
- **错误隔离**：单个任务失败不影响其他任务
- **会话保存**：命令列表自动保存到 `logs/session_commands.json`

#### 数据加载自适应增强

- **自适应探针CSV**：探针CSV不再是必须的，自动使用标签CSV中的温度数据
- **完全无CSV支持**：仅从图片训练也可正常运行
- **探针CSV缺失警告**：日志中显示 `[自适应] 使用标签CSV的温度数据生成时序`

#### 评估输出优化

- **JSON行输出**：评估完成后输出 `__EVAL_JSON__{...}__EVAL_JSON__` 供批量工具解析
- **指标精确解析**：避免 ANSI 颜色代码干扰

### 快捷命令格式

```bash
# 格式: backbone_2d,epochs,offset,task
python tools/batch_train_gui.py \
  "resnet18,100,0,detection" \
  "vit_small,100,0,detection" \
  "resnet18,100,1,detection"
```

### 使用示例

```bash
# 启动交互式界面
python tools/batch_train_gui.py

# 添加预设
选择 [a] → 选择 1 (CNN vs Transformer 对比 6条)

# 开始训练
选择 [7] 开始批量训练

# 查看对比表格
选择 [9] 生成对比表格
```

---

## [v4.2.0] - 2026-07-29

### 新增：可视化与分析工具

#### Streamlit 交互式工具 (tools/streamlit_app.py)

- **实时参数调整**：滑块控制裂纹/气孔阈值、合并距离、最小面积、边界排除
- **三种视图模式**：二值掩膜 / 结构分解 / 对比视图
- **气孔与裂纹分离**：使用多级阈值 + 连通域分析自动分离
- **像素级诊断面板**：输入坐标查看该点是否被识别为裂纹/气孔
- **裁剪区域可视化**：自动检测基板边界，显示裁剪框（绿色框 + 蓝色角标）
- **自动调参功能**：网格搜索最佳参数组合以提取最长裂纹
- **高清图像显示**：使用 st.image() 替代 matplotlib，支持 2x 放大

#### 结构分析模块 (tools/structure_analyzer.py)

- `separate_crack_pore()`: 分离裂纹和气孔掩膜
- `auto_optimize_params()`: 自动寻优最佳参数组合
- 支持边界伪影过滤、面积过滤、长宽比过滤
- 正方形裁剪输出保证（使用 min(h,w)）

#### 命令行可视化工具 (tools/visualization.py)

- `--check-preprocess`: 预处理检查（原始图 + 掩膜）
- `--compare`: 单样本对比（温度场、应力场、掩膜、叠层）
- `--batch-compare`: 批量对比生成
- 支持等值线去除效果对比

#### 图像预处理修复

- **正方形裁剪**：修复 `ImagePreprocessor` 使用 `min(h,w)` 确保 1:1 长宽比
- **自动基板检测**：使用灰度阈值检测基板边界（排除色标/标签）

#### 诊断面板修复

- **简化判定逻辑**：直接显示最终判定结果（裂纹/气孔/其他/边界）
- **十字高亮标记**：选中像素位置用青色十字标记
- **悬停坐标提示**：使用 Plotly 实现悬停查看坐标和灰度

---

## [v4.1.0] - 2026-07-27

### 重大升级：从"矩形框检测"到"像素级裂纹掩膜分割"

本次更新是项目的重要里程碑，从单一检测任务扩展为支持三种任务模式。

#### 新增功能

- **三种任务模式** (`--task` 参数)
  - `detection`: 矩形框检测（原有功能）
  - `segmentation`: 像素级裂纹掩膜分割（新增）
  - `multitask`: 同时输出检测框和分割掩膜（新增）

- **分割标签支持**
  - 使用 `d_hist/*.png` 损伤场图像作为分割标签
  - 图像为主，表格为辅（无表格也能训练分割任务）
  - 充分利用参数化扫描3/4的数据（有图像但缺表格）

- **MaskDecoder 分割头**
  - 8层转置卷积上采样
  - 从576维融合特征逐步上采样到256×256掩膜
  - 使用Sigmoid激活输出二值掩膜

- **新增损失函数**
  - `DiceLoss`: 用于分割任务的Dice系数损失
  - `SegmentationLoss`: Dice Loss + BCE 组合
  - `MultimodalSegmentationLoss`: 多任务损失（检测+分割）

#### 代码修复

- **run_train.py**
  - 修复：损失函数根据 `--task` 参数自动选择
  - 修复：模型创建时传递 `task` 参数
  - 修复：`evaluate_model()` 支持不同任务类型的输出
  - 修复：评估结果根据任务类型显示不同指标

- **dataset_multimodal.py**
  - 修复：multitask 模式标签顺序统一为 `(mask, detection)`

- **models/pe_tsnet_multimodal.py**
  - 修复：`_init_weights()` 不再覆盖预训练的 ResNet 权重

#### API 变更

```bash
# 检测模式（默认，保持向后兼容）
python run_train.py --mode train --task detection

# 分割模式（新增）
python run_train.py --mode train --task segmentation

# 多任务模式（新增）
python run_train.py --mode train --task multitask
```

#### 检查点命名规范

格式：`{backbone_2d}_{backbone_1d}_{fusion}_task{task}_offset{offset}_best.pt`

示例：
- `resnet18_cnn_attn_cross_attn_taskdetection_offset0_best.pt`
- `resnet18_cnn_attn_cross_attn_tasksegmentation_offset0_best.pt`
- `resnet18_cnn_attn_cross_attn_taskmultitask_offset0_best.pt`

---

## [v4.0.0] - 2026-07-XX

### 初始版本

PE-MMNet v4 多模态融合网络

#### 核心功能

- 非对称双分支架构（1D时序 + 2D图像）
- Cross-Attention 多模态融合
- 物理单调性约束损失
- 多批次数据合并加载
- 交互式训练启动器

#### 模型变体

- `full`: 完整MM-DBFNet
- `1d_only`: 仅时序分支
- `2d_only`: 仅空间分支
- `concat`: 双分支拼接
- `add`: 双分支加法
- `cross_attn`: Cross-Attention

---

## 迁移指南

### 从 v4.0.0 升级到 v4.1.0

**向后兼容性**：原有命令 `python run_train.py --mode train` 仍然有效，默认使用 `detection` 任务模式。

**检查点兼容性**：
- v4.0.0 的检查点可以继续用于 detection 任务
- 分割和多任务模型需要使用新格式的检查点

**训练脚本变更**：
- 新增 `--task` 参数（可选，默认 `detection`）
- 模型会根据任务类型自动选择损失函数
- 评估指标会根据任务类型变化
