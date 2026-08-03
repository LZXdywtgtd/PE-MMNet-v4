# 更新日志 (CHANGELOG)

所有重大更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v4.6.0] - 2026-08-03

### 新增：动态训练时间估算与团队协作增强

#### 动态训练时间估算

- **训练前估算**：训练开始前先测量2个epoch的实际速度
  - `estimate_training_time()` 函数自动保存/恢复模型状态
  - 显示 "预估总时间: ~X小时"
  - 避免训练中途发现时间过长

- **实时 ETA 显示（EMA + 置信区间）**
  - `ETAEstimator` 类：使用指数移动平均 (EMA) 计算
  - 置信区间：至少5个epoch后显示 ±Xs
  - 预计完成时间：如 "14:32"
  - 每10轮显示训练进度盒子

#### 检查点元数据增强

- **新增 `--task_id` 参数**
  - 团队协作时用于标记检查点所属任务
  - 检查点保存时包含 `task_id`, `epoch`, `timestamp` 元数据
  - team_train.py 执行任务时自动传递 task_id

#### 团队协作增强

- **任务执行日志**
  - `logs/team_training.log` 记录每个任务的执行情况
  - 记录内容：任务ID、状态、耗时、错误信息、主机名
  - 支持任务追溯和团队协作追踪

- **增强命令行参数**
  - `--auto`: 自动执行所有可执行任务
  - `--force`: 强制执行硬件警告任务（无需确认）
  - `--import`: 导入队友检查点

- **改进文件名解析**
  - `parse_task_id_from_filename()` 函数
  - 支持正则表达式精确匹配多种格式
  - 备用方案：子字符串匹配

- **颜色兼容性增强**
  - 自动检测终端颜色支持能力
  - 支持 `NO_COLOR` 环境变量
  - Windows Terminal / cmd.exe 兼容

#### 默认参数调整

- **统一训练默认设置**
  - `epochs` 默认值：100 → **150**
  - `patience` 默认值：20 → **30**
  - 影响范围：run_train.py、launcher.py、batch_train_gui.py
  - 指令指定参数时优先使用指定值

#### 代码修改

- `run_train.py` 新增：
  - `ETAEstimator` 类（第731-800行）
  - `estimate_training_time()` 函数（第803-860行）
  - `--task_id` 参数
  - 检查点保存增强

- `team_train.py` 新增：
  - `log_task_execution()` 函数
  - `parse_task_id_from_filename()` 函数
  - `get_color_support()` 函数
  - `--auto`, `--force` 命令行参数

---

## [v4.5.0] - 2026-08-02

### 新增：导师反馈8项优化建议实现

#### P0 优先级优化

**分阶段训练**
- 新增 `--staged_train` 参数：先短序列预训练，再长序列微调
- 新增 `--short_batches` 参数：指定短序列批次列表
- 新增 `--freeze_2d`/`--freeze_1d` 参数：阶段2冻结策略
- 新增 `freeze_model_backbone()` 函数

**SE/CoordAttention 注意力模块**
- `models/pe_tsnet_multimodal.py` 新增：
  - `SEBlock`: Squeeze-and-Excitation 通道注意力
  - `CoordAtt`: 坐标注意力（保留位置信息）
  - `BackboneWithAttention`: 骨干网络包装器
- 新增 `--use_se`/`--use_coord_attn` 参数

#### P1-P2 优先级优化

**PatchTST 1D 骨干**
- 新增 `models/pe_tsnet_patchtst.py`：
  - `PatchTST1D`: 分块Transformer时序编码器
  - `PatchTSTWithRate`: 增强版（处理初始温度+变化率）
- 新增变体：`swin_yolo_patchtst`, `vit_yolo_patchtst`

**门控多模态融合**
- 新增 `models/pe_tsnet_fusion.py`：
  - `CrossAttentionBranch`: 交叉注意力分支
  - `GatedMultimodalFusion`: 温度/应力分治门控融合
- 新增 `--fusion gated` 选项

#### P3-P5 优先级优化

**ThermalCutMix 增强（物理安全版）**
- 新增 `training/augmentation.py`：
  - `ThermalCutMix`: 仅混合温度通道，应力通道保持不变
- 新增 `--aug_cutmix_prob` 参数（默认0，关闭）

**密度一致性损失**
- `training/mono_loss.py` 新增：
  - `DensityConsistencyLoss`: 邻域网格密度平滑约束
  - `CombinedDensityLoss`: 组合密度损失

**三通道时序输入**
- `data/dataset_multimodal.py` 新增：
  - `create_triple_channel_seq()`: 生成 [初始温度, 当前温度, 温度变化率]
- 新增 `--triple_channel` 参数

#### 训练稳定性增强

- 显存自动降级策略
- 混合精度训练 (FP16) + GradScaler
- 评估时关闭 FP16 保持精度

#### 团队协作功能

- 新增 `team_train.py`：团队协作训练脚本
  - 团队成员只需运行 `python team_train.py`
  - 选择数字即可开始训练，无需了解命令行参数
  - 预置10种训练任务 + 批量YOLO系列训练
- 新增 `team_configs.txt`：团队配置文件
- 新增 `export_team_configs()` 函数：自动导出团队配置
- `launcher.py` 新增团队协作预设快捷键（`team_base`, `team_yolo`, `team_opt`）

---

## [v4.4.0] - 2026-08-02

### 新增：YOLO-FPN 和 DETR 风格变体

#### 新增模型文件

- **`models/pe_tsnet_yolo.py`**：Swin-YOLO-FPN 和 ViT-YOLO-FPN 变体
  - `SwinBackbone2D`: Swin-Tiny 骨干网络（使用 timm）
  - `ViTYOLOBackbone2D`: ViT-Small 骨干网络
  - `FPNNeck`: Feature Pyramid Network 颈部网络
  - `YOLOFPNHead`: YOLO 检测头
  - `SwinYOLOFPN`: Swin-Tiny + FPN + YOLO 网格回归
  - `ViTYOLOFPN`: ViT-Small + FPN + YOLO 网格回归

- **`models/pe_tsnet_detr.py`**：DETR 风格变体
  - `PositionalEncoding2D`: 2D 正弦位置编码
  - `DETRDecoder`: Transformer 解码器 + 可学习 Object Queries
  - `DETRHead`: DETR 预测头
  - `DETRStyle`: ResNet-18 + Transformer + Hungarian Matching

#### 新增损失函数

- **`training/mono_loss.py`** 新增：
  - `YOLOTargetAssigner`: YOLO 目标分配器（多网格正样本策略）
  - `YOLOLoss`: YOLO 检测损失（DIoU + BCE + 密度）
  - `HungarianMatcher`: Hungarian 匹配器（使用 scipy.optimize.linear_sum_assignment）
  - `DETRLoss`: DETR 检测损失（匹配 + Smooth L1 + BCE）

#### 架构设计

- **训练/推理模式分离**：使用 `self.training` 属性区分，无需额外参数
  - 训练模式：返回完整网格/query 预测
  - 推理模式：返回最高 conf 的预测
- **单调性约束适配**：简化为单值密度预测损失（YOLO/DETR 无时序信息）
- **任务支持**：仅 detection（符合架构设计原则）

#### 新增变体命令

```bash
# Swin-YOLO-FPN
python run_train.py --mode train --variant swin_yolo_fpn --epochs 100

# ViT-YOLO-FPN
python run_train.py --mode train --variant vit_yolo_fpn --epochs 100

# DETR风格（需要更小学习率）
python run_train.py --mode train --variant detr_style --epochs 100 --lr 1e-4
```

#### 依赖更新

- `requirements.txt` 新增 `timm>=0.9.0`

#### 代码修复

- **`evaluate_model()` 函数增强**
  - 新增 `variant_key` 参数
  - 自动解包新变体的 `(output, global_density)` 元组
  - 支持推理模式下取最优预测

---

## [v4.3.0] - 2026-07-30

### 文档重写 + P2 优化项完成

#### 文档重写

- **README.md**：精简为核心功能说明
- **快速配置指南.md**：简化为快速上手指南
- **架构设计文档.md**：精简为当前架构状态（移除待修复内容）
- **CUDA安装指南.md**：简化为标准安装流程
- **调参文档.md**：简化为损失函数调参指南
- **算法报告.md**：精简为核心算法原理

#### 评估函数增强

- **`eval_checkpoint()` 从检查点加载配置**
  - 从检查点的 `config` 字典读取 task、image_size、variant 等参数
  - 回退机制：检查点无配置时从文件名推断
  - 确保评估配置与训练配置一致

- **`ResultsComparator` 传递 image_size**
  - `_eval_checkpoint()` 增加 `image_size` 参数
  - 从 TrainingCommand 获取 image_size，传递给评估函数

- **批量评估修复**
  - 修复 Windows 下的编码问题（使用 UTF-8 + errors='replace'）
  - JSON 解析失败时输出警告而非静默跳过
  - 批量评估现在返回正确指标（不再是全0）

#### 交互界面增强

- **配置保存/加载功能**
  - 重命名 `launcher.load_config()` → `load_command_config()`
  - 避免与 `utils.config.load_config()` 混淆
  - 主菜单添加 [l] 加载配置、[s] 保存配置 选项

- **消融结果表格格式优化**
  - 使用展平数据直接打印，修复嵌套字典格式问题
  - 清晰展示所有变体的 R²、RMSE、MAE、mIoU、违反率

#### 代码质量

- **修复 `get_arch_suggestions()` 返回字段名**
  - `seq_mode` → `seq_interp_mode`
  - 与实际参数名保持一致

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
