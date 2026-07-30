# API 参考文档

本文档详细描述 `project_v4` 中所有公开可调用的 Python 接口。

---

## utils.config 模块

### 配置中心

路径配置已统一迁移到 `utils/config.py`，通过 `config.json` 管理所有路径。

#### `load_config`

加载配置文件。

```python
from utils.config import load_config

config = load_config()
# 返回: dict 或 None
# {
#     "data_root": "D:\\Desktop\\team_project\\simulation\\参考输入",
#     "output_dir": "./output",
#     "checkpoints_dir": "./checkpoints",
#     "results_dir": "./benchmark_results"
# }
```

#### `get_data_root`

获取数据根目录路径。

```python
from utils.config import get_data_root

data_root = get_data_root()
# 返回: str 或 None（未配置时）
```

#### `get_data_batches`

获取已配置的数据批次列表。

```python
from utils.config import get_data_batches

batches = get_data_batches()
# 返回: list[str] 或 None（未配置时）
# 示例: ["单次扫描", "参数化扫描1", "参数化扫描2"]
```

#### `ensure_config`

确保配置文件存在，必要时引导用户配置。

```python
from utils.config import ensure_config

success = ensure_config(interactive=True)
# interactive=True: 交互式提示用户输入
# interactive=False: 仅检查，返回是否已配置
```

---

## data 模块

### `data.dataset_multimodal`

#### 常量定义

| 常量 | 值 | 说明 |
|------|-----|------|
| `DATA_ROOT` | `_LazyDataRoot` | **已移除硬编码路径**，请使用 `utils.config.get_data_root()` |
| `DATA_BATCHES` | `None` | **已移除硬编码路径**，请使用 `utils.config.get_data_batches()` |
| `IMAGE_SIZE` | `256` | 2D图像最终尺寸 |
| `CROP_RATIO` | `0.70` | Center Crop比例 |
| `SEQ_LEN` | `300` | 1D序列目标长度 |

---

#### `create_multibatch_dataloaders`

创建多批次数据加载器（推荐使用）。

```python
def create_multibatch_dataloaders(
    data_roots,         # list[str]: 数据目录列表
    batch_size=16,      # int: 批次大小
    seq_len=300,        # int: 序列长度
    image_size=256,     # int: 图像尺寸
    num_workers=0,      # int: 数据加载线程数
    train_ratio=0.8,    # float: 训练集比例
    augment=True        # bool: 是否启用数据增强
) -> tuple[DataLoader, DataLoader]
```

**示例**：

```python
from data.dataset_multimodal import create_multibatch_dataloaders
from utils.config import get_data_root, get_data_batches
import os

# 获取配置的路径
data_root = get_data_root()
batches = get_data_batches()

if data_root and batches:
    data_roots = [os.path.join(data_root, batch) for batch in batches]
    train_loader, test_loader = create_multibatch_dataloaders(
        data_roots=data_roots,
        batch_size=16,
        augment=True
    )
else:
    print("请先运行 python run_train.py --mode train 配置数据路径")
```

**返回值**：
```python
(train_loader, test_loader)  # (训练DataLoader, 测试DataLoader)
```

---

#### `MultiBatchDataset`

多批次数据集管理类。

```python
class MultiBatchDataset:
```

**构造函数**：

```python
MultiBatchDataset(
    data_roots,         # list[str]: 数据目录列表
    split='train',      # str: 'train' 或 'test'
    train_ratio=0.8,    # float: 训练集比例
    seq_len=300,        # int: 序列长度
    image_size=256,     # int: 图像尺寸
    augment=True        # bool: 是否启用数据增强
)
```

**特性**：
- 自动检测并加载所有数据批次
- 支持不同CSV格式（单次扫描、参数化扫描）
- 统一训练/测试集划分（跨批次随机划分）

---

#### `MultiBatchCollateDataset`

使用 ConcatDataset 实现的多批次数据集。

```python
class MultiBatchCollateDataset(Dataset)
```

**构造函数**：

```python
MultiBatchCollateDataset(
    data_roots,         # list[str]: 数据目录列表
    split='train',      # str: 'train' 或 'test'
    train_ratio=0.8,    # float: 训练集比例
    seq_len=300,        # int: 序列长度
    image_size=256,     # int: 图像尺寸
    augment=True,       # bool: 是否启用数据增强
    seed=42,            # int: 随机种子
    predict_offset=0,   # int: 时间偏移量
    seq_interp_mode='interpolate',  # str: 'interpolate' 或 'pool'
    remove_contours=False,  # bool: 是否启用等值线去除
    disabled_batches=None,  # list[str]: 禁用的批次列表
    task='detection'    # str: 'detection' | 'segmentation' | 'multitask'
)
```

**task 参数**：
- `'detection'`: 返回检测标签 (batch, 6)
- `'segmentation'`: 返回分割掩膜 (batch, 1, 256, 256)
- `'multitask'`: 返回 (mask, detection) 元组

---

#### `SingleBatchDataset`

单批次数据集类（用于加载单个数据目录）。

```python
class SingleBatchDataset(Dataset)
```

**构造函数**：

```python
SingleBatchDataset(
    data_root,          # str: 数据目录根路径
    seq_len=300,        # int: 序列长度
    image_size=256,     # int: 图像尺寸
    augment=True,       # bool: 是否启用数据增强
    sample_indices=None,# list[int]: 指定使用的样本索引
    predict_offset=0,   # int: 时间偏移量
    seq_interp_mode='interpolate',  # str: 序列处理模式
    remove_contours=False,  # bool: 是否启用等值线去除
    task='detection'    # str: 'detection' | 'segmentation' | 'multitask'
)
```

**`__getitem__` 返回格式**：

```python
# detection 模式
(
    (seq_1d, img_2d),   # tuple:
    label               # torch.Tensor: [x, y, l, w, confidence, density]
)
# seq_1d:   torch.Size([300])   归一化后的温度时序
# img_2d:   torch.Size([2, 256, 256]) 温度场+应力场图像（2通道）
# label:    torch.Size([6])  预测目标

# segmentation 模式
(
    (seq_1d, img_2d),   # tuple:
    mask                # torch.Tensor: (1, 256, 256) 二值掩膜
)

# multitask 模式
(
    (seq_1d, img_2d),   # tuple:
    (mask, detection)   # tuple: (掩膜, 检测标签)
)
```

---

#### `MaskLabelProcessor`

分割标签处理器，将 d_hist/*.png 损伤场图像转换为二值掩膜。

```python
class MaskLabelProcessor
```

**构造函数**：

```python
MaskLabelProcessor(
    image_size=256,        # int: 目标图像尺寸
    crop_ratio=0.70,       # float: 中心裁剪比例
    binary_threshold=0.1,  # float: 二值化阈值
    invert=True            # bool: 是否反转（d_hist 裂纹区域为暗色）
)
```

**处理流程**：
1. Center Crop（与 ImagePreprocessor 一致）
2. 归一化到 [0, 1]
3. 反转（如需要，d_hist 裂纹区域为暗色）
4. 二值化
5. Resize 到目标尺寸

---

#### `ImagePreprocessor`

图像预处理器（核心预处理流水线）。

```python
class ImagePreprocessor
```

**处理流程**：
1. 灰度转换（RGB → 灰度）
2. Max-Min归一化（解决跨批次色标差异）
3. Center Crop（70%比例，聚焦物理区域）
4. Resize到目标尺寸
5. 转为Tensor [0, 1]

```python
ImagePreprocessor(
    image_size=256,     # int: 最终尺寸
    crop_ratio=0.70     # float: 裁剪比例
)
```

---

#### `load_csv_safely`

安全加载 COMSOL 导出的 CSV 文件（跳过 `%` 注释行）。

```python
def load_csv_safely(filepath: str) -> pd.DataFrame
```

**返回**：无列名的 DataFrame（列位置固定）。

---

#### `parse_label_csv`

解析标签CSV（支持多种格式）。

```python
def parse_label_csv(df: pd.DataFrame) -> dict
```

**支持的格式**：
- 单次扫描（10列）：time, temp, x, y, density, hist_x, hist_y, hist_d...
- 参数化扫描（11列）：h, time, temp, x, y, density, hist_x, hist_y, hist_d...

**返回字典键**：

| 键 | 类型 | 说明 |
|----|------|------|
| `time` | np.ndarray | 时间序列（秒） |
| `temp` | np.ndarray | 温度（K） |
| `x` | np.ndarray | 裂纹中心 X 坐标（米） |
| `y` | np.ndarray | 裂纹中心 Y 坐标（米） |
| `density` | np.ndarray | 裂纹密度（0~1） |
| `h_ceramic` | np.ndarray | 对流换热系数 |

---

#### `interpolate_seq`

将序列插值到目标长度。

```python
def interpolate_seq(
    time_points,        # np.ndarray: 原始时间点
    values,            # np.ndarray: 对应数值
    target_len=300     # int: 目标长度
) -> np.ndarray        # shape: (target_len,)
```

**关键特性**：使用 `scipy.interpolate.interp1d` 线性插值，`fill_value="extrapolate"` 防止边界 NaN。

---

#### `compute_lw_from_density`

根据裂纹密度计算裂纹边界框尺寸（启发式公式）。

```python
def compute_lw_from_density(density: float) -> tuple[float, float]
```

**物理依据**：
- `l = sqrt(density) * 0.3`
- `w = sqrt(density) * 0.2`

---

#### `PhysicalSafeTransform1D`

1D 序列的物理安全数据增强类。

```python
class PhysicalSafeTransform1D:
    def __init__(
        self,
        mask_ratio=0.05,  # 时间遮蔽比例
        noise_std=0.005,  # 噪声标准差
        enabled=True      # 是否启用
    )
```

**允许的增强**：
- **Time Masking**：随机遮蔽 5% 时间窗口
- **极微小噪声**：模拟传感器噪声

---

#### `PhysicalSafeTransform2D`

2D 图像的物理安全数据增强类。

```python
class PhysicalSafeTransform2D:
    def __init__(
        self,
        noise_std=0.01,     # 高斯噪声标准差
        brightness=0.05,    # 亮度调整范围
        contrast=0.05,      # 对比度调整范围
        enabled=True
    )
```

**允许的增强**：
- **GaussianNoise**：极微小高斯噪声
- **BrightnessContrastAdjust**：极微小亮度/对比度调整

**禁止的增强**：旋转、翻转、裁剪、缩放（破坏物理边界条件）。

---

## models 模块

### `models.pe_tsnet_multimodal`

#### `PETSNetMultimodal`

多模态融合网络（主模型类）。

```python
class PETSNetMultimodal(nn.Module)
```

**构造函数**：

```python
PETSNetMultimodal(
    seq_len=300,           # int: 1D 序列长度
    image_channels=2,      # int: 2D 图像通道数
    pretrained_2d=True,    # bool: 2D 分支是否使用 ImageNet 预训练权重
    dropout=0.2,           # float: Dropout 比例
    task='detection'       # str: 'detection' | 'segmentation' | 'multitask'
)
```

**task 参数**：
- `'detection'`: 输出 (batch, 6) 检测向量
- `'segmentation'`: 输出 (batch, 1, 256, 256) 掩膜
- `'multitask'`: 输出 (mask, detection) 元组

**前向传播**：

```python
def forward(
    x_1d,   # torch.Tensor: (batch, 300)
    x_2d    # torch.Tensor: (batch, 2, 256, 256)
) -> torch.Tensor  # 根据 task 参数返回不同格式

# detection 模式
# -> torch.Tensor: (batch, 6) -> [x, y, l, w, confidence, density]

# segmentation 模式
# -> torch.Tensor: (batch, 1, 256, 256) 二值掩膜

# multitask 模式
# -> tuple[torch.Tensor, torch.Tensor]: (mask, detection)
```

**网络架构**：

```
分支1（2D）：ResNet-18 backbone → 512 维特征向量
分支2（1D）：Micro 1D-CNN + Macro 1D-CNN → LayerNorm → MultiHead Self-Attention → 64 维特征向量
融合：Cross-Attention（双向交叉注意力）→ 576 维融合特征
输出（根据 task）：
├── detection: MLP → 6 个目标值（Sigmoid 激活 confidence 和 density）
├── segmentation: MaskDecoder → 256×256 二值掩膜
└── multitask: 同时输出检测向量和分割掩膜
```

**MaskDecoder 架构**：
```
输入: (batch, 576) 融合特征 → reshape → (batch, 576, 1, 1)
├── ConvTranspose2d 576→256 (4×4, stride=2, pad=1) → 2×2
├── ConvTranspose2d 256→128 → 4×4
├── ConvTranspose2d 128→64  → 8×8
├── ConvTranspose2d 64→32   → 16×16
├── ConvTranspose2d 32→16   → 32×32
├── ConvTranspose2d 16→8    → 64×64
├── ConvTranspose2d 8→4     → 128×128
├── ConvTranspose2d 4→1     → 256×256
└── Sigmoid → (batch, 1, 256, 256)
```

共 8 次转置卷积，逐步从 1×1 上采样到 256×256。

**参数量**：约 12,144,838（检测模式），分割模式略多

---

#### `MaskDecoder`

掩膜解码器，将融合特征上采样回 256×256 二值掩膜。

```python
class MaskDecoder(nn.Module)
```

**构造函数**：

```python
MaskDecoder(
    in_channels=576,   # int: 输入特征维度
    hidden_dim=128     # int: 隐藏层维度（未使用，保留）
)
```

---

#### `ResNet18Backbone2D`

ResNet-18 视觉骨干网络类。

```python
class ResNet18Backbone2D(nn.Module)
```

**输入**：(batch, in_channels, H, W)
**输出**：(batch, 512)

---

#### `TemporalFeatureExtractor`

1D 时序特征提取器（Micro + Macro 并行 → Concat → Self-Attention）。

```python
class TemporalFeatureExtractor(nn.Module)
```

- Micro + Macro 双分支并行
- Concat → 1×1 Conv → LayerNorm → MultiHead Self-Attention → FFN → GlobalAvgPool
- 输出：`(batch, 64)`

---

#### `CrossAttentionFusion`

双向交叉注意力融合模块。

```python
class CrossAttentionFusion(
    dim_2d=512,    # 2D 分支特征维度
    dim_1d=64,     # 1D 分支特征维度
    num_heads=4,   # 注意力头数
    dropout=0.1
)
```

---

#### `MultiTaskHead`

多任务输出头。

```python
class MultiTaskHead(
    input_dim=576,  # 输入特征维度
    hidden_dim=256,
    dropout=0.2
)
```

**输出格式**：[x, y, l, w, confidence, density]
- x, y, l, w：ReLU 确保非负
- confidence, density：Sigmoid 激活到 [0,1]

---

## training 模块

### `training.mono_loss`

#### `MultimodalCrackLoss`

多模态裂纹预测组合损失函数类（用于检测任务）。

```python
class MultimodalCrackLoss(nn.Module)
```

**构造函数**：

```python
MultimodalCrackLoss(
    lambda_mse_density=1.0,  # MSE 损失权重
    lambda_mono=0.1,          # 单调性损失权重
    lambda_loc=1.0,           # 定位损失权重
    lambda_conf=1.0,          # 置信度损失权重
    use_ciou=False            # 是否使用 CIoU（默认 DIoU）
)
```

**前向传播**：

```python
def forward(
    pred,   # torch.Tensor: (batch, 6) 预测值
    target  # torch.Tensor: (batch, 6) 真实标签
) -> tuple[float, dict]:
    # 返回 (总损失, 各损失分量字典)
```

**损失分量**：

| 键 | 说明 |
|----|------|
| `mse_density` | 预测密度与真实密度的 MSE |
| `mono` | 单调性约束损失（裂纹密度随时间单调递增） |
| `loc` | 定位损失（DIoU + MSE for [x, y, l, w]） |
| `conf` | 置信度损失（BCE for 裂纹存在概率） |
| `total` | 加权总损失 |

---

#### `DiceLoss`

Dice Loss 用于图像分割任务。

```python
class DiceLoss(nn.Module)
```

**构造函数**：

```python
DiceLoss(
    smooth=1.0  # float: 平滑项，防止除零
)
```

**前向传播**：

```python
def forward(
    pred,   # torch.Tensor: (batch, 1, H, W) 或 (batch, H, W)
    target  # torch.Tensor: 同上
) -> torch.Tensor:  # 标量损失
```

**原理**：Dice = 2 * |A ∩ B| / (|A| + |B|)，对类别不平衡更鲁棒。

---

#### `SegmentationLoss`

分割任务损失：Dice Loss + BCE Loss 组合。

```python
class SegmentationLoss(nn.Module)
```

**构造函数**：

```python
SegmentationLoss(
    lambda_dice=1.0,  # Dice Loss 权重
    lambda_bce=0.5    # BCE Loss 权重
)
```

**前向传播**：

```python
def forward(
    pred,   # torch.Tensor: (batch, 1, 256, 256)
    target  # torch.Tensor: (batch, 1, 256, 256)
) -> torch.Tensor:  # 标量损失
```

---

#### `MultimodalSegmentationLoss`

多任务损失：检测 + 分割。

```python
class MultimodalSegmentationLoss(nn.Module)
```

**构造函数**：

```python
MultimodalSegmentationLoss(
    lambda_seg=1.0,  # 分割损失权重
    lambda_det=0.5   # 检测损失权重
)
```

**前向传播**：

```python
def forward(
    outputs,  # tuple: (pred_mask, pred_det)
    targets   # tuple: (target_mask, target_det)
) -> torch.Tensor:  # 标量损失
```

**特性**：支持检测标签为 None 的情况（长序列数据只有分割标签）。

---

#### `MonotonicityLossV3`

物理单调性约束损失类（v4 版本）。

```python
class MonotonicityLossV3(nn.Module)
```

**物理原理**：格里菲斯断裂准则——裂纹密度随温度下降（热应力增加）不可逆地单调递增。

---

#### `diou_loss`

计算 DIoU（Distance-IoU）损失。

```python
def diou_loss(
    pred_boxes,    # torch.Tensor: (batch, 4) -> [x, y, w, h]
    target_boxes,  # torch.Tensor: (batch, 4)
    eps=1e-7
) -> torch.Tensor: 标量损失
```

---

#### `verify_monotonicity`

验证预测结果是否满足单调性约束。

```python
def verify_monotonicity(
    predictions  # torch.Tensor or np.ndarray: (N,) or (N, 1)
) -> dict:
```

**返回值**：

```python
{
    'violations': int,           # 违反单调性的次数
    'total_pairs': int,          # 总比较对数
    'monotonicity_score': float, # 单调性得分（1 - violations/total_pairs）
    'max_decrease': float,       # 最大下降幅度
    'max_increase': float        # 最大上升幅度
}
```

---

## run_train 模块

### `run_train`

统一训练脚本，支持训练、消融实验、评估三种模式。

#### 常量

| 常量 | 说明 |
|------|------|
| `VARIANT_MODELS` | 模型变体映射字典 |
| `VARIANT_DISPLAY_NAMES` | 变体显示名称映射 |

#### CLI 参数

```bash
python run_train.py [OPTIONS]

# 关键参数
--mode              # train | eval | ablation
--task              # detection | segmentation | multitask（默认 detection）
--epochs            # 训练轮数（默认 150）
--batch_size        # 批次大小（默认 16）
--lr                # 学习率（默认 3e-4）
--variant           # full | 1d_only | 2d_only | concat | add | cross_attn
--predict_offset    # 时间偏移量（默认 0）
```

#### `train_model`

训练单个模型。

```python
def train_model(
    model,           # nn.Module: 模型
    train_loader,    # DataLoader: 训练数据
    test_loader,     # DataLoader: 测试数据
    config,          # dict: 训练配置（包含 'task' 键）
    device,          # torch.device: 设备
    checkpoint_path  # str: 保存路径
) -> tuple[model, dict]
```

**注意**：根据 `config['task']` 自动选择损失函数：
- `'detection'`: `MultimodalCrackLoss`
- `'segmentation'`: `SegmentationLoss`
- `'multitask'`: `MultimodalSegmentationLoss`

#### `train_variant`

训练单个消融变体。

```python
def train_variant(
    variant_key,  # str: 变体标识符
    config,       # dict: 训练配置
    device        # torch.device: 设备
) -> nn.Module
```

#### `evaluate_model`

评估模型性能。

```python
def evaluate_model(
    model,                 # nn.Module: 模型
    device,                # torch.device: 设备
    data_roots=None,       # list[str]: 数据目录列表
    predict_offset=0,      # int: 时间偏移量
    seq_len=300,           # int: 序列长度
    seq_interp_mode='interpolate',  # str: 序列处理模式
    remove_contours=False, # bool: 是否启用等值线去除
    disabled_batches=None, # list[str]: 禁用的批次列表
    task='detection'       # str: 'detection' | 'segmentation' | 'multitask'
) -> dict
```

**返回值**（根据 task 不同）：

| 键 | detection | segmentation | multitask |
|----|-----------|--------------|-----------|
| `r2` | 决定系数 | 0.0 | 决定系数 |
| `rmse` | 均方根误差 | 0.0 | 均方根误差 |
| `mae` | 平均绝对误差 | 0.0 | 平均绝对误差 |
| `mIoU` | 平均IoU | 0.0 | 平均IoU |
| `violation_rate` | 单调性违反率 | 0.0 | 单调性违反率 |
| `dice` | 0.0 | Dice 系数 | Dice 系数 |

#### `eval_checkpoint`

评估检查点文件。

```python
def eval_checkpoint(
    checkpoint_path: str,  # str: 检查点路径
    device                # torch.device: 设备
) -> dict
```

**特性**：从检查点文件名推断任务类型（`tasksegmentation`、`taskmultitask`）。

---

## main_v2 模块

### `main_v2`

#### `train_model`

训练模式主函数。

```python
def train_model(args: argparse.Namespace)
```

---

#### `infer_model`

推理模式主函数。

```python
def infer_model(args: argparse.Namespace) -> list[dict]
```

**返回值**：每个时间步的预测结果字典列表。

---

## multi_dataset_trainer 模块（旧脚本）

### `multi_dataset_trainer`

多批次数据训练入口。**注意**：推荐使用 `run_train.py` 统一脚本。

#### `train_model`

训练函数（与 main_v2 共享逻辑）。

```python
def train_model(
    model,           # nn.Module: 模型
    train_loader,    # DataLoader: 训练数据
    test_loader,     # DataLoader: 测试数据
    config,          # dict: 训练配置
    device,          # torch.device: 设备
    checkpoint_path  # str: 保存路径
) -> tuple[model, history, best_metrics]
```

---

## 常量定义

### data/dataset_multimodal.py

| 常量 | 值 | 说明 |
|------|-----|------|
| `DATA_ROOT` | `_LazyDataRoot` | **已移除硬编码路径**，请使用 `utils.config.get_data_root()` |
| `DATA_BATCHES` | `None` | **已移除硬编码路径**，请使用 `utils.config.get_data_batches()` |
| `IMAGE_SIZE` | `256` | 2D图像最终尺寸 |
| `CROP_RATIO` | `0.70` | Center Crop比例 |
| `SEQ_LEN` | `300` | 1D序列目标长度 |
