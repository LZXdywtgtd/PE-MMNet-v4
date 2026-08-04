# API 参考文档

> PE-MMNet v4 公开接口

---

## 1. 数据模块 (data.dataset_multimodal)

### create_multibatch_dataloaders

创建多批次数据加载器。

```python
from data.dataset_multimodal import create_multibatch_dataloaders

train_loader, test_loader = create_multibatch_dataloaders(
    data_roots=None,           # list[str]: 数据目录（None自动扫描config.json）
    batch_size=16,             # int: 批次大小
    image_size=512,            # int: 图像尺寸
    seq_len=300,              # int: 序列长度
    augment=True,             # bool: 数据增强
    predict_offset=0,          # int: 时间偏移
    task='detection',         # str: 'detection', 'segmentation', 'multitask'
    triple_channel=False       # bool: 三通道时序输入
)
```

**返回**：
- `train_loader`: 训练数据加载器
- `test_loader`: 测试数据加载器

### create_triple_channel_seq

生成三通道时序数据。

```python
from data.dataset_multimodal import create_triple_channel_seq

result = create_triple_channel_seq(seq_1d, seq_len=300)
# result: (3, seq_len) - [初始温度, 当前温度, 温度变化率]
```

---

## 2. 模型模块

### 2.1 基础模型 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import PETSNetMultimodal

model = PETSNetMultimodal(
    seq_len=300,
    image_channels=2,
    image_size=512,
    pretrained_2d=True,
    dropout=0.2,
    task='detection'
)
```

**输出**：
- 训练模式：`(B, 6)`, `(B, 1)`
- 推理模式：`(B, 6)`, `(B, 1)`

### 2.2 YOLO-FPN 变体 (models.pe_tsnet_yolo)

```python
from models.pe_tsnet_yolo import SwinYOLOFPN, ViTYOLOFPN

# Swin-YOLO-FPN
model = SwinYOLOFPN(
    seq_len=300,
    image_channels=2,
    image_size=512,
    grid_size=16,
    num_queries=256,
    pretrained_2d=True,
    dropout=0.2
)

# ViT-YOLO-FPN
model = ViTYOLOFPN(
    seq_len=300,
    image_channels=2,
    image_size=512,
    grid_size=16,
    num_queries=256,
    pretrained_2d=True,
    dropout=0.2
)
```

**输出**：
- 训练模式：`(B, 256, 6)`, `(B, 1)`
- 推理模式：`(B, 6)`, `(B, 1)`

### 2.3 DETR 风格变体 (models.pe_tsnet_detr)

```python
from models.pe_tsnet_detr import DETRStyle

model = DETRStyle(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2,
    d_model=512,
    num_queries=100,
    encoder_layers=6,
    decoder_layers=6
)
```

**输出**：
- 训练模式：`(B, 100, 6)`, `(B, 1)`
- 推理模式：`(B, 6)`, `(B, 1)`

### 2.4 骨干网络 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import (
    ResNet18Backbone2D,
    TemporalFeatureExtractor
)

# 2D 骨干
backbone_2d = ResNet18Backbone2D(
    in_channels=2,
    pretrained=True
)
backbone_2d.set_spatial_output(True)  # 启用空间特征输出

# 1D 特征提取器
backbone_1d = TemporalFeatureExtractor(
    input_dim=1,
    hidden_dim=32,
    num_heads=4,
    dropout=0.2
)
```

### 2.5 融合模块 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import CrossAttentionFusion, AdaptiveFusion

# 交叉注意力融合
fusion = CrossAttentionFusion(
    dim_2d=6,   # 输出维度（来自模型预测）
    dim_1d=64,  # 1D特征维度
    num_heads=4
)

out = fusion(query_feat, feat_1d)
# query_feat: (B, 6) 或 (B, seq, 6)
# feat_1d: (B, dim_1d)
# out: (B, dim_1d) 或 (B, seq, dim_1d)
```

### 2.6 注意力模块 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import SEBlock, CoordAtt, BackboneWithAttention

# SE注意力
se = SEBlock(channels=512, reduction=16)
out = se(x)  # (B, C, H, W)

# 坐标注意力
coord = CoordAtt(inp=512, oup=512, reduction=32)
out = coord(x)

# 骨干包装器
model = BackboneWithAttention(
    backbone=resnet18,
    attention_type='coord',
    reduction=32
)
```

### 2.7 PatchTST 时序骨干 (models.pe_tsnet_patchtst)

```python
from models.pe_tsnet_patchtst import PatchTST1D, PatchTSTWithRate

# 标准 PatchTST
model = PatchTST1D(
    seq_len=300,
    patch_size=10,
    d_model=64,
    nhead=4,
    num_layers=2,
    output_dim=64
)

# 增强版（处理三通道：初始温度+变化率）
model = PatchTSTWithRate(
    seq_len=300,
    d_model=64,
    output_dim=64
)
```

### 2.8 门控融合模块 (models.pe_tsnet_fusion)

```python
from models.pe_tsnet_fusion import GatedMultimodalFusion, AdaptiveFusion

# 门控多模态融合
fusion = GatedMultimodalFusion(
    dim_2d=512,
    dim_1d=64,
    split_ratio=0.5,
    num_heads=4
)

out = fusion(feat_2d, feat_1d)
# feat_2d: (B, dim_2d)
# feat_1d: (B, dim_1d)
# out: (B, dim_1d)
```

---

## 3. 损失函数 (training.mono_loss)

### MultimodalCrackLoss

检测任务损失函数（用于 resnet18）。

```python
from training.mono_loss import MultimodalCrackLoss

criterion = MultimodalCrackLoss(
    lambda_mse_density=1.0,
    lambda_mono=0.1,
    lambda_loc=1.0,
    lambda_conf=1.0
)
```

### YOLOLoss

YOLO检测损失（用于 swin_yolo, vit_yolo）。

```python
from training.mono_loss import YOLOLoss, YOLOTargetAssigner

assigner = YOLOTargetAssigner(grid_size=16, nearby_range=2)
criterion = YOLOLoss(
    lambda_box=1.0,
    lambda_conf=1.0,
    lambda_mono=0.1
)
```

### DETRLoss

DETR检测损失（用于 detr）。

```python
from training.mono_loss import DETRLoss, HungarianMatcher

matcher = HungarianMatcher()
criterion = DETRLoss(
    matcher=matcher,
    lambda_bbox=1.0,
    lambda_conf=1.0,
    lambda_mono=0.1
)
```

### SegmentationLoss

分割任务损失函数。

```python
from training.mono_loss import SegmentationLoss

criterion = SegmentationLoss(
    lambda_dice=1.0,
    lambda_bce=1.0
)
```

### DensityConsistencyLoss

密度一致性损失（邻域平滑约束）。

```python
from training.mono_loss import DensityConsistencyLoss, CombinedDensityLoss

loss = DensityConsistencyLoss(
    grid_size=16,
    neighbor_range=1,
    lambda_consistency=0.5
)

# 组合损失
criterion = CombinedDensityLoss(
    lambda_mse=1.0,
    lambda_consistency=0.5,
    grid_size=16
)
```

---

## 4. 数据增强 (training.augmentation)

### ThermalCutMix

物理安全版CutMix增强（仅混合温度通道）。

```python
from training.augmentation import ThermalCutMix, RandomNoise, RandomFlip

cutmix = ThermalCutMix(alpha=1.0, prob=0.3)
img_aug, labels_aug = cutmix(img1, img2, labels1, labels2)

# img: (2, H, W) - 通道0=温度，通道1=应力
# 仅混合温度通道，应力通道保持不变
```

### RandomNoise

随机噪声增强。

```python
noise = RandomNoise(prob=0.5, scale=0.1)
img_aug = noise(img)
```

### RandomFlip

随机翻转增强。

```python
flip = RandomFlip(prob=0.5, mode='horizontal')
img_aug = flip(img)
```

---

## 5. 训练脚本 (run_train.py)

### train_model

训练模型。

```python
from run_train import train_model

model, metrics = train_model(
    model,
    train_loader,
    test_loader,
    config,
    device,
    checkpoint_path=None,  # 可选：检查点保存路径
    task_id=None          # 可选：任务ID（团队协作）
)

# 返回:
# - model: 训练后的模型
# - metrics: 最终评估指标 dict
```

### evaluate_model

评估模型。

```python
from run_train import evaluate_model

metrics = evaluate_model(
    model,
    device,
    data_roots=None,
    task='detection',
    image_size=512,
    variant_key='resnet18'  # 可选：模型变体
)

# 返回:
# - metrics: dict，包含 r2, rmse, mae, 违反率, mIoU 等
```

### estimate_training_time

动态训练时间估算（训练前测量）。

```python
from run_train import estimate_training_time

avg_time, total_minutes = estimate_training_time(
    model, train_loader, test_loader,
    criterion, optimizer, device, config
)

# 返回:
# - avg_time: float, 平均每epoch秒数
# - total_minutes: float, 预估总时间（分钟）
# 自动保存/恢复模型状态，不影响训练
```

### freeze_model_backbone

冻结骨干网络。

```python
from run_train import freeze_model_backbone

model = freeze_model_backbone(
    model,
    freeze_2d=True,    # 冻结2D骨干
    freeze_1d=False,   # 保持1D骨干可训练
    freeze_names=None  # 可选：额外要冻结的参数名
)
```

### ETAEstimator

指数移动平均 ETA 估算器。

```python
from run_train import ETAEstimator

# 初始化
eta = ETAEstimator(total_epochs=150, alpha=0.3)

# 每轮更新
eta.update(epoch_time)

# 获取 ETA 信息
info = eta.get_eta(current_epoch)
# 返回:
# {
#     'ema': 1.3,              # 指数移动平均
#     'eta_seconds': 175,      # 剩余秒数
#     'eta_str': '2m55s',     # 格式化字符串
#     'finish_time': '14:32', # 预计完成时间
#     'confidence': '±0.2s',  # 置信区间
#     'remaining_epochs': 135 # 剩余轮次
# }
```

---

## 6. 启动器 (launcher.py)

### build_command

构建训练命令。

```python
from launcher import build_command

cmd = build_command([
    '--variant', 'resnet18',
    '--epochs', '100',
    '--lr', '1e-4'
])
# 返回: 'py "run_train.py" --variant resnet18 --epochs 100 --lr 1e-4'
```

### export_team_configs

导出团队配置文件。

```python
from launcher import export_team_configs

export_team_configs('team_configs.txt')
```

### validate_args

验证命令行参数。

```python
from launcher import validate_args

valid, errors = validate_args(['--variant', 'resnet18', '--epochs', '100'])
```

---

## 7. 团队协作 (team_train.py)

### log_task_execution

记录任务执行日志。

```python
from team_train import log_task_execution

log_task_execution(
    task_id='BASELINE_1',
    status='completed',           # 'started', 'completed', 'failed', 'skipped'
    duration_seconds=43200,       # 可选：执行时长
    error=None                   # 可选：错误信息
)
# 写入 logs/team_training.log
```

### parse_task_id_from_filename

从文件名解析任务ID。

```python
from team_train import parse_task_id_from_filename

all_tasks = {'BASELINE_1': {}, 'OPT_GATED': {}}
task_id = parse_task_id_from_filename('checkpoint_BASELINE_1_best.pt', all_tasks)
# 返回: 'BASELINE_1' 或 None
```

### get_hardware_level

获取当前硬件等级。

```python
from team_train import get_hardware_level

level, gpu_mem = get_hardware_level()
# level: 'L1', 'L1+', 'L2', 'L3'
# gpu_mem: float, 显存大小(GB)
```

### load_tasks_from_files

从 tasks/ 目录加载 JSON 配置。

```python
from team_train import load_tasks_from_files

tasks = load_tasks_from_files()
# 返回: dict {task_id: task_config}
```

### topological_sort

按依赖拓扑排序。

```python
from team_train import topological_sort

tasks = {'T1': {'deps': []}, 'T2': {'deps': ['T1']}}
completed = set()
sorted_tasks = topological_sort(tasks, completed)
# 返回: ['T1', 'T2']
```

---

## 8. 配置模块 (utils.config)

```python
from utils.config import (
    load_config,
    save_config,
    get_data_root,
    get_checkpoints_dir,
    get_results_dir,
    get_data_batches,
    ensure_config
)

# 加载配置
config = load_config()

# 确保配置存在（不存在则创建）
ensure_config(interactive=True)

# 获取数据批次列表
batches = get_data_batches()
```

---

## 9. 控制台输出 (utils.console)

```python
from utils.console import (
    print_title,       # 白色加粗标题
    print_section,     # 分节标题
    print_result,      # 青色关键指标
    print_info,        # 灰色普通信息
    print_warning,     # 黄色警告
    print_error,       # 红色错误
    print_success,     # 绿色成功
    print_progress,    # 进度条
    print_header,      # 分隔线标题
    print_divider,     # 分隔线
    print_results_table, # 结果表格
    print_metric_row   # 指标行
)

print_title("开始训练...")
print_result("R2", 0.9399)
print_warning("显存不足")
print_error("训练失败")
print_success("训练完成")
```

---

## 10. 模型创建 (models.pe_tsnet_multimodal)

### create_model

统一模型创建接口。

```python
from models.pe_tsnet_multimodal import create_model

model = create_model(
    variant_key='resnet18',  # 变体名称
    config={'seq_len': 300, 'image_size': 512}
)
```

### get_arch_specific_config

获取架构特定配置。

```python
from models.pe_tsnet_multimodal import get_arch_specific_config

config = get_arch_specific_config('swin_yolo')
# 返回: dict，包含变体特定的默认参数
```
