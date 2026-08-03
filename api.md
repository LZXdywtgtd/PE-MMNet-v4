# API 参考文档

> PE-MMNet v4 公开接口

---

## 1. 数据模块 (data.dataset_multimodal)

### create_multibatch_dataloaders

创建多批次数据加载器。

```python
from data.dataset_multimodal import create_multibatch_dataloaders

train_loader, test_loader = create_multibatch_dataloaders(
    data_roots=None,           # list[str]: 数据目录（None自动扫描）
    batch_size=16,             # int: 批次大小
    image_size=512,            # int: 图像尺寸
    seq_len=300,              # int: 序列长度
    augment=True,             # bool: 数据增强
    predict_offset=0,          # int: 时间偏移
    task='detection',         # str: 'detection', 'segmentation', 'multitask'
    triple_channel=False       # bool: 三通道时序输入
)
```

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
    task='detection'
)
```

### 2.2 YOLO-FPN 变体 (models.pe_tsnet_yolo)

```python
from models.pe_tsnet_yolo import SwinYOLOFPN, ViTYOLOFPN

# Swin-YOLO-FPN
model = SwinYOLOFPN(
    grid_size=16,
    num_queries=256,
    dropout=0.1
)

# ViT-YOLO-FPN
model = ViTYOLOFPN(
    grid_size=16,
    num_queries=256,
    dropout=0.1
)
```

### 2.3 DETR 风格变体 (models.pe_tsnet_detr)

```python
from models.pe_tsnet_detr import DETRStyle

model = DETRStyle(
    num_queries=100,
    dropout=0.1
)
```

### 2.4 PatchTST 时序骨干 (models.pe_tsnet_patchtst)

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

# 增强版（处理三通道）
model = PatchTSTWithRate(
    seq_len=300,
    d_model=64,
    output_dim=64
)
```

### 2.5 门控融合模块 (models.pe_tsnet_fusion)

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

### 2.6 注意力模块 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import SEBlock, CoordAtt, BackboneWithAttention

# SE注意力
se = SEBlock(channels=64, reduction=16)
out = se(x)  # (B, C, H, W)

# 坐标注意力
coord = CoordAtt(inp=64, oup=64, reduction=32)
out = coord(x)

# 骨干包装器
model = BackboneWithAttention(
    backbone=resnet18,
    attention_type='coord',
    reduction=32
)
```

---

## 3. 损失函数 (training.mono_loss)

### MultimodalCrackLoss

检测任务损失函数。

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

YOLO检测损失。

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

DETR检测损失。

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

### DensityConsistencyLoss

密度一致性损失。

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

物理安全CutMix增强。

```python
from training.augmentation import ThermalCutMix, RandomNoise, RandomFlip

cutmix = ThermalCutMix(alpha=1.0, prob=0.5)
img_aug, _, labels_aug, _ = cutmix(img1, img2, labels1, labels2)

# 仅混合温度通道，应力通道保持不变
```

---

## 5. 训练脚本 (run_train.py)

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
#     'eta_str': '2m55s',      # 格式化字符串
#     'finish_time': '14:32', # 预计完成时间
#     'confidence': '±0.2s',  # 置信区间
#     'remaining_epochs': 135 # 剩余轮次
# }
```

### estimate_training_time

动态训练时间估算（训练前测量）。

```python
from run_train import estimate_training_time

avg_time, total_minutes = estimate_training_time(
    model, train_loader, test_loader,
    criterion, optimizer, device, config
)
# 返回: (avg_epoch_time, estimated_total_minutes)
# 自动保存/恢复模型状态
```

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
    checkpoint_path=None,  # 可选：保存路径
    task_id=None          # 可选：任务ID（团队协作）
)
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
    image_size=512
)
# 返回: {'r2': float, 'rmse': float, 'mIoU': float, ...}
```

### freeze_model_backbone

冻结骨干网络。

```python
from run_train import freeze_model_backbone

model = freeze_model_backbone(
    model,
    freeze_2d=True,    # 冻结2D骨干
    freeze_1d=False    # 保持1D骨干可训练
)
```

### staged_training

分阶段训练。

```python
from run_train import staged_training

model = staged_training(
    variant_key='resnet18',
    config=config,
    device=device,
    data_roots=data_roots,
    task_id=None       # 可选：任务ID
)
```

---

## 6. 启动器 (launcher.py)

### build_command

```python
from launcher import build_command

cmd = build_command(['--variant', 'resnet18', '--epochs', '100'])
```

### export_team_configs

导出团队配置文件。

```python
from launcher import export_team_configs

export_team_configs('team_configs.txt')
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

---

## 8. 配置模块 (utils.config)

```python
from utils.config import load_config, get_data_root, get_data_batches

config = load_config()
data_root = get_data_root()
batches = get_data_batches()
```

---

## 9. 控制台输出 (utils.console)

```python
from utils.console import (
    print_title, print_section, print_result,
    print_info, print_warning, print_error,
    print_success, print_progress, print_header
)

print_title("开始训练...")
print_result("R2", 0.9399)
```
