# API 参考文档

> PE-MMNet v4 公开接口

---

## 1. 配置模块 (utils.config)

### load_config

加载配置文件。

```python
from utils.config import load_config

config = load_config()
# 返回: dict 或 None
# {
#     "data_root": "D:\\path\\to\\data",
#     "checkpoints_dir": "./checkpoints",
#     "results_dir": "./benchmark_results"
# }
```

### get_data_root

获取数据根目录路径。

```python
from utils.config import get_data_root

data_root = get_data_root()
# 返回: str 或 None
```

### get_data_batches

获取已配置的数据批次列表。

```python
from utils.config import get_data_batches

batches = get_data_batches()
# 返回: list[str] 或 None
# 示例: ["参数化扫描1", "参数化扫描2"]
```

### ensure_config

确保配置文件存在。

```python
from utils.config import ensure_config

success = ensure_config(interactive=True)
# interactive=True: 交互式提示用户输入
# interactive=False: 仅检查
```

---

## 2. 数据模块 (data.dataset_multimodal)

### create_multibatch_dataloaders

创建多批次数据加载器。

```python
from data.dataset_multimodal import create_multibatch_dataloaders

train_loader, test_loader = create_multibatch_dataloaders(
    data_roots=data_roots,    # list[str]: 数据目录列表
    batch_size=16,            # int: 批次大小
    image_size=512,           # int: 图像尺寸
    augment=True,             # bool: 数据增强
    predict_offset=0,         # int: 时间偏移
    seq_len=300,              # int: 序列长度
    seq_interp_mode='interpolate',  # str: 'interpolate' 或 'pool'
    task='detection'          # str: 'detection', 'segmentation', 'multitask'
)
```

**示例**：

```python
from data.dataset_multimodal import create_multibatch_dataloaders
from utils.config import get_data_root, get_data_batches
import os

data_root = get_data_root()
batches = get_data_batches()
data_roots = [os.path.join(data_root, b) for b in batches]

train_loader, test_loader = create_multibatch_dataloaders(
    data_roots=data_roots,
    batch_size=16,
    augment=True
)
```

---

## 3. 模型模块 (models.pe_tsnet_multimodal)

### PETSNetMultimodal

完整多模态融合网络。

```python
from models.pe_tsnet_multimodal import PETSNetMultimodal

model = PETSNetMultimodal(
    seq_len=300,              # 1D 序列长度
    image_channels=2,         # 图像通道数（温度场+应力场）
    image_size=512,           # 图像尺寸
    pretrained_2d=True,       # 使用 ImageNet 预训练权重
    task='detection'          # 'detection', 'segmentation', 'multitask'
)
```

### 预定义模型变体

```python
from models.pe_tsnet_multimodal import (
    Model1DOnly,
    Model2DOnly,
    ModelConcat,
    ModelAdd,
    ModelCrossAttn,
    VARIANT_MODELS
)

VARIANT_MODELS = {
    '1d_only': Model1DOnly,
    '2d_only': Model2DOnly,
    'concat': ModelConcat,
    'add': ModelAdd,
    'cross_attn': ModelCrossAttn,
    'full': PETSNetMultimodal,
}
```

---

## 4. 损失函数 (training.mono_loss)

### MultimodalCrackLoss

检测任务损失函数。

```python
from training.mono_loss import MultimodalCrackLoss

criterion = MultimodalCrackLoss(
    lambda_mse_density=1.0,   # MSE 损失权重
    lambda_mono=0.1,          # 单调性损失权重
    lambda_loc=1.0,           # 定位损失权重
    lambda_conf=1.0           # 置信度损失权重
)

loss, loss_dict = criterion(pred, target)
# loss: 总损失
# loss_dict: {'mse': x, 'mono': x, 'loc': x, 'conf': x}
```

### SegmentationLoss

分割任务损失函数。

```python
from training.mono_loss import SegmentationLoss

criterion = SegmentationLoss()
loss = criterion(pred_mask, target_mask)
```

### MultimodalSegmentationLoss

多任务损失函数。

```python
from training.mono_loss import MultimodalSegmentationLoss

criterion = MultimodalSegmentationLoss(
    lambda_seg=1.0,           # 分割损失权重
    lambda_det=0.5            # 检测损失权重
)

loss = criterion(outputs, labels)
# outputs: (mask, detection) 元组
# labels: (mask, detection) 元组
```

---

## 5. 训练脚本 (run_train.py)

### train_model

训练单个模型。

```python
from run_train import train_model

model, metrics = train_model(
    model,                    # nn.Module: 模型
    train_loader,             # DataLoader: 训练数据
    test_loader,              # DataLoader: 测试数据
    config,                   # dict: 配置参数
    device,                   # torch.device: 计算设备
    checkpoint_path=None      # str: 检查点保存路径
)

# 返回: (训练后的模型, 评估指标字典)
```

### evaluate_model

评估模型。

```python
from run_train import evaluate_model

metrics = evaluate_model(
    model,                    # nn.Module: 模型
    device,                   # torch.device: 计算设备
    data_roots=None,          # list[str]: 数据目录
    task='detection',         # str: 任务类型
    image_size=512            # int: 图像尺寸
)

# 返回: {
#     'r2': float,
#     'rmse': float,
#     'mae': float,
#     'mIoU': float,
#     'violation_rate': float,
#     'dice': float
# }
```

### eval_checkpoint

从检查点评估。

```python
from run_train import eval_checkpoint

metrics = eval_checkpoint(
    checkpoint_path,          # str: 检查点路径
    device,                   # torch.device: 计算设备
    image_size=None           # int: 图像尺寸（从检查点读取）
)
```

### auto_select_config

显存自适应配置。

```python
from run_train import auto_select_config

config = auto_select_config(args)
# args: argparse 参数对象
# 返回: {'image_size': 512, 'batch_size': 8, 'fp16': True}
```

---

## 6. 启动器 (launcher.py)

### build_command

构建训练命令。

```python
from launcher import build_command

cmd = build_command(['--backbone_2d', 'resnet18', '--epochs', '100'])
# 返回: "python run_train.py --mode train --backbone_2d resnet18 --epochs 100"
```

### interactive_mode

交互式配置。

```python
from launcher import interactive_mode

args_list = interactive_mode()
# 返回: ['--backbone_2d', 'resnet18', '--epochs', '100', ...]
```

### queue_config_wizard

队列配置向导。

```python
from launcher import queue_config_wizard

result = queue_config_wizard(cli)
# 返回: "start_training" 或 None
```

### load_command_config

加载保存的配置。

```python
from launcher import load_command_config

config_list = load_command_config('config_launcher.json')
# 返回: [['--backbone_2d', 'resnet18'], ['--backbone_2d', 'vit_small']]
```

---

## 7. 批量训练 (tools.batch_train_gui)

### TrainingCommand

训练命令封装。

```python
from tools.batch_train_gui import TrainingCommand

cmd = TrainingCommand(
    id='T001',
    variant='full',
    backbone_2d='resnet18',
    backbone_1d='cnn_attn',
    fusion='cross_attn',
    epochs=100,
    predict_offset=0,
    task='detection'
)

# 方法
cmd.to_args()              # 转换为命令行参数列表
cmd.to_command_string()    # 转换为命令字符串
cmd.get_checkpoint_path()  # 获取检查点路径
cmd.get_display_name()     # 获取显示名称

# 快捷创建
cmd = TrainingCommand.from_quick_format('resnet18,100,0,detection')
```

### BatchTrainer

批量训练管理器。

```python
from tools.batch_train_gui import BatchTrainer

trainer = BatchTrainer()
trainer.add_command(cmd)           # 添加命令
trainer.remove_command(cmd_id)     # 删除命令
trainer.execute_all()              # 执行所有命令
trainer.clear_commands()           # 清空命令
```

### InteractiveCLI

交互式界面。

```python
from tools.batch_train_gui import InteractiveCLI

cli = InteractiveCLI()
cli.run()  # 启动交互式界面
```

---

## 8. 控制台输出 (utils.console)

```python
from utils.console import (
    print_title,    # 章节标题
    print_section,  # 小节标题
    print_result,   # 关键指标（青色高亮）
    print_info,     # 普通信息（灰色）
    print_warning,  # 警告信息（黄色）
    print_error,    # 错误信息（红色）
    print_success,  # 成功信息（绿色）
    print_progress, # 进度信息
    print_header,   # 主标题
    print_divider   # 分隔线
)

print_title("开始训练...")
print_result("R2", 0.9399)
print_info("训练样本: 2249")
```
