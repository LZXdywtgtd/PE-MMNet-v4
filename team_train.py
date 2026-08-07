#!/usr/bin/env python3
"""
PE-MMNet v4 团队协作训练系统

功能：
1. 配置驱动：任务由 JSON 配置文件定义
2. 依赖管理：任务间可定义前置依赖
3. 硬件感知：根据显存推荐任务等级
4. 状态可视化：已完成/可执行/警告/锁定 状态显示
5. 检查点导入：支持导入队友的 .pt 文件
6. 自动批量执行：按拓扑排序执行所有可执行任务
7. 任务日志：记录每个任务的执行情况

使用方法：
    python team_train.py              # 交互式菜单
    python team_train.py --auto      # 自动执行所有可执行任务
    python team_train.py --auto --force  # 自动执行（包括硬件警告任务）
    python team_train.py --import   # 导入队友检查点
"""

import os
import sys
import json
import shutil
import socket
import re
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
import torch

# =============================================================================
# 颜色支持（增强版）
# =============================================================================

def get_color_support():
    """检测终端颜色支持"""
    # 1. 检查 NO_COLOR 环境变量
    if os.environ.get('NO_COLOR'):
        return 'none'

    # 2. 检查 colorama
    try:
        import io
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        from colorama import init, Fore, Style
        init(autoreset=True, strip=False)
        return 'colorama'
    except ImportError:
        pass

    # 3. 检测 ANSI 支持（Windows 10+）
    if sys.platform == 'win32':
        try:
            version = sys.getwindowsversion()
            if version.major >= 10:
                return 'ansi'
        except:
            pass

    # 4. 检测 TERM
    term = os.environ.get('TERM', '')
    if 'xterm' in term or 'screen' in term or term == 'ANSI':
        return 'ansi'

    return 'none'

color_mode = get_color_support()
if color_mode == 'colorama':
    from colorama import Fore, Style
    COLORS = {
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'yellow': Fore.YELLOW,
        'gray': Fore.LIGHTBLACK_EX,
        'red': Fore.RED,
        'bold': Style.BRIGHT,
        'reset': Style.RESET_ALL,
    }
elif color_mode == 'ansi':
    COLORS = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'gray': '\033[90m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'reset': '\033[0m',
    }
else:
    # 无颜色模式
    COLORS = {k: '' for k in ['green', 'blue', 'yellow', 'gray', 'red', 'bold', 'reset']}

# 路径配置
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / 'tasks'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
LOG_DIR = SCRIPT_DIR / 'logs'
RUN_TRAIN = SCRIPT_DIR / 'run_train.py'

# 全局任务字典
all_tasks: Dict = {}

# =============================================================================
# 任务执行日志
# =============================================================================

def log_task_execution(task_id, status, duration_seconds=None, error=None):
    """记录任务执行日志

    Args:
        task_id: 任务ID
        status: 'started', 'completed', 'failed', 'skipped'
        duration_seconds: 执行时长（秒）
        error: 错误信息
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / 'team_training.log'

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'task_id': task_id,
        'status': status,
        'duration_seconds': duration_seconds,
        'error': str(error) if error else None,
        'hostname': socket.gethostname(),
    }

    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[警告] 写入日志失败: {e}")

# =============================================================================
# 内置默认任务
# =============================================================================

DEFAULT_TRAIN_TASKS = {
    # ========== A组：基线变体 ==========
    "A1": {
        "id": "A1",
        "name": "ResNet18基线",
        "desc": "轻量级CNN骨干，所有优化的基准线",
        "args": "--variant resnet18 --epochs 150 --patience 30",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~2GB",
        "deps": [],
    },
    "A2": {
        "id": "A2",
        "name": "Swin-YOLO基线",
        "desc": "推荐！Swin-Tiny骨干+YOLO网格回归，大幅提升空间定位",
        "args": "--variant swin_yolo --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1+",
        "time": "~18小时",
        "gpu_mem": "~4GB",
        "deps": [],
    },
    "A3": {
        "id": "A3",
        "name": "ViT-YOLO基线",
        "desc": "基于ViT-Small的YOLO变体，验证Transformer做骨干的效果",
        "args": "--variant vit_yolo --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~3GB",
        "deps": [],
    },
    "A4": {
        "id": "A4",
        "name": "DETR基线",
        "desc": "纯Transformer的DETR端到端检测，验证全局上下文感知",
        "args": "--variant detr --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1+",
        "time": "~18小时",
        "gpu_mem": "~5GB",
        "deps": [],
    },
    "A5": {
        "id": "A5",
        "name": "Swin-PatchTST",
        "desc": "升级1D时序骨干为PatchTST，提升热应力时序建模能力",
        "args": "--variant swin_yolo_patchtst --epochs 150 --patience 30 --lr 1e-4",
        "level": "L2",
        "time": "~20小时",
        "gpu_mem": "~4GB",
        "deps": [],
    },

    # ========== B组：单项优化 ==========
    "B1": {
        "id": "B1",
        "name": "门控融合优化",
        "desc": "基于ResNet18：温度/应力通道分治门控融合",
        "args": "--variant resnet18 --fusion gated --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B2": {
        "id": "B2",
        "name": "坐标注意力优化",
        "desc": "基于ResNet18：引入空间坐标注意力模块(CoordAtt)",
        "args": "--variant resnet18 --use_coord_attn --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B3": {
        "id": "B3",
        "name": "分阶段训练优化",
        "desc": "基于ResNet18：先短序列预训练，后长序列微调",
        "args": "--variant resnet18 --staged_train --epochs 150 --patience 30",
        "level": "L2+",
        "time": "~20小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B4": {
        "id": "B4",
        "name": "三通道输入优化",
        "desc": "基于ResNet18：增加初始温度、温度变化率通道",
        "args": "--variant resnet18 --triple_channel --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B5": {
        "id": "B5",
        "name": "ThermalCutMix增强",
        "desc": "基于ResNet18：引入物理安全的温度场CutMix数据增强",
        "args": "--variant resnet18 --aug_cutmix_prob 0.3 --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },

    # ========== C组：组合优化 ==========
    "C1": {
        "id": "C1",
        "name": "门控+坐标注意力",
        "desc": "组合优化：门控融合+坐标注意力",
        "args": "--variant resnet18 --fusion gated --use_coord_attn --epochs 150 --patience 30",
        "level": "L2",
        "time": "~18小时",
        "gpu_mem": "~2GB",
        "deps": ["B1", "B2"],
    },
    "C2": {
        "id": "C2",
        "name": "分阶段+坐标注意力",
        "desc": "组合优化：分阶段训练+坐标注意力",
        "args": "--variant resnet18 --staged_train --use_coord_attn --epochs 150 --patience 30",
        "level": "L2+",
        "time": "~24小时",
        "gpu_mem": "~2GB",
        "deps": ["B2", "B3"],
    },
    "C3": {
        "id": "C3",
        "name": "全组合优化",
        "desc": "终极组合：门控+坐标注意力+分阶段训练，最强性能验证",
        "args": "--variant resnet18 --fusion gated --use_coord_attn --staged_train --epochs 150 --patience 30",
        "level": "L3",
        "time": "~24小时",
        "gpu_mem": "~2GB",
        "deps": ["C1", "C2"],
    },
}

# =============================================================================
# 核心函数
# =============================================================================

def get_hardware_level():
    """检测 GPU 显存，返回硬件等级"""
    if not torch.cuda.is_available():
        return 'L1', 0.0

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_mem = total_mem - torch.cuda.memory_allocated() / 1e9

    if total_mem >= 8:
        return 'L3', total_mem
    elif total_mem >= 6:
        return 'L2', total_mem
    elif total_mem >= 4:
        return 'L1+', total_mem
    else:
        return 'L1', total_mem


def get_completed_tasks():
    """扫描 checkpoints/ 目录（含子目录），识别已完成的任务

    支持新分层目录结构（checkpoints/{variant}/best.pt）和旧格式。
    优先从检查点元数据中读取 task_id，兼容文件名解析作为备用。
    跳过 backup/ 目录。
    """
    completed = set()
    if not CHECKPOINT_DIR.exists():
        return completed

    for ckpt_file in CHECKPOINT_DIR.rglob('*_best.pt'):
        # 跳过 backup 目录
        if 'backup' in ckpt_file.parts:
            continue
        task_id = None

        # 方法1: 从元数据读取 task_id（优先）
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'task_id' in checkpoint:
                task_id = checkpoint['task_id']
        except Exception as e:
            pass

        # 方法2: 从文件名解析（增强版）
        if task_id is None:
            task_id = parse_task_id_from_filename(ckpt_file.name, all_tasks)

        if task_id and task_id in all_tasks:
            completed.add(task_id)

    return completed


def parse_task_id_from_filename(filename, all_tasks):
    """从文件名中解析 task_id（增强版）

    支持格式：
    - checkpoint_A1_best.pt           → A1
    - resnet18_A1_best.pt             → A1
    - resnet18_detection_off0_best.pt → 尝试匹配 A1 等任务ID
    - A1_best.pt                      → A1

    Args:
        filename: 检查点文件名
        all_tasks: 所有任务的字典

    Returns:
        task_id 或 None
    """
    stem = Path(filename).stem  # 去掉扩展名

    # 方法1: 使用正则表达式精确提取
    # 查找 TASK_ID 格式：字母+下划线+数字/字母组合
    patterns = [
        r'([A-Z][A-Z0-9]*_\d+)',      # BASELINE_1, OPT_GATED_1
        r'([A-Z]+)_\d+',               # BASELINE, OPT (后面跟数字)
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            candidate = match.group(1)
            if candidate in all_tasks:
                return candidate

    # 方法2: 子字符串匹配（作为备用）
    for task_id in all_tasks:
        # 使用单词边界匹配，避免误匹配
        if task_id in stem:
            return task_id

    return None


def check_deps_satisfied(task_id: str, completed: Set[str]) -> bool:
    """检查任务依赖是否满足"""
    task = all_tasks.get(task_id, {})
    deps = task.get('deps', [])
    return all(dep in completed for dep in deps)


def get_task_status(task_id: str, completed: Set[str], hardware_level: str) -> str:
    """获取任务状态"""
    if task_id in completed:
        return 'completed'

    if not check_deps_satisfied(task_id, completed):
        return 'locked'

    task = all_tasks.get(task_id, {})
    task_level = task.get('level', 'L1')

    levels = {'L1': 1, 'L1+': 2, 'L2': 3, 'L2+': 3, 'L3': 4}
    if levels.get(task_level, 0) > levels.get(hardware_level, 0):
        return 'warning'

    return 'executable'


def load_tasks_from_files():
    """从 tasks/ 目录加载 JSON 配置文件"""
    tasks = {}
    if not TASKS_DIR.exists():
        return tasks

    for json_file in TASKS_DIR.glob('*.json'):
        # 跳过模板参考文件
        if json_file.name in ('examples.json', 'example.json', 'template.json'):
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for task in data:
                    if 'id' in task:
                        tasks[task['id']] = task
            elif isinstance(data, dict) and 'tasks' in data:
                if 'id' in data:
                    tasks[data['id']] = data
        except Exception as e:
            print(f"{COLORS['yellow']}[警告]{COLORS['reset']} 加载 {json_file.name} 失败: {e}")

    return tasks


def merge_tasks(external: Dict, defaults: Dict) -> Dict:
    """合并外部配置和内置默认任务，外部优先"""
    merged = defaults.copy()
    merged.update(external)
    return merged


def topological_sort(tasks: Dict, completed: Set[str]) -> list:
    """按依赖拓扑排序返回可执行任务列表"""
    result = []
    remaining = {k: v for k, v in tasks.items() if k not in completed}

    while remaining:
        # 找出入度为0的任务
        executable = [
            k for k, v in remaining.items()
            if all(dep in completed or dep not in tasks for dep in v.get('deps', []))
        ]

        if not executable:
            break

        for task_id in executable:
            result.append(task_id)
            completed.add(task_id)
            del remaining[task_id]

    return result


def run_training_task(task_id: str) -> bool:
    """执行单个训练任务"""
    task = all_tasks.get(task_id)
    if not task:
        print(f"{COLORS['red']}[错误]{COLORS['reset']} 未知任务: {task_id}")
        return False

    print(f"\n{'=' * 60}")
    print(f"开始训练: {COLORS['bold']}{task['name']}{COLORS['reset']}")
    print(f"描述: {task['desc']}")
    print(f"预计时间: {task['time']}")
    print(f"任务ID: {task_id}")
    print(f"{'=' * 60}\n")

    # 添加 --task_id 参数以便检查点记录任务ID
    cmd = f'py "{RUN_TRAIN}" --mode train {task["args"]} --task_id {task_id}'
    print(f"执行命令: {cmd}\n")

    result = os.system(cmd)

    if result == 0:
        print(f"\n{COLORS['green']}[完成]{COLORS['reset']} {task['name']} 训练完成!")
        return True
    else:
        print(f"\n{COLORS['red']}[失败]{COLORS['reset']} {task['name']} 训练失败 (错误码: {result})")
        return False


def _get_task_checkpoint_info(task_id: str) -> dict:
    """获取任务的检查点信息（epoch、is_complete）"""
    task = all_tasks.get(task_id, {})
    args_str = task.get('args', '')
    variant = _extract_arg(args_str, '--variant') or 'resnet18'
    task_mode = _extract_arg(args_str, '--task') or 'detection'
    offset = _extract_arg(args_str, '--predict_offset') or '0'
    subdir = CHECKPOINT_DIR / variant
    ckpt_path = subdir / f"{variant}_{task_mode}_off{offset}_best.pt"
    last_path = subdir / f"{variant}_{task_mode}_off{offset}_last.pt"

    for path in [ckpt_path, last_path]:
        if path.exists():
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                return {
                    'epoch': ckpt.get('epoch', 0),
                    'is_complete': ckpt.get('is_complete', False),
                    'path': path.name
                }
            except Exception:
                pass
    return {'epoch': 0, 'is_complete': False, 'path': None}


def _extract_arg(args_str: str, arg: str) -> str:
    """从 args 字符串中提取指定参数的值"""
    import re
    pattern = re.compile(re.escape(arg) + r'\s+(\S+)')
    match = pattern.search(args_str)
    return match.group(1) if match else None


def _print_execution_plan():
    """打印执行计划（依赖树 + 检查点状态）"""
    completed = get_completed_tasks()
    hardware_level, _ = get_hardware_level()

    print(f"\n{'=' * 60}")
    print(f"  执行计划预览")
    print(f"{'=' * 60}")

    # 按依赖深度分层显示
    def get_depth(task_id, visited=None):
        if visited is None:
            visited = set()
        if task_id in visited:
            return 0
        visited.add(task_id)
        task = all_tasks.get(task_id, {})
        deps = task.get('deps', [])
        if not deps:
            return 0
        return 1 + max((get_depth(d, visited) for d in deps), default=0)

    # 收集所有任务的深度和状态
    task_info = {}
    for task_id in all_tasks:
        status = get_task_status(task_id, completed, hardware_level)
        ckpt_info = _get_task_checkpoint_info(task_id)
        depth = get_depth(task_id)
        task_info[task_id] = {
            'status': status,
            'depth': depth,
            'epoch': ckpt_info['epoch'],
            'is_complete': ckpt_info['is_complete'],
        }

    # 打印每个深度的任务
    for depth in range(max(t['depth'] for t in task_info.values()) + 1):
        tasks_at_depth = [(tid, info) for tid, info in task_info.items() if info['depth'] == depth]
        tasks_at_depth.sort(key=lambda x: x[0])
        indent = "  " * depth
        for task_id, info in tasks_at_depth:
            task = all_tasks[task_id]
            deps_str = f" ← {', '.join(task.get('deps', []))}" if task.get('deps') else ""

            # 状态图标
            if info['is_complete']:
                icon = f"{COLORS['green']}✓已完成{COLORS['reset']}"
            elif info['status'] == 'locked':
                icon = f"{COLORS['gray']}🔒锁定{COLORS['reset']}"
            elif info['status'] == 'warning':
                icon = f"{COLORS['yellow']}!显存警告{COLORS['reset']}"
            elif info['status'] == 'executable':
                icon = f"{COLORS['cyan']}▶待执行{COLORS['reset']}"
            else:
                icon = f"{COLORS['gray']}?未知{COLORS['reset']}"

            # 检查点信息
            if info['is_complete']:
                ckpt_str = f"Epoch {info['epoch']}/{task.get('args', '')}"
            elif info['epoch'] > 0:
                ckpt_str = f"{COLORS['yellow']}中断于 Epoch {info['epoch']}{COLORS['reset']}"
            else:
                ckpt_str = "新任务"

            print(f"{indent}├── [{task_id}] {task['name']} {icon}")
            print(f"{indent}│   {ckpt_str}{deps_str}")

    print(f"{'=' * 60}\n")


def auto_run_executable(force_warnings=False):
    """自动执行所有可执行任务（动态拓扑重排）

    每次任务完成后重新扫描依赖状态，新解锁的任务会立即加入执行队列。
    优先恢复被中断的任务（is_complete=False）。"""

    hardware_level, _ = get_hardware_level()
    total_tasks = len(all_tasks)

    # 启动前打印执行计划
    _print_execution_plan()

    # 动态扫描：每次执行后重新获取已完成状态
    def get_next_task():
        """获取下一个可执行且未被执行过的任务（优先恢复被中断的）"""
        completed = get_completed_tasks()
        interrupted = []  # 被中断的任务（优先）
        executable = []  # 正常可执行

        for task_id in all_tasks:
            status = get_task_status(task_id, completed, hardware_level)
            if status not in ('executable', 'warning'):
                continue
            ckpt_info = _get_task_checkpoint_info(task_id)
            if ckpt_info['epoch'] > 0 and not ckpt_info['is_complete']:
                interrupted.append(task_id)  # 被中断，优先恢复
            else:
                executable.append(task_id)

        # 优先返回被中断的任务，其次按任务ID排序
        if interrupted:
            return sorted(interrupted)[0], 'executable'
        if executable:
            return sorted(executable)[0], 'executable'
        return None, None

    # 统计
    success_count = 0
    fail_count = 0
    skip_count = 0
    started_tasks = set()  # 记录已启动过的任务（避免重复）

    # 首次扫描：检查是否有任何可执行任务
    first_task, _ = get_next_task()
    if first_task is None:
        completed = get_completed_tasks()
        print(f"\n{COLORS['yellow']}[信息]{COLORS['reset']} 没有可执行的任务 (已完成: {len(completed)}/{total_tasks})")
        return

    mode_str = "强制" if force_warnings else "标准"
    print(f"\n{COLORS['bold']}自动执行模式 ({mode_str}){COLORS['reset']}")
    print(f"总任务: {total_tasks} | 实时检测依赖满足情况\n")

    task_idx = 0
    while True:
        task_id, status = get_next_task()
        if task_id is None:
            break  # 无更多可执行任务

        task = all_tasks[task_id]
        task_idx += 1
        started_tasks.add(task_id)

        # 警告任务处理
        if status == 'warning':
            if not force_warnings:
                print(f"\n{COLORS['yellow']}[警告]{COLORS['reset']} 任务 {task['name']} 需要更多显存")
                confirm = input("是否继续? (y/n): ").strip().lower()
                if confirm != 'y':
                    print(f"跳过 {task['name']}")
                    log_task_execution(task_id, 'skipped', error='用户取消')
                    skip_count += 1
                    continue
            else:
                print(f"\n{COLORS['yellow']}[强制执行]{COLORS['reset']} {task['name']} (显存可能不足)")

        print(f"\n[{task_idx}/{total_tasks}] 执行: {task['name']}")

        # 记录任务开始
        log_task_execution(task_id, 'started')
        task_start = datetime.now()

        if run_training_task(task_id):
            success_count += 1
            duration = (datetime.now() - task_start).total_seconds()
            log_task_execution(task_id, 'completed', duration_seconds=duration)
        else:
            fail_count += 1
            duration = (datetime.now() - task_start).total_seconds()
            log_task_execution(task_id, 'failed', duration_seconds=duration)
            if not force_warnings:
                retry = input("\n训练失败，是否继续下一个任务? (y/n): ").strip().lower()
                if retry != 'y':
                    break

    print(f"\n{'=' * 60}")
    print(f"自动执行完成: {COLORS['green']}{success_count} 成功{COLORS['reset']}, "
          f"{COLORS['red']}{fail_count} 失败{COLORS['reset']}, "
          f"{COLORS['gray']}{skip_count} 跳过{COLORS['reset']}")
    print(f"{'=' * 60}")


def import_checkpoint():
    """导入队友的检查点"""
    print(f"\n{COLORS['bold']}导入检查点{COLORS['reset']}")
    print(f"检查点目录: {CHECKPOINT_DIR}")

    source_dir = input("\n请输入队友检查点所在的文件夹路径 (回车取消): ").strip()
    if not source_dir:
        return

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"{COLORS['red']}[错误]{COLORS['reset']} 目录不存在")
        return

    imported = 0
    skipped = 0
    for pt_file in source_path.glob('*.pt'):
        dest_file = CHECKPOINT_DIR / pt_file.name
        if dest_file.exists():
            print(f"跳过 (已存在): {pt_file.name}")
            skipped += 1
            continue

        shutil.copy2(pt_file, dest_file)
        print(f"{COLORS['green']}[导入]{COLORS['reset']} {pt_file.name}")
        imported += 1

    print(f"\n{COLORS['green']}[完成]{COLORS['reset']} 导入了 {imported} 个检查点"
          + (f"，跳过 {skipped} 个已存在" if skipped > 0 else ""))


def print_menu(completed: Set[str], hardware_level: str):
    """显示任务菜单"""
    print(f"\n{COLORS['bold']}{'=' * 60}")
    print("  PE-MMNet v4 团队协作训练")
    print(f"{'=' * 60}{COLORS['reset']}")

    _, gpu_mem = get_hardware_level()
    print(f"\n当前硬件等级: {COLORS['blue']}{hardware_level}{COLORS['reset']} (显存: {gpu_mem:.1f}GB)")
    print(f"已完成任务: {len(completed)}/{len(all_tasks)}")

    print(f"\n状态标记: {COLORS['green']}[OK] 已完成{COLORS['reset']} | "
          f"{COLORS['blue']}[>] 可执行{COLORS['reset']} | "
          f"{COLORS['yellow']}[!] 硬件警告{COLORS['reset']} | "
          f"{COLORS['gray']}[X] 依赖未完成{COLORS['reset']}")

    print()

    # 按等级分组显示
    levels = {'L1': [], 'L1+': [], 'L2': [], 'L2+': [], 'L3': []}
    for task_id in all_tasks:
        task = all_tasks[task_id]
        level = task.get('level', 'L1')
        if level in levels:
            levels[level].append(task_id)

    idx = 1
    task_index = {}
    for level in ['L1', 'L1+', 'L2', 'L2+', 'L3']:
        if not levels[level]:
            continue
        print(f"  【{level}级别】")

        for task_id in levels[level]:
            task = all_tasks[task_id]
            status = get_task_status(task_id, completed, hardware_level)

            status_symbols = {
                'completed': (f"{COLORS['green']}OK{COLORS['reset']}", COLORS['green']),
                'executable': (f"{COLORS['blue']}>{COLORS['reset']}", COLORS['blue']),
                'warning': (f"{COLORS['yellow']}!{COLORS['reset']}", COLORS['yellow']),
                'locked': (f"{COLORS['gray']}X{COLORS['reset']}", COLORS['gray']),
            }
            prefix, color = status_symbols[status]

            print(f"  [{idx:>2}] {prefix} {color}{task['name']}{COLORS['reset']}")
            print(f"       {task['desc']}")
            # 显示等级、时间和显存（如果有）
            level = task.get('level', 'N/A')
            time_str = task.get('time', 'N/A')
            gpu_mem = task.get('gpu_mem', 'N/A')
            print(f"       [L] {level} | [T] {time_str} | [G] {gpu_mem}")

            if status == 'locked':
                deps = task.get('deps', [])
                print(f"       {COLORS['gray']}依赖: {', '.join(deps)}{COLORS['reset']}")

            if status == 'warning':
                print(f"       {COLORS['yellow']}⚠️ 建议显存 ≥ {task['gpu_mem']}{COLORS['reset']}")

            task_index[idx] = task_id
            idx += 1

        print()

    print("-" * 60)
    print(f"  [a]   自动执行所有可执行任务")
    print(f"  [i]   导入队友检查点")
    print(f"  [q]   退出")
    print("-" * 60)

    return task_index


def main():
    global all_tasks

    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='PE-MMNet v4 团队协作训练系统')
    parser.add_argument('--auto', action='store_true', help='自动执行所有可执行任务')
    parser.add_argument('--force', action='store_true', help='强制执行硬件警告任务（auto模式）')
    parser.add_argument('--import', dest='import_mode', action='store_true', help='导入队友检查点')
    args = parser.parse_args()

    # 加载任务配置
    external_tasks = load_tasks_from_files()
    all_tasks = merge_tasks(external_tasks, DEFAULT_TRAIN_TASKS)

    # 检测硬件
    hardware_level, _ = get_hardware_level()

    # 显示配置来源
    if external_tasks:
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 已加载 {len(external_tasks)} 个外部任务")
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 配置来源: {TASKS_DIR}")
    else:
        print(f"{COLORS['yellow']}[信息]{COLORS['reset']} 使用内置默认任务列表")

    # 命令行模式处理
    if args.auto:
        auto_run_executable(force_warnings=args.force)
        return

    if args.import_mode:
        import_checkpoint()
        return

    # 交互式菜单模式
    while True:
        completed = get_completed_tasks()
        task_index = print_menu(completed, hardware_level)

        choice = input("\n请输入选项: ").strip().lower()

        if choice == 'q':
            print("\n已退出")
            break

        if choice == 'a':
            auto_run_executable()
            continue

        if choice == 'i':
            import_checkpoint()
            continue

        # 数字选择
        if choice.isdigit():
            idx = int(choice)
            if idx in task_index:
                task_id = task_index[idx]
                status = get_task_status(task_id, completed, hardware_level)

                if status == 'completed':
                    print(f"\n{COLORS['yellow']}[跳过]{COLORS['reset']} "
                          f"{all_tasks[task_id]['name']} 已完成")
                    continue

                if status == 'locked':
                    deps = all_tasks[task_id].get('deps', [])
                    print(f"\n{COLORS['red']}[锁定]{COLORS['reset']} "
                          f"请先完成以下依赖任务: {', '.join(deps)}")
                    continue

                if status == 'warning':
                    print(f"\n{COLORS['yellow']}[警告]{COLORS['reset']} "
                          f"{all_tasks[task_id]['name']} 可能需要更多显存")
                    confirm = input("是否继续? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                run_training_task(task_id)
            else:
                print(f"\n{COLORS['red']}[错误]{COLORS['reset']} 无效选项")
        else:
            print(f"\n{COLORS['red']}[错误]{COLORS['reset']} 无效选项")


if __name__ == '__main__':
    main()
