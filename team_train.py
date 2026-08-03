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

使用方法：
    python team_train.py              # 交互式菜单
    python team_train.py --auto      # 自动执行所有可执行任务
    python team_train.py --import   # 导入队友检查点
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Set, Optional
import torch

# 颜色支持
# 注意：在 Windows 下使用 colorama 时需要处理 Unicode 编码问题
try:
    import sys
    import io
    # 设置标准输出为 UTF-8 模式
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    from colorama import init, Fore, Style
    init(autoreset=True, strip=False)
    COLORS = {
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'yellow': Fore.YELLOW,
        'gray': Fore.LIGHTBLACK_EX,
        'red': Fore.RED,
        'bold': Style.BRIGHT,
        'reset': Style.RESET_ALL,
    }
except ImportError:
    # 回退：使用 ANSI 转义码
    COLORS = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'gray': '\033[90m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'reset': '\033[0m',
    }

# 路径配置
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / 'tasks'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
RUN_TRAIN = SCRIPT_DIR / 'run_train.py'

# 全局任务字典
all_tasks: Dict = {}

# =============================================================================
# 内置默认任务
# =============================================================================

DEFAULT_TRAIN_TASKS = {
    "BASELINE_1": {
        "id": "BASELINE_1",
        "name": "ResNet18基线",
        "desc": "轻量骨干，适合快速验证",
        "args": "--variant resnet18 --epochs 150",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~2GB",
        "deps": [],
    },
    "BASELINE_2": {
        "id": "BASELINE_2",
        "name": "Swin-YOLO基线",
        "desc": "推荐！网格回归，适合空间定位",
        "args": "--variant swin_yolo --epochs 150 --lr 1e-4",
        "level": "L1+",
        "time": "~12-13小时",
        "gpu_mem": "~4GB",
        "deps": [],
    },
    "BASELINE_3": {
        "id": "BASELINE_3",
        "name": "ViT-YOLO基线",
        "desc": "ViT风格注意力",
        "args": "--variant vit_yolo --epochs 150 --lr 1e-4",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~3GB",
        "deps": [],
    },
    "BASELINE_4": {
        "id": "BASELINE_4",
        "name": "DETR基线",
        "desc": "Transformer风格，全局上下文感知",
        "args": "--variant detr --epochs 150 --lr 1e-4",
        "level": "L1+",
        "time": "~13-14小时",
        "gpu_mem": "~5GB",
        "deps": [],
    },
    "OPT_GATED": {
        "id": "OPT_GATED",
        "name": "门控融合优化",
        "desc": "基于ResNet18测试门控融合",
        "args": "--variant resnet18 --fusion gated --epochs 150",
        "level": "L2",
        "time": "~14小时",
        "gpu_mem": "~2GB",
        "deps": ["BASELINE_1"],
    },
    "OPT_COORD": {
        "id": "OPT_COORD",
        "name": "坐标注意力优化",
        "desc": "基于Swin-YOLO测试坐标注意力",
        "args": "--variant swin_yolo --use_coord_attn --epochs 150",
        "level": "L2",
        "time": "~14小时",
        "gpu_mem": "~4GB",
        "deps": ["BASELINE_2"],
    },
    "OPT_STAGED": {
        "id": "OPT_STAGED",
        "name": "分阶段训练",
        "desc": "先短序列预训练，再长序列微调",
        "args": "--variant resnet18 --staged_train --epochs 150",
        "level": "L2+",
        "time": "~16小时",
        "gpu_mem": "~2GB",
        "deps": ["BASELINE_1"],
    },
    "OPT_FULL": {
        "id": "OPT_FULL",
        "name": "Swin-YOLO完整优化",
        "desc": "最终交付模型，300轮+坐标注意力",
        "args": "--variant swin_yolo --use_coord_attn --epochs 300 --batch_size 16 --image_size 384",
        "level": "L3",
        "time": "~24小时",
        "gpu_mem": "~6GB",
        "deps": ["OPT_COORD"],
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
    """扫描 checkpoints/ 目录，识别已完成的任务

    优先从检查点元数据中读取 task_id，兼容文件名解析作为备用
    """
    completed = set()
    if not CHECKPOINT_DIR.exists():
        return completed

    for ckpt_file in CHECKPOINT_DIR.glob('*_best.pt'):
        task_id = None

        # 方法1: 从元数据读取 task_id（优先）
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'task_id' in checkpoint:
                task_id = checkpoint['task_id']
        except:
            pass

        # 方法2: 从文件名解析（备用）
        if task_id is None:
            filename = ckpt_file.stem
            for task_id_candidate in all_tasks.keys():
                if task_id_candidate in filename:
                    task_id = task_id_candidate
                    break

        if task_id and task_id in all_tasks:
            completed.add(task_id)

    return completed


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
    print(f"{'=' * 60}\n")

    cmd = f'py "{RUN_TRAIN}" --mode train {task["args"]}'
    print(f"执行命令: {cmd}\n")

    result = os.system(cmd)

    if result == 0:
        print(f"\n{COLORS['green']}[完成]{COLORS['reset']} {task['name']} 训练完成!")
        return True
    else:
        print(f"\n{COLORS['red']}[失败]{COLORS['reset']} {task['name']} 训练失败 (错误码: {result})")
        return False


def auto_run_executable():
    """自动执行所有可执行任务"""
    completed = get_completed_tasks()
    hardware_level, _ = get_hardware_level()

    # 获取可执行任务列表（按拓扑排序）
    executable = []
    for task_id in all_tasks:
        status = get_task_status(task_id, completed, hardware_level)
        if status in ('executable', 'warning'):
            executable.append((task_id, status))

    if not executable:
        print(f"\n{COLORS['yellow']}[信息]{COLORS['reset']} 没有可执行的任务")
        return

    print(f"\n{COLORS['bold']}自动执行模式{COLORS['reset']}")
    print(f"找到 {len(executable)} 个可执行任务\n")

    success_count = 0
    fail_count = 0

    for i, (task_id, status) in enumerate(executable, 1):
        task = all_tasks[task_id]

        # 警告任务需要确认
        if status == 'warning':
            print(f"\n{COLORS['yellow']}[警告]{COLORS['reset']} 任务 {task['name']} 需要更多显存")
            confirm = input("是否继续? (y/n): ").strip().lower()
            if confirm != 'y':
                print(f"跳过 {task['name']}")
                continue

        print(f"\n[{i}/{len(executable)}] 执行: {task['name']}")

        if run_training_task(task_id):
            success_count += 1
        else:
            fail_count += 1
            retry = input("\n训练失败，是否继续下一个任务? (y/n): ").strip().lower()
            if retry != 'y':
                break

    print(f"\n{'=' * 60}")
    print(f"自动执行完成: {COLORS['green']}{success_count} 成功{COLORS['reset']}, "
          f"{COLORS['red']}{fail_count} 失败{COLORS['reset']}")
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
