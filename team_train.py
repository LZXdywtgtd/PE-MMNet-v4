"""
PE-MMNet v4 团队协作训练脚本

使用方法：
    python team_train.py

团队成员只需运行此脚本，选择对应的训练任务即可。
无需了解复杂的命令行参数！
"""

# =============================================================================
# 团队协作训练配置
# =============================================================================

# 训练任务配置：key -> (显示名称, 命令参数)
TRAIN_TASKS = {
    "1": {
        "name": "快速验证 (30分钟)",
        "desc": "验证代码能跑通，50轮训练",
        "args": "--variant full --epochs 50 --batch_size 8",
        "time": "~30分钟"
    },
    "2": {
        "name": "Swin-YOLO基线 (1小时)",
        "desc": "推荐！YOLO网格回归，适合空间定位",
        "args": "--variant swin_yolo_fpn --epochs 100 --lr 1e-4",
        "time": "~1小时"
    },
    "3": {
        "name": "ViT-YOLO基线 (1小时)",
        "desc": "ViT风格，强大注意力机制",
        "args": "--variant vit_yolo_fpn --epochs 100 --lr 1e-4",
        "time": "~1小时"
    },
    "4": {
        "name": "DETR风格 (1小时)",
        "desc": "Transformer风格，全局上下文感知",
        "args": "--variant detr_style --epochs 100 --lr 1e-4",
        "time": "~1小时"
    },
    "5": {
        "name": "PatchTST时序 (1小时)",
        "desc": "分块Transformer，适合时序预测",
        "args": "--variant swin_yolo_patchtst --epochs 100 --lr 1e-4",
        "time": "~1小时"
    },
    "6": {
        "name": "门控融合优化 (1小时)",
        "desc": "温度/应力分治策略",
        "args": "--variant full --fusion gated --epochs 100",
        "time": "~1小时"
    },
    "7": {
        "name": "坐标注意力优化 (1小时)",
        "desc": "保留位置信息的注意力机制",
        "args": "--variant full --use_coord_attn --epochs 100",
        "time": "~1小时"
    },
    "8": {
        "name": "分阶段训练 (1.5小时)",
        "desc": "先短序列预训练，再长序列微调",
        "args": "--variant full --staged_train --epochs 100",
        "time": "~1.5小时"
    },
    "9": {
        "name": "三通道输入 (1小时)",
        "desc": "初始温度+当前温度+温度变化率",
        "args": "--variant full --triple_channel --epochs 100",
        "time": "~1小时"
    },
    "10": {
        "name": "完整训练 (2小时)",
        "desc": "最终交付模型，200轮训练",
        "args": "--variant full --epochs 200 --batch_size 16",
        "time": "~2小时"
    },
}


def run_training_task(task_id):
    """执行指定的训练任务"""
    import subprocess
    import sys
    import os

    task = TRAIN_TASKS.get(task_id)
    if not task:
        print(f"[错误] 未找到任务 {task_id}")
        return False

    print("\n" + "=" * 60)
    print(f"开始训练: {task['name']}")
    print(f"描述: {task['desc']}")
    print(f"预计时间: {task['time']}")
    print("=" * 60)

    cmd = f"py run_train.py --mode train {task['args']}"
    print(f"\n执行命令: {cmd}\n")

    # 执行训练
    result = os.system(cmd)

    if result == 0:
        print("\n" + "=" * 60)
        print(f"[完成] {task['name']} 训练完成!")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print(f"[失败] {task['name']} 训练失败 (错误码: {result})")
        print("=" * 60)
        return False


def main():
    print("\n" + "=" * 60)
    print("  PE-MMNet v4 团队协作训练")
    print("=" * 60)
    print("\n选择训练任务:\n")

    for task_id, task in TRAIN_TASKS.items():
        print(f"  [{task_id:>2}] {task['name']}")
        print(f"      {task['desc']} ({task['time']})")
        print()

    print("  [a]   运行全部 YOLO 系列 (b1+b2+b3)")
    print("  [q]   退出")

    choice = input("\n请输入选项: ").strip().lower()

    if choice == 'q':
        print("\n已退出")
        return

    if choice == 'a':
        # 运行全部 YOLO 系列
        yolo_tasks = ['2', '3', '4', '5']
        for task_id in yolo_tasks:
            print(f"\n\n{'#' * 60}")
            print(f"# 任务 {task_id}/4: {TRAIN_TASKS[task_id]['name']}")
            print(f"{'#' * 60}")
            success = run_training_task(task_id)
            if not success:
                retry = input("训练失败，是否继续下一个任务? (y/n): ").strip().lower()
                if retry != 'y':
                    break
        print("\n全部训练完成!")
        return

    if choice in TRAIN_TASKS:
        run_training_task(choice)
    else:
        print(f"\n[错误] 无效选项: {choice}")


if __name__ == "__main__":
    main()
