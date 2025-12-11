import argparse
import sys

from trainer.pretrainTrainer import PretrainTrainer
from trainer.sftTrainer import SFTTrainer
from trainer.dpoTrainer import DPOTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DeepSpeed训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        choices=['pretrain', 'sft', 'dpo'],
        help='训练任务类型：pretrain（预训练）、sft（监督微调）、dpo（偏好优化）'
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='训练配置文件路径（YAML格式）'
    )
    
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='DeepSpeed本地rank（由DeepSpeed启动器自动传入）'
    )
    
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='DeepSpeed配置文件路径（可选，会覆盖YAML中的配置）'
    )
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    if args.local_rank in [-1, 0]:
        print("=" * 80)
        print("DeepSpeed训练")
        print("=" * 80)
        print(f"任务类型: {args.task_type}")
        print(f"配置文件: {args.config_path}")
        print(f"Local Rank: {args.local_rank}")
        print("=" * 80)
    
    # 根据任务类型选择训练器
    if args.task_type == 'pretrain':
        trainer = PretrainTrainer(
            config_path=args.config_path,
            local_rank=args.local_rank
        )
    elif args.task_type == 'sft':
        trainer = SFTTrainer(
            config_path=args.config_path,
            local_rank=args.local_rank
        )
    elif args.task_type == 'dpo':
        trainer = DPOTrainer(
            config_path=args.config_path,
            local_rank=args.local_rank
        )
    else:
        raise ValueError(f"不支持的任务类型: {args.task_type}")
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        if args.local_rank in [-1, 0]:
            print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        if args.local_rank in [-1, 0]:
            print(f"\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

