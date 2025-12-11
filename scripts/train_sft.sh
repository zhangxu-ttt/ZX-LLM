#!/bin/bash

# SFT（监督微调）启动脚本

# 设置可见GPU（根据实际情况修改）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置文件路径
CONFIG_PATH="config/sft_config.yaml"

# DeepSpeed启动命令
# 单GPU训练使用: python train.py --task_type sft --config_path $CONFIG_PATH
# 多GPU训练使用下面的deepspeed命令

deepspeed train.py \
    --task_type sft \
    --config_path $CONFIG_PATH

# 如果需要指定master端口（避免端口冲突）
# deepspeed --num_gpus=$NUM_GPUS --master_port 29500 train.py \
#     --task_type sft \
#     --config_path $CONFIG_PATH

# 多机训练示例（需要配置hostfile）
# deepspeed --num_gpus=$NUM_GPUS --num_nodes=2 --hostfile=hostfile train.py \
#     --task_type sft \
#     --config_path $CONFIG_PATH

