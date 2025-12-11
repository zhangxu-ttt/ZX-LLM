import os
import json
import yaml
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import deepspeed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import TransformerModel, ModelConfig


class BaseTrainer(ABC):
    """
    DeepSpeed训练器基类
    
    提供所有训练器的公共功能：
    - 配置加载
    - 模型初始化
    - DeepSpeed引擎初始化
    - 训练循环
    - 检查点保存/加载
    - 日志记录
    """
    
    def __init__(self, config_path: str, local_rank: int = -1):
        """
        初始化训练器
        
        Args:
            config_path: 训练配置文件路径（YAML）
            local_rank: DeepSpeed本地rank（由DeepSpeed启动器自动传入）
        """
        self.config_path = config_path
        self.local_rank = local_rank
        
        # 加载配置
        self.config = self.load_config(config_path)
        
        # 设置随机种子
        self.set_seed(self.config['training'].get('seed', 42))
        
        # 初始化tokenizer
        self.tokenizer = self.prepare_tokenizer()
        
        # 准备数据集
        self.train_dataset, self.eval_dataset = self.prepare_dataset()
        
        # 初始化模型
        self.model = self.create_model()
        
        # 初始化DeepSpeed引擎
        self.model_engine, self.optimizer, self.lr_scheduler = self.initialize_deepspeed()
        
        # 准备输出目录
        self.output_dir = Path(self.config['output']['output_dir'])
        if self.is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化WandB（仅主进程）
        self.wandb_run = None
        if self.is_main_process() and self.config['wandb']['enabled']:
            self.init_wandb()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        
        # 从检查点恢复（如果指定）
        if self.config['output'].get('resume_from_checkpoint'):
            self.load_checkpoint(self.config['output']['resume_from_checkpoint'])
    
    def load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_tokenizer(self):
        """准备tokenizer"""
        tokenizer_path = self.config.get('tokenizer_path', 'tokenizer/minimind')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    
    @abstractmethod
    def prepare_dataset(self):
        """
        准备数据集 - 子类必须实现
        
        Returns:
            train_dataset, eval_dataset
        """
        raise NotImplementedError
    
    def create_model(self) -> nn.Module:
        """创建模型"""
        model_config = ModelConfig(**self.config['model'])
        
        # 如果指定了预训练模型路径，从检查点加载
        pretrained_path = self.config['model'].get('pretrained_model_path')
        if pretrained_path:
            self.print_main_process(f"从预训练模型加载: {pretrained_path}")
            model = TransformerModel.from_pretrained(pretrained_path)
        else:
            model = TransformerModel(model_config)
        
        # 启用梯度检查点
        if self.config['training'].get('gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.print_main_process("已启用梯度检查点")
        
        return model
    
    def initialize_deepspeed(self):
        """
        初始化DeepSpeed引擎
        
        Returns:
            model_engine, optimizer, lr_scheduler
        """
        # 加载DeepSpeed配置
        ds_config_path = self.config['deepspeed']['config_path']
        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)
        
        # 设置批次大小参数
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        train_batch_size = (
            self.config['training']['per_device_train_batch_size'] *
            self.config['training']['gradient_accumulation_steps'] *
            world_size
        )
        ds_config['train_batch_size'] = train_batch_size
        ds_config['train_micro_batch_size_per_gpu'] = self.config['training']['per_device_train_batch_size']
        ds_config['gradient_accumulation_steps'] = self.config['training']['gradient_accumulation_steps']
        
        # 初始化DeepSpeed
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )
        
        # 如果DeepSpeed没有创建学习率调度器，手动创建
        if lr_scheduler is None:
            num_training_steps = self.calculate_total_steps()
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config['training']['warmup_steps'],
                num_training_steps=num_training_steps
            )

        self.print_main_process(f"DeepSpeed引擎已初始化 (Rank {self.local_rank})")
        self.print_main_process(f"训练批次大小: {train_batch_size}")
        self.print_main_process(f"每设备批次大小: {self.config['training']['per_device_train_batch_size']}")
        self.print_main_process(f"梯度累积步数: {self.config['training']['gradient_accumulation_steps']}")
        
        return model_engine, optimizer, lr_scheduler
    
    def calculate_total_steps(self) -> int:
        """计算总训练步数"""
        max_steps = self.config['training'].get('max_steps', -1)
        if max_steps > 0:
            return max_steps
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        num_epochs = self.config['training']['num_epochs']
        steps_per_epoch = len(self.train_dataset) // (
            self.config['training']['per_device_train_batch_size'] *
            self.config['training']['gradient_accumulation_steps'] *
            torch.cuda.device_count()
        )
        return num_epochs * steps_per_epoch
    
    def init_wandb(self):
        """初始化WandB"""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['run_name'],
                entity=self.config['wandb'].get('entity'),
                config=self.config,
                resume='allow'
            )
            self.print_main_process("WandB已初始化")
        except ImportError:
            self.print_main_process("警告: wandb未安装，跳过WandB日志记录")
            self.config['wandb']['enabled'] = False
    
    def is_main_process(self) -> bool:
        """判断是否为主进程"""
        return self.local_rank in [-1, 0]

    def print_main_process(self, message: str):
        """仅在主进程打印消息"""
        if self.is_main_process():
            print(message)
    
    def train(self):
        """
        主训练循环
        """
        self.print_main_process("=" * 80)
        self.print_main_process("开始训练")
        self.print_main_process("=" * 80)
        
        # 准备DataLoader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['per_device_train_batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0),
            pin_memory=True
        )
        
        num_epochs = self.config['training']['num_epochs']
        logging_steps = self.config['training']['logging_steps']
        eval_steps = self.config['training']['eval_steps']
        save_steps = self.config['training']['save_steps']
        max_steps = self.config['training'].get('max_steps', -1)
        
        self.model_engine.train()
        
        # 训练循环
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            self.print_main_process(f"\n{'=' * 80}")
            self.print_main_process(f"Epoch {epoch + 1}/{num_epochs}")
            self.print_main_process(f"{'=' * 80}")
            
            epoch_loss = 0.0
            epoch_steps = 0
            
            # 使用tqdm显示进度（仅主进程）
            if self.is_main_process():
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            else:
                progress_bar = train_dataloader
            
            for step, batch in enumerate(progress_bar):
                start_time = time.time()
                
                # 训练步骤
                loss = self.training_step(batch)
                
                # 反向传播
                self.model_engine.backward(loss)
                
                # 优化器步进
                self.model_engine.step()
                
                # 更新全局步数
                self.global_step += 1
                epoch_steps += 1
                epoch_loss += loss.item()
                
                step_time = time.time() - start_time
                
                # 日志记录
                if self.global_step % logging_steps == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    metrics = {
                        'loss': loss.item(),
                        'avg_loss': avg_loss,
                        'learning_rate': lr,
                        'epoch': epoch + 1,
                        'step': self.global_step,
                        'step_time': step_time
                    }

                    self.log_metrics(metrics)
                    
                    if self.is_main_process():
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'avg_loss': f"{avg_loss:.4f}",
                            'lr': f"{lr:.2e}"
                        })
                
                # 评估
                if eval_steps > 0 and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.model_engine.train()
                
                # 保存检查点
                if save_steps > 0 and self.global_step % save_steps == 0:
                    self.save_checkpoint(tag=f"step-{self.global_step}")
                
                # 检查是否达到最大步数
                if max_steps > 0 and self.global_step >= max_steps:
                    self.print_main_process(f"\n已达到最大步数: {max_steps}")
                    break
            
            # Epoch结束，保存检查点
            if self.is_main_process():
                avg_epoch_loss = epoch_loss / epoch_steps

            self.print_main_process(f"\nEpoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.4f}")
            
            self.save_checkpoint(tag=f"epoch-{epoch + 1}")
            
            # 检查是否达到最大步数
            if max_steps > 0 and self.global_step >= max_steps:
                break
        
        # 训练结束
        self.print_main_process("\n" + "=" * 80)
        self.print_main_process("训练完成！")
        self.print_main_process("=" * 80)
            
        if self.wandb_run:
            self.wandb_run.finish()
    
    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        单个训练步骤 - 子类必须实现
        
        Args:
            batch: 批次数据
        
        Returns:
            loss: 损失值
        """
        raise NotImplementedError
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            评估指标字典
        """
        if self.eval_dataset is None:
            return {}
        
        self.model_engine.eval()
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config['training']['per_device_eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            pin_memory=True
        )
        
        total_loss = 0.0
        total_steps = 0
        
        if self.is_main_process():
            print("\n开始评估...")
            progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        else:
            progress_bar = eval_dataloader
        
        with torch.no_grad():
            for batch in progress_bar:
                loss = self.training_step(batch)
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_step': self.global_step
        }
        
        if self.is_main_process():
            print(f"评估结果 - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        self.log_metrics(metrics)
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
        """
        if self.is_main_process():
            # 记录到WandB
            if self.wandb_run:
                self.wandb_run.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, tag: str):
        """
        保存检查点
        
        Args:
            tag: 检查点标签（如 "step-1000" 或 "epoch-1"）
        """
        if not self.is_main_process():
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
        
        # 使用DeepSpeed保存检查点
        self.model_engine.save_checkpoint(
            save_dir=str(self.output_dir),
            tag=tag,
            client_state={'step': self.global_step, 'epoch': self.epoch}
        )
        
        print(f"检查点已保存: {checkpoint_dir}")
        
        # 管理检查点数量
        self.manage_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        print(f"从检查点恢复: {checkpoint_path}")
        
        # 使用DeepSpeed加载检查点
        _, client_state = self.model_engine.load_checkpoint(
            load_dir=checkpoint_path,
            tag=None
        )
        
        if client_state:
            self.global_step = client_state.get('step', 0)
            self.epoch = client_state.get('epoch', 0)
            print(f"已恢复到步数: {self.global_step}, 轮数: {self.epoch}")
    
    def manage_checkpoints(self):
        """管理检查点数量，删除旧的检查点"""
        save_total_limit = self.config['output'].get('save_total_limit', None)
        if save_total_limit is None or save_total_limit <= 0:
            return
        
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
            key=lambda x: x.stat().st_mtime
        )
        
        # 删除旧检查点
        while len(checkpoints) > save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            print(f"删除旧检查点: {old_checkpoint}")
