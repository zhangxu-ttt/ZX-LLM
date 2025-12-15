from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from trainer.baseTrainer import BaseTrainer
from dataset import TextDataset


class PretrainTrainer(BaseTrainer):
    """
    预训练训练器
    
    用于语言模型的预训练任务
    """
    
    def prepare_dataset(self) -> Tuple:
        """
        准备预训练数据集
        
        Returns:
            train_dataset, eval_dataset
        """
        self.print_main_process("准备预训练数据集...")
        
        # 训练集
        train_data_path = self.config['data']['train_data_path']
        train_dataset = TextDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        self.print_main_process(f"训练集大小: {len(train_dataset)}")
        
        # 验证集
        eval_dataset = None
        if self.config['data'].get('eval_data_path'):
            eval_data_path = self.config['data']['eval_data_path']
            eval_dataset = TextDataset(
                data_path=eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config['data']['max_length']
            )
            self.print_main_process(f"验证集大小: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        预训练的训练步骤
        
        Args:
            batch: 包含 'x', 'y', 'loss_mask' 的批次数据
        
        Returns:
            loss: 损失值
        """
        input_ids = batch['x'].to(self.model_engine.device)
        labels = batch['y'].to(self.model_engine.device)
        loss_mask = batch['loss_mask'].to(self.model_engine.device)
        attention_mask = None

        outputs = self.model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        
        return loss

