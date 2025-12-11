from typing import Dict, Tuple
import torch

from trainer.baseTrainer import BaseTrainer
from dataset import ChatMLDataset


class SFTTrainer(BaseTrainer):
    """
    SFT（监督微调）训练器
    
    用于对话模型的监督微调任务
    只计算assistant回答部分的损失
    """
    
    def prepare_dataset(self) -> Tuple:
        """
        准备SFT数据集
        
        Returns:
            train_dataset, eval_dataset
        """
        print("准备SFT数据集...")
        
        # 训练集
        train_data_path = self.config['data']['train_data_path']
        train_dataset = ChatMLDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        
        # 验证集
        eval_dataset = None
        if self.config['data'].get('eval_data_path'):
            eval_data_path = self.config['data']['eval_data_path']
            eval_dataset = ChatMLDataset(
                data_path=eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config['data']['max_length']
            )
            print(f"验证集大小: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        SFT的训练步骤
        
        与预训练类似，但loss_mask只计算assistant回答部分
        
        Args:
            batch: 包含 'x', 'y', 'loss_mask' 的批次数据
        
        Returns:
            loss: 损失值
        """
        # 将数据移到设备
        input_ids = batch['x'].to(self.model_engine.device)
        labels = batch['y'].to(self.model_engine.device)
        loss_mask = batch['loss_mask'].to(self.model_engine.device)
        
        # 前向传播
        outputs = self.model_engine(
            input_ids=input_ids,
            labels=labels
        )
        
        # 重新计算masked loss（只计算assistant部分）
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        labels_shifted = labels[:, :]  # [batch, seq_len-1]
        
        # 展平
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch * seq_len, vocab]
        labels_flat = labels_shifted.reshape(-1)  # [batch * seq_len]
        mask_flat = loss_mask.reshape(-1)  # [batch * seq_len]
        
        # 计算每个位置的损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(logits_flat, labels_flat)  # [batch * seq_len]
        
        # 应用mask（只计算assistant部分）
        masked_loss = loss_per_token * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)
        
        return loss

