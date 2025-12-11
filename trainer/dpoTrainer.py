from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from trainer.baseTrainer import BaseTrainer
from dataset import DPODataset
from model import TransformerModel


class DPOTrainer(BaseTrainer):
    """
    DPO（Direct Preference Optimization）训练器
    
    用于基于人类偏好的模型对齐训练
    """
    
    def __init__(self, config_path: str, local_rank: int = -1):
        """
        初始化DPO训练器
        
        Args:
            config_path: 训练配置文件路径
            local_rank: DeepSpeed本地rank
        """
        # 调用父类初始化
        super().__init__(config_path, local_rank)
        
        # DPO特定参数
        self.beta = self.config['dpo'].get('beta', 0.1)
        self.label_smoothing = self.config['dpo'].get('label_smoothing', 0.0)
        self.reference_free = self.config['dpo'].get('reference_free', False)
        
        # 初始化参考模型（用于计算KL散度）
        if not self.reference_free:
            self.ref_model = self.create_reference_model()
        else:
            self.ref_model = None
        
        print(f"DPO训练器初始化完成，beta={self.beta}")
    
    def create_reference_model(self) -> TransformerModel:
        """
        创建参考模型
        
        参考模型用于计算KL散度，保持不变
        """
        reference_path = self.config['model'].get('reference_model_path')
        
        if reference_path:
            print(f"从指定路径加载参考模型: {reference_path}")
            ref_model = TransformerModel.from_pretrained(reference_path)
        else:
            print("使用当前模型作为参考模型")
            # 复制当前模型作为参考模型
            ref_model = TransformerModel(self.model.config)
            ref_model.load_state_dict(self.model.state_dict())
        
        # 冻结参考模型
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # 移到设备
        ref_model.to(self.model_engine.device)
        
        return ref_model
    
    def prepare_dataset(self) -> Tuple:
        """
        准备DPO数据集
        
        Returns:
            train_dataset, eval_dataset
        """
        print("准备DPO数据集...")
        
        # 训练集
        train_data_path = self.config['data']['train_data_path']
        train_dataset = DPODataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        
        # 验证集
        eval_dataset = None
        if self.config['data'].get('eval_data_path'):
            eval_data_path = self.config['data']['eval_data_path']
            eval_dataset = DPODataset(
                data_path=eval_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config['data']['max_length']
            )
            print(f"验证集大小: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def get_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算log probabilities
        
        Args:
            logits: 模型输出的logits [batch, seq_len, vocab]
            labels: 标签 [batch, seq_len]
            mask: mask [batch, seq_len]
        
        Returns:
            log_probs: 序列的平均log probability
        """
        # 获取每个token的log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab]
        
        # 选择标签对应的log probabilities
        per_token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len]
        
        # 应用mask并计算平均
        masked_log_probs = per_token_log_probs * mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)  # [batch]
        
        return sequence_log_probs
    
    def compute_dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算DPO损失
        
        DPO损失公式：
        loss = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
        
        Args:
            chosen_logps: 策略模型对chosen response的log probability
            rejected_logps: 策略模型对rejected response的log probability
            ref_chosen_logps: 参考模型对chosen response的log probability
            ref_rejected_logps: 参考模型对rejected response的log probability
        
        Returns:
            loss: DPO损失
            metrics: 额外的指标字典
        """
        # 计算log ratios
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO损失
        logits = pi_logratios - ref_logratios
        
        if self.label_smoothing > 0:
            # 使用标签平滑
            loss = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - \
                   F.logsigmoid(-self.beta * logits) * self.label_smoothing
        else:
            loss = -F.logsigmoid(self.beta * logits)
        
        loss = loss.mean()
        
        # 计算额外的指标
        with torch.no_grad():
            # 隐式奖励
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
            reward_margin = chosen_rewards - rejected_rewards
            
            # 准确率（chosen的奖励是否高于rejected）
            accuracy = (reward_margin > 0).float().mean()
            
            metrics = {
                'dpo_loss': loss.item(),
                'chosen_rewards': chosen_rewards.mean().item(),
                'rejected_rewards': rejected_rewards.mean().item(),
                'reward_margin': reward_margin.mean().item(),
                'accuracy': accuracy.item()
            }
        
        return loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DPO训练步骤
        
        Args:
            batch: 包含chosen和rejected数据的批次
        
        Returns:
            loss: DPO损失
        """
        # 将数据移到设备
        chosen_input_ids = batch['chosen_x'].to(self.model_engine.device)
        chosen_labels = batch['chosen_y'].to(self.model_engine.device)
        chosen_mask = batch['chosen_mask'].to(self.model_engine.device)
        
        rejected_input_ids = batch['rejected_x'].to(self.model_engine.device)
        rejected_labels = batch['rejected_y'].to(self.model_engine.device)
        rejected_mask = batch['rejected_mask'].to(self.model_engine.device)
        
        # 1. 前向传播chosen
        chosen_outputs = self.model_engine(
            input_ids=chosen_input_ids,
            labels=chosen_labels
        )
        chosen_logits = chosen_outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        chosen_logps = self.get_log_probs(chosen_logits, chosen_labels, chosen_mask)
        
        # 2. 前向传播rejected
        rejected_outputs = self.model_engine(
            input_ids=rejected_input_ids,
            labels=rejected_labels
        )
        rejected_logits = rejected_outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        rejected_logps = self.get_log_probs(rejected_logits, rejected_labels, rejected_mask)
        
        # 3. 计算参考模型的log probs
        if self.reference_free:
            # Reference-free DPO
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        else:
            with torch.no_grad():
                # Chosen
                ref_chosen_outputs = self.ref_model(
                    input_ids=chosen_input_ids,
                    labels=chosen_labels
                )
                ref_chosen_logits = ref_chosen_outputs.logits[:, :-1, :]
                ref_chosen_logps = self.get_log_probs(ref_chosen_logits, chosen_labels, chosen_mask)
                
                # Rejected
                ref_rejected_outputs = self.ref_model(
                    input_ids=rejected_input_ids,
                    labels=rejected_labels
                )
                ref_rejected_logits = ref_rejected_outputs.logits[:, :-1, :]
                ref_rejected_logps = self.get_log_probs(ref_rejected_logits, rejected_labels, rejected_mask)
        
        # 4. 计算DPO损失
        loss, metrics = self.compute_dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )
        
        # 记录额外的指标
        if self.is_main_process():
            self.log_metrics(metrics)
        
        return loss

