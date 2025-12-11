from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModelConfig(PretrainedConfig):
    model_type = 'zx_model'

    def __init__(self,
                 vocab_size: int = -1,
                 n_layers: int = 8,
                 d_model: int = 512,
                 q_head: int = 8,
                 kv_head: int = 2,
                 d_ff: int = 2048,
                 dropout_p: float = 0.1,
                 max_seq_length: int = 2048,
                 rope_theta: float = 1000000.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.q_head = q_head
        self.kv_head = kv_head
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.max_seq_length = max_seq_length
        self.rope_theta = rope_theta


@dataclass
class KVCache:
    """
    KV 缓存，用于推理加速

    在自回归生成时，之前计算过的 K 和 V 可以缓存起来，
    避免重复计算，大幅提升推理速度
    """
    k: torch.Tensor  # [batch_size, seq_length, n_kv_head, head_dim]
    v: torch.Tensor  # [batch_size, seq_length, n_kv_head, head_dim]

    def update(
            self,
            new_k: torch.Tensor,
            new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新 KV 缓存

        Args:
            new_k: 新的 K，形状 [batch_size, new_seq_length, n_kv_head, head_dim]
            new_v: 新的 V，形状 [batch_size, new_seq_length, n_kv_head, head_dim]

        Returns:
            更新后的完整 (K, V)
        """

        # 更新缓存
        self.k = torch.cat([self.k, new_k], dim=1)
        self.v = torch.cat([self.v, new_v], dim=1)

        # 返回截止到当前位置的所有 K 和 V
        return self.k, self.v


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 层

        Args:
            d_model: 输入维度（模型维度）
            eps: 小常数，防止除零
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算 RMSNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_length: int = 2048,
            theta: float = 1e6,
    ):
        """
        初始化旋转位置编码

        Args:
            dim: 每个注意力头的维度（head_dim）
            max_seq_length: 最大序列长度
            theta: RoPE 的基础频率参数
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta

        # 预计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos 和 sin 缓存
        self._set_cos_sin_cache(max_seq_length)

    def _set_cos_sin_cache(self, seq_length: int):
        """预计算 cos 和 sin 缓存"""
        self.max_seq_length_cached = seq_length
        t = torch.arange(seq_length, device=self.inv_freq.device).type_as(self.inv_freq)

        # 计算频率矩阵: [seq_length, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # 拼接得到完整的频率: [seq_length, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # 计算 cos 和 sin: [1, seq_length, 1, dim]
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            seq_length: Optional[int] = None,
            seq_index: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到查询和键

        Args:
            q: 查询张量，形状 [batch_size, seq_length, n_head, head_dim]
            k: 键张量，形状 [batch_size, seq_length, n_head, head_dim]
            seq_length: 序列长度（如果为 None，从 q 推断）

        Returns:
            应用 RoPE 后的 (q, k)，形状与输入相同
        """
        if seq_length is None:
            seq_length = q.shape[seq_index]

        # 如果序列长度超过缓存，重新计算
        if seq_length > self.max_seq_length_cached:
            self._set_cos_sin_cache(seq_length)

        # 获取 cos 和 sin: [1, seq_length, 1, dim]
        cos = self.cos_cached[:, :seq_length, :, :]
        sin = self.sin_cached[:, :seq_length, :, :]

        # 确保 cos 和 sin 的数据类型与 q 一致（用于 FP16 训练）
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        # 应用旋转，确保结果是连续的
        q_embed = ((q * cos) + (self._rotate_half(q) * sin))
        k_embed = ((k * cos) + (self._rotate_half(k) * sin))

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        旋转张量的一半维度

        将 [x1, x2, x3, x4, ...] 变换为 [-x2, x1, -x4, x3, ...]

        注意：使用 clone() 而不是切片视图，避免 DeepSpeed 多 GPU 环境下的跨设备视图问题
        """
        # 使用 clone() 创建独立副本，避免视图问题
        x1 = x[..., : x.shape[-1] // 2].clone()
        x2 = x[..., x.shape[-1] // 2:].clone()

        # 拼接并返回连续张量
        return torch.cat((-x2, x1), dim=-1)


class GroupQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped Query Attention）
    """

    def __init__(self, d_model: int, q_head: int, kv_head: int, dropout_p: float = 0.0, max_seq_length: int = 2048,
                 rope_theta: float = 1000000.0):
        """
        初始化分组查询注意力层

        Args:
            d_model: 输入维度（模型维度）
            q_head: 查询头数
            kv_head: 键值头数
        """
        super().__init__()

        assert d_model % q_head == 0, "d_model 必须能被 q_head 整除"
        assert (q_head % kv_head == 0) and (q_head >= kv_head), "q_head 必须能被 kv_head 整除"

        self.d_model = d_model
        self.q_head = q_head
        self.kv_head = kv_head
        self.dropout_p = dropout_p
        self.max_seq_length = max_seq_length

        self.head_dim = d_model // q_head

        # 线性变换层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_head * self.head_dim, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_length=max_seq_length, theta=rope_theta)

        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                attn_mask: torch.Tensor,
                use_cache: bool = False,
                kv_cache: KVCache = None,
                start_pos: int = 0
                ):
        batch_size, seq_length, dim = q.shape

        q = self.q_proj(q)

        if use_cache:
            if kv_cache is None:
                k = self.k_proj(k).reshape(batch_size, seq_length, self.kv_head, self.head_dim)
                v = self.v_proj(v).reshape(batch_size, seq_length, self.kv_head, self.head_dim)
                kv_cache = KVCache(
                    k=k,
                    v=v
                )
            else:
                new_k = self.k_proj(k).reshape(batch_size, seq_length, self.kv_head, self.head_dim)
                new_v = self.v_proj(v).reshape(batch_size, seq_length, self.kv_head, self.head_dim)
                k, v = kv_cache.update(new_k, new_v)

            k = k.to(q.dtype)
            v = v.to(q.dtype)

        else:
            k = self.k_proj(k)
            v = self.v_proj(v)

        q = q.reshape(batch_size, -1, self.q_head, self.head_dim)  # batch all_seq_len q_head head_dim
        k = k.reshape(batch_size, -1, self.kv_head, self.head_dim)  # batch all_seq_len kv_head head_dim
        v = v.reshape(batch_size, -1, self.kv_head, self.head_dim)  # batch all_seq_len kv_head head_dim

        q, k = self.rope(q, k)

        q = q.transpose(1, 2)  # batch q_head seq_len head_dim
        k = k.repeat(1, 1, self.q_head // self.kv_head, 1).transpose(1, 2)  # batch q_head all_seq_len head_dim
        v = v.repeat(1, 1, self.q_head // self.kv_head, 1).transpose(1, 2)  # batch q_head all_seq_len head_dim

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=False).transpose(
                1, 2)  # batch all_seq_len q_head head_dim
        else:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p, is_causal=True).transpose(
                1, 2)  # batch all_seq_len q_head head_dim
        output = output.reshape(batch_size, -1, self.d_model)
        output = self.o_proj(output)

        return output, kv_cache


class FFN(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）
    """

    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.0):
        """
        初始化前馈神经网络层

        Args:
            d_model: 输入维度（模型维度）
            d_ff: 前馈网络隐藏层维度
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.linear3 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        x1 = self.linear1(x)
        x2 = self.act(self.linear2(x))
        x = self.linear3(x1 * x2)
        x = self.drop(x)
        return x


class TransformerLayer(nn.Module):
    """
    Transformer Layer
    """

    def __init__(self,
                 d_model: int,
                 q_head: int,
                 kv_head: int,
                 d_ff: int,
                 dropout_p: float = 0.0,
                 max_seq_length: int = 2048,
                 rope_theta: float = 1000000.0
                 ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = GroupQueryAttention(d_model, q_head, kv_head, dropout_p, max_seq_length, rope_theta)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout_p)

    def forward(self,
                x: torch.Tensor,
                attn_mask: torch.Tensor,
                use_cache: bool = False,
                kv_cache: KVCache = None,
                start_pos: int = 0
                ):
        residual = x
        x = self.attn_norm(x)
        x, kv_cache = self.attention(q=x, k=x, v=x, attn_mask=attn_mask, use_cache=use_cache, kv_cache=kv_cache,
                                     start_pos=start_pos)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + residual

        return x, kv_cache


class TransformerModel(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    supports_gradient_checkpointing = True 

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.q_head = config.q_head
        self.kv_head = config.kv_head
        self.d_ff = config.d_ff
        self.dropout_p = config.dropout_p
        self.max_seq_length = config.max_seq_length
        self.rope_theta = config.rope_theta

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.q_head, self.kv_head, self.d_ff, self.dropout_p, self.max_seq_length,
                             self.rope_theta)
            for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                use_cache: bool = False,
                past_key_values: List[KVCache] = None,
                start_pos: int = 0,
                output_hidden_states: bool = False,
                return_dict: bool = True,
                **kwargs  
                ):
        x = self.embedding(input_ids)
        past_kv_caches = past_key_values

        new_kv_caches = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None

        if output_hidden_states:
            all_hidden_states.append(x)

        for i, layer in enumerate(self.layers):
            kv_cache = past_kv_caches[i] if past_kv_caches is not None else None

            if self.gradient_checkpointing and self.training and not use_cache:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs: (x, attn_mask, use_cache, kv_cache, start_pos)
                        return module(x=inputs[0], attn_mask=inputs[1], 
                                    use_cache=inputs[2], kv_cache=inputs[3], 
                                    start_pos=inputs[4])
                    return custom_forward

                x, kv_cache = checkpoint(
                    create_custom_forward(layer),
                    x,
                    attention_mask,
                    use_cache,
                    kv_cache,
                    start_pos,
                    use_reentrant=False 
                )
            else:
                x, kv_cache = layer(x=x, attn_mask=attention_mask, use_cache=use_cache, kv_cache=kv_cache,
                                    start_pos=start_pos)

            if output_hidden_states:
                all_hidden_states.append(x)

            if use_cache:
                new_kv_caches.append(kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=new_kv_caches,
                hidden_states=all_hidden_states,
                attentions=None
            )
        else:
            return (loss, logits, new_kv_caches, all_hidden_states)
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: List[KVCache] = None,
        use_cache: bool = True,
        **kwargs
    ):
        if use_cache and past_key_values is not None:
            start_pos = past_key_values[0].k.shape[1]
            input_ids = input_ids[:, -1:]
        else:
            start_pos = 0


        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "start_pos": start_pos
        }

        return model_inputs
