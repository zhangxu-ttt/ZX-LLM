from typing import List, Dict, Optional, Union
from pathlib import Path
import random
import json

import torch
import pandas as pd
from torch.utils.data import Dataset
from typing_extensions import override

from utils import kmp_search, read_jsonl

class TextDataset(Dataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256
    ):
        """
        Args:
            data_path: 文件路径
            tokenizer: 分词器 (AutoTokenizer)
            max_length: 最大序列长度
                - conversation: JSONL格式对话数据
                - text: 纯文本数据
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_data(data_path)

    def load_data(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            texts = read_jsonl(data_path)
        else:
            texts = []
            for path in data_path:
                texts += read_jsonl(path)
        return texts

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本"""
        sample = self.data[idx]

        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        # 使用 attention_mask 而不是 (input_ids != pad_token_id)
        # 原因：当 tokenizer.pad_token 被设成 eos_token 时，
        # (input_ids != pad_token_id) 会把真实 eos 也当成 padding 屏蔽掉。
        loss_mask = encoding.attention_mask.squeeze().to(dtype=torch.bool)

        x = input_ids[:-1]
        y = input_ids[1:]

        return {
            'x': x,
            'y': y,
            'loss_mask': loss_mask[1:],
        }

class ChatMLDataset(Dataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_data(data_path)

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.answer_start_token = f"<|im_start|>assistant\n"
        self.answer_start_token_id_list = None
        self.answer_end_token_id_list = [self.eos_token_id]

    def load_data(self, data_path: Union[str, List[str]]) -> List[str]:
        if isinstance(data_path, str):
            texts = read_jsonl(data_path)
        else:
            texts = []
            for path in data_path:
                texts += read_jsonl(path)
        return texts

    def process_sample(self, text):
        """处理样本"""
        if self.answer_start_token_id_list is None:
            self.answer_start_token_id_list = self.tokenizer(self.answer_start_token)['input_ids']  # 是一个list

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze()
        input_ids_list = input_ids.tolist()  # 转换为列表供KMP算法使用

        # 处理loss_mask，只计算assistant的回答部分，这里需要找到
        # 1. <im_start>assistant 出现的位置(KMP算法)
        start_idx = kmp_search(input_ids_list, self.answer_start_token_id_list)

        # 2. <im_end> 出现的位置(KMP算法)
        end_idx = kmp_search(input_ids_list, self.answer_end_token_id_list)

        # 3. 将 <im_start>assistant 和它后面第一个出现的 <im_end> 之间的部分保留，其他部分mask
        loss_mask = [0] * len(input_ids_list)

        # 配对算法：为每个 assistant 起始位置找到对应的 <im_end> 结束位置
        start_len = len(self.answer_start_token_id_list)
        end_len = len(self.answer_end_token_id_list)

        j = 0  # end_idx 的指针
        for i in range(len(start_idx)):
            start_pos = start_idx[i]

            # 跳过在当前 assistant 之前的所有 <im_end>
            while j < len(end_idx) and end_idx[j] + end_len - 1 < start_pos:
                j += 1

            # 如果找到了有效的结束位置，标记这个范围
            if j < len(end_idx):
                # 从 assistant 标记之后开始，到 <im_end> 结束（包含）
                range_start = start_pos + start_len
                range_end = end_idx[j]

                for idx in range(range_start, range_end + 1):
                    if 0 <= idx < len(loss_mask):
                        loss_mask[idx] = 1

                j += 1  # 移动到下一个可能的结束位置
            # 如果没有找到对应的 end_idx，跳过这个 assistant（避免死循环）

        x = input_ids[:-1]
        y = input_ids[1:]

        return x, y, torch.tensor(loss_mask[1:])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        text = self.tokenizer.apply_chat_template(
            sample['conversations'],
            tokenize=False,
            add_generation_prompt=True
        )

        x, y, loss_mask = self.process_sample(text)

        return {
            'x': x,
            'y': y,
            'loss_mask': loss_mask.to(dtype=torch.bool)
        }


class DPODataset(ChatMLDataset):
    def __init__(
            self,
            data_path: Union[str, List[str]],
            tokenizer,
            max_length: int = 256,
    ):
        super().__init__(data_path, tokenizer, max_length)

    @override
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        chosen_text = self.tokenizer.apply_chat_template(
            sample['chosen'],
            tokenize=False,
            add_generation_prompt=True
        )

        rejected_text = self.tokenizer.apply_chat_template(
            sample['rejected'],
            tokenize=False,
            add_generation_prompt=True
        )

        chosen_x, chosen_y, chosen_loss_mask = self.process_sample(chosen_text)
        rejected_x, rejected_y, rejected_loss_mask = self.process_sample(rejected_text)

        return {
            'chosen_x': chosen_x,
            'chosen_y': chosen_y,
            'chosen_mask': chosen_loss_mask,
            'rejected_x': rejected_x,
            'rejected_y': rejected_y,
            'rejected_mask': rejected_loss_mask,
        }



