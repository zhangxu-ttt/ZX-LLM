from typing import List, Dict, Optional
import json

def pmt_table(P):
    j = 0
    pmt = [0]
    for i in range(1, len(P)):
        while j > 0 and P[i] != P[j]:
            j = pmt[j - 1]
        if j == 0 and P[i] != P[j]:
            pmt += [0]
        if P[i] == P[j]:
            j += 1
            pmt += [j]
    return pmt

def kmp_search(S, P):
    """KMP算法搜索模式P在文本S中的所有出现位置

    Args:
        S: 文本序列（列表）
        P: 模式序列（列表）

    Returns:
        list: 所有匹配位置的起始索引列表
    """
    if not P or not S or len(P) > len(S):
        return []

    j = 0
    pmt = pmt_table(P)
    idx = []
    for i in range(len(S)):
        while j > 0 and S[i] != P[j]:
            j = pmt[j - 1]
        if S[i] == P[j]:
            j += 1
        if j == len(P):
            idx.append(i - len(P) + 1)
            j = pmt[j - 1]
    return idx


def read_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]


