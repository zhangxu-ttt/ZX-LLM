# test_single_gpu.py
import torch
from transformers import AutoTokenizer
from model import TransformerModel, ModelConfig
from dataset import TextDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("单GPU测试（不使用DeepSpeed）")
print("=" * 80)

# 1. 加载tokenizer
print("\n1. 加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("tokenizer/minimind")
print("✅ Tokenizer加载成功")

# 2. 创建数据集
print("\n2. 加载数据集...")
dataset = TextDataset(
    data_path="data/pretrain_hq.jsonl",
    tokenizer=tokenizer,
    max_length=512
)
print(f"✅ 数据集加载成功，大小: {len(dataset)}")

# 3. 创建DataLoader
print("\n3. 创建DataLoader...")
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0  # 重要：先用0
)
print("✅ DataLoader创建成功")

# 4. 测试数据迭代
print("\n4. 测试数据迭代...")
batch = next(iter(dataloader))
print(f"✅ 数据批次获取成功")
print(f"   x shape: {batch['x'].shape}")
print(f"   y shape: {batch['y'].shape}")

# 5. 创建模型
print("\n5. 创建模型...")
config = ModelConfig(
    vocab_size=6400,
    n_layers=12,
    d_model=32,  # 你改的小模型
    q_head=4,
    kv_head=2,
    d_ff=128,
    max_seq_length=512
)
model = TransformerModel(config)
print("✅ 模型创建成功")

# 6. 移动到GPU
print("\n6. 移动模型到GPU...")
device = torch.device("cuda:0")
model = model.to(device)
print("✅ 模型移动成功")

# 7. 测试前向传播
print("\n7. 测试前向传播...")
input_ids = batch['x'].to(device)
labels = batch['y'].to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
print(f"✅ 前向传播成功，Loss: {outputs.loss.item():.4f}")

# 8. 测试反向传播
print("\n8. 测试反向传播...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i in range(3):
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   Step {i + 1}, Loss: {loss.item():.4f}")

print("\n✅ 所有测试通过！")
print("=" * 80)