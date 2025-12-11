from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model import TransformerModel, ModelConfig

AutoConfig.register("zx_model", ModelConfig)
AutoModelForCausalLM.register(ModelConfig, TransformerModel)

tokenizer = AutoTokenizer.from_pretrained("/Users/zhangxu/PycharmProjects/ZX-LLM/tokenizer/minimind")

model_config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_length=2048,
    n_layers=12,
    q_head=8,               
    kv_head=4,
    d_model=768,
    d_ff=768*4,
    dropout_p=0.1,          
    rope_theta=1000000.0,   
)
model = TransformerModel(model_config)


prompt = '你好！我是一个学生。'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids


output_ids = model.generate(
        input_ids,
        attention_mask=None,
        max_new_tokens=50,      # 生成最多 50 个新 token
        do_sample=True,         # 使用采样
        temperature=0.7,        # 温度
        top_k=50,               # Top-K 采样
        top_p=0.9,              # Top-P 采样
        use_cache=True,         # 使用 KV 缓存
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))