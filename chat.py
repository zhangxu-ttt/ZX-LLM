import argparse
import vllm
from transformers import AutoTokenizer
from enum import Enum


def chat(model, tokenizer):
    message = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            message = []
            print("=" * 80)
            continue

        message.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        response = model.generate(prompt, params)[0]
        message.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}")
        print("-" * 20)

def generate(model):
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            print("=" * 80)
            continue
        prompt = user_input
        response = model.generate(prompt, params)[0]
        print(f"Assistant: {response}")
        print("-" * 20)

class GenerateEnum(Enum):
    chat = chat
    generate = generate

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chat with the ZX-LLM model")
    parser.add_argument("--model", type=str, required=True, help="Path to the ZX-LLM model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--prompt", type=str, default="你好", help="Prompt to start the conversation")
    parser.add_argument("--mode", type=str, default="chat", help="Mode: 'chat' or 'generate'")
    args = parser.parse_args()

    model = vllm.LLM(model=args.model_path)
    params = vllm.SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.mode == "chat":
        if args.tokenizer is None:
            args.tokenizer = args.model
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        GenerateEnum.chat(model, tokenizer)

    elif args.mode == "generate":
        GenerateEnum.generate(model)

    else:
        ValueError("Invalid mode. Please choose 'chat' or 'generate'.")


