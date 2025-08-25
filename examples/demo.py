from transformers import AutoTokenizer

# 换成你实际用的模型，比如 "gpt2" 或 "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "/home/cache/modelscope/hub/models/facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 打印指定 token 的解码结果
target_ids = [67, 128]
for tid in target_ids:
    token_str = tokenizer.decode([tid])
    print(f"Token {tid} -> {token_str}")
