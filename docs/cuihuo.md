## vllm-ascend 拉起一个可用的大模型  
  
Ref: https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen2.5-7B.html

### 首先设置两个环境变量
```
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="./Qwen2.5-7B-Instruct"
```

### 接下来使用 vllm serve 这个模型
```
vllm serve ${MODEL_PATH} \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name qwen-2.5-7b-instruct \
          --trust-remote-code \
          --max-model-len 32768
```

### 验证

```
# 验证模型服务正常
vllm chat
```

### 使用 vllm bench 来测试模型性能
```
vllm bench serve \
  --model ./Qwen2.5-7B-Instruct/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

### 测试结果
```json
{"date": "20260722-060114", "endpoint_type": "openai", "backend": "openai", "label": null, "model_id": "./Qwen2.5-VL-3B-Instruct", "tokenizer_id": "./Qwen2.5-VL-3B-Instruct", "num_prompts": 200, "request_rate": 1.0, "burstiness": 1.0, "max_concurrency": null, "duration": 204.45408433000557, "completed": 199, "failed": 1, "total_input_tokens": 39800, "total_output_tokens": 25472, "request_throughput": 0.9733236714351853, "request_goodput": null, "output_throughput": 124.58542994370372, "total_token_throughput": 319.2501642307408, "max_output_tokens_per_s": 308.0, "max_concurrent_requests": 14, "rtfx": 0.0, "mean_ttft_ms": 97.50969301155726, "median_ttft_ms": 98.94429985433817, "std_ttft_ms": 10.283567228285902, "p99_ttft_ms": 113.88637554366143, "mean_tpot_ms": 34.43461520473263, "median_tpot_ms": 34.45028747967261, "std_tpot_ms": 0.3581634321016956, "p99_tpot_ms": 35.01290820802995, "mean_itl_ms": 34.434617229796345, "median_itl_ms": 34.34124984778464, "std_itl_ms": 1.2341968366262888, "p99_itl_ms": 38.25465443544089}
```
