# Qwen/Qwen3-8B-Base

- **vLLM Version**: vLLM: 0.10.1.1 ([1da94e6](https://github.com/vllm-project/vllm/commit/1da94e6)), **vLLM Ascend Version**: v0.10.1rc1 ([7e16b4a](https://github.com/vllm-project/vllm-ascend/commit/7e16b4a))  
- **Software Environment**: **CANN**: 8.2.RC1, **PyTorch**: 2.7.1, **torch-npu**: 2.7.1.dev20250724  
- **Hardware Environment**: Atlas A2 Series  
- **Parallel mode**: TP1
- **Execution mode**: ACLGraph

**Command**:  

```bash
export MODEL_ARGS='pretrained=Qwen/Qwen3-8B-Base,tensor_parallel_size=1,dtype=auto,trust_remote_code=False,max_model_len=4096'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks gsm8k,ceval-valid \
 --apply_chat_template True   --fewshot_as_multiturn True   --num_fewshot 5   --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
| gsm8k | exact_match,strict-match | ✅0.8271 | ± 0.0104 |
| gsm8k | exact_match,flexible-extract | ✅0.8294 | ± 0.0104 |
| ceval-valid | acc,none | ✅0.815 | ± 0.0103 |
