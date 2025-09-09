# deepseek-ai/DeepSeek-V2-Lite

- **vLLM Version**: vLLM: 0.10.1.1 ([1da94e6](https://github.com/vllm-project/vllm/commit/1da94e6)), **vLLM Ascend Version**: v0.10.1rc1 ([7e16b4a](https://github.com/vllm-project/vllm-ascend/commit/7e16b4a))  
- **Software Environment**: **CANN**: 8.2.RC1, **PyTorch**: 2.7.1, **torch-npu**: 2.7.1.dev20250724  
- **Hardware Environment**: Atlas A2 Series  
- **Parallel mode**: TP2
- **Execution mode**: ACLGraph

**Command**:  

```bash
export MODEL_ARGS='pretrained=deepseek-ai/DeepSeek-V2-Lite,tensor_parallel_size=2,dtype=auto,trust_remote_code=True,max_model_len=4096,enforce_eager=True'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks gsm8k \
    --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
| gsm8k | exact_match,strict-match | ✅0.3813 | ± 0.0134 |
| gsm8k | exact_match,flexible-extract | ✅0.3836 | ± 0.0134 |
