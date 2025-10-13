# Qwen/Qwen2.5-VL-7B-Instruct

- **vLLM Version**: vLLM: 0.10.1.1 ([1da94e6](https://github.com/vllm-project/vllm/commit/1da94e6)), **vLLM Ascend Version**: v0.10.1rc1 ([7e16b4a](https://github.com/vllm-project/vllm-ascend/commit/7e16b4a))  
- **Software Environment**: **CANN**: 8.2.RC1, **PyTorch**: 2.7.1, **torch-npu**: 2.7.1.dev20250724  
- **Hardware Environment**: Atlas A2 Series  
- **Parallel mode**: TP1
- **Execution mode**: ACLGraph

**Command**:  

```bash
export MODEL_ARGS='pretrained=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=1,dtype=auto,trust_remote_code=False,max_model_len=8192'
lm_eval --model vllm-vlm --model_args $MODEL_ARGS --tasks mmmu_val \
 --apply_chat_template True   --fewshot_as_multiturn True    --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
| mmmu_val | acc,none | ✅0.52 | ± 0.0162 |
