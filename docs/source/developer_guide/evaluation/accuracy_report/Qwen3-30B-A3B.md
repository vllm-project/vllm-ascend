# Qwen/Qwen3-30B-A3B

**vLLM Version**: vLLM: 0.10.0 ([6d8d0a2](https://github.com/vllm-project/vllm/commit/6d8d0a2)),
**vLLM Ascend Version**: v0.10.0rc1 ([4604882](https://github.com/vllm-project/vllm-ascend/commit/4604882))  
**Software Environment**: CANN: 8.2.RC1, PyTorch: 2.7.1, torch-npu: 2.7.1.dev20250724  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: gsm8k,ceval-valid  
**Parallel Mode**: TP  
**Execution Mode**: ACLGraph  

**Command**:  

```bash
export MODEL_ARGS='pretrained=Qwen/Qwen3-30B-A3B,tensor_parallel_size=2,dtype=auto,trust_remote_code=False,max_model_len=4096,gpu_memory_utilization=0.6,enable_expert_parallel=True'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks gsm8k,ceval-valid \
--apply_chat_template False --fewshot_as_multiturn False  --num_fewshot 5  \
--limit None --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
|                   gsm8k | exact_match,strict-match |✅0.9006823351023503 | ± 0.0082 |
|                   gsm8k | exact_match,flexible-extract |✅0.8574677786201668 | ± 0.0096 |
|             ceval-valid |        acc,none |✅0.837295690936107 | ± 0.0099 |
