# Qwen/Qwen3-8B-Base

**vLLM Version**: vLLM: 0.10.0 ([6d8d0a2](https://github.com/vllm-project/vllm/commit/6d8d0a2)),
**vLLM Ascend Version**: v0.10.0rc1 ([4604882](https://github.com/vllm-project/vllm-ascend/commit/4604882))  
**Software Environment**: CANN: 8.2.RC1, PyTorch: 2.7.1, torch-npu: 2.7.1.dev20250724  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: gsm8k,ceval-valid  
**Parallel Mode**: TP  
**Execution Mode**: ACLGraph  

**Command**:  

```bash
export MODEL_ARGS='pretrained=Qwen/Qwen3-8B-Base,tensor_parallel_size=1,dtype=auto,trust_remote_code=False,max_model_len=4096'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks gsm8k,ceval-valid \
--apply_chat_template True --fewshot_as_multiturn True  --num_fewshot 5  \
--limit None --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
|                   gsm8k | exact_match,strict-match |✅0.8278999241849886 | ± 0.0104 |
|                   gsm8k | exact_match,flexible-extract |✅0.8294162244124337 | ± 0.0104 |
|             ceval-valid |        acc,none |✅0.8179791976225854 | ± 0.0103 |
