# Qwen/Qwen2.5-VL-7B-Instruct

**vLLM Version**: vLLM: 0.10.0 ([6d8d0a2](https://github.com/vllm-project/vllm/commit/6d8d0a2)),
**vLLM Ascend Version**: v0.10.0rc1 ([4604882](https://github.com/vllm-project/vllm-ascend/commit/4604882))  
**Software Environment**: CANN: 8.2.RC1, PyTorch: 2.7.1, torch-npu: 2.7.1.dev20250724  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: mmmu_val  
**Parallel Mode**: TP  
**Execution Mode**: ACLGraph  

**Command**:  

```bash
export MODEL_ARGS='pretrained=Qwen/Qwen2.5-VL-7B-Instruct,tensor_parallel_size=1,dtype=auto,trust_remote_code=False,max_model_len=8192'
lm_eval --model vllm-vlm --model_args $MODEL_ARGS --tasks mmmu_val \
--apply_chat_template True --fewshot_as_multiturn True  \
--limit None --batch_size auto
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
|                mmmu_val |        acc,none |✅0.5211111111111111 | ± 0.0162 |
