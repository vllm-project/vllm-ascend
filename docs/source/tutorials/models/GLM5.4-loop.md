# GLM-5.4 loop profiling on Atlas 850E

This experimental text decoder uses the 78-layer GLM-5.3 W4A4C8/MXFP4
checkpoint to exercise the [published GLM-5.4 loop design](https://www.zhipuai.cn/zh/news/164).
It executes layers 26 through 53 (zero-based) twice, sharing the attention
and expert weights. These boundaries align with seven complete GLM-5.3
IndexShare groups. Each group retains its first-pass retrieval indices for
the second pass. There are 106 decoder-layer calls per forward.

The checkpoint has no trained loop gates. Both per-token gates start at
zero logits and write back half of their block's update. The reference
implementation chooses a 128-token causal window. In pass two, sparse
attention reads the first-pass global cache and the second-pass local
cache, merging their outputs with their softmax normalization statistics.
Both representations can contribute at overlapping positions.

## Deployment

Use the [single-node GLM-5.3 950DT environment and checkpoint](GLM5.2.md#glm52-53-5-1-1-950dt).
The example uses eight 96 GiB devices, TP8, EP, eager execution and
colocated prefill/decode. Set `MODEL_PATH` to the local GLM-5.3
W4A4C8/MXFP4 checkpoint directory.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve "$MODEL_PATH" --served-model-name glm54-loop \
    --tensor-parallel-size 8 --enable-expert-parallel --quantization ascend \
    --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8 \
    --hf-overrides '{"architectures":["Glm54LoopProfileForCausalLM"]}' \
    --enforce-eager --max-model-len 4096 --max-num-seqs 8 \
    --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 --trust-remote-code
```

```bash
curl --fail-with-body http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm54-loop","prompt":"Explain a loop.","temperature":0,"max_tokens":8,"ignore_eos":true}'
```

## Interpreting measurements

For a GLM-5.3 control run, remove the architecture override and keep the
checkpoint, eager mode, parallelism, cache dtypes, scheduler settings and
request shapes identical. Measure prefill and decode separately. Each
replayed layer uses two sparse attention calls and an explicit merge.
The allocator reserves ordinary paged storage for the local bank, even
though attention reads only a window. These choices affect latency and KV
capacity. The experiment targets operator profiling. Output quality and
production GLM-5.4 throughput require the trained model and its serving implementation.
