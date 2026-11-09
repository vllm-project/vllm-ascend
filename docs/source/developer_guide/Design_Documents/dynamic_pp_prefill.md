# Experimental dynamic PP prefill

This prototype connects Ascend Model Runner V2 to the shared vLLM activation
buffer and dynamic PP runtime. Each PP stage loads overlapping DSV4 layers and
executes the range carried with each batch. Device-completed producer work,
consumer credits, and matched stage timings guide the split. The default is
unchanged: no output ring and no boundary movement.

## Dependencies and scope

Install the matching [vLLM dynamic PP code branch](https://github.com/vllm-project/vllm/pull/63745) before enabling this feature. The
shared implementation lives in `vllm/v1/worker/pp/`; this plugin does not maintain
another controller or buffer implementation. Ordinary configurations do not
import that optional module.

The first adapter supports DeepSeek-V4 complete fresh prefills with one output
token, eager execution, PP=2 and TP. Prefix caching, chunked prefill, decode, KV
connectors and pooling, speculation, LoRA, DP, EP, sequence/context parallelism,
and microbatching are outside this version. Historical KV replication is not
implemented. The runtime rejects requests that would need unsynchronized KV.

For reproducible allocation-independent C4 results, apply the independent
[physical KV page order fix](https://github.com/vllm-project/vllm-ascend/pull/19864).
It changes attention rounding, and is not included in this feature PR.

## Run

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/DeepSeek-V4-Flash-w4a8 \
    --tensor-parallel-size 4 --pipeline-parallel-size 2 \
    --quantization ascend --tokenizer-mode deepseek_v4 --block-size 128 \
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill \
    --max-model-len 20992 --max-num-batched-tokens 20480 \
    --additional-config '{
      "enable_dsa_cp": false,
      "enable_flashcomm1": false,
      "pp_send_buffer_size": 2,
      "dynamic_pp_config": {
        "enabled": true,
        "redundant_layers": 3,
        "initial_offset": -3,
        "cooldown_steps": 8,
        "high_watermark": 2,
        "log_interval_seconds": 10
      }
    }'
```

Use `max_tokens=1` for every request. Each prompt must fit in one scheduled
batch. `initial_offset=-3` starts PP0 with three fewer layers; `redundant_layers`
bounds both directions of movement. Set `adaptive=false` to hold a cut for
comparison. Set only `pp_send_buffer_size` for buffering with a fixed partition.

The buffer reserves independent output storage before profiling assigns the KV
budget. More slots and replicated weights reduce the memory left for KV and
add activation copies. Buffering does not support sleep mode.

## Observe and validate

`Dynamic PP strategy` logs report the executed split, version, completed queue
watermark, stage timings, decision reason, and split counts over the reporting
window. Submitted send count is separate from ready data. Device events are
polled without a host read of device tensors. Timing feedback includes PP1's
logits and sampling work. Hash-routing layers stay on PP0, which owns token IDs.

Compare fixed, buffered, forced-cut, and adaptive runs with the same request
packing. A stable split is specific to the prompt/batch shape and device load.
Layer indivisibility, communication, startup, and drain can still create bubbles;
this feature does not promise a completely bubble-free pipeline.

The two-card regression delays the consumer until two outputs have been posted,
then checks twelve changing-size payloads and paired device timings. The
four-card model regression compares actual DSV4 checkpoint outputs with fixed
PP, including both cut directions and adaptive feedback. Performance claims
require a separate end-to-end comparison under sustained load.
