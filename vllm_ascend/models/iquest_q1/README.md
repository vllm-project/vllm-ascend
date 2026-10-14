# IQuest-Q1 on Ascend NPU (v0)

Ascend port of `IQuestLab/IQuest-Q1` (320B MoE, `model_type: iquest_q1`).
Upstream vLLM does not ship this architecture.

## Files

| File | Origin |
|---|---|
| `configs.py` | Ported verbatim from the upstream plugin `IQuestLab/vllm-iquest-q1` |
| `model.py` | Ported from the same plugin; only the attention layer was re-pointed |
| `attention.py` | **New** — Ascend learned-key sink attention (replaces the plugin's CUDA FA backend) |
| `__init__.py`, `README.md` | New |

## How the sink works here

IQuest-Q1 gives every layer a learned *sink key* `sink_k` of shape `(num_kv_heads, head_dim)`.
It behaves as an always-visible, zero-valued KV entry, so it only rescales the
softmax denominator:

```
sink_logit = scale * (q . sink_k[head // num_queries_per_kv])
out        = out_raw * sigmoid(-(sink_logit - lse))
```

The CANN FIA operand `learnable_sink` cannot express this: it is a per-query-head
**constant** bfloat16 tensor of shape `(Q_N,)` (the gpt-oss / DeepSeek-V4 style sink),
whereas `q . sink_k` varies per token and per head. So instead we ask
`torch_npu.npu_fused_infer_attention_score_v2(..., return_softmax_lse=True)` for the
real softmax LSE — `softmax_max + log(softmax_sum)`, shape `(T, Q_N, 1)` for TND —
and apply the correction with plain torch ops (see `_apply_sink_key`).

No custom Ascend kernel is needed at this stage. A Triton-ascend fusion of
`q·sink_k → exp → div` is the obvious next optimization.

## v0 scope / limitations

- **Eager only.** The sink path is not ACL-graph compatible; the impl raises
  `NotImplementedError` when `_EXTRA_CTX.capturing`. **Always pass `--enforce-eager`.**
- The paged-attention decode path is bypassed (it cannot return the LSE), so every
  layer goes through FIA. Expect decode to be slower than a native GQA model.
- No MTP / speculative decoding (the checkpoint ships an `mtp/` recursive draft).
- No quantization (bf16 only), no LoRA weights, no context parallelism.
- No reasoning / tool parser registered (upstream ships `iquest_q1` parsers).

## Running

```bash
MODEL_ROOT=/path/to/IQuest-Q1

vllm serve "$MODEL_ROOT" \
  --served-model-name IQuest-Q1 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --max-model-len 8192        # start small; 512K needs real KV budget planning
```

OpenAI-compatible smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"IQuest-Q1","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"temperature":0}'
```

## Verification checklist (in order)

1. **Loads at all** — config parsing, 175 shards, expert weight mapping.
   Watch for: `IQuestQ1ForCausalLM` resolving; `sinks_k_weight_loader` sharding `sink_k` by TP.
2. **FIA accepts `return_softmax_lse=True` here** — this is the single unverified
   assumption (see the plan doc §4.3). Confirm it works for both
   `sparse_mode=4` (the 63 sliding-window layers) and `sparse_mode=3` (the 25 full layers),
   and for both prefill and paged decode.
3. **Sink is actually applied** — construct a 1-token prompt and compare against a
   run with `enable_sink_attention` forced off; the outputs must differ.
4. **Token-level agreement** vs the GPU reference (SGLang image
   `iquestlabworkspace/sglang-iquest-q1:cu130` or HF transformers 4.57.1) at
   `temperature=0`.

## Known risk areas

- `"sink" in name` in `worker/model_runner_v1.py` flips `_has_sinks` and sets
  `forward_context.sinks`. That flag is currently **not consumed** anywhere in
  vllm-ascend, so it should be inert — but worth re-checking if attention misbehaves.
- ACL graph is not supported by design here; if vLLM tries to capture anyway,
  the `NotImplementedError` in `forward_fused_infer_attention` will fire.
