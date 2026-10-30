# DFlash2 ACL graphs

DFlash2 uses Model Runner V2 and the same draft graph policy as upstream vLLM.
Set `speculative_config.enforce_eager` to `false` to allow draft capture.
`FULL_DECODE_ONLY`, `FULL`, and `FULL_AND_PIECEWISE` use a full draft graph when
supported by the draft attention backend. The graph includes context KV storage,
the draft forward, candidate top-k, selector scoring, sampling, and proposal cache
updates. Capture sizes are rounded to whole draft query blocks.

The mode policy is inherited from the local upstream implementation in
`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`; DFlash2 does not override it.
The following table describes the **draft** graph, independently of target graphs:

| Configured mode | Upstream vLLM DFlash2 | Ascend DFlash2 |
| --- | --- | --- |
| `NONE` | Eager | Eager |
| `PIECEWISE` | Eager | Eager |
| `FULL_DECODE_ONLY` | Full decode graph | Full decode ACL graph |
| `FULL_AND_PIECEWISE` | Full decode graph | Full decode ACL graph |
| `FULL` | Full decode graph | Full decode ACL graph |

Full graphs require draft attention support of at least `UNIFORM_BATCH`.
A graph request with unsupported attention falls back to eager, as upstream does.
`speculative_config.enforce_eager=true` also forces eager drafting on Ascend.

With `PIECEWISE`, the target uses piecewise graphs and the draft runs eagerly,
matching upstream DFlash's current policy. Unsupported full-graph attention also
falls back to eager drafting. Setting `speculative_config.enforce_eager` to `true`
disables draft graphs independently of the target.

The draft disables Torch compilation to avoid symbolic-shape failures in NPU
RoPE meta kernels; full ACL capture records the NPU operations directly. Target
compilation settings are preserved. V1 remains available through
`VLLM_USE_V2_MODEL_RUNNER=0`.

Validate eager/graph output parity, acceptance rate, and latency on the deployment
NPU, including partial batches, request reordering, and mixed prefill/decode.

DFlash2 probability sampling uses upstream seed/step/position hashing and the
stable FP32 Gumbel transform in both eager and graph execution. This changes
seeded probabilistic outputs compared with the previous Ascend Philox-based
selector. FP64 Gumbel sampling remains unsupported on NPU.

Non-causal sliding-window draft layers retain the symmetric window from the
checkpoint in both eager and graph execution. FIA uses its compressed band-mask
template with matching left/right bounds; removing the mask and selecting full
attention would expose cache entries outside the window and alter draft scores.
