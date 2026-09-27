# GLM-5.3-Flash functional CI — phase one

This PR follows the K3 local-config/dummy-weight/VllmRunner pattern. It is a
functional smoke test, not an accuracy or performance gate. It does not require
a checkpoint path or download tokenizer assets.

## Scope

- Nine main-model layers: seven KDA and two sparse-attention layers; the first
  three FFNs are dense and the remaining six are MoE. Production widths and
  288 routed experts are retained. mHC is enabled; MTP is disabled.
- TP4 with EP and W8A8 dynamic FFNs; attention remains floating point according
  to the checkpoint metadata. No additional C8 switches are enabled.
- Separate eager and FULL_DECODE_ONLY configurations; single and mixed-batch
  generation must finish with eight valid token IDs per request.
- Each worker checks the instantiated layer types, mHC and W8A8 MoE method.
  This does not prove every kernel was used or that graph replay occurred.
- The reduced config retains the production multimodal wrapper and vision tower,
  which initializes before the text layers. Image/video inputs are disabled in
  this text-only smoke; their preprocessing needs a separate test.

## Implementation

`tests/e2e/pull_request/four_card/test_glm5_3_flash.py` constructs the temporary
model metadata, initializes bounded synthetic weights, starts the existing
VllmRunner and checks request completion. The custom dummy initializer is
test-local, including recurrent-state parameters, mHC and positive quant scales.
It is retained for Flash initialization safety rather than baseline capture.

`glm53flash_assets/config.json` is the retained 0–8 text-layer configuration
from the local Eco-Tech/GLM-5.3-Flash-w8a8 metadata snapshot collected on
2026-09-22. `multimodal.json` preserves the matching production outer model and
vision config. A tiny synthetic tokenizer and processor metadata allow the
wrapper to initialize without downloading checkpoint tokenizer files. They do
not validate real tokenization or image/video handling. `quant.json` holds
metadata templates for original layers 0, 3 and 4
(dense KDA, sparse-attention MoE and KDA MoE), including expert zero. The test
expands these to all nine layers and 288 experts. This is metadata only; no
trained weights are included. Dummy weights do not validate checkpoint loading
or the checkpoint's rotation preprocessing.

## Run and validation status

On an available A3 with four logical NPUs and compatible vLLM/vllm-ascend:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 pytest -sv \
  tests/e2e/pull_request/four_card/test_glm5_3_flash.py
```

On 2026-09-27, this multimodal-wrapper smoke passed three consecutive TP4
runs per mode (all exit 0, one test passed per run) on 80.5.9.103. Physical
`/dev/davinci4-7` mapped to container devices 0-3; all eight physical NPUs
had no processes before each run. The image was the September 23
`nightly-main-a3` build (ID `sha256:95af5d4298697fe6b0228e9e3f506e682e997d41940634978c3b1e7c4943c984`).
The staged test file SHA256 was
`9beeea8a00942a5d5531595367dec7824985831a1bc23b6155a3074ddf286738`.
The runtime source was the unmodified `d2a4b485c` tree, with no #17552 NoPE
guard or #17281 draft runtime changes; the image's prebuilt native extension
and custom-op installation artifacts were used. This does not establish a
pass on the latest main runtime or with a newly built native extension.

| Mode | Wall seconds, runs 1/2/3 | Mean wall seconds |
| --- | --- | ---: |
| Eager | 114.586 / 115.573 / 116.263 | 115.474 |
| Graph | 118.469 / 115.928 / 118.441 | 117.613 |

Running both configurations sequentially averaged 233.087 wall seconds.
These are end-to-end test wall times, **not** inference latency or throughput
measurements. A prior single run took 133 seconds eager and 117 seconds graph.
The six repeated-run logs, exit codes, wall times, pre-run NPU snapshots and
JSON summary are in the task directory on 103 with prefix
`glm53flash-103-pr-b45c2f948-bench-r2-20260927`. An earlier measurement
attempt accidentally replaced the image's CANN `PYTHONPATH` and failed to
import `acl` before model initialization; its log remains under the same
prefix without `-r2` and is excluded from the successful-run mean.

The 420-second CI scheduling estimate allows headroom over the measured
combined time; it is not a performance threshold. Failure must remain
visible: do not silently skip, xfail, or install runtime patches from inside
the test. The four-card directory is the existing CI entry.

The old text-only synthetic config hit zero-width NoPE RoPE initialization.
The production multimodal wrapper initializes a positive-width vision RoPE
cache first, so this smoke now follows the same successful initialization
order as formal serving. That does not make zero-width NoPE initialization
safe in every other configuration; the separate focused fix remains relevant.

## Deferred work

Logits capture/comparison, fingerprints as baseline identities, throughput and
latency gates, shared deployment profiles, HTTP/multimodal probes, MTP and full
model nightly are outside this PR. The previous implementation is preserved on
local branch `codex/glm53flash-gates-pending-20260924` at commit
`8afaeb6669873535294de2eeb59201a65c43750a` and in the main local worktree.

Runtime, cache-manager and torch-binding changes are separate review work in
PR #17281. They are not included or automatically applied here. Review their
necessity and architecture independently before claiming clean-main support.
