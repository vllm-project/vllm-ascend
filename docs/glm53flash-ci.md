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

## Implementation

`tests/e2e/pull_request/four_card/test_glm5_3_flash.py` constructs the temporary
model metadata, initializes bounded synthetic weights, starts the existing
VllmRunner and checks request completion. The custom dummy initializer is
test-local, including recurrent-state parameters, mHC and positive quant scales.
It is retained for Flash initialization safety rather than baseline capture.

`glm53flash_assets/config.json` is the retained 0–8 text-layer configuration
from the local Eco-Tech/GLM-5.3-Flash-w8a8 metadata snapshot collected on
2026-09-22. `quant.json` holds metadata templates for original layers 0, 3 and 4
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

The exact rescaled PR still needs clean-runtime A3 validation. Earlier A3 runs
used additional runtime fixes and cannot establish that this standalone PR
works on unmodified main. Failure must remain visible: do not silently skip,
xfail, or install runtime patches from inside the test. The four-card directory
is the existing CI entry. The 900-second scheduling estimate is provisional,
not a measured performance threshold; calibrate it after a clean run.

## Deferred work

Logits capture/comparison, fingerprints as baseline identities, throughput and
latency gates, shared deployment profiles, HTTP/multimodal probes, MTP and full
model nightly are outside this PR. The previous implementation is preserved on
local branch `codex/glm53flash-gates-pending-20260924` at commit
`8afaeb6669873535294de2eeb59201a65c43750a` and in the main local worktree.

Runtime, cache-manager and torch-binding changes are separate review work in
PR #17281. They are not included or automatically applied here. Review their
necessity and architecture independently before claiming clean-main support.
