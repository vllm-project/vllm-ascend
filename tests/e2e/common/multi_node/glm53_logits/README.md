# GLM-5.3 A5 nightly logits regression

Five two-node cases compare each configuration with its own immutable capture:
`disable_megamoe`, `disable_pp`, `disable_flashcomm1`, `disable_lic8`, and
`disable_sfac8`. They do **not** compare disabled features with the all-on case.

## Model and topology

Provision `GLM-5.3-W8A8C8-MXFP8-1D7M1MTP` on both CI pods at the `model` path
in the case YAML. The model is approximately 85.43 GiB. This PR does not ship
checkpoint weights or a checkpoint reduction script. An incomplete or different
checkpoint fails before engine startup. Config, weight index, quantization
description and tokenizer hashes must match `golden/manifest.json`; these are
identity checks, not a fresh checksum of all weight payloads on every run.

The checkpoint has eight backbone layers and one MTP layer. P uses
TP2/PCP2/PP2 with `VLLM_PP_LAYER_PARTITION=4,4`; D uses TP2/DP4, for 16 devices
total. Disabling PP keeps TP2/PCP2 unchanged, uses four P devices and eight D
devices, and removes the manual partition. MTP is outside the backbone partition.
All cases use MRV2 and MooncakeConnectorV2, with no DSACP configuration.

## Nightly integration

The checkpoint and baseline assets must be mounted on both pods.
The five YAMLs in `tests/e2e/cases/models/configs/GLM` are registered in the A5
nightly matrix. The common multi-node pytest entry dispatches configs containing
`logits_regression` to this runner. LWS provides the two pod addresses and node
indices; no host credentials, fixed machine addresses, or SSH are needed.

For a standalone two-host reproduction in prepared NPU containers, copy one
case YAML, set its existing model path and `cluster_hosts: [P_IP, D_IP]`, and run
on each host from the repository root:

```bash
# Use 0 on the P host, 1 on the D host. LOG_PREFIX must be a fresh run directory.
LWS_WORKER_INDEX=0 LOG_PREFIX=/tmp/glm53-logits-run \
  python -m tests.e2e.common.multi_node.glm53_logits.runner /path/to/case.yaml
```

Both nodes run the same command with their own index. The leader checks every
worker's effective MRV2/PP/PCP/TP/DP, MegaMoE, FlashComm1, LIC8 and SFAC8 state,
then makes real P-to-D KV handoffs. Transfer failure is fatal; local recomputation
must not substitute for the PD path. An HTTP artifact endpoint retrieves D-side
captures even without a shared filesystem. The leader returns the test outcome
to D; missing completion, startup failure and timeouts fail the test. Only the
server process group created by the runner is stopped.

Reports, effective config, raw arrays and server logs are written beneath
`LOG_PREFIX/glm53_logits/<case>/<role>/<run-id>`. Pytest propagates failure into
the existing nightly result handling. A changed return string alone is not the
numerical oracle.

## Baseline and gate

The manifest identifies the arrays accepted by the case owner on
2026-11-02. Each of four fixed input lengths (64, 512, 4096 and 16384 tokens)
has one P position and eight D positions over the full 154880-token vocabulary.
SHA256 hashes identify every capture and the fixed input/history corpus.
Provision the 40 arrays under the YAML
`logits_regression.baseline_path`, preserving the `<case>/<sample>-<role>.npz`
layout. Binary arrays are CI assets, not Git payloads. Before enabling the cases,
validate the provisioned bundle with:

```bash
python -m tests.e2e.common.multi_node.glm53_logits.verify_baseline \
  /root/.cache/glm53-logits-golden/20261102
```

Missing, corrupt, incomplete, non-finite or misaligned captures fail closed.
There is no automatic recording, downloading, or replacement of goldens in CI.

`FixedTraceProcessor` uses the MRV2 extension point to fix token histories while
the sampler preserves raw logits before processor masking. This keeps MTP
enabled (P1/D5). The verified vLLM revision rejects its built-in
`trace_decode_token_ids` with MTP, so that interface is not used. Comparing forced
output IDs alone would incorrectly pass every request; the oracle is the full
unmodified score vector. The all-on free-generation versus restarted fixed-trace
experiment reproduced every measured score exactly.

Same-configuration gates are centered relative L2 <= 0.02, KL(reference ||
candidate) <= 0.01, centered cosine >= 0.999, and zero raw-top1 changes at
reference margins >= 0.5. These initial engineering limits are kept in the
manifest; they are not task-accuracy guarantees or calibrated concurrency
bounds. In particular, C8 cases do not receive a looser cross-precision budget.
Changing a baseline or a threshold requires an explicit reviewed change.

## Provenance and limits

Captures used Ascend main `6d2866b7175849ab712d7ccb16835589562b581e` and its verified
vLLM main `544f182d9f0c9bfe8bd7645525951d5e3692a542`, on two eight-device A5 hosts.
Python sources overlaid an existing image; native extensions came from that
image. Image ID and package versions are recorded in the manifest. They are not
measurements of a fresh complete build of today's main.

The paired main left `fp8` mapped to `torch.uint8` even though the registration
API existed, causing MRV2 C8 scale-cache initialization to fail. The captures
used the narrow fallback included in this PR: change only an unregistered
raw-byte FP8 mapping to native FP8, preserving an existing native/custom handler.
Candidate runs use their installed runtime; the harness does not rewrite it.

The five source experiments completed real PD inference. Differences between
on/off configurations are intentionally not nightly pass/fail criteria.
This suite covers reduced W8A8 numerical/function regression, not W4A4, full-model
GPQA, throughput, multi-request concurrency or exhaustive DP routing. CI runner
checkpoint provisioning and the complete nightly image remain prerequisites.
