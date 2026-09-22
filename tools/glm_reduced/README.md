# GLM reduced-checkpoint builder (`tools/glm_reduced`)

Builds layer-reduced ("cropped") copies of real GLM checkpoints for
numerical-precision and performance gates on vllm-ascend. The same reduced
checkpoint feeds both gates.

For the non-Flash GLM5.x **11-layer, >=90% A1 argmax agreement** nightly case,
see [the fixed GSM8K logits gate](data/glm52/logits/README.md). This is a separate
intermediate-logits regression gate, not GSM8K answer accuracy. The currently
registered real-weight case is GLM-5.2 W4A8.

Design properties:

- **Real weights, production dimensions.** Only decoder layers are dropped;
  hidden sizes, head counts, expert counts and quantization metadata are
  preserved. There is no dummy-weight mode in the build path — the synthetic
  checkpoints under `tests/ut/tools/glm_reduced/` are transformation fixtures
  only and are never numerical or performance validation.
- **Streaming.** Safetensors shards are read tensor-by-tensor and copied as
  raw bytes (dtype-agnostic, BF16/FP8 included); the full checkpoint is never
  loaded into memory.
- **Strict parsing.** Malformed headers (duplicate keys, negative/fractional/
  boolean offsets or dimensions, overlapping ranges, truncated data sections)
  are rejected before any byte is copied.
- **Safe.** The source directory is never modified; the output must not exist
  and must not overlap the source (equal, nested, or parent); output is staged
  in a sibling directory and published atomically; index-declared shard paths
  must be plain basenames (no traversal); auxiliary symlinks are resolved
  read-only (HF cache layouts) and never followed for writes.
- **Complete or not at all.** Every retained decoder layer, every declared
  MTP layer, embeddings, final norm and lm_head (unless tied) must be present
  before anything is published.
- **Auditable.** Every build writes `reduction_manifest.json` recording source
  identity (config/index SHA-256, weight file names+sizes, selective-source
  state), the layer mapping (including MTP remap), quantization handling,
  output inventory and per-file plus per-tensor SHA-256 checksums.
- **Explicit failures.** Unknown tensor names, unknown per-layer config
  arrays, non-safetensors weights, missing tokenizer files, unsupported
  quantization schemes, duplicate tensors and incomplete sources all abort
  with actionable messages instead of silently emitting a corrupt model.

## Model inventory

Run `python -m tools.glm_reduced inventory` for the authoritative table, built
from the repo support matrix, tutorials, model registrations, e2e YAML
configs, the model dataset manifest, and the pinned public source descriptors
in `tools/glm_reduced/sources.json` (model IDs, revisions, config/index URLs,
tensor counts, sizes).

Unsupported entries are explicit: the old **GLM-4V** line (support matrix ❌,
issue #2260 — kept distinct from GLM-4.1V), **zai-org/GLM-Image** (root
config.json 404 at review time; not a plain causal-LM checkpoint), and
**RedHatAI/GLM-5.2-speculator.dspark** (draft-only artifact).

## Profiles

Profiles are declarative, per architecture family (`profiles.py`), with
weight-name layouts verified against the pinned upstream index snapshots:

| Profile | Architectures | Layout | Min prefix | Notes |
|---|---|---|---|---|
| `glm4-moe` | Glm4MoeForCausalLM (GLM-4.5/4.6/4.7 + quant variants) | flat | 8 | dense→sparse MLP transition + routed MoE layers |
| `glm4-moe-lite` | Glm4MoeLiteForCausalLM (GLM-4.7-Flash) | flat | 4 | MLA + 64-expert MoE, first_k_dense=1, MTP |
| `glm-moe-dsa` | GlmMoeDsaForCausalLM (GLM-5/5.1/5.2/5.3 + variants) | flat | 8 | truncates `indexer_types`/`mlp_layer_types`; validates shared-indexer producer closure (GLM-5.2/5.3); GLM-5.3 is natively FP8 |
| `glm5-next` | Glm5NextForConditionalGeneration (GLM-5.3-Flash) | nested `text_config` | 8 | KDA/DSA hybrid cycles, mHC, keypool, MTP remap; native FP8; `model.visual.*` kept in full |
| `glm4v` | Glm4vForConditionalGeneration (GLM-4.1V-9B-Thinking) | nested `text_config` | 4 | dense 40-layer text tower `model.language_model.*` + `model.visual.*` kept in full; distinct from the unsupported old GLM-4V line |
| `chatglm` | ChatGLMModel (chatglm3-6b, glm-4-9b-chat) | flat | 4 | `transformer.encoder.layers.*`; `num_layers` and `num_hidden_layers` must agree and are both updated |

Layer reduction keeps a **prefix**. Suffix cropping is rejected: it would
detach GLM-5.2/5.3 `shared` indexer layers from the `full` producer layers
whose weights/top-k indices they reuse (see
`vllm_ascend/patch/worker/patch_deepseek_v2.py`), and a three-layer-only
fixture would miss MoE and sparse-attention paths entirely. MTP layers
(`num_nextn_predict_layers`) are kept by default and remapped to follow the
new prefix.

Loading routes: GLM-5.x/5.3-Flash load through vllm_ascend registrations;
GLM-4.x, GLM-4.1V and ChatGLM load through upstream vLLM's registry under the
Ascend platform (no Ascend-local registration is required for a valid route;
upstream carries a GLM-4.1V processing test). Runtime qualification of reduced
checkpoints on NPU is a separate, explicitly pending step.

## Public source retrieval (no blind multi-TB downloads)

`tools/glm_reduced/sources.json` pins model IDs, revisions, config/index URLs
and sizes. Fetch only the small metadata first, compute the exact shard set a
reduction needs, then fetch just those shards:

```bash
# 1. Pinned descriptors
python -m tools.glm_reduced sources --model zai-org/GLM-5.2

# 2. Metadata only (small)
hf download zai-org/GLM-5.2 --revision <pinned> \
    --include "config.json" "*.index.json" "tokenizer*" "quant_model_description.json"

# 3. Which shards does an 8-layer reduction actually need?
python -m tools.glm_reduced required-shards \
    --config <dir>/config.json --index <dir>/model.safetensors.index.json \
    --profile glm-moe-dsa --layers 8

# 4. Fetch only those shards, then build in explicit selective mode
hf download zai-org/GLM-5.2 --revision <pinned> --include "<shard1>" "<shard2>" ...
python -m tools.glm_reduced build <dir> <out> --profile glm-moe-dsa --layers 8 --selective
```

`--selective` is explicit: the full original index is classified first, every
required shard must be present with all its indexed tensors, absent
dropped-only shards are recorded in the manifest, and the original index
digest is preserved. Without `--selective`, a partially downloaded source
fails the strict completeness checks.

## Usage

```bash
# Dry-run: tensor classification counts, kept bytes, warnings
python -m tools.glm_reduced plan /path/to/GLM-5.2 --profile glm-moe-dsa --layers 8

# Build (source is never modified; output must not exist)
python -m tools.glm_reduced build /path/to/GLM-5.2 /path/to/GLM-5.2-reduced8 \
    --profile glm-moe-dsa --layers 8

# Re-verify output against its manifest (checksums, index, config, inventory)
python -m tools.glm_reduced verify /path/to/GLM-5.2-reduced8
```

Quantized checkpoints:

- `quantization_config` with `quant_method: fp8` (blockwise; native in
  GLM-5.3 and GLM-5.3-Flash) is supported: per-tensor `weight_scale_inv`
  companions follow their weights, and `modules_to_not_convert`/
  `ignored_layers` per-layer entries are filtered/remapped.
- ModelSlim-style `quant_model_description.json` (Eco-Tech w8a8/w4a8/w8a8c8
  variants) is supported: per-layer keys are filtered/remapped atomically
  (values never re-paired), global metadata (`group_size`, `metadata`,
  `optional`, `version`, `is_rot_used`) is preserved, and referenced auxiliary
  safetensors (e.g. `optional/quarot.safetensors`) are copied verbatim and
  checksummed. The Ascend quant index name
  `quant_model_weights.safetensors.index.json` is recognized and never copied
  as a stale aux file.
- Any other scheme (packed MXFP4, AWQ, GPTQ, ...) fails with
  `UnsupportedQuantError` explaining why, instead of emitting a corrupt model.

## Precision gate

Two runs of the **same checkpoint** (identical `checkpoint_id`, distinct
`run_id`s) with a fixed prompt set, compared with caller-supplied tolerances:

```bash
python -m tools.glm_reduced.run_logits_dump REDUCED --profile glm-moe-dsa \
    --mode eager --enforce-eager --output baseline.jsonl
python -m tools.glm_reduced.run_logits_dump REDUCED --profile glm-moe-dsa \
    --mode graph --output candidate.jsonl
python -m tools.glm_reduced compare-logits baseline.jsonl candidate.jsonl \
    --atol <per-profile> --rtol <per-profile>
```

The gate requires: matching checkpoint ids and seeds; complete dumps
(prompt_count records, unique prompt indices, exactly output_tokens produced
per record — truncation on both sides fails); identical prompt token ids and
greedy token sequences (first divergence reported); sampled token present in
both top-k sets; at least one compared logprob pair; finite values and finite
tolerances. Independent A/A runs (identical settings, different run ids) are
valid noise characterization. Tolerances have no defaults. Logit parity alone
does not prove semantic task accuracy; it complements the accuracy suites in
`tests/e2e/`.

## Performance gate

The regression gate compares two runs of the **same checkpoint** — typically
one reduced checkpoint across two runtime revisions — with identical dtype
and engine settings (TP/EP/caching policy):

```bash
python -m tools.glm_reduced.run_perf REDUCED --profile glm-moe-dsa \
    --hardware <tag> --mode baseline-rev-A --output perf-baseline.json
python -m tools.glm_reduced.run_perf REDUCED --profile glm-moe-dsa \
    --hardware <tag> --mode candidate-rev-B --output perf-candidate.json
python -m tools.glm_reduced compare-perf perf-baseline.json perf-candidate.json \
    --max-latency-regression-pct <per-profile/hardware>
```

`compare-perf` hard-fails on: missing/malformed baselines; checkpoint,
workload, hardware, dtype or engine mismatches (a TP=1 vs TP=8 comparison is
not a regression); missing thresholds (a comparison without an explicit
pass/fail criterion is not a gate); and token-count violations (early EOS or
cache-shortened work). Full-vs-reduced latency ratios measure the crop, not a
regression, and are rejected via checkpoint identity. Prefix caching is
disabled by default in the runner so repeated iterations measure equal work.

## Scope notes and runtime boundary

- `run_logits_dump.py` and `run_perf.py` are the only components that need a
  vllm + vllm_ascend NPU runtime; they exit with code 2 and a clear message
  when it is unavailable. Everything else — reduction, verification,
  comparison logic, CLI — is CPU-tested and requires `regex` without Torch or NPU dependencies.
- Both runners drive the **text route only** (`prompt_token_ids`). For
  GLM-4.1V / GLM-5.3-Flash this is valid partial coverage of the language
  tower; it is NOT visual precision/performance coverage. A deterministic
  image-input workload with fixed images/processors in the workload identity
  would be required before claiming multimodal gates — not yet implemented.

```bash
# CI (cpu-ut partition discovers all of tests/ut automatically):
pytest tests/ut/tools/glm_reduced

# Local run without the vllm-dependent tests/ut conftest:
pytest tests/ut/tools/glm_reduced --confcutdir=tests/ut/tools/glm_reduced
```

## Relationship to existing coverage

`tests/e2e/pull_request/four_card/test_kimi_k3.py` is dummy-weight functional
coverage for K3, not an accuracy/performance gate; this tool targets the
missing reduced-checkpoint gates for GLM families. Open PRs #16603,
PRs #15103/#15393, #16529 and #17027 are related prior art (reduced-layer parity,
K3 loading, a GLM5.2 dummy profile, GLM5.2 weekly ITL) and were used as
references only.
