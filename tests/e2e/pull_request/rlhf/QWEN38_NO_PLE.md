# Qwen3.8-Flash-Next four-layer no-PLE weight transfer

This BF16 profile preserves original tensors from decoder layers 0–3, all
512 experts, top-10 routing, GDN, QSA and hyperconnections. It excludes PLE,
vision and MTP. It tests loading and transfer consistency, not full-model
quality, performance or PLE behavior.

## Prepare and verify

Run from the repository root with a read-only original checkpoint. The output
must be a new directory separate from the source. Use an immutable source
revision; preparation records the original config/index hashes as well.

```bash
python tests/e2e/pull_request/rlhf/prepare_qwen38_checkpoint.py \
  --source /path/to/original-checkpoint \
  --output /path/to/four-layer-no-ple-checkpoint \
  --source-revision SOURCE_COMMIT_SHA
```

Preparation copies bounded tensor byte ranges into independent safetensors
shards and prints the manifest SHA-256. It rejects profile drift: the output
must contain exactly 101 BF16 tensors and 23,294,733,120 tensor bytes. It changes
only layer count, layer types and `text_config.ple_layer_ids` in the config.

Unresolved Git LFS pointers are rejected before writing weights. If the original
tokenizer is a pointer, supply `--resolved-tokenizer /path/to/tokenizer.json`.
Its size and SHA-256 must exactly match the original pointer; the source stays
read-only. A different model's tokenizer cannot be substituted.

Each source validates all output file hashes against the externally supplied
manifest digest. Each iteration reads the same original tensor values; there
is no random source or renamed/split expert payload.

## Run on A3

Allocate two NPUs visible to pytest. HCCL uses logical NPU 0 for rollout and
logical NPU 1 for trainer. IPC uses logical NPU 0 for both processes and checks
the worker/trainer physical-device UUIDs. Device mapping respects
`ASCEND_RT_VISIBLE_DEVICES`.

```bash
pytest -sv tests/e2e/pull_request/two_card/rlhf/state_transitions/test_qwen38_no_ple_hccl_weight_transfer.py \
  --qwen38-no-ple-checkpoint /path/to/four-layer-no-ple-checkpoint \
  --qwen38-no-ple-manifest-sha256 MANIFEST_SHA256
pytest -sv tests/e2e/pull_request/two_card/rlhf/state_transitions/test_qwen38_no_ple_npu_ipc_weight_transfer.py \
  --qwen38-no-ple-checkpoint /path/to/four-layer-no-ple-checkpoint \
  --qwen38-no-ple-manifest-sha256 MANIFEST_SHA256
```

Each packed mode starts an independent normal-load reference, then a dummy
live server. Two live updates must match the reference text and token logprobs
exactly. Worker checks reject PLE modules/parameters and require execution of
GDN, QSA, MoE and hyperconnection modules during prefill. The long prompt has
3072 token IDs; this does not prove QSA sparse top-k truncation.

## CI prerequisites

Missing checkpoint/digest and invalid content fail rather than skip. The two
files route through the existing A3 two-card runner. The time entries are
provisional budgets, not measurements.

Before enabling routine CI, publish a real no-PLE resource with a fixed model
revision and manifest digest, register it in the A3 model cache, and measure
all four cases on the verified CI environment. No public model ID is assumed
by these tests. Backend compatibility, TP1 memory peaks and four-case results
must be verified before this change is ready for review.
