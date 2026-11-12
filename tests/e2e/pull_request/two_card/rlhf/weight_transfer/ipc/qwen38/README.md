# Qwen3.8-Flash-Next PLE/MRoPE IPC CI

This opt-in test follows [PR #20535](https://github.com/vllm-project/vllm-ascend/pull/20535):
an independent server loads generated weights at startup, a target starts with
dummy weights, and two unpacked IPC updates must exactly match the reference's
text and token logprobs. The initial dummy signature must differ.

The generated model retains four layers (three linear attention, one full
attention), PLE, interleaved MRoPE, 512 experts and the original hidden size and
token vocabulary. CI reduces the PLE ngram vocabulary base from 20,000,000 to
4,096 and the full-attention head dimension from 256 to 128. Both rotary factors
are adjusted to 0.5 to retain the original 64 rotary channels. Q/K/V/output/norm
shapes and PLE embedding shards are regenerated accordingly. Integer PLE hash
buffers follow the model's prime layout and deterministic multiplier rules.

The conditional-generation architecture preserves MRoPE. Image/video limits
are zero, so the unused vision tower is skipped; no image inputs are tested.
MTP is disabled. This lane does not establish correctness for the original
256-wide attention path or full-size PLE tables.

PLE tables stay on the NPU through `--engram-config '{"cpu_offload":false}'`.
The default CPU-offload path requires an accelerator view of pinned host memory,
which is unsupported on the validated NPU stack.
The lane uses an explicit 4 GiB KV-cache budget per worker. Forward profiling
and full-decode graph capture still run; the fixed budget avoids depending on
the automatic free-memory estimate during co-located process startup. Temporary
NPU generator allocations are synchronized and released before reference startup.
The Ascend Qwen4Exp model state registers its immutable PLE ngram offsets as a
nonpersistent model buffer. MRV2 creates this state inside the weights pool;
registration lets the existing level-2 sleep backup restore the offsets in
place. Without it, discarded offsets become zero and fresh prompts use their
first token instead of EOS as the preceding PLE context. No attention kernels
or cache layout implementations are changed.

Only local configuration, tokenizer/processor assets and safetensors headers
are read. Pretrained tensor values are never loaded. Floating parameters use
the existing per-name deterministic BF16 generator for both startup and IPC.

## Preparing and running

Run from the branch checkout in the prepared Ascend container. Allocate two
idle devices and load the intended CANN environment **before** running tests:

```bash
source /mnt/share/w00893744/cann_b020/ascend-toolkit/set_env.sh

python -m tests.e2e.common.rlhf.weight_transfer.qwen38_manifest \
    --source /mnt/weight/Qwen3.8-Flash-Next \
    --output /path/to/workspace/Qwen3.8-4layers-small-ple-mrope

python -m pytest -q --confcutdir=tests/ut/_tools tests/ut/_tools/test_qwen38_manifest.py

# Replace ID_A,ID_B with the allocated physical IDs.
ASCEND_RT_VISIBLE_DEVICES=ID_A,ID_B python -m pytest -sv \
    tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/qwen38/test_qwen38_ipc.py \
    --qwen38-enable-npu \
    --qwen38-model-dir /path/to/workspace/Qwen3.8-4layers-small-ple-mrope \
    --junitxml=/path/to/workspace/logs/ipc-junit.xml
```

Without the execution flag the test skips. The output directory must be new;
original model assets are not modified. Supply a sufficiently short temporary
directory for Unix IPC sockets and enough temporary storage for the generated
reference checkpoint. The shared transaction retains full-decode graph capture
and sleeps/wakes the co-located rollout workers around each update.
