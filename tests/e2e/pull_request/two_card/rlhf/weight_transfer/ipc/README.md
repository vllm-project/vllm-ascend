# Qwen3.8-Flash-Next PLE/MRoPE IPC CI

This independently scheduled TP2 unpacked case follows PR #20535: model
configuration lives in `case.py`, generated weights come from the shared
`FixedRandomWeightSource`, and `npu_ipc.py` runs two level-2 sleep/update cycles.
Each update must exactly match an independent startup-load reference in text
and token logprobs; the initial dummy-loaded result must differ.

The case retains four layers (three linear attention, one full attention),
PLE, interleaved MRoPE, 512 experts, hidden size and token vocabulary. It reduces
the PLE vocabulary base to 4096 and full-attention head dimension to 128, with
both rotary factors set to 0.5 to keep 64 rotary channels. The indexed full-
attention layer uses the normalized `indexed_attention` name in HF overrides,
which are applied after the original config is normalized. MTP and image/video
inputs are disabled. This does not cover full-size PLE tables, the original
256-wide attention path, or image/video inference.

The inference servers keep the conditional-generation architecture for MRoPE.
Transformers constructs only its text model on the meta device to enumerate
parameters and integer buffers, with the original language-model name prefix.
Only small PLE integer constants are materialized on CPU using Transformers'
own prime-vocabulary and hash initialization. Floating parameters use the
existing per-name deterministic BF16 generator. The PLE embedding is emitted
under the meta model's full `ngram_embedding.weight` name, supported by the
existing loader. Reference checkpoints are streamed in 2 GiB shards using the
same iterator as IPC; original checkpoint tensor values are never loaded.

PLE stays on NPU via `--engram-config '{"cpu_offload":false}'`. Each worker has
a fixed 4 GiB KV-cache budget. Forward profiling and full-decode graph capture
remain enabled. The model-state buffer registration preserves PLE context
offsets across level-2 sleep; its CPU regression test is retained.

## Running

Run from the branch checkout in the prepared Ascend container. This case uses
local configuration and tokenizer/processor assets at
`/mnt/weight/Qwen3.8-Flash-Next`; a CI worker must provision that directory
before executing this case. No descriptor preparation or custom pytest flags
are needed. Running the test directly now executes the NPU case.

Allocate two idle devices and load the intended CANN environment first:

```bash
source /mnt/share/w00893744/cann_b020/ascend-toolkit/set_env.sh

# Replace ID_A,ID_B with the allocated physical device IDs.
ASCEND_RT_VISIBLE_DEVICES=ID_A,ID_B python -m pytest -sv \
    tests/e2e/pull_request/two_card/rlhf/weight_transfer/ipc/test_qwen38_ipc.py \
    --junitxml=/path/to/workspace/logs/ipc-junit.xml
```

Use a short temporary path for Unix IPC sockets and enough temporary disk
space for the generated reference checkpoint. The test includes reference
startup, target startup, two weight updates, strict output checks and cleanup.
