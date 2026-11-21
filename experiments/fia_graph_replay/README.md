# Shared KV lengths for FIA ACL Graph replay

This local experiment targets single-card BF16 Qwen3 with the V2 runner and
single-token `FULL_DECODE_ONLY` mode. The experiment caps the capture size
at eight requests and the context length at 2,048 tokens. The original
`torch_npu.npu_fused_infer_attention_score.out` call captures the attention
kernel. After capture, the experiment binds that kernel in each layer to the
same device tensor of KV lengths. Capture omits the update wait and reset.
The runner copies the current lengths into that tensor once on the replay
stream. The block table already has stable storage in the V2 runner.

The prototype relies on the CANN 9.1 experimental graph task APIs and the
observed FIA launch argument layout. Rebinding is attempted only when the
captured tasks match that layout. The prototype is enabled by building the
local extension below. A production implementation needs a supported runtime
or operator API for binding the device metadata. [CANN task parameter API](https://gitcode.com/cann/runtime/blob/master/docs/zh/api_ref/15_model_running_instance_management.md#aclmdlRITaskSetParams)

## Local validation

Use the same vLLM and CANN environment for both worktrees. Build the probe
from the patched worktree root.

```bash
c++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC \
  $(python -m pybind11 --includes) \
  experiments/fia_graph_replay/rebind.cpp \
  -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -lascendcl \
  -o vllm_ascend/_experimental_fia_replay.so
```

Run this benchmark with the unmodified vLLM Ascend checkout on the Python
path, then with the patched checkout on the Python path. Both use
`VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
Pass the same local Qwen3-0.6B weights to each invocation.

```bash
python experiments/fia_graph_replay/benchmark.py "$MODEL" --expected-updates 28 --trials 1
python experiments/fia_graph_replay/benchmark.py "$MODEL" --expected-updates 0 --trials 1
```

Run `python experiments/fia_graph_replay/check_kernel.py` with the patched
checkout on the Python path to reproduce stale KV lengths in an unmodified
capture and compare shared-length replay against eager FIA.

Compare the generated token IDs from both runs. For latency, run each command
without `--expected-updates` to omit instrumentation. The script warms each
workload and reports end-to-end latency and per-request time per output
token.

## Results

The original and patched worktrees both used vLLM Ascend `820ff30`,
vLLM 0.33.0, torch_npu 2.10.0.post7 and CANN 9.1.0 on the same shared
Ascend 910B3. The Qwen3-0.6B greedy check produced identical token IDs for
1, 3 and 8 concurrent requests, including 3 requests in the size-4 graph.
The task update begin, update end and event record counts each went from
28 to 0 per FULL decode step. The standalone FIA regression varied KV
lengths up to 2,048 with block-table changes and padded rows. Its largest
absolute difference against eager FIA was 0.00098 in BF16. A forced
layout mismatch recaptured the original graph and restored the 28 updates
per step with identical generated tokens.

The timing runs left instrumentation off and generated 64 tokens per request.
The table reports medians over eight warmed trials per workload and variant,
with process runs ordered baseline, patch, patch, baseline.

| Concurrent requests | Baseline TPOT (ms/token) | Patched TPOT (ms/token) | Baseline E2E (ms) | Patched E2E (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.49 | 6.09 | 1,099.90 | 437.50 |
| 3 | 16.09 | 7.03 | 1,080.65 | 497.03 |
| 8 | 16.22 | 7.56 | 1,094.24 | 536.27 |
