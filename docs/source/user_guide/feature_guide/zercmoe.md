# ZercMoE

## Overview

ZercMoE fuses token dispatch, SHMEM communication, two expert matrix
multiplications, SwiGLU, and token combine into one operator. This integration
migrates the ZercMoE functionality from `feature/zercmoe-a2-integration`, based on
`0.26.0rc`, to `zercmoe-new`, based on `0.30.0rc2`. The ZercMoE kernel and its
SHMEM runtime are supplied by an external wheel; this repository contains the
integration code.

Eligible prefill batches use ZercMoE. Decode retains the existing communication
path and ACL graph execution. The integration packs expert weights once after
loading and calls `torch.ops.npu.zerc_moe` directly with
`return_expert_tokens=0`. Expert-token statistics are not consumed in this path,
so their synchronous return is avoided.

## Supported configuration

| Setting | Requirement |
| --- | --- |
| Hardware | Atlas A2 |
| Model | Unquantized MoE with BF16 activations and weights |
| Expert FFN | Standard bias-free SiLU/SwiGLU, with default alpha/beta and no clamp; router weights applied to the expert output |
| Parallelism | Tensor parallelism with expert parallelism enabled; 2 to 64 EP ranks |
| Data and pipeline parallelism | `data_parallel_size=1`, `pipeline_parallel_size=1` |
| Context parallelism | Prefill and decode context parallel sizes must both be `1` |
| Model runner | MRv1, or MRv2 without speculative decoding |
| Hidden size | `[1024, 8192]`, divisible by 16 |
| MoE intermediate size | `[512, 3072]`, divisible by 16 |
| Sequence parallelism | Disabled, including FlashComm1 |
| LoRA | Disabled |
| Dynamic EPLB | Disabled for both legacy Ascend EPLB and MRv2 upstream EPLB (`enable_eplb=false`) |
| Sleep mode | Disabled (`enable_sleep_mode=false`) |
| Online weight transfer | Disabled (`weight_transfer_config=null`); static inference weights only |
| ZercMoE input capacity | At most 4096 tokens per TP rank, also bounded by the scheduler token budget |

The intermediate size is the per-expert size. With EP, each rank owns complete
experts, so it equals the model's `moe_intermediate_size`.

Both switches below are opt-in and default to `0`. Setting only
`enable_zerc_moe=1` does not enable ZercMoE; `enable_fused_mc2=1` is also required.
The ZercMoE path is selected only when the wheel and runtime requirements pass.
Unsupported models or batches continue through an existing MoE path.
After the configuration gate passes, weight loading also checks each routed
layer's actual activation, bias, router weighting, dtype, and weight shapes.
An incompatible layer raises an error asking you to disable ZercMoE, rather
than silently executing different FFN semantics.

The integration keeps both the existing expert weights for decode and packed
zN copies for ZercMoE prefill. Budget approximately one additional copy of the
routed expert weights per rank, plus the SHMEM heap. This reduces the memory
available for the KV cache and can change throughput under memory pressure.
Sleep mode and online weight transfer are excluded from the selection gate;
restoring or updating the packed copies and SHMEM state is outside this
static-weight migration's supported lifecycle.

## Installation and configuration

Install vLLM and vLLM Ascend versions matching the `0.30.0rc2` base, then install
a ZercMoE wheel built for the deployed CANN, PyTorch, Python, and host
architecture. Use a wheel that registers `torch.ops.npu.zerc_moe` with the final
`return_expert_tokens` argument and supports `zercmoe.pack_weights` and
`zercmoe.get_symm_buffer`. A wheel version label alone does not establish ABI or
operator-schema compatibility.

```bash
python -m pip install /path/to/compatible_zercmoe.whl
python - <<'PY'
import torch
import zercmoe

assert callable(zercmoe.pack_weights)
assert callable(zercmoe.get_symm_buffer)
print(torch.ops.npu.zerc_moe.default._schema)
PY
```

Confirm that the printed schema includes `return_expert_tokens` before starting
the service. An older operator schema cannot accept this integration's direct
call. Run this probe in a separate process; the normal integration imports the
wheel only in workers that use it.

| `additional_config` field | Default | Meaning |
| --- | --- | --- |
| `enable_fused_mc2` | `0` | Enable the fused MC2 family. Required for ZercMoE. |
| `enable_zerc_moe` | `0` | Enable the ZercMoE selection gate. |
| `zercmoe_ipport` | `null` | SHMEM bootstrap address; defaults to `tcp://127.0.0.1:28765`. |
| `zercmoe_shmem_mb` | `null` | Positive integer heap size in MB per rank; unset uses the wheel's automatic estimate. |

`VLLM_ASCEND_ENABLE_ZERC_MOE=1` is the equivalent ZercMoE environment switch.
An explicit `additional_config.enable_zerc_moe` value takes precedence.
For the `0.30.0rc2` base, configure fused MC2 through
`additional_config.enable_fused_mc2`; its old environment switch is no longer
supported.

All EP ranks must use the same bootstrap address, heap size, model shapes, and
token capacity. Each independent job needs a unique bootstrap port. The
loopback default is for a single node; multi-node deployments must explicitly
set a reachable master address and provide the transport environment required
by the installed wheel. The integration publishes worker `RANK` and
`WORLD_SIZE` for SHMEM, so these values must describe the same EP group if they
are already present in the environment.

MemFabric is imported only when sparse KV offload is actually used. Its
`libmf_hybm_core.so` exports symbols that can interpose ZercMoE's bundled SHMEM
symbols and cause bootstrap failures. Lazy import prevents unused sparse KV
offload from loading those symbols. Combining ZercMoE with an active MemFabric
connector still requires separate runtime compatibility validation.

## Start a service

The following single-node example uses four A2 devices. Replace the model path
with a compatible unquantized BF16 MoE model and adjust the memory budget to
allow for the extra expert-weight copy.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL=/path/to/bf16-moe-model

vllm serve "$MODEL" \
  --served-model-name zercmoe-test \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_fused_mc2":1,"enable_zerc_moe":1,"enable_flashcomm1":false,"zercmoe_ipport":"tcp://127.0.0.1:28765"}' \
  --port 8000
```

The integration selects ZercMoE for eligible prefill batches above 64 tokens and
within capacity. Both eager decode and graph decode use the existing path.
When the requested graph mode is `FULL`, the platform changes it to
`FULL_DECODE_ONLY` for an eligible ZercMoE configuration. This keeps decode
graphs while leaving prefill eager. Explicitly selecting `FULL_DECODE_ONLY`,
as above, also makes performance comparisons easier to reproduce.

MRv1 uses the runtime prefill flag to distinguish prefill from eager decode.
MRv2 reads prefill counts across the attention metadata objects. When MRv2
metadata is unavailable, ordinary forwards conservatively use the existing
path; profiling can still initialize ZercMoE. MRv2 with speculative decoding is
excluded because its draft-forward context does not reliably distinguish draft
models for this integration. MRv1 draft forwards remain on the existing path.

With TP4, a scheduler budget of 16384 tokens gives at most 4096 tokens per rank.
Increasing the scheduler budget does not increase the ZercMoE per-rank limit.
The symmetric buffer is allocated lazily from fixed configuration when the
first eligible MoE layer executes. Check the `ZercMoE sym-buffer alloc` log
across ranks to confirm identical settings.

## Validate functionality and performance

Local migration checks use an isolated Python harness that runs production
configuration, selection, packing, and operator-call code with mocked vLLM and
NPU dependencies. These checks verify the Python contracts and import
boundaries; they do not execute the ZercMoE kernel, SHMEM collectives, ACL graph
capture, or an NPU accuracy/performance workload. In an installed development
environment, run the corresponding repository tests directly:

```bash
python -m pytest -q \
  tests/ut/ops/test_zercmoe.py \
  tests/ut/test_ascend_forward_context.py \
  tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py
```

On an A2 two-card development machine with the compatible wheel and model
available, run the optional end-to-end test for both model runners:

```bash
python -m pytest -q tests/e2e/pull_request/two_card/test_zercmoe.py
```

This checks that both ranks enter ZercMoE on long prefills and compares prefill
and decode logprobs against the existing MoE path with decode graphs enabled.

Run the same fixed prompts with ZercMoE disabled and enabled, checking successful
responses and output correctness before measuring speed. Use a fresh service
process for each configuration. For a baseline, change only
`"enable_zerc_moe":1` to `"enable_zerc_moe":0` in the launch command. Confirm the
actual baseline communication path; disabling ZercMoE does not guarantee that
MegaMoE is selected on A2.

Warm up each service with the same workload, then collect repeated benchmark
runs. For example:

```bash
vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --model zercmoe-test \
  --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-prompts 200 \
  --max-concurrency 64 \
  --request-rate inf \
  --seed 42 \
  --ignore-eos \
  --save-result \
  --result-dir ./benchmark-results/zercmoe-on
```

Use a different result directory for the disabled run. Compare TTFT, TPOT,
output-token throughput, request failures, and available KV-cache capacity.
Record the Git revisions, wheel SHA256, CANN/PyTorch versions, model revision,
topology, graph mode, warm-up procedure, and complete launch arguments.
Keep them identical across comparisons. Include short and long prefill and the
target concurrency; a faster fused call alone does not establish an
end-to-end throughput improvement.

The migration requires accuracy and performance validation on the target NPU
system. The commands above are a reproducible validation procedure, not
evidence of hardware validation or a performance guarantee.
