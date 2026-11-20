# DeepSeek-V4.1 mHC boundary fusion

## Shared/routed Add + FFN HcPost

`enable_hc_post_fusion` is an experimental, default-off A3 switch. Merge it
into the existing `--additional-config` object:

```json
{"enable_hc_post_fusion": true}
```

It supports both model runners through the shared DeepSeek-V4.1 model path,
including the inherited DSpark decoder layer. It does not change the model
runner, speculative length, weights, or MoE communication layout.

The fused Triton kernel consumes the finalized shared and routed outputs
directly, removing the BF16 Add intermediate and one kernel launch. It keeps
the BF16 rounding after Add, the A3 HcPost FP32 product/add order, and the
final BF16 cast. Its input/output contract is:

- Shared/routed: BF16 `[tokens, hidden_size]`.
- Residual/output: BF16 `[tokens, 4, hidden_size]`.
- Post: FP32 `[tokens, 4]`; combination: FP32 `[tokens, 4, 4]`.
- Last dimensions must be contiguous; row-strided inputs are supported.
- This initial kernel is limited to at most 128 local tokens; larger
  prefill batches use the native operators until benchmarked separately.

Column tiles follow the hidden size, with power-of-two widths bounded by
256 and 2048. At hidden size 5120, each token uses three masked column
tiles instead of twenty 256-column tiles. This changes the launch grid,
while retaining the same arithmetic order and rounding points.

Scaling, routed transforms and shared-expert stream waits remain before the
fusion. The runtime custom op retains the original path when the combined
output still needs a TP collective, zero-expert addition, width truncation,
or an unsupported tensor layout/dtype. A communication decision is never
specialized from the tracing batch into the compiled graph. This does not
extend the existing NPU graph scheduler's communication/capture contract.

Other hardware and legacy MoE runners retain the original model path.
Backends such as DeepGEMM MegaMoE that already combine the expert outputs
also retain native HcPost through the fallback; this switch does not replace
their fused MoE implementation.
Disable the switch and restart normally to restore the unfused path. A
configuration edit alone does not change an already-running worker.

## Verification

CPU contracts cover opt-in defaults, hardware scope, fake tensors, output
shapes, communication fallback and full-graph compilation with changing
runtime communication context:

```bash
pytest -q tests/ut/ops/test_hc_post_fusion.py \
  tests/ut/ops/test_fused_moe.py \
  tests/ut/models/test_deepseek_v41_hyper_connection.py \
  tests/ut/device/test_hardware_profile.py
```

On an **idle, allocated A3** device, compare with the native operators and
verify graph replay:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_hc_post_add.py
python benchmarks/hc_post_add.py --device 0 --tokens 10 12 --banks 40
python benchmarks/hc_post_add.py --device 0 --tokens 10 12 --dim 5120 \
  --banks 40 --blocks 256 512 1024 2048
```

The benchmark alternates baseline/candidate measurement order and uses
multiple tensor banks in NPU graphs. Its microseconds are single-rank
operator latency, **not service TPOT or a distributed performance result**.
After native-kernel equivalence passes, run the same frozen-input Herd
suite with the switch off/on; retain acceptance, MAL, ITL and throughput
alongside TPOT. CPU checks alone do not establish NPU numerical agreement.

## HcPost to the following HcPre

That second boundary is deliberately unchanged in this implementation. The
following HcPre needs both the complete four-branch residual for its FP32/HF32
projection and the delayed `pre_mix` for collapse. Later HcPost also needs
that residual. Therefore neither dropping the residual nor multiplying the
combination matrices in advance preserves the current numeric contract.

A useful follow-up needs a new HcPre producer/consumer ABI: produce the
rounded residual plus reusable collapse/RMS partials and allow HcPre to skip
the corresponding work. Computing those extra outputs while leaving the
current HcPre unchanged would duplicate work. This switch does not claim
that second fusion or its potential performance gain.
