# Native Store decode-reuse validation

## Measured pairing

NPU records: 2026-11-07/08, Qwen3.8-27B-W8A8, Model Runner V2,
HND, eager TP1/2/4, MTP3, context 2048, match unit 128,
physical blocks 896, Store logical TP4 and 4-GiB KV per rank.
Ascend baseline `1bf75989c152c353ae0254b3e2aefa717d74f982` plus the native
Store device adapter; vLLM baseline
`24420b7a341af37a9bc3186e6fc6062512e0aa58` plus the Mooncake/Multi portion
of merged #64096 and device/layout extensions. Software: Mooncake 0.3.13,
CANN 9.2.1 and torch/torch_npu 2.10. The final lifecycle run additionally
uses the five-line scheduler completion-polling correction.

A [production-source reconstruction](https://github.com/KeyanHu-git/vllm/tree/b6ff840c57a7d0ad9e1c64c61786cb187e7f21d0)
preserves the pinned vLLM pairing and required production extensions. It was
published after measurement; the run versions remain those recorded above.

## Restoration and lifecycle

| Writer TP | Reader TP | Store hit | Local hit | Failed keys | Natural answer | MTP accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1792 | 0 | 0 | 392 | 3 |
| 1 | 4 | 1792 | 0 | 0 | 392 | 3 |
| 2 | 1 | 1792 | 0 | 0 | 392 | 3 |
| 2 | 4 | 1792 | 0 | 0 | 392 | 3 |
| 4 | 1 | 1792 | 0 | 0 | 392 | 3 |
| 4 | 2 | 1792 | 0 | 0 | 392 | 3 |

Each direction reads 68 objects / 281,673,728 bytes in total across ranks.
The source computed-token count crosses 1792 during decode (1790 to 1793).
Three continuation points of one conversation have identical natural-EOS
output token sequences with decode saving off/on (402/352/4 output tokens).

The final native MultiConnector 1P1D run uses legacy matching. P restores
1664 tokens; D has Store lookup disabled and receives 344,064,000 bytes
through native PD. The answer is 392; both native local-cache resets succeed,
with no new Store PUT. This case verifies composition/release; decode-origin
checkpoint consumption is established separately by the Store-only matrix.

## Layout and capacity evidence

Three real SDK PUT/GET fixtures cover mixed FP16/FP32 packed Mamba,
BF16 MLA and INT8 C8 MLA (512+16 components, scale_dim0). Entire raw
allocations match, including padding and untouched regions. These are
layout-transfer tests; GLM5Next registration also has CPU coverage.

A 32-GiB Store pool completed 64 fills/65 fixed probes with GET median 65.47 ms,
P95 71.25 ms and maximum 82.35 ms. A further 776 pressure requests exercised 109
evictions with no GET failure and allocation balance residual 0. The earlier
GET degradation at 64–256 GiB live occupancy remains open.

## Paired timing records

The adjacent JSON retains all ten actual A/B pairs, with serial requests
and alternating AB/BA order. A disables decode saving; B enables it.
Both use Store. A/B run on physical devices 15/8 without crossover.
Source generation has 1536 tokens; paired restoration uses 1815 input and 32
output tokens (`ignore_eos=true`). All restore local hits are 0.

| Metric | A median | B median |
| --- | ---: | ---: |
| External hit, tokens | 256 | 1792 |
| Remaining input, tokens | 1559 | 23 |
| Restore TTFT, ms | 795.247 | 323.661 |
| End-to-end, s | 1.806 | 1.318 |
| Source TPOT, ms | 33.229 | 32.377 |
| Restore TPOT, ms | 32.472 | 32.116 |

The median paired TTFT change is −468.017 ms. Sampled HBM peaks are
40,832 / 41,141 MiB, with 284 samples per device and median sample interval 3.76 s.
These timing data precede the final scheduler completion correction.

Recheck the measurements without starting an engine:

```bash
python benchmarks/check_mooncake_store_reuse.py \
    benchmarks/data/mooncake_store_reuse_qwen38_27b.json
```

The CLI evaluates the recorded acceptance gates and returns 1 on regression.
The CPU tests also reject deliberately slower restore times and local-only
hits presented as Store reuse. This rechecks frozen evidence, not new NPU runs.
