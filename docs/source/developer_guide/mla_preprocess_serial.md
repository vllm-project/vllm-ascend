# Serial MLA Preprocess on A3

This BF16/no-quant/split-ND fallback prepares query tensors and updates the
compressed KV cache. C uses two public `torch.matmul` calls and one public
`torch.bmm`; V uses independent Ascend C through op_host/op_kernel, ACLNN,
EXEC_NPU_CMD and Torch PrivateUse1/Meta. There are no custom Cube kernels.
Original fused compute/tiling/golden/disassembly was not inspected.

The default path retains M20 ND weights and behavior. An explicit optional
single-token weight preparation uses public Torch-NPU NZ conversion at load
time. Only T=1 selects those extra weights. No Vector kernel/tiling, schema,
Meta implementation, environment variable or global weight cache is added.
SuperKernel remains disabled. This is standalone operator evidence, not
model-serving or complete-ABI acceptance.

## Data flow and interface

```mermaid
flowchart TD
    ND["Default ND down/up weights"] --> SELECT{"T=1 and extra weights supplied?"}
    NZ["Load-time optional KN-NZ down/up weights"] --> SELECT
    SELECT --> C1["torch.matmul: down"]
    X["x: T,D"] --> C1
    C1 --> V1["NormCache: Q/KV RMSNorm, K RoPE, cache writes"]
    V1 --> C2["torch.matmul: Q up"]
    C2 --> B{"T=1?"}
    B -->|yes| VP["QPrepare: Q-nope and Q RoPE"]
    VP --> C3S["torch.bmm; output layout is a view"]
    B -->|no| PK["QPack: T,H,P to H,T,P"]
    PK --> C3["torch.bmm"]
    C3 --> VF["QFinalize: layout and Q RoPE"]
    C2 --> VF
```

T>1 runs3 C and3 V stages; T=1 runs3 C and2 V stages. Four Vector interfaces
are registered: NormCache, QPrepare, QPack, QFinalize. QPrepare is mutually
exclusive with QPack/QFinalize. All run serially on the current stream;
Torch chooses the matrix kernels and their tiling.

| Argument | Shape and meaning |
| --- | --- |
| x | BF16 NPU `[T,D]` |
| weight_down | Default persistent BF16 ND `[L+576,D]`, rows `[KV,K-rope,Q]` |
| weight_up | Default persistent BF16 ND `[H*(P+64),L]` |
| weight_key | Persistent BF16 ND `[H,P,512]` |
| gamma_q, gamma_kv | BF16 `[L]`, `[512]` |
| cos, sin | Paired BF16 `[T,64]`, or both absent |
| slots | INT32 `[T]`; negative slots preserve cache contents |
| cache, cache_rope | BF16 `[blocks,128,1,512]`, `[blocks,128,1,64]` |
| single_token_weights | Keyword-only optional pair of prepared down/up tensors; default None |
| Returned queries | `q_nope [T,H,512]`, `q_rope [T,H,64]` |

Cache outputs alias their inputs. Only the cache block axis may have physical
stride padding. Nonnegative slots must be unique and within capacity.
RMS epsilon is1e-5; Norm/RoPE intermediates are FP32.

`prepare_mla_preprocess_serial_weights(wdqkv,wuq,wuk)` still returns exactly
three tensors and preserves the old API. Inputs use logical down rows
`[Q,KV,K-rope]` and key `[H,512,P]`. The new helper accepts its down/up outputs:

```python
from vllm_ascend.ops.mla_preprocess_serial import (
    mla_preprocess_serial,
    prepare_mla_preprocess_serial_weights,
    prepare_mla_preprocess_single_token_weights,
)

down, up, key = prepare_mla_preprocess_serial_weights(wdqkv, wuq, wuk)
single = prepare_mla_preprocess_single_token_weights(down, up)
outputs = mla_preprocess_serial(
    x, down, up, key, gamma_q, gamma_kv, cos, sin, slots, cache, cache_rope,
    single_token_weights=single,
)
```

The caller explicitly holds both weight versions and rebuilds them when
weights change. The optional helper prepares contiguous RHS `[K,N]`, converts
it to FRACTAL_NZ29, then returns a logical `[N,K]` transposed view. Forward's
existing `.t()` restores the RHS view. BF16 logical values are checked bit
exact. For T>1 the supplied extra pair is ignored; default None preserves
the previous path. No explicit format conversion is added to forward.
After replacing prepared tensors or their device addresses, recapture any
Graph that consumes them; an existing replay retains its captured addresses.

Extra logical weight storage is `2*(D*(L+576)+L*H*(P+64))` bytes:33.375–100.875
MiB for the tested H8–128, excluding allocator overhead. Preparation and
transient preparation storage are outside steady Graph timing. All runtime
projections, layouts, Norm/RoPE and cache updates remain inside the full call.

## CANNBot tiling experiments and selected scope

CANNBot generic core-shrink, DMA batching and transpose-fusion guidance was
consulted at revision393564ae43f304426db2c206e361cd43b2811413. A5 MicroAPI
templates are not imported into A3. Platform queries provide AIV capability;
the original fused tiling is never a source.

Retained V tiling uses one NormCache token task, grouped-head QPrepare,
bounded8192-element double-buffered QPack and up-to2-token/128-cell QFinalize.
The following additional V strategies were measured and rejected:

| Trial | Strategy | Paired M20/new result and decision |
| --- | --- | --- |
| MLA01 | Split small-T Finalize heads for more AIV tasks | 0.960–1.000x in8 probes; REJECT |
| MLA02 | Balance Pack token groups when head parallelism is low | 0.934–0.999x in6 probes; REJECT |
| MLA04 | Separate small-T Q norm and KV/cache tasks | 0.918–1.038x; protected regression; REJECT |
| MLA05 | Finalize64-cell double buffer at same total UB budget | 0.989–1.002x; no reliable gain; REJECT |
| MLA06 | Fill128-cell tiles across more tokens at few heads | 0.986–1.004x; no reliable gain; REJECT |

All probes passed independent precision/cache gates. More tasks did not
reliably improve full-call latency, so these branches are not retained.

MLA07 compared six public weight-format policies. NZ key/all paths passed
eager checks but failed Graph capture through aclop BatchMatMul; they have no
speed result. Unconditionally using KN-NZ down/up passed61 precision/Graph
points and met0.8x at57, but regressed protected points by up to9.3% against
M20. That general policy is REJECTED. MLA08 confines the optional choice to
the existing structural T=1 path, with no per-shape winner table.

MLA09 additionally tested down-only KN-NZ while up aliases ND, reducing extra
logical storage to28.875MiB. It passed61 checks,146 UTs and the14 False-option
checks, but was slower than the both-weight option at most small heads in the
paired five-head probe. The apparent H128 recovery did not repeat in full-range
confirmation. It is REJECTED as a performance/storage tradeoff, not a numerical
failure. No head-specific winner table is introduced; MLA08 is retained.

## Expanded shapes and accuracy

The61 business points retain all original27. Fixed dimensions are D7168,
Q latent L1536,Q-nope P128,KV512,RoPE64. Local H8/16/32/64/128 are compatible
subdivisions of128 global heads; no TP/model-serving run is claimed.
T counts flattened active tokens, not necessarily one sequence.

* Main grid: T1/2/4/8/16/32/64/128/256/512/1024 times H8/16/32/64/128.
* Protected prior points: T257/1023 times H16/32/128.
* Timed inputs: BF16/no quantization/split ND cache, block128, stride1,
  supplied RoPE, valid unique nonnegative slots.
* T>1024 is excluded because the installed fused ABI previously rejected it.
  FP16, quantization, combined/NZ cache and serving remain unvalidated.

The single device UT passed146 cases: original132 plus14 optional T=1 eager/
Graph cases across all five head counts and no-RoPE/negative-slot/stride2
variants. It verifies bit-exact prepared weights, Graph reads updated inputs,
numerical outputs, cache aliasing, untouched cache and physical padding.
Independent CPU FP32 BF16 acceptance is atol=rtol=2^-6,matched>=0.99,max_abs<=1.

The14 optional cases also pass in an isolated process explicitly setting
allow_internal_format=False. Native option readback is `disable` before and
after those tests. The product never writes this option. The public attribute
has no getter in this package and returns None; it is not the state readback.

## Graph measurement and workload-sorted results

Same A3 Ascend910_9362,CANN9.2.0,Torch2.10.0+cpu and
Torch-NPU2.10.0.post7.dev20261020. All three implementations are compared in
the same process on the same inputs. Graph captures20 complete calls; NPU
events bracket one replay and divide elapsed time by20. Two groups each use25
replays, discard5 and retain20, alternating order. Speed is the smaller
group-mean ratio. The event interval includes Graph runtime gaps, not pure
kernel sum. Weight preparation and CPU reference are outside timing.

The baseline weight-preparation harness requests ALLOW_INTERNAL_FORMAT=enable;
timing uses that common setup for all three implementations. The separate
disable run establishes accuracy and Graph support only, not performance
for that configuration. The delivered helper never changes this option.

Workload MACs is `T*D*(L+576)+T*L*H*(P+64)+T*H*P*512`; FLOPs is2*MACs.
It describes projection work and excludes vector/cache traffic. Rows sort by
MACs then T/H, not by observed latency. Costs can reverse locally with shape,
runtime placement and Torch dispatch. Times below average the two group means;
speed ratios remain conservative per-group values.

Same-capture default M20: 0.736–1.227x fused, 51/61 >=0.8x. Optional T1: 0.738–1.237x, 53/61 >=0.8x. Eight optional-path points remain below0.8x; there is no full-range performance pass.

Remaining points: t2-h64 (0.764x), t4-h32 (0.774x), t128-h128 (0.771x), t256-h128 (0.738x), t512-h128 (0.761x), t1024-h64 (0.750x), t1023-h128 (0.739x), t1024-h128 (0.778x).

| T | Local H | GMAC | Fused us | M20 us | Optional T1 us | Fused/optional | M20/optional |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.018022 | 36.442 | 43.215 | 31.809 | 1.144x | 1.357x |
| 1 | 16 | 0.020906 | 39.604 | 45.365 | 35.224 | 1.123x | 1.287x |
| 1 | 32 | 0.026673 | 45.110 | 57.115 | 46.426 | 0.971x | 1.230x |
| 2 | 8 | 0.036045 | 39.347 | 45.358 | 45.412 | 0.864x | 0.999x |
| 1 | 64 | 0.038207 | 53.806 | 63.713 | 58.844 | 0.913x | 1.083x |
| 2 | 16 | 0.041812 | 42.812 | 46.675 | 46.705 | 0.916x | 0.998x |
| 2 | 32 | 0.053346 | 45.621 | 57.023 | 56.812 | 0.803x | 1.003x |
| 1 | 128 | 0.061276 | 108.320 | 100.260 | 100.078 | 1.081x | 1.001x |
| 4 | 8 | 0.072090 | 41.646 | 45.361 | 45.211 | 0.921x | 1.001x |
| 2 | 64 | 0.076415 | 54.393 | 71.873 | 71.029 | 0.764x | 1.012x |
| 4 | 16 | 0.083624 | 42.516 | 46.294 | 46.232 | 0.919x | 1.001x |
| 4 | 32 | 0.106693 | 45.430 | 58.576 | 58.665 | 0.774x | 0.998x |
| 2 | 128 | 0.122552 | 104.801 | 114.811 | 114.261 | 0.917x | 1.004x |
| 8 | 8 | 0.144179 | 42.579 | 43.928 | 43.686 | 0.975x | 1.005x |
| 4 | 64 | 0.152830 | 54.731 | 67.318 | 63.048 | 0.867x | 1.067x |
| 8 | 16 | 0.167248 | 42.689 | 47.301 | 47.522 | 0.898x | 0.995x |
| 8 | 32 | 0.213385 | 47.679 | 59.223 | 59.036 | 0.807x | 1.003x |
| 4 | 128 | 0.245105 | 113.795 | 108.232 | 105.981 | 1.073x | 1.021x |
| 16 | 8 | 0.288358 | 43.463 | 35.615 | 35.441 | 1.225x | 1.004x |
| 8 | 64 | 0.305660 | 55.602 | 65.174 | 64.698 | 0.859x | 1.007x |
| 16 | 16 | 0.334496 | 47.277 | 39.842 | 39.511 | 1.196x | 1.008x |
| 16 | 32 | 0.426770 | 51.143 | 46.793 | 45.911 | 1.114x | 1.019x |
| 8 | 128 | 0.490209 | 106.588 | 108.530 | 108.040 | 0.987x | 1.004x |
| 32 | 8 | 0.576717 | 51.649 | 42.079 | 42.243 | 1.222x | 0.996x |
| 16 | 64 | 0.611320 | 60.402 | 60.134 | 59.315 | 1.018x | 1.013x |
| 32 | 16 | 0.668991 | 55.535 | 45.483 | 44.885 | 1.237x | 1.013x |
| 32 | 32 | 0.853541 | 57.048 | 52.263 | 51.885 | 1.098x | 1.006x |
| 16 | 128 | 0.980419 | 122.835 | 117.106 | 118.456 | 1.037x | 0.988x |
| 64 | 8 | 1.153434 | 56.913 | 49.687 | 49.659 | 1.144x | 1.000x |
| 32 | 64 | 1.222640 | 70.866 | 71.835 | 72.299 | 0.980x | 0.993x |
| 64 | 16 | 1.337983 | 57.683 | 52.610 | 52.348 | 1.102x | 1.005x |
| 64 | 32 | 1.707082 | 61.531 | 67.391 | 62.068 | 0.991x | 1.086x |
| 32 | 128 | 1.960837 | 128.271 | 131.177 | 136.500 | 0.939x | 0.960x |
| 128 | 8 | 2.306867 | 67.734 | 60.913 | 60.474 | 1.120x | 1.007x |
| 64 | 64 | 2.445279 | 78.048 | 85.683 | 84.478 | 0.924x | 1.014x |
| 128 | 16 | 2.675966 | 69.266 | 65.516 | 65.673 | 1.055x | 0.997x |
| 128 | 32 | 3.414163 | 84.857 | 77.386 | 76.071 | 1.115x | 1.017x |
| 64 | 128 | 3.921674 | 154.611 | 161.299 | 164.426 | 0.940x | 0.981x |
| 256 | 8 | 4.613734 | 97.119 | 90.681 | 90.470 | 1.073x | 1.002x |
| 128 | 64 | 4.890558 | 97.655 | 111.570 | 115.691 | 0.844x | 0.964x |
| 256 | 16 | 5.351932 | 108.099 | 106.101 | 101.753 | 1.062x | 1.043x |
| 257 | 16 | 5.372838 | 157.993 | 129.137 | 127.826 | 1.235x | 1.010x |
| 256 | 32 | 6.828327 | 131.152 | 129.660 | 131.668 | 0.995x | 0.985x |
| 257 | 32 | 6.855000 | 180.374 | 157.431 | 158.300 | 1.139x | 0.994x |
| 128 | 128 | 7.843348 | 164.094 | 215.793 | 212.887 | 0.771x | 1.014x |
| 512 | 8 | 9.227469 | 154.228 | 142.937 | 142.141 | 1.085x | 1.006x |
| 256 | 64 | 9.781117 | 162.842 | 196.959 | 192.583 | 0.845x | 1.023x |
| 512 | 16 | 10.703864 | 165.910 | 162.486 | 161.223 | 1.029x | 1.008x |
| 512 | 32 | 13.656654 | 193.702 | 201.452 | 206.119 | 0.940x | 0.977x |
| 256 | 128 | 15.686697 | 252.054 | 340.847 | 341.567 | 0.738x | 0.998x |
| 257 | 128 | 15.747973 | 371.027 | 370.738 | 372.847 | 0.995x | 0.994x |
| 1024 | 8 | 18.454938 | 252.366 | 269.514 | 269.696 | 0.936x | 0.999x |
| 512 | 64 | 19.562234 | 262.224 | 312.407 | 314.128 | 0.835x | 0.994x |
| 1023 | 16 | 21.386822 | 280.558 | 309.131 | 310.870 | 0.902x | 0.994x |
| 1024 | 16 | 21.407728 | 279.350 | 296.828 | 296.054 | 0.943x | 1.002x |
| 1023 | 32 | 27.286634 | 353.981 | 417.264 | 411.404 | 0.860x | 1.014x |
| 1024 | 32 | 27.313308 | 355.135 | 420.411 | 418.450 | 0.849x | 1.005x |
| 512 | 128 | 31.373394 | 440.827 | 581.402 | 578.846 | 0.761x | 1.004x |
| 1024 | 64 | 39.124468 | 501.729 | 658.976 | 668.336 | 0.750x | 0.986x |
| 1023 | 128 | 62.685512 | 855.109 | 1161.245 | 1156.740 | 0.739x | 1.004x |
| 1024 | 128 | 62.746788 | 859.946 | 1101.779 | 1105.489 | 0.778x | 0.996x |

At T1,H8/16/32/64 paired M20/optional speed is1.357/1.287/1.230/1.083x.
H128 is1.001x in the main capture and0.976x in the swapped-capture repeat;
its benefit is unresolved. T>1 uses the original ND path and adds no device
work, but independent Graph measurements range0.960–1.086x. Three apparent
regressions remain visible. Targeted swapped-order and identical-function
controls also vary; these data do not establish universal zero regression or
attribute timing differences to added device work. No better repeat replaces
the primary61 rows. The different floor counts are observed counts, not proof
that all threshold crossings were caused by the optional T1 weights.

## Independent complete-Graph profiler data (2026-10-31)

These 61 captures are separate from the 2026-10-30 Graph event measurements above.
They use an idle A3 Ascend910_9362 on a different host, CANN 9.1.0
(B243), PyTorch 2.10.0+cpu, torch_npu 2.10.0.post4 and Python 3.12.13.
Actual device properties report 20 Cube cores and 40 Vector cores; the platform query reports
196608 bytes of UB. The unchanged custom Vector package and Torch wrapper
were built with CANN 9.2.0 and matched their recorded hashes.
The public source is commit 0353622162ba3f35541200b571b7a3a6058149c0;
the MLA08 Python SHA256 is
`c33d20434f10b2d8f3e2341bf5869e69deb6632f90e6842a14a7fe849b618c92`.

The original 146 device UTs passed in this runtime without skips. Every
profiled case also passed the existing independent CPU FP32 output/cache
reference after Graph replay. The test process sets
`ALLOW_INTERNAL_FORMAT=enable` and reads back the native value; the product
does not change that option. Optional weights are supplied, with their
existing selection confined to T=1.

Each shape runs in its own fresh application and full msprof process.
Five complete eager warmups precede capture of one complete pipeline call
in a Graph. The Graph replays 25 times; the first five whole sequences are
discarded, and all tasks in the final 20 sequences are retained. No kernel
is benchmarked independently. Weight preparation, CPU reference and
validation copies are outside the captured Graph. Task counts below come
from the actual periodic trace, with all extra tasks retained.

C is the sum of the three Torch matrix-task durations; V is the sum of
the owned Vector-task durations. Extra layout and other tasks are shown
separately and included in Total. Fractions are ratios of these means.
They describe software stage durations, not Cube pipeline utilization.
Task-duration sums exclude host submission and Graph runtime gaps. They
must not be combined with or subtracted from the older Graph event
intervals, and do not change the earlier 53/61 performance-floor result.

This 9.1 profiler reports Input Shapes and Task Type as N/A. Shape identity
comes from each recorded one-case configuration, command and accuracy
check; task identity/count comes from that case's actual trace. A prior
same-process 61-case attempt reused Graph IDs and exported stale T1
metadata with missing T>1 tasks. All of its component timings are
INVALID and excluded. Its 61 accuracy checks remain independent valid
checks. The older four CANN 9.2 profiles also remain only in the historical
archive; no old/new timing is selected to improve this table.

Values below are means in microseconds, rounded to three decimals. Raw
CSVs and JSON retain all 25 durations per task and the selected 20 samples.

| Case | GMAC | Tasks/call | C us | Owned V us | Extra layout us | Other us | Total us | C/(C+V) | C/Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t1-h8 | 0.018022 | 5 | 31.501 | 7.259 | 0.000 | 0.000 | 38.760 | 81.27% | 81.27% |
| t1-h16 | 0.020906 | 5 | 34.219 | 10.106 | 0.000 | 0.000 | 44.325 | 77.20% | 77.20% |
| t1-h32 | 0.026673 | 5 | 42.884 | 12.107 | 0.000 | 0.000 | 54.991 | 77.98% | 77.98% |
| t2-h8 | 0.036045 | 6 | 46.636 | 8.573 | 0.000 | 0.000 | 55.209 | 84.47% | 84.47% |
| t1-h64 | 0.038207 | 5 | 58.435 | 10.995 | 0.000 | 0.000 | 69.430 | 84.16% | 84.16% |
| t2-h16 | 0.041812 | 6 | 49.829 | 10.327 | 0.000 | 0.000 | 60.156 | 82.83% | 82.83% |
| t2-h32 | 0.053346 | 6 | 58.110 | 12.534 | 0.000 | 0.000 | 70.644 | 82.26% | 82.26% |
| t1-h128 | 0.061276 | 5 | 103.119 | 11.275 | 0.000 | 0.000 | 114.394 | 90.14% | 90.14% |
| t4-h8 | 0.072090 | 6 | 45.937 | 9.226 | 0.000 | 0.000 | 55.163 | 83.28% | 83.28% |
| t2-h64 | 0.076415 | 6 | 63.703 | 14.621 | 0.000 | 0.000 | 78.324 | 81.33% | 81.33% |
| t4-h16 | 0.083624 | 6 | 49.444 | 10.294 | 0.000 | 0.000 | 59.738 | 82.77% | 82.77% |
| t4-h32 | 0.106693 | 6 | 56.427 | 12.989 | 0.000 | 0.000 | 69.416 | 81.29% | 81.29% |
| t2-h128 | 0.122552 | 6 | 107.705 | 14.669 | 0.000 | 0.000 | 122.374 | 88.01% | 88.01% |
| t8-h8 | 0.144179 | 6 | 46.714 | 10.265 | 0.000 | 0.000 | 56.979 | 81.98% | 81.98% |
| t4-h64 | 0.152830 | 6 | 60.923 | 15.009 | 0.000 | 0.000 | 75.932 | 80.23% | 80.23% |
| t8-h16 | 0.167248 | 6 | 49.747 | 10.945 | 0.000 | 0.000 | 60.692 | 81.97% | 81.97% |
| t8-h32 | 0.213385 | 6 | 58.249 | 13.978 | 0.000 | 0.000 | 72.227 | 80.65% | 80.65% |
| t4-h128 | 0.245105 | 6 | 102.817 | 22.595 | 0.000 | 0.000 | 125.412 | 81.98% | 81.98% |
| t16-h8 | 0.288358 | 6 | 30.228 | 17.266 | 0.000 | 0.000 | 47.494 | 63.65% | 63.65% |
| t8-h64 | 0.305660 | 6 | 62.126 | 16.476 | 0.000 | 0.000 | 78.602 | 79.04% | 79.04% |
| t16-h16 | 0.334496 | 6 | 33.517 | 13.427 | 0.000 | 0.000 | 46.944 | 71.40% | 71.40% |
| t16-h32 | 0.426770 | 6 | 47.764 | 16.224 | 0.000 | 0.000 | 63.988 | 74.65% | 74.65% |
| t8-h128 | 0.490209 | 6 | 102.833 | 17.832 | 0.000 | 0.000 | 120.665 | 85.22% | 85.22% |
| t32-h8 | 0.576717 | 6 | 37.966 | 24.743 | 0.000 | 0.000 | 62.709 | 60.54% | 60.54% |
| t16-h64 | 0.611320 | 6 | 49.974 | 30.859 | 0.000 | 0.000 | 80.833 | 61.82% | 61.82% |
| t32-h16 | 0.668991 | 6 | 40.847 | 19.833 | 0.000 | 0.000 | 60.680 | 67.32% | 67.32% |
| t32-h32 | 0.853541 | 6 | 43.509 | 21.622 | 0.000 | 0.000 | 65.131 | 66.80% | 66.80% |
| t16-h128 | 0.980419 | 6 | 97.012 | 22.021 | 0.000 | 0.000 | 119.033 | 81.50% | 81.50% |
| t64-h8 | 1.153434 | 6 | 40.354 | 22.243 | 0.000 | 0.000 | 62.597 | 64.47% | 64.47% |
| t32-h64 | 1.222640 | 6 | 54.528 | 38.065 | 0.000 | 0.000 | 92.593 | 58.89% | 58.89% |
| t64-h16 | 1.337983 | 6 | 44.155 | 25.154 | 0.000 | 0.000 | 69.309 | 63.71% | 63.71% |
| t64-h32 | 1.707082 | 6 | 49.853 | 26.376 | 0.000 | 0.000 | 76.229 | 65.40% | 65.40% |
| t32-h128 | 1.960837 | 6 | 137.975 | 28.628 | 0.000 | 0.000 | 166.603 | 82.82% | 82.82% |
| t128-h8 | 2.306867 | 6 | 49.372 | 43.712 | 0.000 | 0.000 | 93.084 | 53.04% | 53.04% |
| t64-h64 | 2.445279 | 6 | 74.422 | 28.953 | 0.000 | 0.000 | 103.375 | 71.99% | 71.99% |
| t128-h16 | 2.675966 | 6 | 49.357 | 34.060 | 0.000 | 0.000 | 83.417 | 59.17% | 59.17% |
| t128-h32 | 3.414163 | 6 | 55.951 | 54.578 | 0.000 | 0.000 | 110.529 | 50.62% | 50.62% |
| t64-h128 | 3.921674 | 6 | 130.050 | 36.744 | 0.000 | 0.000 | 166.794 | 77.97% | 77.97% |
| t256-h8 | 4.613734 | 6 | 68.825 | 43.044 | 0.000 | 0.000 | 111.869 | 61.52% | 61.52% |
| t128-h64 | 4.890558 | 6 | 80.175 | 58.336 | 0.000 | 0.000 | 138.511 | 57.88% | 57.88% |
| t256-h16 | 5.351932 | 6 | 76.001 | 38.181 | 0.000 | 0.000 | 114.182 | 66.56% | 66.56% |
| t257-h16 | 5.372838 | 6 | 111.892 | 42.037 | 0.000 | 0.000 | 153.929 | 72.69% | 72.69% |
| t256-h32 | 6.828327 | 6 | 94.520 | 65.105 | 0.000 | 0.000 | 159.625 | 59.21% | 59.21% |
| t257-h32 | 6.855000 | 6 | 134.752 | 41.020 | 0.000 | 0.000 | 175.772 | 76.66% | 76.66% |
| t128-h128 | 7.843348 | 6 | 185.368 | 43.051 | 0.000 | 0.000 | 228.419 | 81.15% | 81.15% |
| t512-h8 | 9.227469 | 6 | 107.359 | 48.683 | 0.000 | 0.000 | 156.042 | 68.80% | 68.80% |
| t256-h64 | 9.781117 | 6 | 158.193 | 46.652 | 0.000 | 0.000 | 204.845 | 77.23% | 77.23% |
| t512-h16 | 10.703864 | 6 | 127.347 | 71.439 | 0.000 | 0.000 | 198.786 | 64.06% | 64.06% |
| t512-h32 | 13.656654 | 6 | 153.833 | 69.093 | 0.000 | 0.000 | 222.926 | 69.01% | 69.01% |
| t256-h128 | 15.686697 | 6 | 289.032 | 66.321 | 0.000 | 0.000 | 355.353 | 81.34% | 81.34% |
| t257-h128 | 15.747973 | 6 | 328.222 | 84.265 | 0.000 | 0.000 | 412.487 | 79.57% | 79.57% |
| t1024-h8 | 18.454938 | 6 | 207.411 | 91.675 | 0.000 | 0.000 | 299.086 | 69.35% | 69.35% |
| t512-h64 | 19.562234 | 6 | 252.586 | 64.532 | 0.000 | 0.000 | 317.118 | 79.65% | 79.65% |
| t1023-h16 | 21.386822 | 6 | 250.769 | 94.771 | 0.000 | 0.000 | 345.540 | 72.57% | 72.57% |
| t1024-h16 | 21.407728 | 6 | 236.277 | 78.257 | 0.000 | 0.000 | 314.534 | 75.12% | 75.12% |
| t1023-h32 | 27.286634 | 6 | 338.610 | 85.891 | 0.000 | 0.000 | 424.501 | 79.77% | 79.77% |
| t1024-h32 | 27.313308 | 6 | 351.125 | 81.860 | 0.000 | 0.000 | 432.985 | 81.09% | 81.09% |
| t512-h128 | 31.373394 | 6 | 413.475 | 190.158 | 0.000 | 0.000 | 603.633 | 68.50% | 68.50% |
| t1024-h64 | 39.124468 | 6 | 485.400 | 211.213 | 0.000 | 0.000 | 696.613 | 69.68% | 69.68% |
| t1023-h128 | 62.685512 | 6 | 758.066 | 406.739 | 0.000 | 0.000 | 1164.805 | 65.08% | 65.08% |
| t1024-h128 | 62.746788 | 6 | 719.432 | 399.389 | 0.000 | 0.000 | 1118.821 | 64.30% | 64.30% |

The following table retains each task position. C1 is down projection,
V1 is NormCache, C2 is Q up, V2 is QPrepare at T=1 or QPack otherwise,
C3 is batched key projection, and V3 is QFinalize when present. A dash
means the task is absent. The sequence column identifies every actual
task, including any extra task, using the complete kernel-name legend.

| ID | Actual profiler task name |
| --- | --- |
| K1 | `MlaSerialNormCache` |
| K2 | `MlaSerialQFinalize` |
| K3 | `MlaSerialQPack` |
| K4 | `MlaSerialQPrepare` |
| K5 | `aclnnBatchMatMul` / `BatchMatMulV2` (API/kernel endpoints; full native name retained unchanged in CSV) |
| K6 | `aclnnMatmulWeightNz_MatMulCommon_MatMulV2` |
| K7 | `aclnnMatmul_MatMulCommon_MatMulV2` |
| K8 | `aclnnMatmul_MatMulV3Common_MatMulV3` |

| Case | C1 us | V1 us | C2 us | V2 us | C3 us | V3 us | Extra task us | Actual task sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| t1-h8 | 17.990 | 3.833 | 5.395 | 3.426 | 8.116 | — | — | K6 → K1 → K6 → K4 → K5 |
| t1-h16 | 18.190 | 3.730 | 7.167 | 6.376 | 8.862 | — | — | K6 → K1 → K6 → K4 → K5 |
| t1-h32 | 18.355 | 3.855 | 14.719 | 8.252 | 9.810 | — | — | K6 → K1 → K6 → K4 → K5 |
| t2-h8 | 29.219 | 3.873 | 9.185 | 2.349 | 8.232 | 2.351 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1-h64 | 21.094 | 3.958 | 24.236 | 7.037 | 13.105 | — | — | K6 → K1 → K6 → K4 → K5 |
| t2-h16 | 28.731 | 3.896 | 12.734 | 4.103 | 8.364 | 2.328 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t2-h32 | 29.549 | 3.592 | 18.110 | 6.317 | 10.451 | 2.625 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1-h128 | 26.682 | 4.178 | 55.041 | 7.097 | 21.396 | — | — | K6 → K1 → K6 → K4 → K5 |
| t4-h8 | 29.297 | 4.275 | 8.921 | 2.492 | 7.719 | 2.459 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t2-h64 | 29.307 | 3.828 | 19.935 | 7.347 | 14.461 | 3.446 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t4-h16 | 29.305 | 4.043 | 11.716 | 3.685 | 8.423 | 2.566 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t4-h32 | 28.673 | 4.128 | 17.974 | 6.096 | 9.780 | 2.765 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t2-h128 | 30.458 | 3.866 | 56.821 | 6.734 | 20.426 | 4.069 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t8-h8 | 29.062 | 4.767 | 9.250 | 2.465 | 8.402 | 3.033 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t4-h64 | 28.871 | 4.014 | 18.578 | 7.299 | 13.474 | 3.696 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t8-h16 | 29.093 | 4.645 | 11.799 | 3.348 | 8.855 | 2.952 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t8-h32 | 30.256 | 4.990 | 18.150 | 5.865 | 9.843 | 3.123 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t4-h128 | 31.432 | 4.452 | 51.552 | 13.749 | 19.833 | 4.394 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t16-h8 | 18.127 | 9.451 | 5.437 | 3.220 | 6.664 | 4.595 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t8-h64 | 29.042 | 4.840 | 18.864 | 7.320 | 14.220 | 4.316 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t16-h16 | 18.663 | 6.259 | 7.595 | 3.669 | 7.259 | 3.499 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t16-h32 | 20.622 | 6.184 | 18.174 | 6.166 | 8.968 | 3.874 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t8-h128 | 31.328 | 5.192 | 50.820 | 7.374 | 20.685 | 5.266 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t32-h8 | 21.938 | 14.462 | 9.006 | 3.062 | 7.022 | 7.219 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t16-h64 | 17.951 | 9.750 | 19.154 | 15.391 | 12.869 | 5.718 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t32-h16 | 22.197 | 10.982 | 11.757 | 3.608 | 6.893 | 5.243 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t32-h32 | 21.276 | 10.332 | 13.700 | 6.093 | 8.533 | 5.197 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t16-h128 | 25.634 | 6.922 | 52.028 | 8.921 | 19.350 | 6.178 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t64-h8 | 24.337 | 12.047 | 9.429 | 2.582 | 6.588 | 7.614 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t32-h64 | 22.016 | 17.456 | 20.516 | 12.988 | 11.996 | 7.621 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t64-h16 | 24.333 | 13.458 | 12.708 | 4.151 | 7.114 | 7.545 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t64-h32 | 24.412 | 12.799 | 16.780 | 5.377 | 8.661 | 8.200 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t32-h128 | 34.959 | 11.038 | 80.939 | 8.619 | 22.077 | 8.971 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t128-h8 | 29.798 | 23.788 | 12.929 | 5.670 | 6.645 | 14.254 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t64-h64 | 29.326 | 13.885 | 30.961 | 7.045 | 14.135 | 8.023 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t128-h16 | 29.450 | 18.334 | 12.502 | 6.079 | 7.405 | 9.647 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t128-h32 | 28.781 | 22.954 | 17.989 | 13.650 | 9.181 | 17.974 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t64-h128 | 35.150 | 13.467 | 71.338 | 9.038 | 23.562 | 14.239 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t256-h8 | 49.381 | 25.760 | 11.853 | 6.775 | 7.591 | 10.509 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t128-h64 | 31.622 | 24.113 | 33.403 | 16.023 | 15.150 | 18.200 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t256-h16 | 48.456 | 19.020 | 18.032 | 7.341 | 9.513 | 11.820 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t257-h16 | 67.180 | 22.893 | 31.123 | 7.669 | 13.589 | 11.475 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t256-h32 | 49.932 | 31.179 | 30.695 | 15.239 | 13.893 | 18.687 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t257-h32 | 67.225 | 20.083 | 48.503 | 7.922 | 19.024 | 13.015 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t128-h128 | 46.273 | 17.127 | 116.242 | 9.553 | 22.853 | 16.371 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t512-h8 | 80.361 | 28.589 | 17.420 | 7.824 | 9.578 | 12.270 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t256-h64 | 60.498 | 21.099 | 76.946 | 10.334 | 20.749 | 15.219 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t512-h16 | 82.091 | 35.924 | 29.319 | 15.809 | 15.937 | 19.706 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t512-h32 | 83.241 | 31.825 | 47.579 | 14.529 | 23.013 | 22.739 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t256-h128 | 75.839 | 21.151 | 167.393 | 14.954 | 45.800 | 30.216 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t257-h128 | 97.022 | 25.234 | 175.797 | 18.394 | 55.403 | 40.637 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1024-h8 | 158.285 | 53.574 | 35.460 | 14.823 | 13.666 | 23.278 | — | K8 → K1 → K7 → K3 → K5 → K2 |
| t512-h64 | 109.247 | 26.498 | 111.355 | 13.582 | 31.984 | 24.452 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1023-h16 | 174.477 | 52.449 | 48.682 | 16.809 | 27.610 | 25.513 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1024-h16 | 172.587 | 49.650 | 44.829 | 10.054 | 18.861 | 18.553 | — | K8 → K1 → K7 → K3 → K5 → K2 |
| t1023-h32 | 196.734 | 42.084 | 101.814 | 12.964 | 40.062 | 30.843 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1024-h32 | 208.240 | 42.395 | 110.056 | 13.071 | 32.829 | 26.394 | — | K8 → K1 → K8 → K3 → K5 → K2 |
| t512-h128 | 119.882 | 35.266 | 223.563 | 25.683 | 70.030 | 129.209 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1024-h64 | 212.683 | 51.264 | 203.462 | 25.505 | 69.255 | 134.444 | — | K8 → K1 → K7 → K3 → K5 → K2 |
| t1023-h128 | 205.798 | 51.739 | 379.133 | 45.990 | 173.135 | 309.010 | — | K7 → K1 → K7 → K3 → K5 → K2 |
| t1024-h128 | 210.823 | 45.504 | 379.118 | 44.958 | 129.491 | 308.927 | — | K8 → K1 → K7 → K3 → K5 → K2 |

## Build and test

No new Vector/schema/build integration is needed for the optional helper.
The original A3 vendor build remains:

```bash
cd csrc
bash build.sh --pkg --soc=ascend910_93 --vendor_name=mla_serial \
  --ops=mla_serial_norm_cache,mla_serial_q_prepare,mla_serial_q_pack,mla_serial_q_finalize -j8
```

This builds the Vector vendor package; standard repository installation also
builds the Torch wrapper. Source the vendor environment before loading it.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_serial.py
```

Focused vendor/wrapper builds, the device UT and Graph/profile gates above
are separate from full-repository builds, CI or model serving. Raw samples,
failed experiments and compiler/profile evidence stay in the task archive;
the earlier optional-helper change updated Python and the existing single device UT.
This profiler-data supplement updates only this document.
