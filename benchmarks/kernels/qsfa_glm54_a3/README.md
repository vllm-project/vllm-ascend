# GLM-5.4 MTP5 QSFA on Ascend A3

## Operators

This benchmark measures the decode-side quantized sparse flash attention (QSFA) operator on one A3 die. The default candidate is `qsfa_native_reference:qsfa_glm54_mtp5_blocked_kv`. It follows the original TileLang port's CANN Arch22 Cube/Vector schedule and online softmax. Gather writes BF16 KV into blocked storage for the existing QK and PV copies. The change is confined to the Gather-to-Cube workspace layout.

The default baseline is `torch_npu.npu_kv_quant_sparse_flash_attention`. To compare only the workspace change, add `--baseline-candidate qsfa_native_reference:qsfa_glm54_mtp5` to the benchmark and profiler commands. The `native_*` result columns refer to the selected baseline.

## Configuration

The synthetic fixture uses six query rows per request (MTP5), 64 heads, a 512-element compressed latent, 64 RoPE elements, 2,048 selected tokens per query row, 128-token pages, and a 131,072-token context. Each 656-byte KV cache row contains the 512 INT8 latent elements, the 64 BF16 RoPE elements, and 16 bytes of scales. Gather and the two Cube stages use a 512-token panel. The blocked layout uses four 128-token stripes per panel.

Run in this directory on Linux with an idle Ascend A3 die. Use CANN 9.2.1, `torch_npu` 2.10.0.post7 with matching PyTorch, and Ascend TileLang commit `9b5ac505c801ce038e31cc8ab6f1f83e14df1497`. Counter collection requires `msprof` and an Ascend 910_9362 die.

## Reproducibility

```bash
common=(--native-impl torch-npu --index-pattern token --rope-dim 64
        --block-i 512 --packed-gather --seed 93211)

python check_blocked_kv_boundary.py --seed 93211
python benchmark.py "${common[@]}" --batch 10 --output b10
python profile_qsfa.py "${common[@]}" --mode pipe --batch 10 --output b10-pipe

python benchmark.py "${common[@]}" --batch $(seq 1 80) --output sweep
python profile_qsfa.py "${common[@]}" --mode hbm --batch $(seq 1 80) --output sweep-hbm
python export_paired.py --graph sweep/graph.csv --hbm sweep-hbm \
    --require-full-hbm --output sweep-export
```

Use fresh output directories for each run. The paired benchmark feeds identical inputs to both operators, compares both with an independent FP32 reference, and checks captured output against eager output. The boundary test also covers filtered sparse indices and causal lengths. Nine forwards per graph timing and 23 paired `ABBA` / `BAAB` samples per batch are the defaults. `graph.csv` reports median latency and median paired speedup with a bootstrap 95% confidence interval.

Useful MFU divides the valid QK and PV FLOPs by latency and the dense BF16 peak of 280 TFLOP/s per die. Logical MBU divides the logical packed KV, query, metadata and output bytes by latency and 1,600 decimal GB/s per die. Physical MBU instead uses measured HBM traffic, including internal traffic and cache effects. The separate HBM capture assumes an otherwise idle die. The PipeUtilization and HBM sessions run separately from graph latency measurement.

Run the CPU-side fixture, profiler attribution, layout and export checks with `python -B -m unittest -v test_benchmark test_blocked_kv test_export_paired`.

## Results

The [B1–80 sweep](results/b1-b80.csv) contains paired P50 latency, confidence intervals, logical MBU, physical HBM counters and useful MFU. The [summary](results/summary.json) includes complete HBM coverage and key batches. The separate [B10 retest](results/b10-retest.csv) and [HBM capture](results/b10-hbm.csv) compare stock CANN with blocked KV. The [B10 pipe capture](results/b10-tilelang-pipe.csv) compares the original TileLang layout with blocked KV.

The HBM measurement replays the same captured inputs within each window, so cache reuse can affect physical MBU, particularly at small batch sizes.


### Fused Gather DMA and PV prefetch

The opt-in variant ping-pongs the UB output banks and writes the NoPE and RoPE groups in one Gather DMA per chunk. In the first PV output stripe it queues V and P for two K panels in the existing L1 banks before the first MMAD. Later stripes prefetch V and reuse resident P. The variant preserves the MMAD reduction order and the blocked-KV workspace size.

Run the combined candidate against stock Ascend C with the same fixture and software versions:

```bash
candidate=qsfa_native_reference:qsfa_glm54_mtp5_blocked_output_fused_dma_pv_prefetch
common=(--candidate "$candidate" --native-impl torch-npu --index-pattern token
        --context-length 131072 --topk 2048 --heads 64 --rope-dim 64
        --block-i 512 --packed-gather --warmup 12 --forwards 9 --samples 23)

python check_blocked_kv_boundary.py --candidate "$candidate" \
    --seed 93211 --native-baseline --partial-cast
for seed in 93211 93212; do
    python benchmark.py "${common[@]}" --seed "$seed" --batch 10 --output "combined-b10-$seed"
done
python profile_qsfa.py "${common[@]}" --seed 93211 --mode hbm \
    --profile-samples 4 --batch 10 --output combined-b10-hbm
python profile_qsfa.py "${common[@]}" --seed 93211 --mode pipe \
    --profile-samples 5 --batch 10 --output combined-b10-pipe

python benchmark.py "${common[@]}" --seed 93211 --batch $(seq 1 80) --output combined-sweep
python profile_qsfa.py "${common[@]}" --seed 93211 --mode hbm \
    --profile-samples 4 --batch $(seq 1 80) --output combined-sweep-hbm
python export_paired.py --graph combined-sweep/graph.csv --hbm combined-sweep-hbm \
    --require-full-hbm --output combined-sweep-export
```

| B10 seed | Ascend C P50 (µs) | TileLang P50 (µs) | Ascend C P95 (µs) | TileLang P95 (µs) | Paired speedup (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| 93211 | 296.19 | 259.25 | 296.99 | 260.40 | 1.144× (1.135–1.147) |
| 93212 | 292.74 | 256.43 | 293.45 | 257.69 | 1.144× (1.133–1.145) |

The B10 fixture produces element-wise equal output to stock QSFA in eager and captured graphs, including a boundary with trailing `-1` indices. The independent FP32 reference agrees within the BF16 tolerance. At seed 93211, physical MBU rises from 12.97% to 14.36% and useful MFU from 20.63% to 23.57%. Estimated HBM traffic is 59.54 to 60.13 MB per forward, so the measured HBM bandwidth rises with the replay rate.

The [combined B10 results](results/combined-b10.csv) record the two seeds and the independent hardware counter capture. The [B1–80 comparison](results/combined-b1-b80.csv) includes P50, P95, useful MFU, physical MBU, and output agreement. Across the 80 batches, P50 falls by 0.78–19.39% with a median decrease of 2.80%. The median decrease across B1–10 is 14.28%. P95 falls at all 80 points, with a median decrease of 2.86%. These are synthetic single-operator decode measurements.

## Attribution

The three `qsfa_native_reference*.py` modules are derived from the CANN Arch22 QSFA implementation and retain the source attribution. They are covered by the [CANN Open Software License Agreement Version 2.0](LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt). The benchmark drivers and tests carry their SPDX license identifiers.
