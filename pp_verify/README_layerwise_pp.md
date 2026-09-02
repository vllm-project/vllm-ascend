# layerwise + PP verification instructions (npu2, 8x 910B3)

## Setup

```bash
# 1) Deploy the feat/dsv4-layerwise-pp branch source code into /home/vllm-ascend
#    (includes the global-layer-addressing patch and the multi-cache-spec patch)
# 2) Start the memcache MetaService (host net, ports 5000/6000)
# 3) Start the layerwise PP2 x TP4 instance:
bash pp_verify/start_layerwise_pp2_tp4.sh
# wait ~10 min for model load; port 8006 = ready
```

## Verification

```bash
# 1) Confirm global layer addressing from the serve log:
grep "PP key space" /home/lsy/npu2_lwpp_full.log
# expected: stage 0/2 writes global layers [0, 22), stage 1/2 writes [21, 42)

# 2) Run the benchmark:
python3 pp_verify/bench_lwpp.py
# results: pp_verify/lwpp_bench_result.json
```

## Results (2026-09-02, 8x Ascend 910B3)

- PP structure: 8 workers (PP0_TP0-3 + PP1_TP0-3)
- memcache: 6+ ranks joined, zero KV transfer errors
- benchmark: 50/50 requests succeeded (see lwpp_bench_result.json)
