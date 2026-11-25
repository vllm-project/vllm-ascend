# Optimized single-launch ReplaySSM verify

The BF16 kernel processes six verification tokens per request from QKV projection through output projection. The CSV includes the native path, PR #16634, the published multi-kernel TileLang path, the first single-kernel measurement, and the optimized kernel. Positive savings indicate a latency reduction from the published multi-kernel path. Each optimized point is the median of 11 device tasks. Model weights are packed before timing.

Atlas 850 的 batch 49 另做了交错复测。固定权重地址时，旧版与优化后单核的设备耗时中位数分别为 554.682 和 550.206 µs。轮换四份数值相同、地址不同的权重时，分别为 553.285 和 548.454 µs。每种配置下两版各测 20 次，两种执行顺序各占 10 次。 [配对实测样本](atlas-850-b49-paired-device-times.json)

Run the following commands from this directory with the captured replay fixture, CANN 9.2.1, the Ascend TileLang environment, torch-npu and vllm-ascend available. Use a fresh profiler output directory for each board.

```sh
batches=$(seq -s, 1 50)
msprof --output=profile --task-time=on --msproftx=on --application="python verify_whole_kda.py --trace trace.pt --out verified.json --batches $batches --mstx"
python extract_whole.py profile timing.json
```

[Latency and accuracy](mtp5-optimized-whole-kda-b1-50.csv) · [Three device timing curves](mtp5-optimized-whole-kda-b1-50.svg) · [Individual device samples and numerical checks](mtp5-optimized-whole-kda-b1-50.json)
