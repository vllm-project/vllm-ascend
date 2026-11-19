# GLM5Next KDA verify benchmark

Run on Ascend950DT_9581 with torch-npu, fla-npu, vllm-ascend, CANN 9.2.1 and TileLang with the Ascend mixed Cube/Vector backend.

```sh
python benchmarks/kernels/kda_verify/benchmark.py --batches 1:80 --out comparison.json
```

Each request verifies six tokens with 64 heads of dimension 128 and hidden size 4096. Timings cover QKV projection through output projection, including padding. The benchmark checks the output, convolution cache, normalized values, correction and KG records, and FP32 checkpoint before timing. Each data point reports the median of 20 alternating graph replays after warmup. Weights and inputs are synthetic. Cache slot zero is reserved and the harness assigns contiguous slots 1 through batch size. Serving integration of the mixed kernel requires scheduler-supplied slot indices and validation with real model weights.
