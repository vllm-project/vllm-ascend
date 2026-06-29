# ascend-vllm-serving-auto-benchmark

Automated benchmark skill for vLLM-Ascend serving on Huawei Ascend 910C NPU. Launches vLLM serving, runs evalscope perf stress tests, and produces a complete Markdown/CSV performance report.

## When to invoke

Use this skill when the user wants to:
- Benchmark LLM serving performance on Ascend 910C NPU
- Compare performance across different model configurations or parallelism settings
- Generate a reproducible performance report for vllm-ascend

Trigger phrases: "benchmark", "pressure test", "性能测试", "压测", "ascend benchmark", "vllm-ascend serving 性能"

## Prerequisites

Before running, the system must have:

1. **Hardware**: Huawei Ascend 910C NPU (Atlas 800T A2 or Atlas 800I A2/A3 series)
2. **CANN**: >= 8.1.0 (recommend 9.0.0)
3. **torch-npu**: matched to your PyTorch version
4. **vllm-ascend**: installed (`pip install vllm-ascend` or from source)
5. **evalscope[perf]**: installed (`pip install evalscope[perf] -U`)
6. **Model weights**: downloaded locally or accessible via HuggingFace/ModelScope path

## Inputs the user must supply

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_path` | Yes | Local path or HuggingFace/ModelScope model ID |
| `model_name` | Yes | Short name for the model (used in reports and API) |
| `tensor_parallel_size` | No | Number of NPU cards (default: 1 for 910C single card) |
| `pipeline_parallel_size` | No | Pipeline parallel size (default: 1) |
| `max_model_len` | No | Max context length (default: 8192) |
| `dtype` | No | Weight dtype: `float16`, `bfloat16`, `auto` (default: `float16`) |
| `quantization` | No | Quantization method: `w8a8`, `w4a16`, `None` (default: None) |
| `port` | No | Server port (default: 5000) |
| `parallel_levels` | No | Concurrency levels to test (default: `1 4 8 16 32 64`) |
| `requests_per_level` | No | Requests per concurrency level (default: `10 40 80 160 320 640`) |
| `input_tokens` | No | Input token length (default: 1024) |
| `output_tokens` | No | Output token length (default: 512) |
| `config_file` | No | Path to a YAML config file from `configs/` directory |

## Workflow

This skill runs in **five phases**:

### Phase 1 — Environment & Config Validation
- Detect available NPU devices via `npu-smi info`
- Validate model path exists
- Validate config file if provided (via `scripts/validate_configs.py`)
- Check evalscope and vllm-ascend installations

### Phase 2 — Launch vLLM-Ascend Server
Start the serving endpoint:
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,...  \
python -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --served-model-name <model_name> \
  --tensor-parallel-size <tp> \
  --pipeline-parallel-size <pp> \
  --max-model-len <max_len> \
  --dtype <dtype> \
  --port <port> \
  --trust-remote-code
```

Wait for server readiness by polling `/health` endpoint (max 300s).

### Phase 3 — Warm-up
Run 20 warm-up requests at concurrency=1 before measuring:
```bash
evalscope perf \
  --parallel 1 \
  --number 20 \
  --model <model_name> \
  --url http://127.0.0.1:<port>/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens <output_tokens> \
  --min-tokens <output_tokens> \
  --min-prompt-length <input_tokens> \
  --max-prompt-length <input_tokens> \
  --tokenizer-path <model_path>
```

### Phase 4 — Stress Test (evalscope perf)
Run the main benchmark across all concurrency levels:
```bash
evalscope perf \
  --parallel <parallel_levels> \
  --number <requests_per_level> \
  --model <model_name> \
  --url http://127.0.0.1:<port>/v1/chat/completions \
  --api openai \
  --stream \
  --dataset random \
  --max-tokens <output_tokens> \
  --min-tokens <output_tokens> \
  --min-prompt-length <input_tokens> \
  --max-prompt-length <input_tokens> \
  --tokenizer-path <model_path>
```

Results are saved to `outputs/<timestamp>/<model_name>/`.

### Phase 5 — Report Generation
Call `scripts/generate_report.py` to produce:
- `benchmark_report.md` — full Markdown report with tables and charts description
- `benchmark_results.csv` — raw metrics for spreadsheet import

## Output report structure

The generated `benchmark_report.md` includes:

1. **Environment Summary** — Hardware (NPU model, count), CANN version, vllm-ascend version, evalscope version
2. **Model Configuration** — model path, TP/PP size, dtype, quantization, max_model_len
3. **Workload Specification** — input/output token lengths, dataset type, request distribution
4. **Performance Overview Table** — per-concurrency: RPS, success rate, throughput (tokens/s)
5. **Latency Distribution Table** — per-concurrency: avg/P50/P90/P99 latency, TTFT, TPOT
6. **Best Configuration Summary** — peak throughput point, best TTFT point, recommended concurrency
7. **Fairness & Reproducibility** — exact server command, evalscope command, git commit hashes

## Constraints & fairness rules

- Always restart the server between different TP/PP configurations; never reuse a warm server across configs
- Warm up before every measurement run
- Record exact vllm-ascend git commit: `git -C $(pip show vllm-ascend | grep Location | awk '{print $2}')/vllm_ascend rev-parse HEAD`
- Report NPU utilization if `npu-smi` is available
- All concurrency levels must use identical model weights and tokenizer
- Never compare results from different NPU types in the same report

## Config files

YAML configs in `configs/` encode model-specific defaults. The user can pass `--config_file configs/qwen2.5-7b-instruct.yaml` to skip specifying individual flags. See `configs/qwen2.5-7b-instruct.yaml` for the format.

## Example invocations

**Quick single-card benchmark:**
```
/ascend-vllm-serving-auto-benchmark
model_path=/data/models/Qwen2.5-7B-Instruct
model_name=Qwen2.5-7B-Instruct
```

**Multi-card benchmark with custom config:**
```
/ascend-vllm-serving-auto-benchmark
config_file=configs/deepseek-r1-7b.yaml
tensor_parallel_size=2
```

**Full custom benchmark:**
```
/ascend-vllm-serving-auto-benchmark
model_path=/data/models/Qwen2.5-72B-Instruct
model_name=Qwen2.5-72B-Instruct
tensor_parallel_size=8
max_model_len=16384
dtype=bfloat16
parallel_levels="1 8 16 32 64 128"
requests_per_level="10 80 160 320 640 1280"
input_tokens=2048
output_tokens=1024
```

## Handling failures

- **Server fails to start**: Check CANN env vars (`source /usr/local/Ascend/ascend-toolkit/set_env.sh`), NPU visibility (`npu-smi info`), and available NPU memory
- **OOM on NPU**: Reduce `--max-model-len` or increase `--tensor-parallel-size`
- **evalscope not found**: Run `pip install evalscope[perf] -U`
- **Low throughput / high latency**: Check if ACLGraph is enabled (`VLLM_ASCEND_ENABLE_ACLGRAPH=1`), ensure no CPU-NPU sync bottlenecks
- **All requests fail**: Verify server health at `curl http://127.0.0.1:<port>/health`
