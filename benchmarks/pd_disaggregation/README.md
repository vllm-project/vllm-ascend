# PD Disaggregation Benchmarks

This directory contains example scripts for running vLLM Ascend PD disaggregation
serving and benchmark flows. The scripts use environment variables for local
paths so private model, dataset, and profiler directories are not committed.

## Set Environment Variables

In Linux shells, use `export` when a variable should be visible to child
processes such as `bash`, `python3`, or `vllm`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.example.json
export MODEL_PATH=/path/to/model
export DATASET_PATH=/path/to/dataset.json
```

These values only apply to the current terminal session. To set a variable for a
single command, put it before the command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model bash mix_pd_serve.sh
```

To make the settings persistent, add the `export ...` lines to `~/.bashrc` or
your shell profile, then reload it with `source ~/.bashrc`.

## Start Mooncake Master

The online serving scripts use Mooncake-backed KV transfer. Start the Mooncake
master service before running any serving or benchmark flow:

```bash
cd examples/disaggregated_prefill_v1
export MOONCAKE_CONFIG_PATH="${PWD}/mooncake.example.json"
bash start_mooncake_master.sh
```

For multi-node runs, copy `mooncake.example.json` and set
`local_hostname` and `master_server_address` to the reachable IP address and
port of the Mooncake master node. The default `127.0.0.1:50088` is for
single-node local testing.

## Fuxi Alpha / HSTU Flow

Start the serving process:

```bash
cd benchmarks/pd_disaggregation/gr
ASCEND_RT_VISIBLE_DEVICES=0 \
MOONCAKE_CONFIG_PATH=/path/to/mooncake.example.json \
MODEL_PATH=/path/to/vllm-ascend/examples/fuxi_alpha \
bash mix_pd_serve.sh
```

Run the benchmark client in another shell:

```bash
cd benchmarks/pd_disaggregation/gr
DATASET_PATH=/path/to/hstu_prompts_d.jsonl \
MODEL_PATH=/path/to/vllm-ascend/examples/fuxi_alpha \
PORT=8100 \
bash mix_pd_benchmark.sh
```

You can generate HSTU prompt JSONL files from the Fuxi Alpha example with
`examples/fuxi_alpha/prompts2json.py`.

## Qwen2.5 Flow

Start the serving process:

```bash
cd benchmarks/pd_disaggregation/Qwen2.5
ASCEND_RT_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
bash mix_pd_serve.sh
```

Run the benchmark client:

```bash
cd benchmarks/pd_disaggregation/Qwen2.5
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
DATASET_PATH=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
PORT=8105 \
bash mix_pd_benchmark.sh
```

## Useful Environment Variables

- `ASCEND_TOOLKIT_PATH`: Ascend toolkit root. Defaults to
  `/usr/local/Ascend/ascend-toolkit/latest`.
- `ASCEND_RT_VISIBLE_DEVICES`: NPU device list. Defaults are examples only.
- `MODEL_PATH`: local model path.
- `DATASET_PATH`: benchmark dataset path.
- `MOONCAKE_CONFIG_PATH`: Mooncake configuration path.
- `MOONCAKE_PORT`: Mooncake master port used by
  `examples/disaggregated_prefill_v1/start_mooncake_master.sh`. Defaults to
  `50088`.
- `VLLM_TORCH_PROFILER_DIR`: optional profiler output directory.
- `RESULT_DIR`: benchmark result directory.
- `PORT`: serving port.
