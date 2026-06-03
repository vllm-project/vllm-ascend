# Fuxi Alpha HSTU Example

This example demonstrates HSTU inference with vLLM Ascend. The scripts are
templates: set local model, dataset, Mooncake, and profiler paths through
environment variables before running them.

## Set Environment Variables

In Linux shells, use `export` when a variable should be visible to child
processes such as `bash`, `python3`, or `vllm`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/fuxi_alpha_model_or_this_directory
export DATASET_PATH=/path/to/kuairand/dataset
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
export LIB_FBGEMM_NPU_API_SO_PATH=/path/to/libfbgemm_npu_api.so
```

These values only apply to the current terminal session. To set a variable for a
single command, put it before the command:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model bash run_model.sh
```

To make the settings persistent, add the `export ...` lines to `~/.bashrc` or
your shell profile, then reload it with `source ~/.bashrc`.

## Run The Demo

```bash
cd examples/fuxi_alpha
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/fuxi_alpha_model_or_this_directory \
DATASET_PATH=/path/to/kuairand/dataset \
MOONCAKE_CONFIG_PATH=/path/to/mooncake.json \
bash run_model.sh
```

`run_model.sh` calls `run.sh` with a 1B random-model configuration. You can
override parameters by editing the command in `run_model.sh` or by invoking
`run.sh` directly.

Example direct run:

```bash
MODEL_PATH=/path/to/fuxi_alpha_model_or_this_directory \
DATASET_PATH=/path/to/kuairand/dataset \
bash run.sh \
  --embedding_dim 4096 \
  --num_heads 16 \
  --dim 256 \
  --max_seq_len 2048 \
  --max_batch_size 2 \
  --use_random 1 \
  --aclgraph 1 \
  --candidate_num 256 \
  --has_ffn 1 \
  --max_vocab_size 4096 \
  --concat_batch 1 \
  --profiler 0 \
  --max_model_len 8832 \
  --range 2 \
  --graph_step 512 \
  --block_size 128
```

## Generate Benchmark Prompts

```bash
cd examples/fuxi_alpha
MODEL_PATH=/path/to/fuxi_alpha_model_or_this_directory \
DATASET_PATH=/path/to/kuairand/dataset \
HSTU_PROMPT_OUTPUT_DIR=/path/to/output \
HSTU_PROMPT_USER_NUM=1000 \
python3 prompts2json.py \
  --embedding_dim 4096 \
  --num_heads 16 \
  --dim 256 \
  --max_seq_len 2048 \
  --max_batch_size 2 \
  --use_random 1 \
  --aclgraph 1 \
  --candidate_num 256 \
  --has_ffn 1 \
  --max_vocab_size 4096 \
  --concat_batch 1 \
  --profiler 0 \
  --max_model_len 8832 \
  --range 2 \
  --graph_step 512 \
  --is_async 0 \
  --block_size 128
```

The generated decode prompt file can be passed to
`benchmarks/pd_disaggregation/gr/mix_pd_benchmark.sh` as `DATASET_PATH`.

## Required Environment Variables

- `MODEL_PATH`: directory containing `config.json` and model artifacts. For
  random-model testing, this directory can be the example directory.
- `DATASET_PATH`: local KuaiRand-style dataset directory used by the example.
- `LIB_FBGEMM_NPU_API_SO_PATH`: path to `libfbgemm_npu_api.so`.

## Optional Environment Variables

- `ASCEND_RT_VISIBLE_DEVICES`: NPU device list.
- `MOONCAKE_CONFIG_PATH`: Mooncake configuration path.
- `VLLM_TORCH_PROFILER_DIR`: profiler output directory.
- `HSTU_PROMPT_OUTPUT_DIR`: output directory for generated prompt JSONL files.
- `HSTU_PROMPT_USER_NUM`: number of users used for generated prompts.
