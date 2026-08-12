# Generated performance datasets

Nightly and weekly AISBench cases can generate deterministic text datasets on
the CI runner instead of downloading them from ModelScope. This is opt-in: an
existing case that only defines `dataset_path` or `dataset_path_local` keeps its
previous behavior.

Generated datasets are supported only for performance cases. Accuracy and
multimodal cases must continue to use their real datasets.

## Fixed-length dataset

```yaml
benchmarks:
  perf:
    case_type: performance
    dataset_generator:
      type: fixed
      input_len: 3500
      num_samples: 2800
      seed: 42
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 2800
    max_out_len: 1500
    batch_size: 700
    request_rate: 0
    baseline: 1
    threshold: 0.97
```

## Prefix dataset

```yaml
benchmarks:
  perf_prefix75:
    case_type: performance
    dataset_generator:
      type: prefix
      input_len: 3500
      num_samples: 210
      prefix_ratio: 0.75
      prefix_num: 1
      warmup_prefix: true
      dp: 4
      seed: 42
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 210
    max_out_len: 1500
    batch_size: 70
    request_rate: 0
    baseline: 1
    threshold: 0.97
```

Generated files are cached under `/tmp/vllm_ascend_datasets` by default. Set
`VLLM_ASCEND_DATASET_CACHE` to use another cache directory. Each cache
directory name contains the dataset shape, model basename, and a configuration
hash, for example `GSM8K-in3500-num2800-DeepSeek-V4-<hash>`.
Each generated dataset directory contains `test.jsonl` and an empty
`train.jsonl`, as required by the AISBench GSM8K dataset loader.

The same benchmark mapping works in the single-node, internal-DP, and
external-DP YAML formats because dataset resolution is shared by
`tools/aisbench.py`.

When `warmup_prefix` is enabled, each distinct prefix is repeated `dp` times in a
prefix-only dataset. AISBench runs that dataset first with `batch_size=dp` and
`max_out_len=1`, then runs the full dataset against the same server process.
This mirrors the original prefix tool's behavior, but it does not guarantee
that a load balancer routes exactly one warmup request to every DP rank.

The generator reads a GSM8K-format JSONL file from `GSM8K_SOURCE_PATH` in
`tools/benchmark_dataset.py`. Its default value is
`/mnt/share/c00893695/datasets/GSM8K.jsonl`. Change this single constant when
running in another local or CI environment; benchmark YAML files do not carry
the source path. Rows must contain a non-empty `question` field. Fixed datasets
sample questions with replacement, while prefix datasets select distinct
questions for prefixes and sample questions with replacement for suffixes.
