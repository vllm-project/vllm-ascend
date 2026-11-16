# KV Pool SLO Concurrency Search

`slo_search.py` uses AISBench to find the highest concurrency that meets latency and request success rate requirements, saving latency, throughput, and cache hit metrics for each trial. For example, it can find the highest concurrency and corresponding throughput with **TTFT P90 < 10 s and TPOT P90 < 20 ms**.

The tool requires Python 3.9+ and the `regex` package, which is included in vllm-ascend's requirements. It invokes `aisbench_test.py` from the external [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) project through its CLI:

```text
slo_search.py → aisbench_test.py → AISBench → model service
```

## Prerequisites

1. Start the model service and storage backend following the [KV Cache Pool deployment guide](../../../../../docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md), and verify that the benchmark environment can access the service.
2. Set up a complete, separate checkout of `aisbench_auto_tools_prefix` and its AISBench dependencies. Configure the model, tokenizer, service address, authentication, and dataset paths in `config.py`. Prefix tests also require `POD_INFO` or `HOST_IP:HOST_PORT` for cache metric queries.
3. Verify that the benchmark Python interpreter can run `aisbench_test.py` and that `ais_bench` is available in the subprocess `PATH`.
4. Ensure that the external project and AISBench configuration, data, and log directories are writable. The external project generates data and updates configuration and dataset symlinks. Do not run multiple searchers or manual benchmarks concurrently against the same checkout.

The tool generates load, searches concurrency, and records results. It does not start the model service, configure KV Pool, or clear the service cache.

The following examples use Bash. From the vllm-ascend repository root, enter the tool directory:

```bash
cd vllm_ascend/distributed/kv_transfer/utils/slo_search
```

Replace `/path/to/aisbench_auto_tools_prefix` with your configured external project directory. The searcher and external project can reside in different directories. Install `regex` into the environment running `slo_search.py`, for example with `uv pip install --python /path/to/search-python regex`. To use a separate Python environment for benchmarking, specify its interpreter with `--python /path/to/python`; this option only controls the external benchmark subprocess.

## Usage Examples

### Search Using P90

Search up to concurrency 128 with 2048 input tokens, 1024 output tokens, TTFT P90 < 10 s, and TPOT P90 < 20 ms:

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --tpot-limit-ms 20 \
  --max-concurrency 128 \
  -- --input_len 2048 --output_len 1024
```

The standalone `--` separates searcher arguments, such as `--max-concurrency`, from arguments passed unchanged to `aisbench_test.py`, such as `--input_len`.

By default, the search starts at concurrency 8, uses four times the concurrency as the full-workload request count for each trial, evaluates SLOs using P90, and performs no extra boundary confirmation. Each trial displays mean and P90 TTFT/TPOT values. When the search finishes, it prints a result summary and the output directory.

### Prefix Cache: 64K Input, 1K Output

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --tpot-limit-ms 20 \
  --max-concurrency 128 \
  --prefix-num-equals-concurrency \
  -- \
  --input_len 65536 \
  --output_len 1024 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test
```

`--prefix_test` asks the external project to warm up prefixes. `--prefix-num-equals-concurrency` sets the number of distinct prefixes to the trial's concurrency. For example, concurrency 8 uses 8 distinct prefixes and 32 full-workload requests by default; warmup requests are excluded from those 32 requests.

`--repeat_rate 0.9` means that the shared prefix accounts for approximately 90% of the generated input. Refer to each trial's metrics for the actual cache hit rate.

### Mean TTFT Only: 64K Input, 1 Output Token

```bash
python slo_search.py \
  --project-dir /path/to/aisbench_auto_tools_prefix \
  --ttft-limit-ms 10000 \
  --percentile mean \
  --max-concurrency 128 \
  --prefix-num-equals-concurrency \
  -- \
  --input_len 65536 \
  --output_len 1 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test
```

This example evaluates latency using only mean TTFT < 10 s while still recording TTFT P90. TPOT does not apply to single-token output: it appears as `N/A` in the terminal and `null` in JSON, with `tpot_status` set to `not_applicable_single_token`. Do not specify `--tpot-limit-ms` for this workload.

### Optional Boundary Confirmation

Add `--confirm-runs 2` to the searcher arguments to rerun the candidate boundary and neighboring points twice each. For example, if concurrency 10 passes and 11 fails, the default `--boundary-radius 1` confirms points in the order 11, 10, 9. Both confirmation trials at a point must pass for it to be confirmed. Search trials do not count toward confirmation runs.

If no point in the neighboring range is confirmed, the tool continues downward one concurrency at a time until a point passes or the minimum concurrency is reached. With `--boundary-radius 0`, the initial confirmation range contains only the candidate point.

## SLO Evaluation and Search Strategy

A trial must meet every specified latency limit and the minimum request success rate:

```text
If a TTFT limit is specified: TTFT for the selected statistic < ttft-limit-ms
If a TPOT limit is specified: TPOT for the selected statistic < tpot-limit-ms
Request success rate >= min-success-rate
```

- Latency limits are in milliseconds and use strict `<` comparisons. A TPOT of 20 ms does not meet a `< 20 ms` limit. The default request success rate requirement is 100%.
- Specify at least one TTFT or TPOT limit. Both metrics share `--percentile`, which accepts `mean` or a percentile present in the logs, such as `90` or `99`. Mixing statistics is not supported.
- Every trial records mean and P90 values, but SLO evaluation uses only the selected statistic. Both metrics meeting their P90 limits does not mean that the same 90% of requests meet both conditions.
- `--stage` selects latency, throughput, and request statistics from the corresponding Stage in the summary tables; the default is `total`. It does not change load generation or enable steady-state testing. Prefix warmup statistics are excluded from SLO evaluation.

| Phase | Behavior |
| --- | --- |
| `expand` | Double concurrency from the starting point until the first failure or the search limit |
| `bisect` | Bisect the boundary between passing and failing points; also search downward if the starting point fails |
| `confirm` | Optionally rerun the candidate boundary and neighboring points, requiring all confirmation trials at each point to pass |

With the default `--confirm-runs 0`, results come from search trials. When confirmation is enabled, results come from the highest confirmed concurrency and its confirmation trials. The search assumes approximately monotonic SLO behavior and does not scan every concurrency. Keep input/output lengths, prefix configuration, service state, and the selected statistic consistent when comparing results.

## Argument Reference

Run `python slo_search.py --help` for the full help text.

### SLOs and Search Range

| Argument | Default | Description |
| --- | --- | --- |
| `--ttft-limit-ms` | Unset | Strict TTFT upper limit in ms; omit to leave TTFT unconstrained |
| `--tpot-limit-ms` | Unset | Strict TPOT upper limit in ms; omit to leave TPOT unconstrained |
| `--percentile` | `P90` | Latency statistic; use `mean` for the average |
| `--stage` | `total` | AISBench Stage name to read |
| `--min-success-rate` | `1.0` | Minimum request success rate, in `(0, 1]` |
| `--min-concurrency` | `1` | Minimum search concurrency |
| `--start-concurrency` | `8` | Starting concurrency |
| `--max-concurrency` | `1024` | Maximum search concurrency |
| `--confirm-runs` | `0` | Extra confirmation trials per point; `0` disables confirmation |
| `--boundary-radius` | `1` | Number of neighboring integer concurrency points on each side of the candidate in the initial confirmation range |

The concurrency range must satisfy `1 <= min-concurrency <= start-concurrency <= max-concurrency`. For example, if the maximum is 4, the starting concurrency must also be adjusted to at most 4.

### Request Counts and Prefixes

| Argument | Default | Description |
| --- | --- | --- |
| `--requests-per-concurrency` | `4` | Full-workload requests per unit of concurrency in each trial; must be a positive integer |
| `--min-data-num` | `0` | Minimum full-workload request count per trial |
| `--data-num` | Unset | Fixed full-workload request count; must be at least the maximum search concurrency |
| `--prefix-num-equals-concurrency` | Disabled | Set `--prefix_num` to the trial's concurrency |

Without `--data-num`, each trial uses `max(min-data-num, current concurrency × requests-per-concurrency)` full-workload requests.

When linking the prefix count to concurrency, pass `--dataset_type prefix_cache` after `--` and do not also pass `--prefix_num` or `--dataset`. Otherwise, a fixed `--prefix_num` can be passed through. `--prefix_test` controls warmup; warmup concurrency depends on the external project's version and configuration.

When using an existing `--dataset`, the dataset determines the actual request count. The tool does not expand it automatically; ensure that it contains enough data for the target concurrency.

The searcher manages `--concurrency`, `--request_rate 0`, `--repeat 1`, and `--data_num`; do not place these arguments after `--`. Passing through `--test_accuracy` is also unsupported.

### Paths and Execution Control

| Argument | Default | Description |
| --- | --- | --- |
| `--project-dir` | Script directory | External project directory containing `aisbench_test.py` |
| `--python` | Current Python interpreter | Python interpreter used to run the external project |
| `--output-dir` | `<project-dir>/slo_results/<timestamp>` | Output directory for this search; must not already exist |
| `--timeout` | `3600` seconds | Subprocess timeout per trial, including data preparation, warmup, and the full benchmark |
| `--latency-unit` | Unset | Specify `ms` or `s` only when the logs omit time units |

`--timeout` applies to each trial, not the entire search. Any trial timeout terminates that trial's subprocess tree and aborts the search.

## Results and Logs

By default, results are stored under the external project directory. An explicitly specified relative `--output-dir` is resolved against the current working directory. Each search requires a new directory and does not overwrite existing results.

```text
<project-dir>/slo_results/<timestamp>/
├── settings.json          Search configuration and passthrough arguments
├── summary.json           Final result or abort reason
├── history.csv            Metrics for each trial
├── history.jsonl          Full trial records and actual commands
└── trial_0001/            Subsequent trials are numbered sequentially
    ├── command.json       Command executed for this trial
    ├── console.log        Subprocess output and metric summary after successful parsing
    ├── metrics.txt        Metric summary after successful parsing
    └── result.json        Trial metrics and outcome, or error reason
```

### Final Result

```bash
RESULT_DIR=/path/to/result_directory
python -m json.tool "$RESULT_DIR/summary.json"
```

| Field | Meaning |
| --- | --- |
| `status` | `complete` when the search finishes; `aborted` on an error, with the reason in `error` |
| `highest_passing_concurrency` | Highest passing concurrency; `null` if no passing point was found |
| `result_source` | `search` for search trials or `confirm` for confirmation trials |
| `active_slo_metrics` | Latency metrics used for evaluation, such as `["ttft"]` |
| `confirmation_performed` | Whether extra confirmation was enabled |
| `result_trial_ids` | Trial IDs used for the final result |
| `throughput.*_tokens_per_s_mean` | Mean output or prefill throughput across result trials, in token/s |
| `throughput.requests_per_s_mean` | Mean request throughput across result trials, in req/s |
| `worst_ttft_ms`, `worst_tpot_ms` | Maximum selected latency statistic across result trials, in ms |
| `result_cache_metrics` | Cache metrics from result trials |
| `confirmation_trial_ids`, `confirmation_cache_metrics` | Confirmation results; empty arrays when confirmation is disabled |
| `at_search_limit` | Whether the search limit was reached; if `true`, consider increasing the limit and rerunning |

If no point passes, throughput and latency result fields are omitted. On an abort, `summary.json` records only the aborted status, error, and recorded trial count.

Without confirmation, throughput comes from the single search trial at the final concurrency. With confirmation, throughput is averaged over confirmation trials at the selected concurrency. The "worst P90" is the maximum of those trials' individual P90 values, not a P90 recalculated from pooled requests.

### Per-Trial Metrics

```bash
cat "$RESULT_DIR/history.csv"
cat "$RESULT_DIR/trial_0001/metrics.txt"
```

History records include concurrency, request count, phase, latency, throughput, request success rate, and cache metrics. `status` is `pass`, `fail`, or `error`, indicating a passing, failing, or invalid trial respectively. Invalid trials abort the search.

`ttft_mean_ms`, `ttft_p90_ms`, `tpot_mean_ms`, and `tpot_p90_ms` record mean and P90 values. `ttft_ms` and `tpot_ms` hold the selected SLO statistic, also in ms. Other selected percentiles are displayed separately in the terminal.

Cache hit rates are recorded as `hbm_hit_rate_pct` and `external_hit_rate_pct` in the range 0–100, along with their respective `*_hits`, `*_queries`, and `*_cache_status` fields:

- Hit rates are calculated from aggregate hit/query counts over the full benchmark, without averaging per-node percentages. `external` retains its meaning as the external KV cache.
- `no_queries` means the query count is zero and the hit rate is undefined. `not_reported` means the logs did not provide the metric. Both appear as `N/A` in the terminal, `null` in JSON, and empty cells in CSV.
- Cache metrics cover the full benchmark interval. Selecting another Stage does not make them specific to that Stage.

Prefill throughput comes from AISBench's `Prefill Token Throughput`. Throughput and cache metrics are recorded for observation and do not affect SLO evaluation.

To save the searcher's complete terminal output, use `set -o pipefail` and append `2>&1 | tee slo_search.log` to the command. Each trial's subprocess output is stored in `trial_*/console.log`.

## Troubleshooting

| Situation | Action |
| --- | --- |
| Missing latency, throughput, or request statistics | Inspect the trial's `console.log`. Logs must include TTFT/TPOT mean, P90, and the selected percentile, plus Prefill/Output Token Throughput, Request Throughput, Total Requests, and Failed Requests. TPOT may be absent for single-token output |
| Missing time units | Specify `--latency-unit ms` or `--latency-unit s`; units present in the logs take precedence |
| Missing or invalid cache summary | With `--prefix_test`, the full benchmark must include a `Prefix cache hit summary` with HBM/external hit and query counts in `ALL_PODS`. Check node configuration and service `/metrics` access |
| Output directory already exists | Use a new `--output-dir`, or omit it to generate a directory automatically |
| Project directory is in use | `.slo_search.lock` prevents overlapping searchers. A `WARNING: could not remove lock` message reports a cleanup failure without changing the search result. Check permissions and verify that related processes have stopped before deleting a stale lock |
| Subprocess exit, timeout, or metric parsing failure | Inspect `error` in `summary.json` and the trial's `console.log`. Recorded trials are preserved. Errors abort the search instead of being treated as ordinary SLO failures |

The log format must match the summary tables supported by the tool. Missing percentiles or Stages, ERROR/Traceback output, skipped cache nodes, or invalid counters cause errors. The tool does not reuse warmup results or metrics from other trials.

| Exit Code | Meaning |
| --- | --- |
| `0` | Search completed with a passing point |
| `1` | Search completed without a passing point |
| `2` | Argument, execution, or parsing error |
