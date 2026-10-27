# Qwen3.5-27B dense MTP comparison

These scripts collect the V1/V2 observations requested by
[Issue #13736](https://github.com/vllm-project/vllm-ascend/issues/13736).
They run on two visible Ascend NPU devices with BF16 and TP=2.

## Environment

Install vLLM Ascend main and its matching vLLM main commit, preserving the
Git-derived development version. Record both commit IDs and the model revision.
The A3 environment also needs the FLA wheel specified by `Dockerfile.a3`.

The latest-main validation on 2026-10-26 used:

- vLLM Ascend: `ccc6564de16230f9aa10e531fa8faf04e704e0f6`.
- vLLM: `c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c`, from
  `.github/vllm-main-verified.commit` in that Ascend checkout.
- vLLM development version: `0.32.0.dev0+empty`.
- CANN 9.1.0, torch 2.10.0, torch-npu 2.10.0.post4, and
  flash-linear-attention-npu-a3 26.9.1+dev812ccc8.

## Run

Run from the repository root in an environment where `python` selects the
prepared Python environment and the Ascend runtime environment is loaded.
Use the same devices, model files, and launch settings for both runners.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
MODEL=/path/to/Qwen3.5-27B
SCRIPT=tests/e2e/nightly/single_node/model_runner_v2/compare_qwen35_mtp.py
mkdir -p results

for graph in eager graph; do
    graph_args=()
    if [ "$graph" = graph ]; then graph_args=(--graph); fi
    for spec in 1 3 5; do
        for runner in 0 1; do
            VLLM_USE_V2_MODEL_RUNNER=$runner python "$SCRIPT" \
                --model "$MODEL" --spec-method qwen3_5_mtp --spec-tokens "$spec" \
                --output "results/${graph}_mtp${spec}_v$((runner+1)).json" \
                "${graph_args[@]}"
        done
    done
done

python tests/e2e/nightly/single_node/model_runner_v2/summarize_qwen35_mtp.py \
    results --output report
```

Each process writes its status and completed measurements atomically. Preserve
failed-run artifacts and logs separately when rerunning a configuration.
The memory collector uses named worker extension RPCs and does not require
pickle-based RPC serialization.

Each run also records the live runner class, drafter or speculator class,
rejection-sampler class, graph-manager classes, KV-cache group descriptions,
block-table class, and cache configuration. This distinguishes V1's
`get_spec_decode_method` proposer path from V2's `init_speculator` path instead
of inferring it from throughput or output text.

### Logits probe

The probe uses one short deterministic prompt and returns ranked logprobs for
generated-token positions. Run V1 and V2 in separate processes with the same
runner environment variables as the main matrix. The top-1000 probe fits the
current A3 sampling kernel:

```bash
for runner in 0 1; do
    VLLM_USE_V2_MODEL_RUNNER=$runner python "$SCRIPT" \
        --model "$MODEL" --spec-method qwen3_5_mtp --spec-tokens 1 --batches 1 --probe-only \
        --probe-logprobs 1000 \
        --output "results/logits_top1000/v$((runner+1)).json"
done

python tests/e2e/nightly/single_node/model_runner_v2/compare_logits_probe.py \
    results/logits_top1000 \
    --output report/logits_top1000_summary.json
```

`--probe-logprobs -1` requests the full vocabulary. In the initial A3 run, V1
completed with 248,320 ranked tokens per position while V2 failed in the MRV2
`logprob.py` kernel with a UB overflow during compilation. Preserve that failed
attempt separately when checking whether the limitation has been fixed.

### KV-cache reuse probe

Run one V1 and one V2 process with prefix caching enabled. The probe first seeds
a long prompt, repeats the exact prompt, then requests a different suffix over
the same long prefix. It records prefix-cache query/hit deltas and greedy output
for each case:

```bash
for runner in 0 1; do
    VLLM_USE_V2_MODEL_RUNNER=$runner python "$SCRIPT" \
        --model "$MODEL" --spec-method qwen3_5_mtp --spec-tokens 1 --batches 1 --kv-cache-check \
        --output "results/kv_cache/v$((runner+1)).json"
done
```

The matrix JSON and KV-cache JSON retain TP-worker snapshots, V1/V2 assembly
classes, graph managers, cache counters, and greedy outputs for report analysis.

## Measurements

- Accuracy: temperature 0 and 0.8, seed 1024, top-p 1, top-k sampling disabled,
  up to 64 output tokens, short/long English, short/long Chinese, and code.
- Logits ranking: top-20 logprobs are recorded during accuracy runs. The
  separate probe records top-1000 or full-vocabulary rankings.
- Speculation: snapshots record draft counts, proposed and accepted draft
  tokens, and accepted tokens by position. Their differences produce the
  accepted draft length histogram; the bonus token is reported separately.
- Performance: batch 1/4/16, exactly 128 input tokens and 128 output tokens per
  request, one warmup batch followed by three timed repeats per batch size.
  Logprobs are disabled during timed runs.
- Latency: offline engine TTFT and queued-to-last-token time from
  `RequestStateStats`; batch wall time includes the `LLM.generate` call.
- Memory: device HBM snapshots and Torch allocated/reserved peaks are distinct
  measurements. RPC collection is outside timed generation calls.
- Graph mode: target `FULL_DECODE_ONLY`, capture sizes adjusted by
  `num_speculative_tokens + 1`, with eager drafting on both runners.

## Interpretation

The report records exact greedy token consistency. Temperature greater than
zero samples randomly, and the same seed produces the same sequence only when
the RNG algorithm, call order, and rejection-sampler draw consumption also
match. Non-greedy comparison therefore uses acceptance rate, per-position
acceptance rate, accepted-length histogram TVD, and logits top-k/KL. The probe
summary distinguishes KL over the returned top-k set, renormalized within that
set, from full-vocabulary KL. HTTP transport latency is outside these
measurements.

Performance comparisons use the median of the three warmed trials. Acceptance
thresholds must be agreed with the task reviewer; the scripts report measured
ratios without inventing a threshold. A complete JSON file indicates that the
measurement procedure completed, and is distinct from Issue acceptance.
