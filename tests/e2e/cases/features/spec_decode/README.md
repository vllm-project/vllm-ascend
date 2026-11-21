# DSpark speculative decoding: Model Runner V2 vs V1

Related issue: [#13745](https://github.com/vllm-project/vllm-ascend/issues/13745).

This directory provides runner-independent tools that collect comparable accuracy and performance data from Model Runner V1 and V2. The default pytest model pair is the existing single-card Qwen3 DSpark combination. Other model pairs and deployed multi-node services can be selected through command-line arguments or environment variables.

## Files

| File | Purpose |
| --- | --- |
| `dspark_v1_v2_collect.py` | Collect one runner arm: token sequences, top-k logprobs, acceptance metrics, and throughput |
| `dspark_v1_v2_compare.py` | Compare two collection dumps and write a Markdown report with a regression verdict |
| `dspark_v1_v2_perf.sh` | Sweep runner, speculative-token count, and batch size with `vllm bench serve`, including peak HBM |
| `dspark_v1_v2_perf_summary.py` | Summarize performance sweep JSON files into paired Markdown tables |
| `test_dspark_v1_v2_parity.py` | Pytest coverage for greedy parity, non-greedy acceptance, and comparator edge cases |

## Runner selection

An unset `VLLM_USE_V2_MODEL_RUNNER` selects V2, not V1. Set it explicitly on both arms; otherwise a supposed V1/V2 comparison can silently compare V2 with itself. The collector exits when the variable is unset.

The environment variable is read while vLLM builds its configuration, so one process cannot switch runners. Run each arm in a separate process or against a separately started service.

## Accuracy comparison

```bash
cd tests/e2e/cases/features/spec_decode

VLLM_USE_V2_MODEL_RUNNER=0 python dspark_v1_v2_collect.py \
    --model Qwen/Qwen3-8B --draft-model deepseek-ai/dspark_qwen3_8b_block7 \
    --num-speculative-tokens 7 --temperature 0 --out v1_greedy.json
VLLM_USE_V2_MODEL_RUNNER=1 python dspark_v1_v2_collect.py \
    --model Qwen/Qwen3-8B --draft-model deepseek-ai/dspark_qwen3_8b_block7 \
    --num-speculative-tokens 7 --temperature 0 --out v2_greedy.json

python dspark_v1_v2_compare.py --v1 v1_greedy.json --v2 v2_greedy.json \
    --out report_greedy.md --fail-on-regression
```

Before applying the normal verdict, the comparator detects outputs in which at least 60% of generated symbols are identical. It uses token IDs when present and non-whitespace characters otherwise. If the baseline is degenerate, the comparison is invalid rather than evidence of a candidate regression; `--fail-on-regression` exits with code 3. A degenerate candidate next to a healthy baseline remains a regression and exits with code 1.

### Remote multi-node collection

The remote mode talks to an already-running OpenAI-compatible service. It does not start or configure the cluster:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 python dspark_v1_v2_collect.py \
    --endpoint http://<host>:<port> --served-model-name <served-model-name> \
    --runner-evidence-file ./v1-server-startup.log \
    --model <target-model> --draft-model <draft-model> \
    --tensor-parallel-size <tp-size> --data-parallel-size <dp-size> \
    --num-speculative-tokens 7 --draft-sample-method greedy --enforce-eager-draft \
    --temperature 0 --out v1_greedy.json
```

Node roles, DP ranks, HCCL settings, and service startup remain deployment-specific. `--runner-evidence-file` should point to a copied startup log containing `using Model Runner V1` or `using Model Runner V2`. Without that evidence, the dump marks the runner label as an unverified client claim.

In remote mode, `--num-speculative-tokens`, `--draft-sample-method`, and `--enforce-eager-draft` describe the service; they do not reconfigure it. If the evidence contains the rank-0 `non-default args` line, the collector verifies those values against the server's `speculative_config` and aborts on a mismatch.

Remote collection requests `return_token_ids=true` by default. Use `--no-return-token-ids` for servers that do not support the extension. When token IDs are unavailable, the comparator falls back to a coarser character-level comparison and labels the report accordingly.

Collection dumps contain prompt previews, generated text and token IDs, model identifiers or paths, and the configured remote endpoint. Treat dumps as raw test evidence: review and sanitize them before attaching them to an issue or pull request.

### Accuracy matrix

Target temperature and draft sampling method are independent controls:

| Group | `--temperature` | `--draft-sample-method` | Criterion |
| --- | --- | --- | --- |
| A | `0` | `greedy` (default) | Generated token sequences must be identical |
| B | `>0` | `greedy` (default) | Compare distributions: per-position acceptance delta <= 0.1; acceptance must not be 1.0 at every position |
| C | `>0` | `probabilistic` | Same distribution checks as B, with draft logprobs used by verification |

For group C, V1 silently falls back to greedy draft sampling when the draft vocabulary is a subset of the target vocabulary (`llm_base_proposer.py`). V2 supports that mapping through draft-to-target scatter. The default Qwen3 draft model has such a mapping, while `K3DSparkForCausalLM.draft_id_to_target_id` is `None`; do not generalize a Qwen3 group-C runner difference to Kimi-K3.

## Performance comparison

The following is a single-card smoke sweep. A large multi-node model requires an independently orchestrated service topology and deployment-specific values.

```bash
MODEL=Qwen/Qwen3-8B \
DRAFT_MODEL=deepseek-ai/dspark_qwen3_8b_block7 \
TP=1 DP=1 \
SPEC_TOKENS="1 3 5 7" BATCH_SIZES="1 4 8 16" \
bash ./dspark_v1_v2_perf.sh

python dspark_v1_v2_perf_summary.py ./perf_results --out perf_report.md
```

### Eager and graph modes

The V1 DSpark draft path is forced eager, while V2 can use FULL graph mode. Run two sweeps:

1. `ENFORCE_EAGER_DRAFT=true` (default) compares V1 and V2 with an eager draft.
2. `ENFORCE_EAGER_DRAFT=false RUNNERS=1` measures the incremental V2 full-graph result.

Running only the eager sweep can understate V2 performance.

## Pytest case

```bash
pytest tests/e2e/cases/features/spec_decode/test_dspark_v1_v2_parity.py
```

The default test uses the single-card Qwen3 pair. Override it with `DSPARK_PARITY_MODEL`, `DSPARK_PARITY_DRAFT_MODEL`, `DSPARK_PARITY_TP`, `DSPARK_PARITY_DP`, `DSPARK_PARITY_QUANTIZATION`, `DSPARK_PARITY_PROMPTS`, `DSPARK_PARITY_MAX_TOKENS`, and `DSPARK_PARITY_NUM_SPEC_TOKENS`.

The two hardware smoke tests require a working vLLM Ascend and NPU environment. The remaining tests cover collection parsing, comparator validation, runner evidence, token-ID handling, and invalid-baseline behavior without loading a model.

This tool is not added to the nightly or weekly matrix here. Maintainers can choose a schedule after selecting an appropriate NPU resource topology and model budget.

## Logprob comparison scope

The public completion API exposes top-k logprobs, not a full-vocabulary distribution:

- Top-k overlap is the intersection size divided by the larger top-k set size.
- KL divergence is computed after renormalizing the union of both top-k supports. Missing entries use that side's smallest retained logprob as a floor.

The resulting KL is a truncated-support approximation. It is useful for comparisons collected with the same `--logprobs` value, but it is not full-vocabulary KL and should not be reported as an absolute model-distribution measurement.
