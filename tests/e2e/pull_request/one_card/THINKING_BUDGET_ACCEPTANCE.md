# V2 thinking-budget acceptance tests

This suite separates broad model and serving acceptance coverage from the
focused operator and request-validation fixes in
[PR #21906](https://github.com/vllm-project/vllm-ascend/pull/21906).
The request-transport boundary assertions require that fix. Validate the
combined revisions until the prerequisite is merged.

```bash
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_state_machine.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_generation.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_nonbinding.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_preemption_constrained.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_preemption_natural.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_serving.py
```

The 12 existing budget cases remain in `test_thinking_budget.py`. This PR
adds the 94 expanded state-machine cases in a separate file; their function
bodies and decorators are unchanged from the verified suite. Run the existing
entry alongside the new files for complete coverage.

The 50 offline cases are split into generation (10), exact nonbinding-budget
comparisons (8), constrained preemption (16), and natural preemption (16).
Serving adds eight configurations and retains individual response artifacts.
Models are Qwen3-1.7B, Qwen3-0.6B for drafting, and Qwen2.5-0.5B-Instruct for
ordinary-model regression. Eager/ACL Graph, greedy/random target sampling,
n-gram speculation and both greedy/probabilistic drafting are covered.

Exact nonbinding comparisons use batch-invariant computation without prefix
caching and compare matching batch positions. Default-mode generation and
serving retain normal computation and prefix caching. Preemption tests verify
actual preemption counters; speculation records draft/accepted-token counters.
Release versions without DraftLogitsCache skip the corresponding component
cases explicitly; this is not reported as exercised coverage.

Run the individual files through `/e2e` when CI capacity permits. Duration
estimates keep the preemption configurations in separate jobs. Avoid placing
all real-model cases in a single job that exceeds its execution limit.

[Historical hardware evidence](https://github.com/jryxxx/vllm-ascend/releases/tag/thinking-budget-a2-evidence-a6162d7)
contains versioned commands, logs, outputs, performance measurements and failed
experiments. A pytest pass is distinct from workflow success: historical
container-hook failures remain recorded. New revisions require their own CI
status; historical evidence does not imply that CI has passed for a new head.
