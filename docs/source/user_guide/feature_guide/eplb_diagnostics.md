# EPLB enablement screening

This is the first diagnostic stage: run representative requests with **EP on
and EPLB off**, then decide whether an EPLB performance trial is warranted.
Output is logs only. The diagnostic never changes expert placement, routes,
or weights and does not run MLP calibration or weight migration.

## Run

Add to the Ascend additional configuration:

```json
{"eplb_diagnostics": {"mode": "observe", "window_size": 32, "warmup_steps": 0, "max_windows": 0}}
```

Keep both upstream EPLB and Ascend dynamic EPLB disabled. `mode="off"` is the
default and disables diagnostics. `benefit` is no longer supported. Defaults
for window size and warmup are 32 aligned worker calls; warmup zero includes
the first real request. `max_windows=0` observes the entire run.

Use a representative dataset and concurrency. GSM8K is a starting point, but
its input/output lengths and prefill/decode mix must represent the intended
service before generalizing the result. All workers must use identical settings.
After generation has finished on all participating workers, finalize the logs:

```python
llm.collective_rpc("finish_eplb_diagnostics")
```

All EP participants must enter the RPC together. Independently controlled DP
engines must invoke it concurrently. Each PP stage reports its own conclusion.
There is no asynchronous teardown collective.

## What is measured

- Per-layer, per-EP-rank expert assignment counts, including owned cold experts.
  A top-k token contributes k assignments, not k input tokens.
- Window and cumulative rank peak/mean, plus a step-aware ratio that sums each
  step's rank maximum before dividing by the corresponding summed rank mean.
  This avoids cancellation when the busiest rank changes across steps/layers.
- Hot experts and Jaccard overlap between consecutive complete windows.
- Candidate placement from window A, evaluated on the actual workload in
  window B, before building the next candidate. Evaluation uses the sum of
  per-step rank maxima under each placement. It never evaluates a candidate
  on its own training window.

Candidate generation reuses the configured production policy on CPU copies:
MRv2 uses the upstream policy and model expert-group/node parameters; MRv1
supports policy 1 (DefaultEplb) and 2 (SwiftBalanceEplb). Unsupported policies
or missing parameters produce insufficient evidence rather than silently
substituting another algorithm. The diagnostic does not simulate the production
update interval, migration delay, or expert replication.

## Decision and evidence

| Conclusion | Meaning |
| --- | --- |
| `recommend_trial` | Persistent imbalance and hotspots; the previous-window candidate repeatedly reduces future peak work. Proceed to a measured EPLB performance trial. |
| `not_recommended_now` | This workload shows no persistent rank imbalance, or the candidate does not consistently improve subsequent work. |
| `insufficient_evidence` | Too few complete windows, changing hotspots, unsupported policy, or invalid/incomplete collection. |

The current screening heuristics require at least three consecutive complete
windows (two held-out evaluations):

- Step-aware rank peak/mean is at least 1.2 in at least 80% of windows.
- Hotspot Jaccard overlap is at least 0.5 in at least 80% of evaluated pairs.
  Hot experts exceed 1.2 times mean expert work and meet the top-four cutoff,
  including ties.
- Candidate peak-work reduction is at least 5% in at least 80% of pairs, and
  at least 5% across all held-out peak work combined.

These are workload-screening heuristics, not hardware performance thresholds.
Logs expose the supporting counts and signed reduction. No timing is inferred
from token counts. The summary recommends a trial when at least one layer
qualifies and lists those layers; a negative summary requires every observed
layer to have a negative assessment. Other cases remain insufficient evidence.

Example (illustrative):

```text
[EPLB diagnostic] ... layer=layer.0 ... evidence={'decision': 'recommend_trial', 'evaluated_pairs': 2, 'heldout_peak_work_reduction': 0.25, ...}
[EPLB summary] ... candidate_layers=['layer.0'] conclusion=recommend_trial scope=workload_screening timing_evidence=not_collected net_benefit=unknown speedup=not_estimated
```

A 25% peak-work reduction is not a 25% speedup. This stage does not establish
that the slowest compute rank limits inference, or that savings cover EPLB
costs. Actual device timing, migration costs and end-to-end ON/OFF validation
belong to the subsequent performance stage.

## Collection boundaries

Real source routes are collected in eager or ACL graph execution for supported
MC2/AllToAll/AllGather preparation paths. Dummy/profile/capture warmup and
padding do not contribute workload. Idle ranks still participate in aligned
collectives; bookkeeping call counts are not real work.

Each window verifies source-token conservation per TP group and per step,
expert ownership and route call counts. Invalid data, placement changes and
window gaps clear predictive evidence. Prefill, decode and mixed prefill/decode
work all participate in screening, including when EP ranks report different
phases. Phase labels remain in the logs as descriptive metadata; phase changes
or missing labels do not discard otherwise valid workload evidence. If the
workload changes, hotspot overlap and next-window candidate evaluation test
whether the earlier placement remains useful. This is assignment-count
screening, not a comparison of prefill and decode execution times. Partial tails
contribute cumulative workload but are not training or evaluation windows.

Initial support requires fixed EP groups, no redundant experts, equal expert
capacity per rank, explicit ownership and supported source-shard layouts.
Unsupported routing cannot become a balanced or positive result. Collection
retains one bounded device history window and one CPU candidate per layer;
window gathering, policy evaluation and logging add diagnostic overhead.
