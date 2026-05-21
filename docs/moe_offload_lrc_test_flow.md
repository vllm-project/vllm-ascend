# MoE Offload LRC Cache Test Flow

This file is for the agent that validates the `moe_offload_v2.0` branch.
The goal is to verify that expert offload now uses the LRC cache policy and
that every decode-layer update can report which routed experts were already
resident on device and which experts had to be loaded again.

## What changed

- Added `vllm_ascend/expert_offload/lrc_policy.py`.
- `ExpertOffloadManager.update_weights()` now accepts `topk_weights`.
- Decode path computes:
  - `already_there = needed & on_device`
  - `need_to_load = needed - already_there`
- Runtime logs expose per-layer cache hits and misses.
- Config options were added under `additional_config.expert_offload_config`:
  - `cache_policy_enabled`
  - `cache_recent_window`
  - `cache_ema_beta`
  - `cache_recent_weight`
  - `cache_ema_weight`
  - `cache_router_weight`
  - `cache_age_weight`
  - `cache_stats_log_interval`

## Static checks

Run from the repo root:

```bash
python -B -c "from pathlib import Path; files=['vllm_ascend/expert_offload/lrc_policy.py','vllm_ascend/expert_offload/expert_offload_manager.py','vllm_ascend/ascend_config.py','vllm_ascend/ops/fused_moe/fused_moe.py','vllm_ascend/quantization/methods/w8a8_dynamic.py','tests/ut/expert_offload/test_lrc_policy.py']; [compile(Path(p).read_text(), p, 'exec') for p in files]; print('syntax checks passed')"
```

Expected result:

- The command prints `syntax checks passed`.
- There is no `SyntaxError`.

If the test machine has the vLLM Ascend test environment, also run:

```bash
pytest tests/ut/expert_offload/test_lrc_policy.py
```

Expected result:

- All tests in `test_lrc_policy.py` pass.
- If pytest fails before collecting tests because `torch_npu` is missing, that is an environment failure, not a policy failure.

## Runtime config

Expert offload and the LRC cache policy must both be enabled. For focused validation,
make the cache log every decode call by setting `cache_stats_log_interval` to `1`.

Example `--additional-config`:

```bash
--additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"expert_offload_config":{"expert_offload":true,"num_device_experts":6,"num_device_layers":2,"cache_policy_enabled":true,"cache_stats_log_interval":1}}'
```

Notes:

- If the model really has 6 total experts per layer and `num_device_experts` is also `6`, then after warmup the expected steady-state behavior is that all routed experts are hits and `last_miss=[]`.
- If total experts is greater than `num_device_experts`, misses are expected. Success is not zero misses; success is that already-resident experts appear in `last_hit` and are not loaded again.
- Keep the rest of the user's online launch script unchanged unless the environment requires normal deployment changes.

## Runtime test

1. Start `vllm serve` with expert offload enabled and `cache_stats_log_interval=1`.
2. Send several decode-heavy requests. Repeated prompts are useful because they should stabilize routing and improve hit rate.

Example:

```bash
curl http://127.0.0.1:7000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds","messages":[{"role":"user","content":"写一段 800 字的技术说明，主题是 MoE 专家缓存命中率。"}],"max_tokens":512,"temperature":0}'
```

Run the same request 5 to 10 times.

## Logs to check

Search the server log for:

```text
[EXPERT-OFFLOAD-CACHE]
```

Expected log shape:

```text
[EXPERT-OFFLOAD-CACHE] layer=<layer> cache_step=<n> calls=<n> policy_step=<m> hit_rate=<r> hits=<h> misses=<m> last_hit=[...] last_miss=[...] resident=[...]
```

`cache_step` / `calls` is the number of cache updates for that layer.
`policy_step` is the number of routed token rows observed by the LRC policy for that layer.

Also useful:

```text
[UPDATE-W] l=<layer> cache_hit=[...] cache_miss=[...]
```

These two lists are the key validation signal:

- `cache_hit` / `last_hit`: experts already prefetched or resident on device; they should not be copied again.
- `cache_miss` / `last_miss`: routed experts not currently resident; these require CPU to NPU loading.

## Success criteria

The change is considered tested successfully when all of the following are true:

1. The server starts with `expert_offload_config.expert_offload=true` and `expert_offload_config.cache_policy_enabled=true`.
2. Decode requests complete successfully and return valid model output.
3. Logs contain `[EXPERT-OFFLOAD-CACHE]` for active MoE layers.
4. Each cache log has sane counters:
   - `calls` increases over time for the same layer.
   - `hits + misses == total routed expert requests accumulated for that layer`.
   - `0.0 <= hit_rate <= 1.0`.
   - `last_hit`, `last_miss`, and `resident` contain valid expert ids.
5. For a 6-expert model/cache case:
   - expert ids in `last_hit`, `last_miss`, and `resident` are within `0..5`;
   - after warmup, repeated similar requests should show increasing hit rate;
   - if all 6 experts fit on device, steady-state `last_miss` should usually become `[]`.
6. No log shows repeated loading for an expert that is already listed in `cache_hit` for the same layer update.
7. There are no crashes, shape errors, invalid config errors, or NPU copy errors from the modified offload path.

## Failure criteria

Treat any of the following as a failed validation:

- No `[EXPERT-OFFLOAD-CACHE]` logs appear while expert offload is enabled and decode requests are running.
- `cache_hit` is always empty after many repeated decode requests, even when the cache can hold the routed experts.
- An expert appears in both hit and miss lists for the same layer update.
- Expert ids are outside the valid range, for example outside `0..5` in a 6-expert setup.
- `hit_rate` is negative, greater than 1, or does not match `hits / (hits + misses)`.
- The process crashes inside `ExpertOffloadManager.update_weights()` or `_update_weights()`.
- Accuracy/output becomes obviously broken compared with the same branch before this change under the same model and prompt.

## Suggested final report

Report these items back:

- commit id tested;
- exact launch command or changed `--additional-config`;
- whether static checks passed;
- whether pytest passed or was blocked by environment;
- sample `[EXPERT-OFFLOAD-CACHE]` lines from at least two layers;
- final observed hit rate range;
- whether the 6-expert resident/hit/miss behavior matched expectations.
