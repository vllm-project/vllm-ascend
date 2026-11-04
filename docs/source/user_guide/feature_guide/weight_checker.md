# Weight Checker

## Overview

The Weight Checker verifies that a weight update landed on every rank. It
hashes each model parameter with SHA-256, so you can save the original
digests, reset the weights, transfer them back, and confirm they match. See
the upstream [Weight Checker](https://docs.vllm.ai/en/latest/training/weight_checker/)
guide for the full API.

The checker's standard use is validating the *weight-sync path itself*
during development and bring-up: hash the weights currently deployed, reset
them, transfer the *same* weights again, and expect an exact match. It does
not produce the expected digests of newly trained weights, so a mismatch
against an old baseline after a real training step is expected — it is not a
per-step correctness proof of RL updates.

## Getting started

Enable it with `VLLM_SERVER_DEV_MODE=1` and use `POST /weight_checker`:

| Request | Response |
| --- | --- |
| `{"action": "checksum"}` | `{"checksums": {key: sha256_hex}}` |
| `{"action": "reset"}` | `{"status": "reset"}` |
| `{"action": "compare", "baseline": {key: sha256_hex}}` | `{"match": bool, "mismatches": [key]}` |

Keys have the form `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{tensor_name}`, so each
shard is checked separately. There is no separate `ep` field: expert-parallel
ranks are identified by their DP/PCP/TP coordinates.

```bash
URL=http://localhost:8000

# 1. Stop serving, then save the original digests.
curl -X POST "$URL/pause?mode=abort"
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}' > baseline.json

# 2. Zero the weights.
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d '{"action":"reset"}'

# 3. Transfer the original weights with the hccl or npu_ipc backend.

# 4. Compare against the baseline; expect {"match": true, "mismatches": []}.
curl -X POST $URL/weight_checker -H 'Content-Type: application/json' \
  -d "{\"action\":\"compare\",\"baseline\":$(jq .checksums baseline.json)}"

curl -X POST $URL/resume
```

## Operational constraints

Follow these operational constraints on Ascend:

1. Stay paused from `checksum` to `compare`: the weights are invalid between
   `reset` and the transfer, and EPLB moves experts while serving. The worker
   methods do not enforce this — pausing is the caller's responsibility.
2. Before `reset`, confirm a recovery path is available: a working weight
   transfer configuration, or a checkpoint that `reload_weights` can
   restore. `weight_nz_mode` defaults to `1`, but RL mode disables FRACTAL_NZ
   automatically (`rl_config.enabled=true` forces `weight_nz_mode=0`). In
   non-RL deployments where `weight_nz_mode` remains enabled, including the
   default configuration, the weight-update transaction refuses to start,
   and `reset_weights` logs a warning that only checkpoint reload can recover.
3. Resume only after `compare` reports a full match. If a transfer fails, do
   not resume serving; restore the weights first, for example with
   `POST /collective_rpc {"method": "reload_weights"}`.
4. The engine must be awake: sleep levels 1 and 2 release the weight memory.

`reset` zeroes every checked parameter and is a destructive development
endpoint. Keep dev-mode servers inside a trusted access boundary (localhost,
a private network, or an authenticated gateway) and never expose them to the
public internet.

## Limitations

Ascend-specific limitations, in addition to the
[upstream list](https://docs.vllm.ai/en/latest/training/weight_checker/#limitations):

- Weights held as buffers rather than parameters (for example some
  quantization repacking paths) are invisible to the checker.
- MoE weights padded beyond the checkpoint shape may report false mismatches;
  validate against your model before relying on `compare`.
- Ascend 310P is not yet verified with the weight checker.
