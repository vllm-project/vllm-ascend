# Step5 weight transfer: draft integration tests

## Prerequisite: Step5 implementation

These tests are preparation for Step5 integration, **not a claim that Step5
works from upstream source**. At base `041cfb328`, the official
[deployment guide](../../../../../docs/source/tutorials/models/Step-5-Preview.md)
explicitly states that the supporting code has not been merged. The integration
PR must remain draft until the model implementation, compatible vLLM pairing,
reduced fixtures and NPU validation are available.

The earlier development-repository draft was submitted to the wrong repository.
This upstream version has no imports of its private `vllm_ascend.models.step5`
package and carries none of that repository's model/kernel changes. The proposed
reshape, language-stream and vision-order unit tests are deferred until their
implementation exists upstream; this change does not claim those three tests.

## Oracle and scope

Following [PR #18634](https://github.com/vllm-project/vllm-ascend/pull/18634), use
the existing shared `assert_weight_update_matches_reference` assertion:

```text
dummy != startup(W)
live_update_1(W) == startup(W)
live_update_2(W) == startup(W)
```

The reference loads the original local checkpoint directly. The transfer source
reads the same safetensors payload without renaming, casting or reshaping. Its
unit tests verify metadata, repeated iteration and preservation of checkpoint
names/dtypes/values for router, shared and routed experts. They do not prove
HF-to-runtime loader mappings or complete model parameter coverage.

Three opt-in E2E functions cover IPC text, unpacked HCCL text, and IPC image
requests. Both servers use the same BF16, TP1, NZ-disabled, FULL_DECODE_ONLY
configuration and tokenizer/processor assets. Reference signatures must repeat
exactly before checking updates. Token/logprob comparison rejects nonfinite
values. IPC uses level-2 sleep, weights-only wake, transfer, then KV wake.
Prefix caching is disabled. The image lane also disables processor caching,
requires zeroing only vision/projector weights to change the signature, and
verifies restoration; an insensitive image probe fails rather than passes.

## Running

Use exclusively allocated Linux NPUs with identity device numbering in pytest
(ASCEND_RT_VISIBLE_DEVICES unset). IPC uses card 0 for trainer and inference;
HCCL uses card 0 for inference and card 1 for trainer. This is not a TP2 test.
Each lane runs in its own subprocess with a one-hour deadline, including native
collectives; HTTP requests also have deadlines. Logs are inherited by pytest.

Supply complete, internally consistent, reduced BF16 checkpoints. Text fixtures
must retain attention/indexer, routed/shared experts, router bias and lm_head.
VL fixtures additionally need vision/projector weights, processor, tokenizer
and chat template assets. No layer override silently slices the supplied model.
The IPC trainer payload and inference weights must fit on the same card; sleep
does not eliminate that simultaneous allocation requirement.

```bash
pytest -sv tests/e2e/pull_request/one_card/rlhf/state_transitions/test_step5_weight_transfer.py \
  --step5-text-checkpoint /fixtures/step5-text-bf16 \
  --step5-vl-checkpoint /fixtures/step5-vl-bf16

pytest -sv tests/e2e/pull_request/two_card/rlhf/state_transitions/test_step5_hccl_weight_transfer.py \
  --step5-text-checkpoint /fixtures/step5-text-bf16

pytest -q tests/ut/distributed/weight_transfer/test_step5_weight_source.py
```

Missing fixture options explicitly skip; invalid configured fixtures fail.
There is no fixed skip for a configured Step5 model. CI fixture provisioning and
option wiring are still pending, so collection or skips are not E2E acceptance.
Packed, quantized updates and TP8/DP2/EP are out of scope.

The development branch's IPC/HCCL lifecycle limitations must not be attributed
to upstream: upstream has different reload implementations. Actual compatibility
must be tested after Step5 is integrated. Hardware logs and normal repository
UT results are required before marking this draft ready for review.
