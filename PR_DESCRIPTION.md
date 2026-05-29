<!--  Thanks for sending a pull request!

BEFORE SUBMITTING, PLEASE READ https://docs.vllm.ai/en/latest/contributing/overview.html

-->
### What this PR does / why we need it?

This PR merges the DeepSeek V4 MTP (Multi-Token Prediction) related changes from the vllm-ascend-deepseekv4 repository into the community mtp-fused-dev branch. The changes include:

**Key Features:**

1. **DSA (DeepSeek Attention) Enhancements**
   - Updated `vllm_ascend/attention/dsa_v1.py` with improved attention implementation for DeepSeek V4
   - Added support for multi-stream overlap and compressed KV cache handling

2. **MTP (Multi-Token Prediction) Support**
   - Enhanced `vllm_ascend/spec_decode/eagle_proposer.py` for MTP drafter integration
   - Updated `vllm_ascend/patch/worker/patch_deepseek_mtp.py` for DeepSeek MTP layer support
   - Added new file `vllm_ascend/patch/worker/patch_module.py` for torch_npu argsort patching

3. **Sampler Improvements**
   - Updated `vllm_ascend/sample/sampler.py` with new sampling constraints and runtime state handling
   - Enhanced `vllm_ascend/sample/rejection_sampler.py` with strict rejection sampling for MTP/ngram-style speculation

4. **Model Runner Updates**
   - Updated `vllm_ascend/worker/model_runner_v1.py` with fused MTP graph capture and runtime buffer management
   - Added `vllm_ascend/core/kv_state_scheduler.py` for KV state scheduling support

5. **Utility Updates**
   - Updated `vllm_ascend/utils.py` with DeepSeek V4 block size handling and compressed position generation

6. **Test Coverage**
   - Added `tests/ut/attention/test_dsa_v1.py` for DSA backend testing
   - Updated existing tests to cover new MTP and sampling features

**Why these changes are needed:**

- Enables proper DeepSeek V4 MTP speculative decoding support on Ascend NPU
- Improves attention efficiency for DeepSeek V4 models with DSA backend
- Provides better integration with the community's mtp-fused-dev branch
- Adds missing patch modules for proper Ascend NPU compatibility

### Does this PR introduce _any_ user-facing change?

Yes. This PR introduces the following user-facing changes:

- **New Features**: DeepSeek V4 MTP speculative decoding support is now available for Ascend NPU users
- **API Changes**: New sampling constraints API in `sampler.py` for runtime state handling
- **Behavior Changes**: Improved rejection sampling behavior for MTP-style speculation

No documentation updates are required as the feature follows existing vLLM speculative decoding patterns.

### How was this patch tested?

Tests were verified in the following ways:

1. **Lint Check**: All Python files pass `ruff check` (except pre-existing style issues from upstream that are intentional)

2. **Unit Tests**: The following tests pass successfully:
   - `tests/ut/sample/test_sampler.py` - 2 tests passed
   - `tests/ut/sample/test_rejection_sampler.py` - 11 tests passed
   - `tests/ut/attention/test_dsa_v1.py` - 6 tests passed
   - `tests/ut/worker/test_model_runner_v1.py` - 35 tests passed (MTP-related tests)

3. **Integration Testing**: The changes were merged from a working vllm-ascend-deepseekv4 repository that has been tested with actual DeepSeek V4 MTP models on Ascend NPU hardware.

**Test Environment:**
- Container: dsv4-19 (quay.io/ascend/vllm-ascend:vllm-20260523)
- Platform: Ascend NPU
- Base branch: origin/feature/mtp-fused-dev