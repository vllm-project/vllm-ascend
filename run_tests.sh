#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Test script for MTP feature integration
# Run on A2 machine in dsv4-19-new container

set -e

echo "========================================"
echo "MTP Feature Integration Test Script"
echo "========================================"

cd /vllm-workspace/vllm-ascend

# Ensure latest code
echo "[Step 1] Syncing code..."
git fetch origin
git reset --hard origin/feature/mtp-fused-dev
git log --oneline -5

# Run lint check
echo ""
echo "[Step 2] Running lint check..."
pip install -q ruff
ruff check \
    tests/ut/attention/test_dsa_v1.py \
    tests/ut/attention/test_mla_v1.py \
    tests/ut/sample/test_rejection_sampler.py \
    tests/ut/sample/test_sampler.py \
    tests/ut/spec_decode/test_eagle_proposer.py \
    vllm_ascend/spec_decode/llm_base_proposer.py \
    vllm_ascend/sample/rejection_sampler.py \
    vllm_ascend/sample/sampler.py \
    vllm_ascend/envs.py \
    vllm_ascend/patch/worker/patch_deepseek_mtp.py

# Run unit tests for modified files
echo ""
echo "[Step 3] Running unit tests..."
python -m pytest \
    tests/ut/attention/test_dsa_v1.py \
    tests/ut/attention/test_mla_v1.py::TestAscendMLAUpdateGraphParams \
    tests/ut/sample/test_rejection_sampler.py \
    tests/ut/sample/test_sampler.py \
    tests/ut/spec_decode/test_eagle_proposer.py::TestEagleProposerInitialization \
    tests/ut/spec_decode/test_eagle_proposer.py::TestDraftProposerHelperMethods \
    -v --tb=short

echo ""
echo "[Step 4] Running full unit tests for attention and sample..."
python -m pytest \
    tests/ut/attention/test_dsa_v1.py \
    tests/ut/attention/test_mla_v1.py \
    tests/ut/sample/ \
    -v --tb=short

echo ""
echo "========================================"
echo "Unit tests completed!"
echo "========================================"

# E2E test requires multi-card TP due to model size
# Uncomment below if running with TP>=4
#
# echo ""
# echo "[Step 5] Running E2E MTP test (requires multi-card TP)..."
# VLLM_USE_MODELSCOPE=true pytest -sv \
#     tests/e2e/singlecard/spec_decode/test_mtp_eagle_correctness.py::test_deepseek_mtp_local \
#     --tb=short

echo ""
echo "All tests completed successfully!"