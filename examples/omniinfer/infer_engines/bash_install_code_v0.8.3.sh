#!/bin/bash
set -e

PATCH_ROOT=../../omni/adaptors/vllm/patches

cd ./vllm

git reset --hard
git clean -fd
git checkout tags/v0.8.3
git apply --whitespace=nowarn $PATCH_ROOT/v0.8.3/support_omni_infer_for_v0.8.3.patch
git apply --whitespace=nowarn $PATCH_ROOT/v0.8.3/solve_triton_issue_on_a_plus_x.patch
git apply --whitespace=nowarn $PATCH_ROOT/v0.8.3/adapt_vllm_to_omniinfer_v0.3.0.patch
git apply --whitespace=nowarn $PATCH_ROOT/v0.8.3/common_dependency.patch

# SETUPTOOLS_SCM_PRETEND_VERSION=0.8.3 VLLM_TARGET_DEVICE=empty pip install -e .