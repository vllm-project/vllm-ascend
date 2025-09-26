#!/bin/bash
set -e

PATCH_ROOT=${1:-../../omni/adaptors/vllm/patches/}
VLLM_PATH=${2:-./vllm}

cd ${VLLM_PATH}
git reset --hard
git clean -fd
git checkout v0.9.0
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_kv_cache_manager_partial_kv_transfer.patch
git apply --whitespace=nowarn $PATCH_ROOT/tokenizer_proc_pool.patch
git apply --whitespace=nowarn $PATCH_ROOT/mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/api_server_keepalive_timeout.patch
git apply --whitespace=nowarn $PATCH_ROOT/async_schedule_multi_step.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_support_fast_path_pull_kv.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_support_prefilled_token_skip_tokenize.patch
git apply --whitespace=nowarn $PATCH_ROOT/common_dependency.patch
git apply --whitespace=nowarn $PATCH_ROOT/omni_attn.patch
git apply --whitespace=nowarn $PATCH_ROOT/chunk_prefill_enable.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_abort_kv_loading_failure_request.patch
git apply --whitespace=nowarn $PATCH_ROOT/tfas_patch_request.patch
git apply --whitespace=nowarn $PATCH_ROOT/prometheus_dp_logging.patch
git apply --whitespace=nowarn $PATCH_ROOT/swap_kv_cache.patch
