#!/bin/bash
set -e

PATCH_ROOT=../../omni/adaptors/vllm/patches/

cd ./vllm
git checkout tags/v0.9.0
git apply $PATCH_ROOT/null_value_handling.patch
git apply --whitespace=nowarn $PATCH_ROOT/tokenizer_proc_pool.patch
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_kv_cache_manager_partial_kv_transfer.patch
# git apply --whitespace=nowarn $PATCH_ROOT/async_schedule_update_output.patch
git apply --whitespace=nowarn $PATCH_ROOT/chunked_prefill_disable.patch
