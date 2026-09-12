#!/usr/bin/env bash
# One-way sync of the V2-entry source subset from the evolution delivery tree
# (source of truth) into this vllm-ascend subtree. Kernel evolution continues
# in the evolution tree; re-run this after any accepted kernel change, then
# rebuild vllm_ascend_C and re-run tests/ut/ops/test_grouped_matmul_situ_quant.py.
#
# NOTE: op_host/gmm_situ_quant_v2.cpp is renamed to gmm_situ_quant_entries.cpp
# on copy ("V2" belongs to the official aclnn family lineage, not to our op —
# registered torch.ops names carry no version suffix).
set -euo pipefail

: "${GMSQ_P0_KERNEL:?Set GMSQ_P0_KERNEL to the external p0_entries/kernel directory}"
SRC="${GMSQ_P0_KERNEL}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "${SRC}/op_kernel/gmm_situ_vcv_dev.cpp" ] || { echo "source of truth not found: ${SRC}"; exit 1; }

cp -f "${SRC}/op_kernel/gmm_situ_vcv_dev.cpp"        "${HERE}/op_kernel/"
cp -f "${SRC}/op_kernel/gmsq_vcv_controller.h"       "${HERE}/op_kernel/"
cp -f "${SRC}/op_kernel/situ_epilogue.h"             "${HERE}/op_kernel/"
rm -rf "${HERE}/op_kernel/vendor/wqbmm" "${HERE}/op_kernel/vendor/gmsq2"
cp -r  "${SRC}/op_kernel/vendor/wqbmm"               "${HERE}/op_kernel/vendor/"
cp -r  "${SRC}/op_kernel/vendor/gmsq2"               "${HERE}/op_kernel/vendor/"
cp -f "${SRC}/op_host/gmm_situ_quant_v2.cpp"         "${HERE}/op_host/gmm_situ_quant_entries.cpp"
cp -f "${SRC}/utils/torch_kernel_helper.h"           "${HERE}/utils/"

# ops.h and the root CMake/torch dispatcher integration are owned here and
# are NOT synced.
echo "[sync] V2 subset refreshed from ${SRC}"
echo "[sync] reminder: reinstall vllm-ascend, then run pytest tests/ut/ops/test_grouped_matmul_situ_quant.py"
