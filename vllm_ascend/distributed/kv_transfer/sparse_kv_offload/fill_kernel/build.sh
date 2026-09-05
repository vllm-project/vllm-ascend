#!/bin/bash
# Build the FillLocalCopy fused HBM->HBM fill kernel with bisheng (CANN >= 8.3.RC1).
# Produces libfill_local_copy.so next to this script; the manager loads it via
# VLLM_ASCEND_FILL_KERNEL_SO or this default module-relative path.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
CANN_ARCH="${FILL_KERNEL_CCE_ARCH:-dav-c220}"   # dav-c310 on A5
 # shellcheck disable=SC1090,SC1091
source "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/bin/setenv.bash" 2>/dev/null || true
cd "$DIR"
bisheng -x asc fill_local_copy.cpp -fPIC -shared -g -o libfill_local_copy.so --cce-aicore-arch="${CANN_ARCH}"
echo "built: $DIR/libfill_local_copy.so"
