#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
build_dir=${BUILD_DIR:-"$root/build"}
python_bin=${PYTHON:-python3}
: "${ASCEND_HOME_PATH:?Source the CANN toolkit set_env.sh before building}"
soc=${SOC_VERSION:-}
if [[ -z "$soc" ]]; then
    soc=$("$python_bin" -c 'import torch_npu; print(torch_npu.npu.get_device_name(0))')
fi

# CANN 9.2 transforms its intermediate kernel objects in place. Reusing these
# after a source change can feed an already transformed object to ld.lld.
mkdir -p "$build_dir"
work=$(mktemp -d "$build_dir/.cmake.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
cmake -S "$root" -B "$work" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DSOC_VERSION="$soc" -DPython3_EXECUTABLE="$(command -v "$python_bin")" "$@"
cmake --build "$work" --target flash_mla_a3_ops --parallel "${MAX_JOBS:-4}"
cmake --build "$work" --target flash_mla_a3_grouped --parallel "${MAX_JOBS:-4}"
cmake --build "$work" --target flash_mla_a3_catlass_ops --parallel "${MAX_JOBS:-4}"
"$root/build_catlass.sh" "$work/catlass" "$work"

mv -f -- "$work"/libflash_mla_a3{_ops,_grouped,_catlass_ops,_catlass}.so "$build_dir/"
