#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
work=$1
dest=$2
mkdir -p "$work/deps"
tar -xzf "$root/third_party/catlass-1.0.0-headers.tar.gz" -C "$work/deps"
cmake -S "$root/native/catlass" -B "$work/cmake" -DCATLASS_INCLUDE_DIR="$work/deps/include"
cmake --build "$work/cmake" --target catlass_kernel --parallel "${MAX_JOBS:-4}"
mv -- "$work/cmake/libcatlass_kernel.so" "$dest/libflash_mla_a3_catlass.so"
