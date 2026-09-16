#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
# Never truncate a library mapped by live workers. Atomic replacement preserves
# their old inode; newly started workers load the new library.
tmp=$(mktemp "$root/engram_vmm/.libengram_host_vmm.XXXXXX.so")
trap 'rm -f -- "$tmp"' EXIT
g++ -std=c++17 -O2 -shared -fPIC -pthread \
  "$root/engram_vmm/native.cpp" -I"$cann/include" -L"$cann/lib64" \
  -lascendcl -o "$tmp"
mv -- "$tmp" "$root/engram_vmm/libengram_host_vmm.so"
