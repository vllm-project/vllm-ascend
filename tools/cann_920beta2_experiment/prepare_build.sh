#!/usr/bin/env bash
set -euo pipefail

install_build_dependencies() {
    local evidence=$1
    local dependency resolved
    local required=(git cmake g++ make pigz dos2unix unzip curl patch pkg-config)
    local missing=()
    # These commands have matching Ubuntu package names. Use one list for installation and verification.
    for dependency in "${required[@]}"; do
        if ! command -v "$dependency" >/dev/null 2>&1; then
            missing+=("$dependency")
        fi
    done
    if (( ${#missing[@]} )); then
        command -v apt-get >/dev/null || { echo 'Missing apt-get in the disposable build container' >&2; return 1; }
        printf 'Installing missing build dependencies: %s\n' "${missing[*]}"
        apt-get -o Acquire::Retries=3 update
        DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends "${missing[@]}"
    fi
    : > "$evidence/build-tools.txt"
    for dependency in "${required[@]}"; do
        resolved=$(command -v "$dependency") || { echo "Missing build tool: $dependency" >&2; return 1; }
        printf '%s=%s\n' "$dependency" "$resolved" | tee -a "$evidence/build-tools.txt"
    done
    {
        pigz --version
        dos2unix --version
    } > "$evidence/build-tool-versions.txt" 2>&1
}

main() {
    local evidence scripts work src
    evidence=$(realpath "$1")
    scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    [[ $(uname -m) == aarch64 && $(id -u) == 0 ]]
    install_build_dependencies "$evidence"

    # Resolve source and package-tool failures before the expensive CANN install and baseline build.
    work=$(mktemp -d /tmp/cann920-opsnn-fixed.XXXXXX)
    src="$work/ops-nn"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v9.2.0-beta.2 \
        https://gitcode.com/cann/ops-nn.git "$src"
    [[ $(git -C "$src" rev-parse HEAD) == 30ef7dd563c8a4b74c3161835c8e47d1d96f87b6 ]]
    git -C "$src" apply --check "$scripts/optional_input_fix.patch"
    git -C "$src" apply "$scripts/optional_input_fix.patch"
    git -C "$src" add -- norm/add_rms_norm_dynamic_quant/op_host/arch22/add_rms_norm_dynamic_quant_tiling.cpp \
        norm/add_rms_norm_dynamic_quant/tests/ut/op_host/arch22/test_add_rms_norm_dynamic_quant_tiling.cpp
    [[ $(git -C "$src" write-tree) == c41d331cdab95233834dbca87e0ada238251a5f0 ]]
    git -C "$src" diff --cached > "$evidence/applied-fix.patch"
    printf '%s\n' "$src" > "$evidence/fix-source-dir.txt"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
