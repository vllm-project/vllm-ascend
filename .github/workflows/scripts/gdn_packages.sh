#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eo pipefail

OPS_REPO=https://gitcode.com/cann/ops-transformer.git
OPS_COMMIT=0340eae051ddf5ab956df0f7e0396dbd6314cfc5
OPS=fused_gdn_gating,chunk_gated_delta_rule_compute_wy
CACHE_DIR="${GITHUB_WORKSPACE:?}/.cache/gdn-packages"
SCRIPT="$(realpath "$0")"
test -n "${GDN_IMAGE_TAG:?Set the full CANN image tag}"
ARCH="$(uname -m)"

cache_key() {
  local recipe
  recipe="$(sha256sum "$SCRIPT" | cut -d ' ' -f1)"
  printf 'gdn-packages-v1-%s-%s-%s-%s\n' \
    "$ARCH" "$GDN_IMAGE_TAG" "$OPS_COMMIT" "$recipe"
}

verify_packages() {
  test "$(cat "$CACHE_DIR/cache-key.txt")" = "$(cache_key)"
  test -f "$CACHE_DIR/cann-ops-transformer-custom_linux-${ARCH}.run"
  test -f "$CACHE_DIR/cann_ops_transformer-1.0.0-py3-none-any.whl"
  (cd "$CACHE_DIR" && sha256sum --check SHA256SUMS)
}

case "${1:-}" in
  key)
    cache_key
    ;;
  verify)
    verify_packages
    ;;
  build)
    # Keep large build trees out of RUNNER_TEMP: hooks can copy it per step.
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 -m pip install build wheel setuptools ninja
    GDN_SOURCE="$(mktemp -d /tmp/gdn-ops.XXXXXX)"
    trap 'cd /tmp; rm -rf -- "$GDN_SOURCE"' EXIT
    git init "$GDN_SOURCE"
    git -C "$GDN_SOURCE" remote add origin "$OPS_REPO"
    git -C "$GDN_SOURCE" fetch --depth 1 origin "$OPS_COMMIT"
    git -C "$GDN_SOURCE" checkout --detach FETCH_HEAD
    test "$(git -C "$GDN_SOURCE" rev-parse HEAD)" = "$OPS_COMMIT"
    cd "$GDN_SOURCE"
    bash build.sh --pkg --experimental --soc=ascend310p \
      --vendor_name=custom --ops="$OPS" -j16
    bash build.sh --torch_extension --experimental
    mkdir -p "$CACHE_DIR"
    cp "build_out/cann-ops-transformer-custom_linux-${ARCH}.run" "$CACHE_DIR/"
    cp build_out/cann_ops_transformer-1.0.0-py3-none-any.whl "$CACHE_DIR/"
    cache_key > "$CACHE_DIR/cache-key.txt"
    (
      cd "$CACHE_DIR"
      sha256sum "cann-ops-transformer-custom_linux-${ARCH}.run" \
        cann_ops_transformer-1.0.0-py3-none-any.whl > SHA256SUMS
    )
    verify_packages
    ;;
  install)
    verify_packages
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    bash "$CACHE_DIR/cann-ops-transformer-custom_linux-${ARCH}.run"
    test -d "$ASCEND_HOME_PATH/python/site-packages"
    python3 -m pip install --target="$ASCEND_HOME_PATH/python/site-packages" \
      --upgrade --force-reinstall --no-deps \
      "$CACHE_DIR/cann_ops_transformer-1.0.0-py3-none-any.whl"
    GDN_VENDOR="$ASCEND_HOME_PATH/opp/vendors/custom_transformer"
    test -d "$GDN_VENDOR/op_api/lib"
    TORCH_EXTENSIONS_DIR="$(mktemp -d /tmp/gdn-torch-extensions.XXXXXX)"
    export TORCH_EXTENSIONS_DIR
    export ASCEND_CUSTOM_OPP_PATH="$GDN_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$GDN_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    {
      echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
      echo "ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
      echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
      echo "PYTHONPATH=$PYTHONPATH"
    } >> "${GITHUB_ENV:?}"
    cd /tmp
    python3 -c "import cann_ops_transformer; print('Loaded package:', cann_ops_transformer.__file__); from cann_ops_transformer.ops import fused_gdn_gating, chunk_gated_delta_rule_compute_wy; print('GDN imports OK', flush=True)"
    ;;
  *)
    echo "Usage: $0 {key|verify|build|install}" >&2
    exit 2
    ;;
esac
