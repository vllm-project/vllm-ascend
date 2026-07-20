#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
CXX=${CXX:-g++}
ASCEND_HOME=${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import pybind11" >/dev/null 2>&1; then
  echo "pybind11 is not installed in $PYTHON_BIN environment. Try: $PYTHON_BIN -m pip install pybind11" >&2
  exit 1
fi

EXT_SUFFIX=$($PYTHON_BIN-config --extension-suffix 2>/dev/null || true)
if [[ -z "${EXT_SUFFIX}" ]]; then
  EXT_SUFFIX=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
fi

PYBIND_INCLUDES=$($PYTHON_BIN -m pybind11 --includes)

OUT="mdaemon_py${EXT_SUFFIX}"

$CXX -O3 -shared -std=c++17 -fPIC \
  $PYBIND_INCLUDES \
  daemon_pybind.cpp daemon.cpp handle_pool.cpp model_management.cpp \
  -I ./inc \
  -I "${ASCEND_HOME}/include" \
  -L "${ASCEND_HOME}/lib64" \
  -lascendcl \
  -o "$OUT"

echo "Built: $OUT"