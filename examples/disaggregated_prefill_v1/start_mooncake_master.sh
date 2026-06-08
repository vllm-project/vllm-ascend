#!/usr/bin/env bash
set -euo pipefail

MOONCAKE_PORT="${MOONCAKE_PORT:-50088}"
MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO="${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO:-0.8}"
MOONCAKE_EVICTION_RATIO="${MOONCAKE_EVICTION_RATIO:-0.05}"
MOONCAKE_LIB_PATH="${MOONCAKE_LIB_PATH:-/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake}"

export LD_LIBRARY_PATH="${MOONCAKE_LIB_PATH}:${LD_LIBRARY_PATH:-}"

exec mooncake_master \
  --eviction_high_watermark_ratio "${MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO}" \
  --eviction_ratio "${MOONCAKE_EVICTION_RATIO}" \
  --port "${MOONCAKE_PORT}"
