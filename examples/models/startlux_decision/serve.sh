#!/usr/bin/env bash
set -euo pipefail
# v1 and Instant share native decision pooling; the checkpoint controls causality.
# Keep full-prompt scheduling in this conservative deployment recipe.
exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_instant.sh" \
    --served-model-name startlux-decision "$@"
