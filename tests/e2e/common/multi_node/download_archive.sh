#!/bin/bash
set -euo pipefail

# Called from run.sh on exit and by CI before deleting running pods.
backup_requested_directory() {
    [ -n "${DOWNLOAD_SPEC:-}" ] || return 0
    local node="node${LWS_WORKER_INDEX:-0}"
    local requested archive temporary_archive shared_archive
    local marker="/tmp/${node}_download.archived"
    archive="$LOG_PREFIX/${node}_download.tar.gz"
    # EXIT and CI finalization run in the same container. Reuse only
    # an archive produced successfully by this container; retry failed attempts.
    if [ -f "$marker" ] && [ -f "$archive" ]; then
        echo "Requested path already archived on $node"
        return 0
    fi
    if ! requested=$(python3 /root/.cache/tests/collect_download.py "$DOWNLOAD_SPEC" \
        --path-for-node --node "$node"); then
        echo "WARNING: Invalid download request for $node" >&2
        return 1
    fi
    [ -n "$requested" ] || return 0
    if ! mkdir -p "$LOG_PREFIX"; then
        echo "WARNING: Could not create shared log directory on $node" >&2
        return 1
    fi
    if ! temporary_archive=$(mktemp "/tmp/${node}-download.XXXXXX.tar.gz"); then
        echo "WARNING: Could not create temporary archive on $node" >&2
        return 1
    fi
    if ! python3 /root/.cache/tests/collect_download.py "$DOWNLOAD_SPEC" \
        "$temporary_archive" --node "$node"; then
        rm -f "$temporary_archive"
        echo "WARNING: Could not archive requested path on $node" >&2
        return 1
    fi
    if ! shared_archive=$(mktemp "$LOG_PREFIX/.${node}_download.XXXXXX.tar.gz"); then
        rm -f "$temporary_archive"
        echo "WARNING: Could not stage requested archive on shared storage for $node" >&2
        return 1
    fi
    if ! cp "$temporary_archive" "$shared_archive"; then
        rm -f "$temporary_archive" "$shared_archive"
        echo "WARNING: Could not copy requested archive to shared storage for $node" >&2
        return 1
    fi
    rm -f "$temporary_archive"
    if ! mv "$shared_archive" "$archive"; then
        rm -f "$shared_archive"
        echo "WARNING: Could not publish requested archive on shared storage for $node" >&2
        return 1
    fi
    touch "$marker"
    echo "Archived requested path on $node: $requested -> $archive"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "${1:-}" = "--strict" ]; then
        backup_requested_directory
    else
        backup_requested_directory || true
    fi
fi
