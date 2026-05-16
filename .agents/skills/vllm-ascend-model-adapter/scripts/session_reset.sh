#!/bin/bash
# session_reset.sh — Kill stale vllm servers and verify port 8000 is free
#
# Usage: session_reset.sh [PORT]
#   PORT  Port to check (default: 8000)
#
# Pass --hard to also git-reset both source trees (destructive, requires confirmation).

set -euo pipefail

PORT=${1:-8000}

echo "==> Stopping stale vllm processes ..."
pkill -f "vllm serve|api_server|EngineCore" 2>/dev/null && echo "    Sent SIGTERM to matching processes" || echo "    No matching processes found"

sleep 1

echo "==> Checking port $PORT ..."
if netstat -ltnp 2>/dev/null | grep -q ":${PORT}"; then
    echo "    WARNING: port $PORT is still in use:"
    netstat -ltnp 2>/dev/null | grep ":${PORT}" | sed 's/^/    /'
    echo "    Kill the process above manually, then re-run this script."
    exit 1
else
    echo "    OK: port $PORT is free"
fi
