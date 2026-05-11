#!/bin/bash
# syntax_check.sh — Python syntax check for one or more source files
#
# Usage: syntax_check.sh <file1.py> [file2.py ...]
#
# Exits 0 if all files pass, 1 if any file fails.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: syntax_check.sh <file1.py> [file2.py ...]"
    exit 1
fi

FAIL=0
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "SKIP (not found): $f"
        continue
    fi
    if python -m py_compile "$f" 2>&1; then
        echo "OK:   $f"
    else
        echo "FAIL: $f"
        FAIL=1
    fi
done

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "==> Syntax check PASSED"
else
    echo "==> Syntax check FAILED"
    exit 1
fi
