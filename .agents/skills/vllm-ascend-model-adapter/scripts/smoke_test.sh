#!/bin/bash
# smoke_test.sh — Readiness poll + OpenAI-compatible smoke requests
#
# Usage: smoke_test.sh <SERVED_NAME> [PORT] [--multimodal]
#
#   SERVED_NAME   The model name passed to --served-model-name
#   PORT          API port (default: 8000)
#   --multimodal  Also run a text+image smoke request (requires a local test image)
#
# Exit codes: 0 = all checks passed, 1 = readiness or smoke failed

set -euo pipefail

SERVED_NAME=${1:?"Usage: smoke_test.sh <SERVED_NAME> [PORT] [--multimodal]"}
PORT=${2:-8000}
MULTIMODAL=0
for arg in "$@"; do
    [ "$arg" = "--multimodal" ] && MULTIMODAL=1
done

BASE_URL="http://127.0.0.1:${PORT}"

# ── 1. Readiness poll ─────────────────────────────────────────────────────────
echo "==> [1] Waiting for server readiness at ${BASE_URL}/v1/models ..."
READY=0
for i in $(seq 1 200); do
    if curl -sf "${BASE_URL}/v1/models" >/tmp/models.json 2>/dev/null; then
        READY=1
        echo "    OK: server ready after $((i * 3))s (attempt $i)"
        break
    fi
    sleep 3
done

if [ "$READY" -eq 0 ]; then
    echo "    ERROR: server did not become ready within 600s"
    exit 1
fi

echo "    /v1/models response:"
cat /tmp/models.json | python -m json.tool 2>/dev/null || cat /tmp/models.json
echo ""

# ── 2. Text smoke request ─────────────────────────────────────────────────────
echo "==> [2] Text smoke request ..."
TEXT_RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\": \"${SERVED_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"say hi\"}],
        \"temperature\": 0,
        \"max_tokens\": 16
    }")

echo "    Response: $TEXT_RESP"

# Check HTTP-level success and non-empty choices
CONTENT=$(echo "$TEXT_RESP" | python - <<'PY'
import sys, json
try:
    d = json.loads(sys.stdin.read())
    choices = d.get("choices", [])
    if not choices:
        print("ERROR: empty choices")
        sys.exit(1)
    msg = choices[0].get("message", {}).get("content", "")
    if not msg:
        print("ERROR: empty content in first choice")
        sys.exit(1)
    print(f"OK: got response: {msg!r}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PY
)
echo "    $CONTENT"
if echo "$CONTENT" | grep -q "^ERROR"; then
    echo "==> Text smoke FAILED"
    exit 1
fi

# ── 3. Multimodal smoke request (optional) ────────────────────────────────────
if [ "$MULTIMODAL" -eq 1 ]; then
    echo ""
    echo "==> [3] Multimodal (text+image) smoke request ..."
    # Use a 1x1 white PNG encoded as base64 as a minimal test image
    TEST_IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    MM_RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"${SERVED_NAME}\",
            \"messages\": [{
                \"role\": \"user\",
                \"content\": [
                    {\"type\": \"text\", \"text\": \"What is in this image?\"},
                    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${TEST_IMAGE_B64}\"}}
                ]
            }],
            \"temperature\": 0,
            \"max_tokens\": 32
        }")
    echo "    Response: $MM_RESP"
    MM_CONTENT=$(echo "$MM_RESP" | python - <<'PY'
import sys, json
try:
    d = json.loads(sys.stdin.read())
    choices = d.get("choices", [])
    if not choices:
        print("ERROR: empty choices in multimodal response")
        sys.exit(1)
    msg = choices[0].get("message", {}).get("content", "")
    if not msg:
        print("ERROR: empty content in multimodal response")
        sys.exit(1)
    print(f"OK: got multimodal response: {msg!r}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PY
)
    echo "    $MM_CONTENT"
    if echo "$MM_CONTENT" | grep -q "^ERROR"; then
        echo "==> Multimodal smoke FAILED"
        exit 1
    fi
fi

echo ""
echo "==> Smoke test PASSED"
