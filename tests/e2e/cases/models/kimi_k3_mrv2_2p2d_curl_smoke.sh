#!/usr/bin/env bash
set -euo pipefail

# SERVICE_URL is the external-DP proxy base URL. Override MODEL_NAME when the
# service uses --served-model-name instead of the checkpoint identifier.
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:1999}"
MODEL_NAME="${MODEL_NAME:-Eco-Tech/Kimi-K3-w4a8}"
CURL_TIMEOUT="${CURL_TIMEOUT:-120}"

endpoint="${SERVICE_URL%/}/v1/chat/completions"
response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT

curl --fail --silent --show-error --max-time "$CURL_TIMEOUT" \
  -H 'Content-Type: application/json' \
  -X POST "$endpoint" \
  -d "$(cat <<JSON
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "Reply with one short sentence explaining why prefix caching helps repeated prompts."}
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": false
}
JSON
)" >"$response_file"

python3 - "$response_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as response:
    payload = json.load(response)

choices = payload.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("curl smoke failed: response has no choices")

message = choices[0].get("message") or {}
content = message.get("content") or message.get("reasoning_content")
if not isinstance(content, str) or not content.strip():
    raise SystemExit("curl smoke failed: first choice has no generated text")

print("curl smoke passed: HTTP response contains generated text")
PY
