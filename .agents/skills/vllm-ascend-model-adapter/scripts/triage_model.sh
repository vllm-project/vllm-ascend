#!/bin/bash
# triage_model.sh — Fast model inventory and config.json field scan
#
# Usage: triage_model.sh <MODEL_PATH>

set -euo pipefail

MODEL_PATH=${1:?"Usage: triage_model.sh <MODEL_PATH>"}

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

echo "==> [1/4] Model directory listing ..."
ls -la "$MODEL_PATH"

echo ""
echo "==> [2/4] Key config.json fields ..."
CONFIG="$MODEL_PATH/config.json"
if [ ! -f "$CONFIG" ]; then
    echo "    ERROR: config.json not found at $CONFIG"
    exit 1
fi
PATTERN="architectures|model_type|quantization_config|torch_dtype|max_position_embeddings|num_nextn_predict_layers|version|num_attention_heads|num_key_value_heads|num_experts|n_routed_experts|kv_lora_rank|sliding_window|vision_config|audio_config|state_size|mamba"
if command -v rg &>/dev/null; then
    rg -n "$PATTERN" "$CONFIG" || true
else
    grep -nE "$PATTERN" "$CONFIG" || true
fi

echo ""
echo "==> [3/4] Weight index files ..."
ls -la "$MODEL_PATH"/*index*.json 2>/dev/null || echo "    (none found)"

echo ""
echo "==> [4/4] Custom model code files ..."
ls -la "$MODEL_PATH"/*.py 2>/dev/null || echo "    (none found)"
