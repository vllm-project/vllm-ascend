#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a local Instant checkpoint}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select the assigned NPU devices}"
# vLLM accepts underscores and unambiguous long-option abbreviations.
for argument in "$@"; do
    option=${argument%%=*}
    option=${option//_/-}
    if [[ $option == --?* ]]; then
        for protected in --enable-chunked-prefill --enable-prefix-caching --max-model-len --max-num-batched-tokens; do
            if [[ $protected == "$option"* ]]; then
                echo "Instant requires a complete prompt; use MAX_MODEL_LEN for its length budget, without chunking or prefix caching." >&2
                exit 2
            fi
        done
    fi
done
python_bin=${PYTHON_BIN:-python}
exec "$python_bin" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name startlux-instant \
    --runner pooling \
    --hf-overrides '{"architectures":["StartLuxDecisionForSequenceClassification"],"num_labels":26}' \
    --pooler-config '{"task":"classify","use_activation":false}' \
    --dtype bfloat16 \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --max-num-seqs 16 \
    --host 127.0.0.1 --port "${PORT:-8000}" "$@" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN:-8192}" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
