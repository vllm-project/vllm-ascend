  vllm serve "$MODEL" \
    --enforce-eager \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 2048 \
    --enforce-eager \
    --speculative-config '{"method":"eagle3","num_speculative_tokens":3, "model":"$model"}' \
    --port 1123