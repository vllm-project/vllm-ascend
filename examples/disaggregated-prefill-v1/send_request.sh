#!/bin/bash

# Make sure the model is same as the one used in the server
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REQUEST_ID=request$RANDOM

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "X-Request-ID: ${REQUEST_ID}" \
    -d '{
        "ignore_eos": false,
        "stream": false,
        "stop": "None",
        "temperature": 0.5,
        "top_k": -1,
        "top_p": 1,
        "model": "'${MODEL_NAME}'",
        "prompt": [
            "In 2020, who won the world series?",
            "In 2019, Who won the world series?"
        ],
        "max_tokens": 40
    }'
