# curl -X POST http://127.0.0.1:8999/start_profile
curl -X POST http://127.0.0.1:8999/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "deepseek",
    "prompt": "hello, introduce yourself please",
    "max_tokens": 200,
    "temperature": 0,
     "top_p": 1,
    "top_k": -1
}'
# curl -X POST http://127.0.0.1:8999/stop_profile

# "prompt": "hello, introduce yourself please.",
