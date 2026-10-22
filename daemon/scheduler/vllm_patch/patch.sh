cp -f new_file/api_server.py /vllm-workspace/vllm/vllm/entrypoints/openai/api_server.py
cp -f new_file/async_llm.py /vllm-workspace/vllm/vllm/v1/engine/async_llm.py
cp -f new_file/core_client.py /vllm-workspace/vllm/vllm/v1/engine/core_client.py
cp -f new_file/core.py /vllm-workspace/vllm/vllm/v1/engine/core.py
cp -f new_file/request_manager.py /vllm-workspace/vllm/vllm/entrypoints/openai/request_manager.py

echo "Patch for intelligent scheduling applied successfully."