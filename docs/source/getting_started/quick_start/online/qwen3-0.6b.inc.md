The following Qwen3-0.6B example has been validated using the default model loading configuration.

```bash
vllm serve Qwen/Qwen3-0.6B &
```

If you see the following logs:

```text
INFO:     Started server process [3594]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Congratulations! You have successfully started the vLLM server.

You can query the model list:

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

You can also send a prompt to the model:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Beijing is a",
        "max_completion_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
```

vLLM is running as a background process. You can use `kill -2 $VLLM_PID` to stop it gracefully, which is similar to pressing `Ctrl+C` for a foreground vLLM process:

???+ warning "Confirm the process before stopping the service"

    If other `vllm serve` processes are running in the current environment, `pgrep -f "vllm serve"` may also match those vLLM services.

    Before running `kill`, confirm that `VLLM_PID` is the service started by this example to avoid stopping another running vLLM process by mistake.

```bash
VLLM_PID=$(pgrep -f "vllm serve")
kill -2 "$VLLM_PID"
```

The output is as follows:

```text
INFO:     Shutting down FastAPI HTTP server.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
```

Finally, press `Ctrl+D` to exit the container.
