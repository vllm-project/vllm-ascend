The following Qwen3-0.6B example has been validated on Atlas 300I DUO and Atlas 200I Pro.

???+ important "Runtime requirements for this hardware path"

    - Use `float16`.
    - Use `FULL_DECODE_ONLY` and limit the graph capture sizes.
    - `enable_npugraph_ex` is not supported. Set `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'`.

<!-- doctest: quickstart-300i-duo-online-serve -->
```bash
vllm serve Qwen/Qwen3-0.6B \
    --dtype float16 \
    --max-num-seqs 8 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}' &
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

<!-- doctest: quickstart-300i-duo-online-model-list -->
```bash
curl --fail http://localhost:8000/v1/models
```

You can also send a prompt to the model:

<!-- doctest: quickstart-300i-duo-online-completion -->
```bash
curl --fail http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
    }'
```

Confirm that the response contains nonempty generated text in `choices[0].text`.
An error response or an empty completion does not confirm successful inference.

vLLM is running as a background process. You can use `kill -2 $VLLM_PID` to stop it gracefully, which is similar to pressing `Ctrl+C` for a foreground vLLM process:

???+ warning "Confirm the process before stopping the service"

    If other `vllm serve` processes are running in the current environment, `pgrep -f "vllm serve"` may also match those vLLM services.

    Before running `kill`, confirm that `VLLM_PID` is the service started by this example to avoid stopping another running vLLM process by mistake.

<!-- doctest: quickstart-300i-duo-online-stop -->
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

### Native data parallelism on 310P

For example, add `--data-parallel-size 2 --tensor-parallel-size 2` to use
four visible NPUs. Every DP replica is a complete model replica: the MoE
parallel config is restricted to the TP group, so the MoE forward pass issues
no cross-DP tensor collectives (CPU-side DP coordination and the Gloo world
group are unchanged). In our tested 310P environment, capturing a FULL graph
for that path failed with an exhausted capture event pool while cross-DP HCCL
collectives were recorded; removing them made graph capture succeed. Each
replica also holds its own copy of the expert weights, which reduces the KV
cache capacity per device.
Pipeline, context, and expert parallelism remain unsupported by the 310P v2
model runner.

vLLM slices preassigned device lists per DP rank before starting each engine
core, so workers index into that slice and must not apply a second DP offset.
Ray and external launchers keep their own rank assignment.
For DP, `HCCL_IF_BASE_PORT` is the base of a contiguous allocation of 32
host ports per replica (default base: 60000), following the
[HCCL host-port allocation](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0019.html). The entire allocation must
fit in ports 1024 through 65535. Set different, nonoverlapping base ranges
for multiple instances on the same host. Port allocation does not probe or
reserve sockets against other applications.

MoE DP still performs CPU-side coordination. Before this change the MoE
forward path sharded experts across the DP group and ran device all-gather
and reduce-scatter on every layer; with the complete-replica layout those
cross-DP tensor collectives are gone. Gloo is used for the world control
group while model-parallel subgroups retain HCCL, so native DP performance
still differs from two independent services at the same tensor-parallel size.
