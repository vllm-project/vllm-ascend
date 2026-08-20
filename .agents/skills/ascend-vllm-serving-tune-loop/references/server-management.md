# Server Management

## Managed process contract

- Every tuning attempt owns exactly one server process.
- Record its PID in the run directory, for example `<work_dir>/server.pid`.
- Clean up that managed PID and its process group only.

## Safe cleanup

Use:

```bash
bash scripts/cleanup_managed_server.sh \
  --pid-file <work_dir>/server.pid \
  --port <port>
```

Rules:

1. Never instruct the agent to run broad `pkill -9 -f vllm`, `pkill -9 -f VLLM`, or `pkill -9 -f ray` on shared machines.
2. If the target port is occupied by a non-managed process, stop and ask the user before killing anything.
3. Wait for graceful shutdown first, then escalate only for the managed process.

## Timeouts

- Server readiness timeout: bounded and recorded as `STARTUP_FAIL`
- Benchmark timeout: bounded and recorded as `BENCHMARK_FAIL`
- A retry may happen once for a transient issue, but the retry reason must be recorded

## Wrapper behavior

`scripts/auto_resume_wrapper.sh` should:

- exit immediately on `SIGINT`
- stop retrying when the wrapped command exits `130`
- relaunch only when the run is incomplete and not interrupted by the user
