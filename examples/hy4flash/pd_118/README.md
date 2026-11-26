# HY4 manual PD launch on server 118

These scripts record the manual launch environment used on November 26. They
retain the existing model and communication-resource paths for that server.
They are not generic deployment defaults: update those paths, the CANN paths
and both source roots in `bin/vllm` when using another environment.

Run inside the prepared CANN 9.3 service container. Keep each service in its
own terminal. The scripts source CANN and call their adjacent Python entry
explicitly, so shell PATH resolution cannot select the system vLLM CLI.
The entry selects the candidate source roots, registers the NPU backend and
checks Triton/RoPE launchability before loading weights, including on spawn.

```bash
# P: devices 0-3, TP4/DP1 eager.
bash examples/hy4flash/pd_118/start_prefill.sh

# Four separate terminals: D devices 4-7, TP1/PCP1/DP4, DFly5.
bash examples/hy4flash/pd_118/start_decode.sh 0 8200 4 0 127.0.0.1 13445 1 1
bash examples/hy4flash/pd_118/start_decode.sh 1 8201 4 1 127.0.0.1 13445 1 1
bash examples/hy4flash/pd_118/start_decode.sh 2 8202 4 2 127.0.0.1 13445 1 1
bash examples/hy4flash/pd_118/start_decode.sh 3 8203 4 3 127.0.0.1 13445 1 1
```

All four D ranks must start before waiting for distributed initialization.
Each API should return HTTP 200 at `/health`. The proxy is a separate service:
it must route P port 8100 and D ports 8200-8203 before clients use port 19383.
Starting P/D alone does not start that proxy. Check its `/healthcheck` endpoint.

The candidate's only HY4 optimization boolean is `hy4flash_dfly_dp_graph=true`
on D. General Ascend settings and communication environment variables are
retained from the completed PD compatibility run. See
[the integration notes](../../../docs/source/developer_guide/hy4_decode_integration.md)
for the complete-run evidence and the later manual launch limitations.
