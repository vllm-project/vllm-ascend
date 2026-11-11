# Shared Qwen PLE table

Independent Qwen services can attach to one read-only BF16 PLE table instead of each owning a separate table. Each service keeps its own request and graph state. The checkpoint must remain unchanged while the owner and consumers are running: the table identity covers the model configuration, checkpoint index and table prefix, but does not hash the weight payload.

The BM backend uses a MemFabric shared pool. The RDMA backend uses the shared Engram HIXL protocol and fetches rows before graph execution. A2 and A3 have been exercised with the Qwen checkpoint; A5 support is limited to code and non-device testing. The implementation incorporates the required BM and HIXL protocol components from the ongoing shared Engram work and requires vLLM per-tensor checkpoint filtering. Install a compatible vLLM version before enabling it.

Set `--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'` for every consumer and specify `shared_ple` in `--additional-config`:

```json
{"shared_ple":{"backend":"bm","root":"/path/to/pool","checkpoint_prefix":"model.language_model.layers.1.ple.ple_embedding.ngram_embedding"}}
```

For the HIXL backend, use `backend: "rdma"`, replace `root` with the owner's reachable `endpoint`, and optionally provide an absolute `library_path` for `libengram_hixl.so`:

```json
{"shared_ple":{"backend":"rdma","endpoint":"192.0.2.1:29591","library_path":"/path/to/libengram_hixl.so","checkpoint_prefix":"model.language_model.layers.1.ple.ple_embedding.ngram_embedding"}}
```

Choose the prefix, shard count and embedding width from the actual checkpoint. Start the owner first using `python -m vllm_ascend.shared_ple --help` for BM or `python -m vllm_ascend.shared_ple_rdma --help` for HIXL. BM consumers need access to the same pool directory and a compatible fabric; HIXL consumers must be able to reach the owner's control and engine endpoints. Use a new BM pool directory for each owner generation. The HIXL control protocol is for trusted serving networks and must not be exposed as a public storage endpoint.

For the A5 code path, configure HIXL device UB communication resources for each process through `ENGRAM_HIXL_LOCAL_COMM_RES`. The resource file path can contain `{physical}` to select the JSON for that process's physical device. Set `ENGRAM_HIXL_FABRIC=ub` on the owner to bypass the A2 same-device RoCE workaround; this switch does not itself select the HIXL transport. The resource file must specify device UB. These settings are inherited from the shared Engram HIXL implementation and have not been validated on A5 for Qwen.

The current HIXL path requires ETP=TP, PCP=DCP=1 and does not support speculative decoding, DBO or elastic EP. It synchronizes the device before reusing its graph input buffer, which can affect latency. Both backends currently support BF16 PLE tables only. They do not provide transparent owner failover.

Stop all consumers cleanly before stopping the owner. Set a positive `--shutdown-timeout` long enough for graph and transport teardown; a forced worker exit can leave leases retained and prevent owner shutdown. Do not release backing memory while consumers may still access it.
