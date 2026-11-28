# Share a Qwen PLE table across services

Independent Qwen services can share one immutable BF16 PLE table. The existing `shared_engram` owner discovers the single table from the checkpoint configuration and safetensors index. Do not modify the checkpoint while the owner or consumers are running: identity covers configuration, index and table prefix, but not the full weight payload.

The worker uses native per-tensor filtering when available. For supported vLLM builds without [vLLM PR #64395](https://github.com/vllm-project/vllm/pull/64395), it supplies a compatibility patch for lazy, single-threaded safetensors loading. Other loading formats and strategies are rejected when external PLE weights are enabled. Remove the compatibility patch once all supported vLLM versions include native per-tensor filtering; this branch’s CI targets vLLM v0.33.0 and verified main commit `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`.

Start the A3 MemFabric owner with the ordinary Engram CLI:

```sh
python -m vllm_ascend.shared_engram --backend a3 --model /path/to/checkpoint --endpoint HOST:PORT --device 0
```

For A2 HIXL use `--backend a2` and add `--engine-endpoint HOST:PORT`; A5 uses `--backend a5` with device UB resources provided by `--local-comm-res` and `--fabric ub`. See [Shared host Engram over HIXL](shared_engram_rdma.md) for HIXL networking and resource options. The owner and consumer must reach the configured endpoints over a trusted fabric.

Configure each consumer with Model Runner V2, a positive shutdown timeout, CPU offload, and the existing `shared_engram` setting:

```sh
--engram-config '{"cpu_offload":true}' \
--additional-config '{"shared_engram":{"backend":"a3","endpoint":"HOST:PORT"}}' \
--shutdown-timeout 120
```

Use the matching `a2` or `a5` backend for HIXL; `capacity`, `library_path`, `listen_port`, and `local_comm_res` follow the shared Engram configuration. Set `capacity` high enough for the maximum tokens times PLE heads per rank. HIXL currently requires ETP=TP and PCP=DCP=1, and does not support speculative decoding, DBO or elastic EP. The external table takes precedence over automatic local DP host-table sharing. Only one PLE layer and BF16 checkpoint shards are supported; dummy loading and sleep mode are not supported.

Stop consumers cleanly before the owner so graph references and transport leases can retire. Forced exit can retain a lease and block owner shutdown. There is no transparent owner failover.
