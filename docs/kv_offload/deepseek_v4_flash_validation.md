# DeepSeek-V4-Flash Mooncake KV offload validation

The prefill (P) worker supports cross-layer slot reuse for attention compressor KV
(compressed history), sliding-window attention (SWA), indexer K and scale, and
the recurrent states of the attention and indexer compressors. Independent
layers and DSpark draft caches retain their own addresses within their groups.

## Layout and runtime configuration

The model layout entry point in `vllm_ascend/models/deepseek_v4/cache_config.py`
uses the common planner to assign shared slots by page stride and cache group.
Components in the same layer and group use distinct slots. A layer that reuses
multiple slots waits for every predecessor's Store save and handoff before
writing those slots.

The decode (D) worker uses the standard cross-group layout. The P worker
allocates each physical slot separately. Capacity estimation, Model Runner V2
(MRV2) allocation, and transfer addressing use the same layout descriptors.
The planner reserves an extra 2 MiB per allocation for alignment and counts
aliased storage once.

Use vLLM commit `24420b7a341af37a9bc3186e6fc6062512e0aa58`. The scheduler
can read the current grouped block table through
`kv_connector_block_state.get_block_ids()`, including null entries left after
sliding-window reclamation. Validate continuation even when a step allocates
zero new blocks.

Use MRV2 with `--enforce-eager --max-num-batched-tokens 4096` and synchronous
scheduling for the initial baseline. Set pipeline parallelism (PP), prefill
context parallelism (PCP), and decode context parallelism (DCP) to 1. The target
configuration uses DP4/TP1 with expert parallelism enabled. For DSpark, use
`--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`.
Keep the model, quantization, and Mooncake settings consistent with the
baseline. The Store connector configuration is:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 2,
    "layerwise_independent_layers": [0, 1],
    "layerwise_prefetch_layers": 2
  }
}
```

Run with 1, 2, 3, and 4 shared slots per compatible set and retain the full
server logs. Set `layerwise_independent_layers` to `"all"` for a fully resident
control that still uses the external Store. Use the original service without
slot sharing to check the standard layout separately.

## Validation criteria

| Area | Procedure | Expected result |
| --- | --- | --- |
| Shared and resident layout | Check component addresses, group IDs, page sizes, strides, and predecessors in host tests with 1, 2, 3, and 4 shared slots. | Only eligible components share slots with compatible layouts. Independent layers and resident components keep separate allocations. |
| Physical memory budget | Compare the budget with unique allocations and alignment reserves in host tests. | Each allocation is counted once, and the shared slot count does not reduce resident allocations. |
| Chunked prefill continuation | Use 4096-token chunks and input lengths of 4096, 4097, 8193, and 12289. Compare deterministic outputs. | Restoration uses the previous chunk's exact token endpoint. Resident data stays in HBM and outputs match. |
| External prefixes and concurrency | Compare hit metrics and outputs for cold, repeated, different-suffix, and mixed-length concurrent requests. | A full hit requires all required groups. Missing data shortens the hit or causes the request to fail when required. |
| Slot overwrite protection | Control predecessor completion events in host tests. | A shared slot becomes writable after all required predecessor saves and handoff reads finish. |
| Lifecycle and failures | Exercise normal completion, cancellation, GET, PUT, and commit failures, and a small pool. | Required transfer failures prevent reuse of incomplete data. Host tests verify cancellation and read-session release. |

During continuation, resident components in a mixed group can still be
published for external readers. The worker reloads the shared components that
the next chunk needs. External prefix restoration uses the shared
coordinator's endpoint and readable-range checks. Public Store keys account
for component specifications, format, model configuration, and parallel
sharding without adding a data-parallel rank namespace.

### Deterministic output and concurrency

Run from the repository root on the server. Use the same model weights,
tokenizer, and sampling settings for both services. The script supplies token
IDs directly to control input length. Its concurrent batches include repeated
and mixed-length inputs.

```bash
python tests/e2e/cases/features/kv_pool/tools/compare_dsv4_offload.py \
  --baseline http://BASELINE_HOST:PORT \
  --candidate http://OFFLOAD_HOST:PORT \
  --model dsv4 --concurrency 4 > comparison.log
```

Retain the full service logs. Report performance metrics, output comparison
results, and the first error with its context in at most 300 lines. After a
cancellation, send a normal request to check that the service still responds.

## Host checks

```bash
python -m pytest --noconftest -q tests/host/test_dsv4_mooncake_offload.py
bash format.sh ci
```

Host tests exercise production layout, addressing, and range-selection
functions and include host memory copy checks. Use the server scenarios above
to validate NPU execution, stream synchronization, service startup, and
multi-device outputs. The platform layout patch calls the model-specific layout
entry point and can be removed once upstream provides the same hook.

## P/D deployment

The standalone Store configuration is for single-node offload. The layerwise
proxy sends the request to D first and dispatches to P after D calls
`/v1/metaserver`. The standalone Store connector does not implement that
callback.

- On P, use `MultiConnector` with `MooncakePushConnector` and
  `AscendStoreConnector`. Store handles P-side save and restore.
- On D, use `MooncakePushConnector` with fully resident caches. Push copies KV
  directly from P device memory to D device memory.
- Use the same Push implementation, model, quantization, cache format, and
  block size on both sides. Preserve MRV2, DP/EP, and Mooncake settings.
- Push uses each component's group ID, live block table, and actual page
  stride. Validate history completion and transfer of live window and recurrent
  state at the final endpoint.
- D computes the last prompt token locally. P prefills through the token
  endpoint N−1. The proxy returns the response from D.
- Shared slots wait for Store writes and D2D reads before reuse. Validate that
  request blocks remain allocated until the associated reads finish.

Use TP=PP=PCP=DCP=1 and DP4 with expert parallelism. Start with eager execution,
synchronous scheduling, and speculative decoding disabled. Then run the
combined configuration with `--async-scheduling --enable-prefix-caching`.
Check that the P worker restores the shared slots from the external prefix and
that the D worker reuses its local resident prefix. For a partial D hit,
check that Push fills the required remaining history and final window. Compare
cold and repeated requests as well as concurrent requests that share a prefix
and have different suffixes.

Add the following arguments to the P startup command:

```bash
--kv-transfer-config "$(cat examples/disaggregated_prefill_v1/dsv4_offload_p.json)" \
--enforce-eager --no-async-scheduling --max-num-batched-tokens 4096
```

Add the following arguments to the D startup command:

```bash
--kv-transfer-config "$(cat examples/disaggregated_prefill_v1/dsv4_offload_d.json)" \
--enforce-eager --no-async-scheduling
```

The example side-channel port ranges are 14500–14503 on P and 14600–14603 on D.
Ensure the peers can reach these ports. Start the proxy from the repository root:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host PROXY_HOST --port 1999 \
  --prefiller-hosts PREFILL_HOST --prefiller-ports 8099 \
  --decoder-hosts DECODE_HOST --decoder-ports 8066
```

Point the benchmark and comparison script at `http://PROXY_HOST:1999`. Start
with a short request and confirm it reaches both workers. Then check
cross-chunk requests of 4097, 8193, and 12289 tokens for completed transfers,
shared-slot waits, timeouts, and output parity.
