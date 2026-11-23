# Shared host Engram over HIXL

This opt-in DeepSeek V4.1 backend keeps one immutable Engram table in an external owner process. Independent prefill and decode instances retrieve selected quantized rows over HIXL instead of loading a complete host table per instance. It supports INT8 codes with E8M0 or FP32 scales, and FP8 E4M3 codes with E8M0 scales.

## Data path

```text
Consumer NPU hashes, gathered across the existing Engram group
  -> one CPU copy of both layers' local head IDs
  -> TCP request containing IDs
Owner ordinary DRAM table
  -> CPU gather into a registered, pinned host buffer for this reader
  -> HIXL READ into the consumer's bounded HBM buffer
  -> NPU dequantization into BF16 rows
  -> existing TP/DP row exchange and graph-input preparation
  -> existing model execution
```

This is explicit selected-row transfer. The NPU does not dereference a remote DRAM pointer. The owner does not stage the entire table into HBM. TP/DP collectives, selected-row HBM storage and graph-input copies remain. Engram lookup runs before decode graph replay; the decoder graph mode does not need to be disabled. KVPool storage and Mooncake P/D KV transfer retain their existing connectors.

On A2 the tested configuration uses device RoCE. On A5 configure HIXL device UB/URMA endpoints; the transport is selected by HIXL's communication resources, not by the host-side engine address. A5 small-table FP8 lookup has been exercised across two nodes; full-model and multi-reader qualification remain separate.

On A5, provide a device UB endpoint JSON with owner `--local-comm-res` and consumer `local_comm_res`. For multiple devices, the path may contain `{physical}`, replaced with the process physical device ID. Each process needs resources generated for its own device. Set owner `--fabric ub` to bypass the A2 RoCE-specific alternate-device workaround; the endpoint resources must still specify device UB.

The two-layer hash copy happens before either layer enqueues its TP exchange. This removes the second ID copy's dependency on the first layer's queued exchange. Each HIXL read still waits for the previous use of its shared local buffer before reuse.

When the owner advertises batch lookup and both layers' selected rows fit within `capacity`, one request packs both layers and one HIXL READ fetches them. Each layer is then dequantized into its own local BF16 result before the existing exchanges. This keeps the registered receive buffer unchanged, with bounded simultaneous per-layer results. Larger batches, older owners and the direct single-rank graph-buffer path retain per-layer lookup. No new collective or decoder graph requirement is introduced.

## Build and start

The native adapter uses the installed CANN HIXL headers and library rather than the HIXL Python extension. The CMake build includes `libengram_hixl.so` when the HIXL library is present. For a standalone build against an installed SDK:

```bash
g++ -O2 -std=c++17 -fPIC -shared csrc/engram_hixl/native.cpp \
  -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" \
  -lcann_hixl -lascendcl -lmetadef -pthread -o libengram_hixl.so
```

Start the owner with reachable, distinct control and HIXL endpoints:

```bash
python -m vllm_ascend.shared_engram \
  --backend a2 \
  --model /path/to/checkpoint \
  --endpoint OWNER_IP:29500 --engine-endpoint OWNER_IP:29501 \
  --device 0 --listen-port 26666 --library-path /absolute/path/libengram_hixl.so \
  --ready-file /path/to/owner-ready.json
```

Wait for `POOL_READY` from this owner process. Enable consumers with `cpu_offload=true` and `dp_shared_memory=true` in `--engram-config`, and merge the following into their existing `--additional-config`:

```json
{
  "shared_engram": {
    "backend": "a2",
    "endpoint": "OWNER_IP:29500",
    "capacity": 65536,
    "library_path": "/absolute/path/libengram_hixl.so"
  }
}
```

For A5, use `--backend a5` for the owner and `"backend": "a5"` for consumers. Select the owner transport explicitly with `--fabric ub` or `--fabric roce` as appropriate.

Owner `--listen-port` and `--local-comm-res` are mutually exclusive. Consumers may set `listen_port` or `local_comm_res` in `shared_engram`; these options are mutually exclusive and omitted by default. Rebuild the native adapter when updating this code.

`capacity` is a minimum number of selected rows in one lookup, not a token count. The consumer chooses the larger of 65536, configured capacity, and the maximum batch or graph size multiplied by the largest per-TP single-layer head count, then sends that capacity to the owner on attach. The supported limit is 262144 rows; configurations requiring more fail at startup. Two-layer batches may fall back to per-layer lookup. A single-layer lookup exceeding the negotiated capacity fails locally without retiring the reader and is not split automatically. The consumer allocates a bounded packed-row HBM buffer; the owner allocates a pinned stage of the same capacity per connected reader. Other Engram result/exchange buffers remain additional allocations.

The consumer requires the accompanying core loader change that invokes the model's unused-checkpoint-weight filter before materializing a tensor. Metadata-only parameters alone do not prevent a loader from temporarily loading the full table. The multimodal wrapper forwards this filter too.

## Placement and lifecycle

The owner runs as a separate process and needs an NPU for HIXL even though its persistent table is in DRAM. It may be co-located with a model instance; reserve distinct HIXL device listen ports to avoid a conflict with model communication. In the A2 device RoCE configuration, a reader using the owner's device connects through another visible owner device. This implementation does not use a direct local-memory lookup. Verify the available devices and resource accounting for the deployment.

Readers compare checkpoint metadata and sampled boundary weights before attaching. Each owner has a generation; mismatched generations and invalid lookup shapes fail explicitly. Sampling is a compatibility check, not a full checkpoint checksum or a model-quality evaluation.
Owner and readers must use matching protocol versions: attachment includes the physical device ID as well as row dtype and index metadata, so mixed old and new versions reject attachment.

On normal shutdown, the reader synchronizes NPU use, disconnects/deregisters its local transport, and asks the owner to retire its leased stage. Set a positive `--shutdown-timeout` on every consumer so its worker processes have time to run this cleanup; the default immediate shutdown can force-kill workers and leave retained leases. An unexpected socket EOF does not prove outstanding RDMA has retired, so the owner retains that stage conservatively. Automatic reclamation after consumer crashes and transparent owner failover are not implemented. Owner shutdown drains normal leases; a retained failed lease can keep it draining.

Set `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` high enough for readers to close; the shared backend defaults it to 120 seconds if unset.

## Validation and performance claims

Focused tests cover protocol bounds, checkpoint compatibility, loading under a non-CPU default device, packed-row handling, batched ID ordering and both supported scale representations. Full-size checkpoint row probes cover both Engram layers, nonzero head offsets and invalid-ID masking. End-to-end validation must additionally preserve the intended P/D topology, decoder graph configuration, KVPool capacity and production request distribution.

Report unique table bytes, owner/staging memory, device buffers and task reservations separately. DP-shared native host mappings must not be multiplied by the worker count. Process PSS may omit driver-managed Host allocations. Restoring a larger KVPool can increase total configured DRAM even while eliminating duplicated Engram tables. Small lookup microbenchmarks alone do not establish serving throughput or SLO equivalence.
