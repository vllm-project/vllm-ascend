# Layerwise pool PD prototype

This opt-in prototype uses one `AscendStoreConnector` on each instance:
P computes the prompt and saves KV to Mooncake Store or MemCache; the proxy waits for a
committed snapshot and sends the original prompt and snapshot descriptor to D.
D loads KV layer by layer, retains it in HBM, and replays the final prompt token
to obtain logits. The token sampled by P is discarded.

Complete blocks retain the existing content-addressed pool keys. An incomplete
final block gets an immutable request-scoped key, `pd-tail-v1:<transfer_id>:<N>`.
The descriptor carries the prompt digest, token count, block size, layout
fingerprint and complete-block hashes. The physical final page is transferred;
only its valid tokens participate in attention. Tail objects remain in the
bounded pool until normal backend eviction.

## Supported scope

- V1 model runner, full Attention/MLA, one KV group, equal P/D TP, PP/CP=1.
- Text or token-ID prompts, one completion, streaming or nonstreaming `/v1/completions`.
- No LoRA, multimodal inputs, speculative decoding, or decode-side pool writes.
- Dedicated pool namespace and identical model files, KV dtype, and layout on P/D.
- P must generate exactly one token; use the proxy to enforce this contract.
- Set `VLLM_USE_V2_MODEL_RUNNER=0`; the launch script selects the supported V1 runner.
- Prototype failure handling: an incomplete P snapshot returns HTTP 502 through
  the proxy. A missing or invalid D snapshot fails closed; restart D if its engine
  exits. Pool eviction/retry recovery is follow-up work. Streaming starts after
  P publishes a complete snapshot and relays D's SSE events, including usage.

## Runtime requirements

Use the vLLM commit recorded in `.github/vllm-main-verified.commit` of this
Ascend checkout. The current experiment uses `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
Mooncake must provide all seven session/range methods checked by
`MooncakeBackend.validate_layerwise_support`; the image's
`mooncake-transfer-engine-npu==0.3.11.post1` does not provide them. The experiment
built Mooncake `16b7ba3c7364eb8d36d04378bc7d28013bea1a1e` with `USE_ASCEND_DIRECT=ON`.

The A5 image is
`vllm-ascend:dev-26.2.0.day20260922-A5-py311-openEuler24.03-lts-aarch64`.
Source CANN/ATB environment scripts and preserve their `PYTHONPATH` entries when
adding source checkouts. Mount the NPU device nodes needed by HIXL topology
discovery, `/dev/ummu`, `/dev/uburma`, the driver, `/usr/bin/urma_admin`,
`/lib/route.conf`, and `/etc/hccl_rootinfo.json`.

## Start

Run commands inside the prepared runtime. The example uses one A5 card per
instance, Qwen3-30B-A3B BF16, block size 128 and chunked prefill of 128 tokens.

```bash
MOONCAKE_MASTER=/path/to/Mooncake/build/mooncake-store/src/mooncake_master
"$MOONCAKE_MASTER" --port 15051 --metrics_port 19003 \
  --default_kv_lease_ttl 120000 --client_ttl 120
bash examples/pool_pd/run_node.sh prefill 0 /models/Qwen3-30B-A3B
bash examples/pool_pd/run_node.sh decode 1 /models/Qwen3-30B-A3B
python examples/pool_pd/proxy.py
```

Run the four foreground commands in separate terminals. All HTTP services bind
to localhost. `mooncake.json` points both instances at the same master.
After restarting a client, its old memory segment can remain registered until
the master's client liveness timeout expires. For a clean experiment, stop P/D,
restart their dedicated master, then start P/D; do not reset a shared pool.

```bash
curl http://127.0.0.1:18082/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-30b","prompt":"Explain KV cache briefly:","max_tokens":32,"temperature":0}'
```

### MemCache backend

The A5 image includes `memcache_hybrid==1.2.0`. Pool PD requires its explicit
`batch_write_finish` API; older implicit-publication SDKs are rejected.
The example configuration uses a dedicated DRAM pool with `device_urma` and
disables SSD storage. Run these commands in separate terminals:

```bash
export MMC_META_CONFIG_PATH="$PWD/examples/pool_pd/mmc-meta.conf"
python -c 'from memcache_hybrid import MetaService; MetaService.main()'
bash examples/pool_pd/run_node.sh prefill 2 /models/Qwen3-30B-A3B python memcache
bash examples/pool_pd/run_node.sh decode 3 /models/Qwen3-30B-A3B python memcache
python examples/pool_pd/proxy.py --port 18182 \
  --prefill-url http://127.0.0.1:18180 --decode-url http://127.0.0.1:18181
```

P/D use ports 18180/18181, and meta/config-store/metrics use 15350/15351/15352.
Override `MMC_LOCAL_CONFIG_PATH` to use another prepared pool configuration.
These localhost addresses are for the single-machine experiment; separate hosts
need mutually reachable addresses and the corresponding transport configuration.

MemCache retains its existing GVA data path: allocate a whole block object,
copy each layer, then publish with `batch_write_finish` after the final layer.
Positive GVA metadata alone does not prove publication. PD readiness probes
acquire short read leases and release them; D acquires its own transfer leases.
Complete blocks use P's snapshot hashes, and the tail key is shared across P/D
despite their different request IDs. Committed prefixes are never overwritten.

### Prefix reuse and graph experiments

Set `pool_pd_prefix_reuse: true` in P's `kv_connector_extra_config` to load a
contiguous committed prefix from the pool before computing the remaining prompt.
The default remains full prefill. Only complete blocks count as reusable prefix;
the final partial block belongs to the new prompt snapshot. D still validates
and restores the complete snapshot and replays one prompt token.

The V1 runner executes synchronous layerwise KV restoration eagerly. Subsequent
decode steps can replay graphs. `AscendStoreConnector` currently requires
`PIECEWISE`, with compilation mode 3, when layerwise transfer is enabled; remove
`--enforce-eager` and use that supported configuration for graph experiments.
Verify graph capture and an actual `ACL graph replay is active` log message,
and use the same graph settings on both sides of a performance comparison.

For prefix-hit benchmarks, generate prefixes and unique suffixes once, prewarm
only the shared prefixes, and exclude warmup from timing. Record measured pool
hit counters independently on P and D. With 131072 input tokens, a 90% textual
prefix and 128-token blocks yield 117888 reusable tokens (89.9414%). Local HBM
prefix caching should be disabled when measuring pool reads. Keep both model
RoPE settings and TP identical. To compare against ordinary PD plus layerwise
pooling, use `MultiConnector` with `MooncakeConnectorV1` for direct P-to-D KV
transfer and `AscendStoreConnector` for P-side layerwise pooling. Both deployments
then use four cards with P TP=2 and D TP=2. Match pool client counts and configured
capacity as well as model, workload, and graph settings. A standalone TP=2
pooling instance uses only two cards and answers a different comparison.

## Validate

First run a standalone model with the same settings, without a connector, on
port 18079. Record its output, then stop it before starting P on the same card:

```bash
python examples/pool_pd/validate.py --url http://127.0.0.1:18079 --output baseline.json
python examples/pool_pd/validate.py --url http://127.0.0.1:18082 \
  --reference baseline.json --output pool-pd.json
```

The cases cover 1, 127, 128, 129, 255, 256, 257, and 513 prompt tokens plus a
repeated prompt. The validator compares every generated token ID. Check
`POOL_PD_SAVE ... status=ready` in P logs and `POOL_PD_LOAD ... replay_tokens=1`
in D logs to distinguish transfer from prompt recomputation. Enable the existing
`VLLM_ASCEND_KVPOOL_RANGE_DEBUG=1` option when byte-range evidence is needed.
Use the same chunk size in the baseline and PD runs. BF16 MoE output is not
guaranteed to be invariant to prefill chunking or reuse of an existing prefix.

## Interfaces added

- `PoolPDTransfer.create/from_wire`: snapshot identity, validation and tail key.
- `KVPoolScheduler._get_pd_matched_tokens`: D validates the descriptor and returns
  `N-1` externally cached tokens; P creates the descriptor.
- `KVPoolScheduler.request_finished`: P publishes `kv_transfer_params.pool_pd`
  only after every prompt object is readable.
- `ReqMeta.pd_transfer`: transports snapshot metadata to workers; existing
  layerwise session methods use P's complete-block hashes and tail key on D.
- `pool_pd: true`: enables the path without changing default pool behavior.
- `prepare_pool_pd_forward`: makes the V1 model runner prepare layerwise sessions
  before attention even on P with zero cache hits. Recent vLLM otherwise defers
  `start_load_kv` until after forward, which is too late to prepare layer PUTs.

## Prepared A5 runtime

The experiment container is `d2rh-pool-20260925`. Its host workspace is
`/root/d2rh-pool-20260925`, mounted at `/workspace`. Enter it with
`docker exec -it d2rh-pool-20260925 bash`, then source `/workspace/runtime_env.sh`.
Python is `/workspace/.venv/bin/python`; the built master is
`/workspace/Mooncake/build/mooncake-store/src/mooncake_master`.
Source is under `/workspace/vllm-ascend` and `/workspace/vllm`; the read-only
model is `/models/Qwen3-30B-A3B`. P uses card 0, D uses card 1. The proxy listens
on `127.0.0.1:18082`; P/D listen on 18080/18081. Logs and validation JSON are
under `/workspace/logs`. No external HTTP listener is required.

## Verification on 2026-09-25

Qwen3-30B-A3B BF16, TP=1 per instance, A5 cards 0/1, eager mode, block/chunk
size 128:

- 99 unit tests passed across pool PD, proxy, scheduler and Mooncake layerwise tests.
- A real NPU-to-pool-to-NPU session/range round trip preserved tensor values.
- From an empty pool, all 9 boundary/repeated-prompt requests completed and each
  generated 16 token IDs identical to the standalone chunk-128 baseline.
- The same 9 cases at concurrency 2 matched every output token ID.
- Three Chinese question/answer prompts also matched the standalone output.
- P readiness, D layerwise range reads and D's external-cache hit metrics were
  checked; D replays one prompt token. MLA and multi-card TP were not run.

Result files are `final-pool-pd-results.json`,
`final-pool-pd-concurrent-results.json`, `natural-results.json` and
`baseline-chunk128-results.json` under the runtime log directory. The initial
chunk-256 baseline differed on the 129-token prompt and its repeat. A standalone
chunk-128 control reproduced PD's output for those cases; the final deployment
uses chunk 128 on both sides. This experiment establishes the first functional
path, not throughput, batch invariance, or production failure recovery.

## MemCache verification on 2026-09-26

Qwen3-30B-A3B BF16, TP=1 per instance, A5 cards 2/3, eager mode, block/chunk
size 128, `memcache_hybrid==1.2.0`, `device_urma`, DRAM pool:

- 381 unit tests passed across PD, MemCache PD, proxy, scheduler, worker,
  backend, layer transfer and Mooncake layerwise tests.
- All 9 boundary/repeated-prompt requests matched the saved standalone
  chunk-128 baseline token for token, both serially and at concurrency 2.
- Three Chinese/English requests completed with readable output, including
  Beijing as China's capital and `12 * 13 = 156`.
- P published complete snapshots and D logged `replay_tokens=1`; the D external
  cache hit metric and healthy P/D endpoints were checked.

The container keeps the original Mooncake deployment on cards 0/1. MemCache
source is `/workspace/vllm-ascend-memcache`; its P/D/proxy ports are
18180/18181/18182. Logs and result JSON are under
`/workspace/.vaws-local/memcache-pd/20260926`.

The standalone SDK probe copied NPU data correctly but aborted during process
cleanup, including after explicit buffer unregister and store close. This
image/SDK cleanup issue remains unresolved; the serving processes stayed
healthy throughout the functional tests. Clean shutdown, larger concurrency,
eviction recovery, graph mode, MLA and multi-card TP are not established by
these results.

## MemCache PD comparison on 2026-09-27

Compared unmodified ordinary PD plus layerwise pooling at
`ba945e40f81894930b6884ce608cd4c1211a0f28` with this pool PD path. Both used the
same four A5 cards, Qwen3-30B-A3B BF16, P/D TP=2, PIECEWISE graphs, four MemCache
data clients configured for 64 GB each, and no local prefix cache or speculative
decoding. Input/output lengths were 131072/1024 tokens with four shared prefixes;
P reused 117888 tokens per request. Each state ran two rounds of four requests
at concurrency 1 and 4, in A/B/B/A order, after separate prefix and serving warmup.
All 32 formal requests completed without preemption.

- Concurrency 1, baseline to pool PD: mean TPOT 70.40 to 58.72 ms, TTFT 4104.3
  to 4531.6 ms, output throughput 13.45 to 15.96 tokens/s. Pool PD's round-to-round
  TPOT CV was 12.34%, exceeding the preset 10% limit; the result is inconclusive
  for a repeatable improvement.
- Concurrency 4: mean TPOT 84.23 to 79.39 ms, TTFT 9369.0 to 9703.8 ms, and output
  throughput 40.85 to 42.81 tokens/s. These two rounds observed 5.75% lower TPOT
  and 4.80% higher throughput, with 3.57% higher TTFT.
- Continuing decode steps selected PIECEWISE in both states and issued no pool
  KV loads. These counters do not identify the cause of the TPOT difference.

This is a deployment comparison, including different connectors and proxies,
with only two rounds per state. The block-aligned 128K prompts do not exercise
tail blocks, and synthetic performance inputs are not an accuracy evaluation.
MLA and production eviction/retry recovery remain unverified.
