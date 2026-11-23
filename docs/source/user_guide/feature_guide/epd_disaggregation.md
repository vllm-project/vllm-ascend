# Disaggregated Encoder with Mooncake on Ascend

`ECMooncakeConnector` transfers multimodal encoder outputs from an Encoder
instance (E) to a language-model instance running prefill and decode (PD).
Separating these stages lets you scale vision encoding independently of the
language model. With cross-Encoder caching enabled, Encoders can also reuse
image embeddings through a shared Mooncake Store.

This guide provides separate **A2/RoCE** and **A3/HCCS with FabricMem**
transport setups. Choose the setup for your hardware, start a single-host
**1E + 1PD** service, then add shared caching and a second Encoder. The serving
commands use the `ascend` Mooncake protocol in both setups.

## Roles and topologies

Disaggregated Encoder (EPD) splits the multimodal path of a vision-language
model into up to three stages, each running in its own vLLM instance group:

- **E (encoder)** runs only the multimodal encoder (for example the ViT of
  Qwen3-VL) and writes the encoder cache (EC) to a shared backend. An E
  instance is encoder-only: it is marked `mm_encoder_only` by the
  `ec_role: ec_producer` configuration, runs the encoder and the EC save path,
  and does not allocate a KV cache.
- **P (prefill)** is optional. It consumes the encoder cache, computes the
  prompt KV cache, and transfers that KV cache to D through a prefill-decode
  connection such as `MooncakeLayerwiseConnector`.
- **PD / D (decode)** generates tokens. In the E+PD topology the PD instance
  also runs prefill and is the encoder-cache consumer; in E+P+D the P instance
  is the encoder-cache consumer and D only consumes the KV cache.

A load-balancing proxy in front of the instances routes every request to one
encoder instance, optionally to one prefill instance, and finally to one decode
instance. A text-only request skips the encoder stage entirely, so decoding does
not wait for vision encoding.

| Topology | Instances | EC consumer | Notes |
| --- | --- | --- | --- |
| E + PD | 2 | PD | Two cards are enough; the Ascend example is `examples/disaggregated_encoder/disagg_1e1pd_example.sh`. |
| E + P + D | 3+ | P | P and D are also connected by a prefill-decode KV connector; see [Separate prefill and decode](#separate-prefill-and-decode). |
| Encoder-only pooling models | 1 | none | Not EPD: `--runner pooling` models have no stage split; see [Model Runner support](#model-runner-support). |

Running the stages in separate processes gives three practical benefits:

- **Independent scaling**: encoders are small compared with the language model,
  so E instances can be added or removed without touching the P/D fleet.
- **Lower time-to-first-token (TTFT)**: the encoder no longer competes with
  prefill and decode for the same device, and the result can be produced ahead
  of the prompt.
- **Cross-process encoder-cache reuse**: a request whose image was already
  encoded on any E instance can load the embedding from the shared cache
  instead of running the vision encoder again.

For the architecture, data/control plane, and interface specification, see the
[design document](../../developer_guide/Design_Documents/epd_disaggregation.md).

## Deployment overview

```text
Router / proxy             Encoder / producer                    PD / consumer
      |                             |                                  |
      |--------image request------->|                                  |
      |<-------image metadata-------|                                  |
      |-------------------metadata-only request----------------------->|
      |                             |-------reserve destination------->|
      |                             |<-------session + address---------|
      |                             |======Mooncake embedding write===>|
      |                             |                                  |
      |                             |                           Prefill + decode
      |<-----------------------generated response----------------------|
                                    ^
                          GET / async PUT on miss
                                    |
                                    v
                       +-------------------------+
                       |     Mooncake Store      |
                       | Shared embedding cache  |
                       +-------------------------+
```

With caching enabled, Encoders GET embeddings from Store and encode on a
miss, then publish through asynchronous PUT. Both cached and newly encoded
outputs reach PD through EC P2P. Store is a shared cache for the Encoders;
it does not forward requests between the proxy and PD.

## Requirements

- Install compatible vLLM, vLLM Ascend, CANN, and Mooncake builds. See [Install Mooncake](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#install-mooncake)
  for installation guidance.
- Each producer must use TP=1, PP=1, and DP=1. Scale encoding by starting
  independent producer instances.
- Reserve NPU memory for the model and EC pools, and provision Store storage
  separately for the selected transport. Tune the example pool sizes and
  memory-utilization fractions for your model and hardware.
- Ensure HTTP services, EC control addresses, and Mooncake transport endpoints
  are reachable from their peers. For cross-node and container setup, follow
  [Verify Multi-Node Communication Environment](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#verify-multi-node-communication-environment)
  and the container instructions in that guide.

For cross-Encoder caching:

- Install Mooncake with Ascend Store support and run a RAM Store with
  `mooncake_master`.
- Leave multimodal processor caching enabled (the default) to preserve content
  identifiers.
- Use the same immutable weights, compatible preprocessing, and matching
  multimodal identifiers on participating Encoders.
- Do not enable LoRA; the current connector rejects LoRA configurations.

## Hardware, connectivity, and software

Choose the communication path independently from the number of Store engines:

| Environment | EC embeddings, E to PD | Store access |
| --- | --- | --- |
| A2 | Ascend Direct RoCE | Non-FabricMem RoCE, shared or independent Store engine |
| A3 within a supported HCCS domain | Ordinary HCCS device-to-device transfer | FabricMem access to Store Host memory over the supported fabric |
| A3 with an operational RoCE network | Ascend Direct RoCE | Non-FabricMem RoCE |

Confirm the actual server/product, visible device mapping, and interconnect.
FabricMem is a Host-memory access mechanism on supported platforms. Ordinary
HCCS device transfer alone does not give a Store access to remote Host memory.
A2 does not support the mixed **EC/HCCS + Store/RoCE** configuration in this
connector. A5 (Atlas 950) has not yet been tested or adapted for this EC Store
integration.

Use the [Ascend compatibility table](kv_cache_pool_ascend_store.md#hdk-and-cann-version-compatibility-reference)
for the installed hardware and software. For FabricMem, Mooncake additionally
requires supported A3 hardware, CANN 9.0+, and HDK 26.0+; see its
[Ascend Direct requirements](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/transport/ascend_direct_transport.md#environment-variables-configuration).
These communication requirements apply to both embedded and standalone Store.

Before serving:

1. Inspect `npu-smi info` and the device mapping available inside the container.
   `ASCEND_RT_VISIBLE_DEVICES` must use the device identifiers accepted by that
   installation; do not assume host physical IDs equal process-local IDs.
2. For RoCE, check each selected device's IP, link, and peer reachability with
   `hccn_tool` and the [Verify Multi-Node Communication Environment](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#verify-multi-node-communication-environment).
3. For HCCS/FabricMem, verify the fabric domain, device visibility, permissions,
   and the A3 communication checks. A missing RoCE IP does not by itself rule
   out HCCS/FabricMem. Conversely, topology output showing HCCS does not verify
   Host-memory allocation or mapping.
4. Mount the devices, driver libraries, configuration files such as
   `/etc/hccn.conf`, and network resources required by the chosen path, following
   the [container guide](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#run-with-docker).
   Keep the driver, userspace libraries, and Mooncake build compatible.

The connector leaves `HCCL_INTRA_PCIE_ENABLE`, `HCCL_INTRA_ROCE_ENABLE`,
`ASCEND_GLOBAL_RESOURCE_CONFIG`, and `ASCEND_ENABLE_USE_FABRIC_MEM` unchanged.
Configure them before starting each process; restarting a service is required
when changing transport. Do not change selectors between EC and Store engine
creation inside a running worker.

## Model Runner support

EPD runs on **Model Runner V2 (MRV2)**, the only model runner. Model Runner V1
has been removed from vllm-ascend, so no extra flag is needed on the Ascend
side and there is no V1 fallback to select. Do **not** set
`VLLM_USE_V2_MODEL_RUNNER=0`: the setting cannot be honored and the vLLM
process fails at startup with a `ValueError` (`vllm_ascend/mrv2_utils.py`).

Runner selection lives in `vllm_ascend/mrv2_utils.py`; E nodes use the upstream
`MMEncoderModelRunner` for the encoder-only instance, and P/D nodes use the
Ascend V2 model runner and the upstream encoder-cache contract.

**The VL encoder graph is supported on MRV2**: setting
`--compilation-config '{"cudagraph_mm_encoder": true}'` enables the encoder ACL
graph on Model Runner V2, and the V1 fallback that older builds used for it no
longer exists. Features the V2 runner cannot serve are rejected at startup
instead of falling back to another runner. The EPD examples do not enable the
encoder graph.

### Encoder-only pooling models are a different entry point

Models such as BERT, BGE, and SigLIP2 are served as **encoder-only pooling
models** with `--runner pooling` and the Ascend encoder-only model state. They
are not EPD deployments: there is no separate E/P/D split, no
`--ec-transfer-config`, and no encoder-cache transfer between instances. In
particular, SigLIP2 is a pooling embedding model and does not support PD
separation; see [SigLIP2](../../tutorials/models/SigLIP2.md#5-online-service-deployment).

## Encoder-cache connectors

Producer and consumer instances must use the same connector and the same
connector configuration. The EC data plane is selected by the connector in
`--ec-transfer-config`; Ascend-specific connectors are registered in
`vllm_ascend/distributed/ec_transfer/__init__.py`.

| Connector | Data plane | Extra configuration keys | When to use |
| --- | --- | --- | --- |
| `ECExampleConnector` | Shared filesystem, one safetensors file per `mm_hash` | `shared_storage_path` | E and P/D instances can see the same directory; simplest reference setup |
| `ECMooncakeConnector` | Mooncake with Ascend direct transfer and a bounce arena | `mooncake_protocol` (must be `ascend` when set; defaults to it), `ascend_mooncake_bounce_arena_size` | A Mooncake cluster with shared storage is deployed; multi-node EC reuse |
| `ECCPUConnector` | Pinned CPU mmap under `/dev/shm` | `ec_cpu_bytes` | Single host, repeated multimodal inputs; no cross-node transfer |
| `ECMemcacheConnector` | MemCache cluster through `memcache_hybrid` | `ec_connector_module_path`, `recv_buffer_tokens` | A MemCache deployment is available and EC entries must survive across hosts |

Notes:

- `ECExampleConnector` and `ECMooncakeConnector` are the cross-process
  connectors for EPD. The E instance acts as producer and P/D as consumer.
- `ECCPUConnector` is a single-host cache, supports `ec_both` only, and does
  not perform cross-node transfer; see
  [Encoder Cache CPU Offload](encoder_cache_cpu_offload.md).
- `ECMemcacheConnector` is configured with an explicit
  `ec_connector_module_path` (the upstream factory cannot find it otherwise)
  and requires the `memcache_hybrid` package plus `MMC_LOCAL_CONFIG_PATH`.
- The Mooncake connector requires a working Mooncake deployment; see
  [Install Mooncake](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md#install-mooncake).

## 1. Prepare the serving environment

The example uses E on NPU 0, PD on NPU 1, and these ports:

| Service | Address |
| --- | --- |
| Encoder HTTP | `127.0.0.1:19534` |
| PD HTTP | `127.0.0.1:19537` |
| PD EC reservation/control | `tcp://127.0.0.1:19019` |
| EPD proxy HTTP | `127.0.0.1:10002` |
| Optional Store master | `127.0.0.1:50051` |

Run this environment setup in each serving terminal. Use the same configured
model path or identifier on E and PD:

```bash
export MODEL=Qwen/Qwen3.5-9B
```

### A2: EC and Store over RoCE

In every Encoder and PD terminal:

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
unset ASCEND_BUFFER_POOL
unset ASCEND_GLOBAL_RESOURCE_CONFIG
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
```

This selects the shared EC/Store engine path when caching is enabled. PD uses
its own EC receive engine; it does not become a Store client. Keep the selectors
on both roles for the lifetime of their processes: successful engine creation
alone does not establish that later HCCL channel creation will use the same path.

For an **independent Store RoCE engine**, use the same selectors and additionally
set this on each Encoder before starting it:

```bash
export ASCEND_GLOBAL_RESOURCE_CONFIG='{
  "comm_resource_config.protocol_desc": ["roce:device"],
  "store": {
    "comm_resource_config.protocol_desc": ["roce:device"]
  }
}'
```

Set the matching top-level EC configuration on PD:

```bash
export ASCEND_GLOBAL_RESOURCE_CONFIG='{
  "comm_resource_config.protocol_desc": ["roce:device"]
}'
```

The top level selects EC resources; the `store` object selects the Store-owned
engine's resources. A separate engine does not create an independent network
or imply that mixed HCCS/RoCE works on A2.

### A3: EC over HCCS, Store over FabricMem

Within a supported HCCS/FabricMem domain, replace the RoCE setup with the
following in every Encoder and PD terminal:

```bash
unset HCCL_INTRA_PCIE_ENABLE
unset HCCL_INTRA_ROCE_ENABLE
unset ASCEND_GLOBAL_RESOURCE_CONFIG
unset ASCEND_BUFFER_POOL
export ASCEND_ENABLE_USE_FABRIC_MEM=1
```

EC uses ordinary Ascend HCCS device transfers. On a caching Encoder, Store
creates its own FabricMem engine and owns the Host storage segment; it does
not reuse the EC engine. PD needs HCCS reachability for EC reception and does
not need a Store client. FabricMem applies to the Store engine, so enabling it
does not turn the EC payload path into a Host Store operation.

This is the A3 HCCS/FabricMem setup described in the
[Ascend Store guide](kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend).
Leave the resource JSON unset for this basic setup.

To use **A3 RoCE instead**, use the A2 RoCE environment block and optionally
its independent-engine block after verifying the A3 RoCE network. Disable
FabricMem by **unsetting** `ASCEND_ENABLE_USE_FABRIC_MEM`, not by assigning
`0`.

### Optional A3 experiment: EC over HCCS, Store over RoCE

You can also try **EC/HCCS + Store/RoCE** on A3 when the participating devices
have both working HCCS and RoCE connectivity, and the installed Mooncake/HIXL
stack supports selecting the two engines' resources independently. We have
not tested this combination for this connector; treat it as an experimental
configuration rather than the validated A3 FabricMem path.

On each Encoder, disable FabricMem and global RoCE forcing, then select the
protocol for each engine explicitly:

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
unset ASCEND_BUFFER_POOL
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE
export ASCEND_GLOBAL_RESOURCE_CONFIG='{
  "comm_resource_config.protocol_desc": ["hccs:device"],
  "store": {
    "comm_resource_config.protocol_desc": ["roce:device"]
  }
}'
```

On PD, use HCCS for its EC receive engine:

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
unset ASCEND_BUFFER_POOL
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE
export ASCEND_GLOBAL_RESOURCE_CONFIG='{
  "comm_resource_config.protocol_desc": ["hccs:device"]
}'
```

Do not enable `HCCL_INTRA_ROCE_ENABLE=1` globally for this experiment: it can
force the EC engine onto RoCE despite its HCCS resource configuration. If that software stack requires the global RoCE switch for
Store operation, use one of the earlier setups instead.

## 2. Start the Encoder and PD

The commands below are a small functional example, not throughput tuning.
The memory fractions and EC pool sizes must fit the selected model and device;
increase the pool if concurrent image payloads do not fit. Use real model weights
on both roles, not `--load-format dummy`, for output validation.

Start E in its own terminal with the common environment and **one** transport
setup from step 1:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --host 0.0.0.0 --port 19534 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --enable-request-id-headers \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_producer",
    "ec_buffer_device": "npu",
    "ec_buffer_size": 134217728,
    "ec_connector_extra_config": {
      "mooncake_protocol": "ascend"
    }
  }'
```

Start PD in another terminal with the same model and the matching transport setup:

```bash
ASCEND_RT_VISIBLE_DEVICES=1 vllm serve "$MODEL" \
  --host 0.0.0.0 --port 19537 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --enable-mm-embeds \
  --enable-request-id-headers \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_ip": "127.0.0.1",
    "ec_port": 19019,
    "ec_buffer_device": "npu",
    "ec_buffer_size": 268435456,
    "ec_connector_extra_config": {
      "mooncake_protocol": "ascend"
    }
  }'
```

### EC configuration fields

`--ec-transfer-config` is a JSON object. Configure the following fields on each
role; `ec_connector_extra_config` holds connector-specific options:

| Field | Location | Meaning in these commands |
| --- | --- | --- |
| `ec_connector` | EC config | `ECMooncakeConnector` selects the Mooncake EC connector; Ascend supplies its NPU implementation. Use it on both roles. |
| `ec_role` | EC config | `ec_producer` encodes and sends outputs on E; `ec_consumer` receives them on P/PD. |
| `ec_ip` | Consumer EC config | Local address on which the ZMQ control service binds and which E uses to contact it. Use a peer-reachable address across hosts. |
| `ec_port` | Consumer EC config | ZMQ control port for the first PD worker, here `19019`. E contacts it to reserve space for embeddings and obtain the destination address. Other TP/DP workers use consecutive ports, as shown below. |
| `ec_buffer_device` | EC config | `npu` places the transfer pool on the NPU. |
| `ec_buffer_size` | EC config | Pool size in bytes: E uses it for staging outgoing embeddings and Store I/O; P/PD uses it to receive embeddings. Here E reserves 128 MiB and PD 256 MiB. It is separate from Store Host capacity. |
| `ec_connector_extra_config` | EC config | Object containing transport and optional Store reuse settings. |
| `mooncake_protocol` | Extra config | Set to `ascend` on both roles. HCCS/RoCE/FabricMem are selected through the process environment, not by changing this value to `tcp` or `rdma`. |
| `control_timeout_s` | Extra config, optional | Reservation RPC timeout; default 30 seconds. |
| `push_wait_timeout_s` | Extra config, optional | Consumer wait for a matching push; default 60 seconds. |

`--enable-mm-embeds` explicitly enables received-embedding inputs on PD.
Cross-Encoder caching requires multimodal processor caching to remain enabled;
the default already satisfies this requirement. Encoder TP/PP/DP must each be
one; these example commands use TP=1 and the default PP=1, DP=1.

For the single-worker example, PD listens for EC control requests on `19019`.
Its OpenAI HTTP API listens on `19537`; these are different services. The
control port handles reservations and transfer coordination, not the embedding
payload itself.

### Consumer TP and DP examples

E remains a single-device producer on device 0 with TP=1, DP=1. The examples
below replace the PD command above and the proxy command in step 3; use only
one example at a time. Each uses devices 1 and 2 for PD, so three devices are
needed in total. Apply the same model and chosen transport environment first.

#### TP=2, DP=1: one PD model split across two devices

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2 vllm serve "$MODEL" \
  --host 0.0.0.0 --port 19537 \
  --tensor-parallel-size 2 --data-parallel-size 1 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --enable-mm-embeds \
  --enable-request-id-headers \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_ip": "127.0.0.1",
    "ec_port": 19019,
    "ec_buffer_device": "npu",
    "ec_buffer_size": 268435456,
    "ec_connector_extra_config": {"mooncake_protocol": "ascend"}
  }'
```

TP ranks 0 and 1 use control ports `19019` and `19020`. The proxy needs only
the TP-rank-0 address; the connector coordinates delivery to the TP workers:

```bash
export VLLM_ROOT=/path/to/vllm
python "$VLLM_ROOT/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
  --host 0.0.0.0 --port 10002 \
  --encode-servers-urls http://127.0.0.1:19534 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19537 \
  --ec-consumer-zmq-addrs tcp://127.0.0.1:19019 \
  --ec-consumer-dp-size 1
```

#### TP=1, DP=2: two PD replicas

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2 vllm serve "$MODEL" \
  --host 0.0.0.0 --port 19537 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --enable-mm-embeds \
  --enable-request-id-headers \
  --ec-transfer-config '{
    "ec_connector": "ECMooncakeConnector",
    "ec_role": "ec_consumer",
    "ec_ip": "127.0.0.1",
    "ec_port": 19019,
    "ec_buffer_device": "npu",
    "ec_buffer_size": 268435456,
    "ec_connector_extra_config": {"mooncake_protocol": "ascend"}
  }'
```

DP replicas 0 and 1 use control ports `19019` and `19020`. This command exposes
one PD HTTP service on `19537`. List both replicas' EC addresses in the proxy
and set its DP size so each request and its embeddings go to the same replica:

```bash
export VLLM_ROOT=/path/to/vllm
python "$VLLM_ROOT/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
  --host 0.0.0.0 --port 10002 \
  --encode-servers-urls http://127.0.0.1:19534 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19537 \
  --ec-consumer-zmq-addrs tcp://127.0.0.1:19019,tcp://127.0.0.1:19020 \
  --ec-consumer-dp-size 2
```

In general, worker `(DP rank d, TP rank r)` listens on
`ec_port + d * TP_size + r`. For example, TP=2 and DP=2 would use:

| PD replica | TP rank 0 | TP rank 1 | Address supplied to the proxy |
| --- | --- | --- | --- |
| DP rank 0 | `19019` | `19020` | `tcp://127.0.0.1:19019` |
| DP rank 1 | `19021` | `19022` | `tcp://127.0.0.1:19021` |

That combined configuration needs four PD devices plus the Encoder device.
Reserve every worker's port, but pass only one TP-rank-0 address per DP
replica to the proxy, in DP order. These addresses are separate from the
HTTP and Store ports.

Wait until both services are ready:

```bash
curl --fail http://127.0.0.1:19534/health
curl --fail http://127.0.0.1:19537/health
```

## 3. Start the proxy and send a request

Use the proxy:

```bash
export VLLM_ROOT=/path/to/vllm
python "$VLLM_ROOT/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
  --host 0.0.0.0 --port 10002 \
  --encode-servers-urls http://127.0.0.1:19534 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19537 \
  --ec-consumer-zmq-addrs tcp://127.0.0.1:19019
```

`--ec-consumer-zmq-addrs` supplies the PD EC control addresses that the proxy
passes to the Encoder. It is an argument of that script, not a `vllm serve`
option. See the upstream
[ECMooncakeConnector usage guide](https://github.com/vllm-project/vllm/blob/main/docs/features/mooncake_ec_connector_usage.md).

The EC address must match PD's `ec_ip` and `ec_port`. Static
`--ec-consumer-zmq-addrs` supports **E + PD** with
`--prefill-servers-urls disable`; it cannot be combined with independent
prefill URLs.

Send an image chat request to the proxy with any OpenAI-compatible client.
For example, the following optional `curl` request uses an HTTP image URL.
Replace the URL with an image reachable from the Encoder, and keep `MODEL`
consistent with the served model name:

```bash
curl --fail-with-body http://127.0.0.1:10002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    {"type": "text", "text": "Describe this image."}
  ]}],
  "max_tokens": 32,
  "temperature": 0
}
EOF
```

## 4. Enable cross-Encoder caching with Mooncake Store

Store reuse is an optional producer-side feature. Configure it only on the
Encoders; PD keeps its EC consumer configuration. For independently managed
capacity, prefer [standalone Store mode](#standalone-store-mode), following the
[Upstream cross-Encoder cache guide](https://github.com/vllm-project/vllm/blob/main/docs/features/cross_encoder_cache.md).
The embedded example below keeps storage in the Encoder processes.

### Start a RAM Store

Start the master in another terminal and keep it running:

```bash
mooncake_master --port 50051
```

`--port 50051` sets the master RPC listen port. Clients must use this
port in `master_server_address`. The master manages object metadata and
storage segments; it does not provide cache capacity.

Create `mooncake_store.json` on each Encoder host or mount it into each
Encoder container:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "4GB",
  "local_buffer_size": "512MB",
  "protocol": "ascend",
  "device_name": "",
  "enable_offload": false
}
```

In **embedded** mode, each Store client contributes `global_segment_size`
when it connects. The JSON describes that client's contribution; it does not
select which serving roles become Store clients.

In this deployment, `cross_encoder_cache: true` creates an EC Store client on
each Encoder. PD uses `ec_role: "ec_consumer"` without cross-Encoder caching
and does not create an EC Store client. Consequently, E0 and E1 each contribute
4 GiB, for 8 GiB after both caching Encoders have started. Store chooses
the segments holding each object.

| Field | Meaning |
| --- | --- |
| `mode` | `embedded` for the deployment shown here. |
| `metadata_server` | `P2PHANDSHAKE` selects handshake-based TransferEngine metadata exchange. |
| `master_server_address` | Shared master address and port; use the same master on all Encoders. |
| `global_segment_size` | Storage segment contributed per Store client, not a cluster-wide capacity limit. |
| `local_buffer_size` | Local buffer size in shared-engine mode. Independent and FabricMem engines use an effective size of zero. Keep a positive value in the JSON configuration. |
| `protocol` | Must be `ascend`. |
| `device_name` | Empty in this example; resources are selected by the transport configuration. |
| `enable_offload` | Must be `false`; disk offload is not supported for cross-Encoder caching. |

Local file paths may differ, but
every Encoder must connect to the same shared Store deployment.

### Configure and restart the Encoders

`MOONCAKE_CONFIG_PATH` is a **process environment variable** containing the
Store JSON file path. Export it before starting each caching Encoder:

```bash
export MOONCAKE_CONFIG_PATH=/absolute/path/to/mooncake_store.json
```

This variable connects the Encoder to Store; it does not start the master.
Keep the serving environment and the selected A2/RoCE or A3/FabricMem setup
from step 1. Restart E using the command
in step 2, replacing its entire `--ec-transfer-config` value with:

```json
{
  "ec_connector": "ECMooncakeConnector",
  "ec_role": "ec_producer",
  "ec_buffer_device": "npu",
  "ec_buffer_size": 134217728,
  "ec_connector_extra_config": {
    "mooncake_protocol": "ascend",
    "cross_encoder_cache": true,
    "embedding_cache_prefix": "ascend-ec-demo-v1"
  }
}
```

The async GET budget defaults to 2 GiB; `ascend_store_get_async_bytes` is
optional and is omitted here. Set it only to change that budget.

For a 2E + 1PD deployment, keep E0 and PD running. In another terminal,
use the same serving and Store environment, then run the Encoder command from
step 2 with the cache-enabled JSON above, changing only:

- `ASCEND_RT_VISIBLE_DEVICES=2`
- `--port 19535`

The resulting process layout is:

| Process | Visible-device setting | HTTP port | TP / DP |
| --- | --- | --- | --- |
| E0 | `0` | `19534` | 1 / 1 |
| PD in the basic example | `1` | `19537` | 1 / 1 |
| E1 | `2` | `19535` | 1 / 1 |

Restart the proxy with both Encoder URLs:

```bash
python "$VLLM_ROOT/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
  --host 0.0.0.0 --port 10002 \
  --encode-servers-urls http://127.0.0.1:19534,http://127.0.0.1:19535 \
  --prefill-servers-urls disable \
  --decode-servers-urls http://127.0.0.1:19537 \
  --ec-consumer-zmq-addrs tcp://127.0.0.1:19019
```

### Verify a remote cache hit

1. Route a request through E0 while E1 has not processed that image. Initially
   run the proxy with E0 as its only Encoder.
2. Wait for E0's `Stored encoder output in Mooncake Store` log. Publication is
   asynchronous; completion of a serving request does not acknowledge Store
   readiness.
3. Restart the proxy with E1 as its only Encoder, keeping E0 and Store running.
   Submit the identical image with the same model and preprocessing.
4. Check E1 for `encoder output(s) from Mooncake Store`. HTTP success alone
   does not prove remote reuse: a miss also succeeds by encoding.
5. Restore the two-Encoder proxy configuration for normal serving.

Use the same configured model path on both Encoders for automatic multimodal
identifiers. `embedding_model_identity` changes only the model field in Store
keys; it does not by itself make different paths share content identifiers.
Change `embedding_cache_prefix` when replacing weights in place or changing
output-affecting settings. Caller-provided identifiers must consistently refer
to the same input.

### Store ownership and engine selection

Storage ownership and transfer-engine ownership are separate settings:

| Setting | Meaning |
| --- | --- |
| Store `mode: "embedded"` | Each caching Encoder contributes its configured Host storage segment. |
| Store `mode: "standalone-store"` | Encoders contribute zero capacity; an external `mooncake_client` owns the RAM pool. |
| FabricMem enabled | Store creates a FabricMem engine. The EC slab remains registered only with the EC engine. |
| FabricMem unset, resource JSON nonempty | Store creates an independent engine; the same fixed NPU slab is registered with EC and Store. |
| FabricMem unset, resource JSON unset | Store borrows the EC engine and its slab registration. |

### Standalone Store mode

A separate `mooncake_client` provides the RAM pool. Keep the master and storage
process running when restarting Encoders. Data held by a storage process is
lost when that process exits.

To use this mode instead of embedded storage, start the storage process below,
then replace the Encoders' Store JSON. Reuse the communication setup from
step 1. See Mooncake's [standalone deployment guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md#method-c--resource-owning-real-client-mooncake_client)
for the storage process.

#### Start the master and configure the storage process

Run a master on the Store host, or reuse the master already running there:

```bash
mooncake_master --port 50051
```

`--port 50051` sets the master RPC listen port. Clients must use this
port in `master_server_address`. The master manages object metadata and
storage segments; it does not provide cache capacity.

In another terminal, start a storage process providing 16 GiB of RAM.
Replace `192.0.2.10` with the Store host IP:

```bash
mooncake_client \
  --host=192.0.2.10 --port=50052 \
  --master_server_address=192.0.2.10:50051 \
  --metadata_server=P2PHANDSHAKE \
  --protocol=ascend \
  --global_segment_size=17179869184 \
  --enable_offload=false \
  --start_offload_rpc_server=false \
  --tenant_id=default --threads=8
```

| Argument | Meaning |
| --- | --- |
| `--host` | Host IP bound by the storage process; replace it with the Store host address. |
| `--port=50052` | Storage-process RPC listen port, separate from the master's port 50051. |
| `--master_server_address` | Master address and RPC port to connect to. |
| `--metadata_server=P2PHANDSHAKE` | Exchange Transfer Engine metadata through peer handshakes. |
| `--protocol=ascend` | Use the Ascend Transfer Engine. |
| `--global_segment_size=17179869184` | Contribute 16 GiB of Host cache capacity from this process. |
| `--enable_offload=false` | Disable SSD offload. |
| `--start_offload_rpc_server=false` | Do not start the SSD offload RPC service. |
| `--tenant_id=default` | Use the default tenant. |
| `--threads=8` | Use eight client worker threads in the storage process. |

#### Connect the Encoders

Each Encoder keeps its own transport environment from step 1 and uses this
Store JSON, replacing the example master address:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "192.0.2.10:50051",
  "global_segment_size": 0,
  "local_buffer_size": "512MB",
  "protocol": "ascend",
  "device_name": "",
  "enable_offload": false
}
```

Replace the Store JSON configured above and restart the caching Encoders.
The vLLM `standalone-store` configuration requires `global_segment_size: 0`
on Encoders; the separate `mooncake_client` supplies the 16 GiB Host pool. The JSON's `local_buffer_size` must
still be positive. The connector passes zero local-buffer bytes to native setup
for FabricMem or independently configured Store engines; the shared-engine
path uses the configured value. Each Encoder still allocates its own NPU EC
pool, and PD keeps its existing EC consumer setup.

## Configuration and memory sizing

### Shared reuse

These options belong in each producer's `ec_connector_extra_config`:

| Option | Default | Meaning |
| --- | --- | --- |
| `cross_encoder_cache` | `false` | Enable Store reuse and background GET admission. |
| `embedding_cache_prefix` | Empty string | Shared namespace; use the same value on participating Encoders. |
| `embedding_model_identity` | Configured model path | Override the key's model field; matching content identifiers are still required. |
| `store_max_pending_items` | `32` | Maximum pending publications per Encoder; must be positive. |
| `store_max_pending_bytes` | `2147483648` (2 GiB) | Retained tensor-storage budget for pending publications; must be positive. |
| `store_read_buffer_bytes` | `134217728` (128 MiB) | Maximum bytes per read chunk, including object headers; must be positive. Individual objects exceeding it fall back to encoding. |
| `ascend_store_get_async_bytes` | `2147483648` (2 GiB) | Expected payload bytes admitted to background GETs until consumed or discarded. Zero admits no background GETs. |

Background GETs run whenever shared reuse is enabled and budget is available.
The scheduler holds affected requests until results are available; misses and
recoverable read errors follow normal encoding. Objects beyond the remaining
async budget use the synchronous Store path.

The async byte budget is an admission limit, **not an extra allocation**.
Ascend Store PUT and GET lease regions from the producer's registered NPU
staging pool. GET chunks must also fit its aligned available capacity; lack
of a lease can cause a hit candidate to fall back to encoding. Loaded outputs
need local-cache storage, and a GET staging lease remains live until cache
copies and P2P readers finish. Size the EC pool for concurrent P2P, PUT, and
GET work, independently of Store storage segments.

Pending-publication bytes, read-chunk bytes, and async-admission bytes control
different resources. Increasing one does not enlarge the EC pool or Store
capacity. Publication is best-effort and can be skipped when its budget or
staging space is unavailable.

## Separate prefill and decode

For E + P + D, put `ECMooncakeConnector` with `ec_role: "ec_consumer"` on
**P**, where embeddings are needed. Configure P-to-D KV transfer separately.
Use the upstream proxy's dynamic registration API and register `ec_zmq_addrs`
on the prefill record. The static `--ec-consumer-zmq-addrs` flag above
supports E + PD only. See the upstream
[proxy README](https://github.com/vllm-project/vllm/blob/main/examples/disaggregated/disaggregated_encoder/README.md#dynamic-registration)
for registration and authentication.

For Ascend P-to-D deployment, use the
[Mooncake PD disaggregation across nodes](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md)
and [Ascend EPD example](https://github.com/vllm-project/vllm-ascend/tree/main/examples/epd_disaggregated).
Those examples are separate deployment paths; do not assume their proxy flags
implement this guide's Mooncake EC routing contract.

## Ascend reference proxy: health, shutdown, and retries

The walkthrough above uses the upstream proxy. The Ascend repository also ships
a standalone reference proxy, `examples/disaggregated_encoder/disagg_epd_proxy.py`,
which speaks the same OpenAI-compatible front end with its own E/P/D URL groups.
The notes below describe that script; `--ec-consumer-zmq-addrs` is not one of
its options.

Besides `--encode-servers-urls`, `--prefill-servers-urls`, `--decode-servers-urls`
(use `disable` or `none` for the E+PD topology) and `--encoder-dispatch-mode`
(`single`, one encoder request per request, or `fanout`, one encoder request per
multimodal item), these optional parameters are opt-in and keep the previous
behavior by default:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--encode-timeout-seconds` | `0` | Per-request timeout for the encode stage; `0` keeps the shared overall client timeout of 100000 seconds |
| `--prefill-timeout-seconds` | `0` | Same for the prefill stage |
| `--decode-timeout-seconds` | `0` | Same for the decode stage, including streaming decode |
| `--encoder-failure-cooldown-seconds` | `30` | Seconds a failed encoder URL is deprioritized when several encoders are configured; `0` or negative disables health-aware ordering |
| `--probe-encoders-on-startup` | off | Probe every encoder `GET /health` on startup (5 s per probe) and fail fast when an instance is unreachable |
| `--shutdown-drain-seconds` | `0` | On shutdown, wait up to this many seconds for in-flight requests before closing the client sessions; `0` closes immediately |

### Health checks and startup order

Every vLLM instance exposes `GET /health`, which returns HTTP 200 once the
engine is ready. The reference proxy aggregates the three groups:

```bash
curl -s http://localhost:10001/health
```

The response is a JSON object with `proxy`, `encode_cluster`,
`prefill_cluster`, and `decode_cluster` fields. Each cluster field is
`healthy`, `unhealthy`, or `empty` (no instance configured); the endpoint
returns 200 when no cluster is unhealthy and 503 otherwise.

This aggregated `/health` endpoint belongs to the reference proxy
`examples/disaggregated_encoder/disagg_epd_proxy.py`. The layerwise proxy used
by the E+P+D guide
(`examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py`)
only exposes `GET /healthcheck`, which reports its own status and the number of
connected encoder, prefill, decode, and PD instances without aggregating
instance health.

With `--probe-encoders-on-startup`, the reference proxy checks every encoder
`/health` once while starting (5 s timeout) and refuses to start when an
encoder is unreachable. The reference proxy also exposes Prometheus metrics at
`GET /metrics`: `epd_requests_total{stage,status}`,
`epd_stage_seconds_bucket/sum/count{stage}`, `epd_retries_total{stage}`,
`epd_encoder_requests_total`, `epd_mm_items_total`, and
`epd_requests_inflight`. The metrics endpoint is additive and does not change
the `/health` response.

Start the services in this order and wait for each instance's `/health` before
starting the next stage:

1. E instance(s).
2. P and D instance(s) (in E+P+D, P before D).
3. The proxy.
4. Wait for the proxy health endpoint before sending traffic (`/health` for
   the reference proxy, `/healthcheck` for the layerwise proxy).

The E+PD example script implements this wait with `curl -sf <url>/health` and
a timeout. On the NPU side, `npu-smi info -i <rank> -t health` can be used to
check device health while the service is starting.

### Graceful shutdown and cleanup

The reference proxy supports an optional drain window via
`--shutdown-drain-seconds` (default `0`, which closes immediately as before).
When it is greater than zero, shutdown waits until no request is in flight or
the window expires, then closes the client sessions. The in-flight counter is
released at the ASGI response boundary, including requests whose client
disconnects before the first streamed chunk. Stop in this order:

1. Stop the proxy; it drains and closes its client sessions in its shutdown
   hook.
2. Stop the P and D instances with SIGTERM, then the E instance(s). On SIGTERM
   the worker calls `ensure_kv_transfer_shutdown()` and then
   `ensure_ec_transfer_shutdown()`, which calls the connector `shutdown()`
   (see `vllm_ascend/worker/worker.py`).
3. Clean the encoder-cache backend:
   - `ECExampleConnector`: files under `shared_storage_path` are not removed
     automatically; remove them explicitly when the cache should not be reused.
   - `ECCPUConnector`: the mmap under `/dev/shm` is unregistered and unlinked
     during shutdown.
   - `ECMooncakeConnector`: the Mooncake deployment owns the storage
     lifecycle; follow the Mooncake guide for cleanup.

### Failure and retry semantics

- **Stage attempts**: each stage request gets up to three attempts in total
  (the initial attempt plus up to two retries).
- **Retryable failures**: connection errors are mapped to HTTP 502, stage
  timeouts to HTTP 504, and downstream 5xx responses to HTTP 502; all of them
  are retried. Every 2xx response is treated as success.
- **Downstream 4xx**: the 4xx status is preserved and the request is **not**
  retried.
- **Multi-encoder routing**: when several encoder URLs are configured, an
  encoder that failed with a connection error, a stage timeout or a downstream
  5xx is removed from selection for `--encoder-failure-cooldown-seconds`
  (default 30). While at least one healthy encoder remains, `single` dispatch
  draws only from the healthy subset and `fanout` round-robins only the healthy
  subset, so retries avoid recently failed encoders. When every encoder is
  failing the proxy falls back to the full list (best effort). 4xx responses
  and other client-side errors do not mark an encoder unhealthy. With a single
  encoder the selection is unchanged.
- **Exhaustion**: when all attempts fail, the client receives HTTP 500.
- **P stage failure**: a prefill failure returns its mapped status (4xx
  preserved, otherwise 502/504) and D is not started, so no partial output is
  produced.
- **D stage**: non-streaming requests use the same attempt rule. Streaming
  requests are retried only before the first chunk is forwarded; once
  streaming has started, an error is surfaced to the client instead of being
  retried.
- **Encoder-cache miss on the consumer**: behavior depends on the connector.
  `ECMemcacheConnector` falls back to local encoding and logs an
  `EC LOAD miss` warning. The shared-filesystem (`ECExampleConnector`) load
  path requires the EC entry to exist; if it is missing, the request fails with
  an error, so make sure the E instance and the P/D instances use the same
  `shared_storage_path`.
- **Timeouts**: the three `--*-timeout-seconds` options configure per-stage
  timeouts. With the default `0` the proxy omits the per-request timeout, so
  the shared 100000-second overall client timeout applies; a value greater
  than zero bounds the corresponding stage request (including streaming
  decode).

## Limitations

- **VL encoder graph**: `--compilation-config '{"cudagraph_mm_encoder": true}'`
  is supported on MRV2 (encoder ACL graph); there is no V1 fallback, and the
  EPD examples do not enable it. EPD with the encoder graph has not been
  validated.
- **Encoder-only pooling models** (`--runner pooling`) are not EPD; they have
  no PD/EPD split. SigLIP2 does not support PD separation.
- **EC + Pipeline Parallel** is not verified. Treat the combination as unknown
  until coverage exists.
- **ECCPUConnector** works on a single host only (pinned `/dev/shm` mapping).
- **ECMooncakeConnector** requires a Mooncake cluster and shared storage, which
  must be provisioned before starting EPD.
- **ECExampleConnector** requires the same `shared_storage_path` on the
  encoder and consumer instances; a mismatch shows up as missing cache entries.
- **Encoder-cache keys are content hashes**: the key of an image is
  `mm_hash` (the multimodal feature identifier). The same image with different
  preprocessing or resolution can produce a different key, so cross-preprocessing
  reuse is not guaranteed. Duplicate images inside one request are deduplicated.
- **Processor cache**: setting `--mm-processor-cache-gb 0` disables the
  multimodal processor cache, so processed multimodal inputs are not reused.
  This cache is independent of the encoder-cache transfer described here.
- **310P**: multi-process EPD is not committed on 310P.
- **E+P+D** is documented as a supported deployment design. End-to-end test
  cases for E+P+D are included in the test suite but have not been validated
  on hardware yet.

## Related guides

- [Upstream ECMooncakeConnector documentation PR #59196](https://github.com/vllm-project/vllm/pull/59196)
- [Upstream cross-Encoder cache guide](https://github.com/vllm-project/vllm/blob/main/docs/features/cross_encoder_cache.md)
- [Ascend encoder-cache CPU offload](encoder_cache_cpu_offload.md) for shared CPU-memory caching with `ECCPUConnector`
- [Mooncake PD disaggregation on one node](../../tutorials/features/pd_disaggregation_mooncake_single_node.md)
- [Mooncake PD disaggregation across nodes](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md)
