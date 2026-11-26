# Step-5-Preview

## 1 Introduction {: #1-introduction }

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM Ascend is recommended.**

**Step-5-Preview** is a next-generation flagship Mixture-of-Experts (MoE) model
developed by StepFun. It has **600B** total parameters with **27B** active per
token, supports a **1M-token** context window and vision input, and is validated
only on **950DT products** (96GB × 8 per node). Other Ascend hardware, such as
the A3 series, is not supported for `Step-5-Preview`.

This document shows the main verification steps of the model, including
supported features, prerequisites, installation, online service deployment,
functional verification, accuracy and performance evaluation, performance
tuning, and FAQs.

This guide covers two deployment scenarios on 950DT products:

- **Multi-node co-located deployment**: two nodes, with Prefill and Decode
  running in one DP group, targeting the full 1M-token context.
- **Prefill-Decode (PD) separation deployment**: six nodes (2 Prefill + 4
  Decode), with an optional KV cache pool for prefix reuse.

Step-5-Preview is supported in **vLLM Ascend v0.34.0rc1 and later**.

## 2 Supported Features {: #2-supported-features }

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

Together, the deployment examples in this document exercise tensor parallelism
(TP), pipeline parallelism (PP), expert parallelism (EP), data parallelism (DP),
native Multi-Token Prediction (MTP) speculative decoding with three tokens, decode graph capture,
shared-expert data parallelism, shared-expert overlap, and FlashComm1. Each
topology in [Section 5](#5-online-service-deployment) enables the subset listed
there: Prefill uses TP with PP, Decode uses TP with DP, and only the Decode role
captures the decode graph. The PD examples additionally enable Prefill-Decode
disaggregation, and the pool example in
[Section 5.3](#53-pd-separation-with-kv-cache-pool) adds the KV cache pool.

!!! note

    Only the features enabled by the commands in this document are covered.
    Other features, such as LoRA, are not validated for `Step-5-Preview`.

## 3 Prerequisites {: #3-prerequisites }

### 3.1 Model Weight {: #31-model-weight }

Download the checkpoint from ModelScope for the hardware listed below. Only the
Ascend-adapted W8A8-MXFP8 checkpoint is used in this guide; the official weights
on Hugging Face use a different quantization format and are not
interchangeable. The deployment commands use the checkpoint directory as a local
mount path, not as a download source.

| Checkpoint | Hardware | Deployment | Download | Notes |
| --- | --- | --- | --- | --- |
| `Step-5-Preview-w8a8-mxfp8` | 950DT products (96GB × 8 per node) | 2 nodes for multi-node co-located deployment; 6 nodes for PD deployment; KV cache pool optional | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Step-5-Preview-w8a8-mxfp8) | Ascend-adapted W8A8-MXFP8 weights; required for all validated configurations. |

It is recommended to download the weights to a shared directory across nodes,
such as `/root/.cache/`, so that every serving node loads the same path.

> **Path description**: Download the model weights to a directory of your choice and record it. Use the same directory on every node and ensure that the model path in the subsequent deployment commands matches this directory.
>
### 3.2 Verify Multi-node Communication (Optional) {: #32-verify-multi-node-communication-optional }

For multi-node deployment, verify the communication environment by following [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect).

## 4 Installation {: #4-installation }

### 4.1 Docker Image Installation {: #41-docker-image-installation }

{% include "getting_started/installation/ascend_docker_runtime.inc.md" %}

#### Download Docker Image {: #download-docker-image }

```bash
docker pull quay.io/ascend/vllm-ascend:nightly-releases-v0.34.0rc1-a5
```

#### Create Docker Container {: #create-docker-container }

Use the following command to create the container.

```bash
# Set the vLLM Ascend image name.
export IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.34.0rc1-a5
export CONTAINER_NAME=step5

# Host directory that holds the KV cache pool configuration files from
# Section 5.3. It must exist on every node and is mounted at the same path.
export MEMCACHE_CONFIG_DIR=<MEMCACHE_CONFIG_DIR>

# 950DT products have 8 NPUs and use Device UB.
docker run --rm \
-e ASCEND_VISIBLE_DEVICES=0-7 \
--name $CONTAINER_NAME \
--net=host \
--shm-size=60g \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /etc/hixlep:/etc/hixlep \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /var/log/npu/:/usr/slog \
-v /lib64:/lib64 \
-v /root/.cache:/root/.cache \
-v "${MEMCACHE_CONFIG_DIR}:${MEMCACHE_CONFIG_DIR}" \
-it $IMAGE bash
```

Inside the container, run `npu-smi info` and confirm that all eight NPUs are visible. Run `vllm --version` to check the installed vLLM version.

The container runs without `--privileged`: Ascend Docker Runtime injects the eight Davinci devices selected by `-e ASCEND_VISIBLE_DEVICES=0-7`, so no `--device` entry is needed for them. Add `--device` only if you start the container without that runtime.

The `-v` option maps directories. **You should map the weight directory into the container** so that the weights can be accessed inside the container. The local weight path is `/mnt/weight`; modify it according to the actual situation.

The `docker run` command above opens an interactive container shell. Use that shell for the following steps. To open an additional shell, run this command from another terminal on the host:

```bash
docker exec -it ${CONTAINER_NAME} bash
```

**Installation Verification:**

In another terminal on the same host, check the container status (replace `step5` if you changed `CONTAINER_NAME`):

```bash
docker ps --filter name=step5
```

Expected result: the container is listed with status `Up`. In the container terminal, verify the installed package versions:

```bash
pip show vllm vllm-ascend
```

Expected result: the version information for both packages is displayed; `vllm-ascend` is the preview build shipped in this image.

Use the same image and absolute checkpoint path on every participating node. Reserve enough host memory and shared memory for Engram CPU offload and local DP sharing.

The KV cache pool in [Section 5.3](#53-pd-separation-with-kv-cache-pool) uses the MemCache backend over the `device_urma` transport on 950DT products. That transport needs the `/dev/uburma` node inside the container; check it with `ls /dev | grep uburma` after the container starts. The pool reads its configuration files from `MEMCACHE_CONFIG_DIR`, which the container mounts at the same path, so create that directory on every node before starting Docker. Remove the mount if you do not use the KV cache pool.

### 4.2 Source Code Installation {: #42-source-code-installation }

To build from source, follow the [software environment installation guide](../../getting_started/installation.md#installation-software-environment) and check out the `main` branch of vLLM Ascend. Use the upstream vLLM commit recorded in `.github/vllm-main-verified.commit` at the selected vLLM Ascend revision.

## 5 Online Service Deployment {: #5-online-service-deployment }

This chapter covers the two supported topologies: multi-node co-located
deployment and multi-node PD separation, the latter with an optional KV cache
pool. Set the model path as described in [Section 3.1](#31-model-weight), and
use the values of the selected role script. Hardware-specific and role-specific
values are not interchangeable between Prefill and Decode.

### 5.1 Multi-Node Co-located Deployment {: #51-multi-node-co-located-deployment }

The validated co-located deployment uses two 950DT product nodes with eight devices
per node. vLLM data parallelism spans the two nodes with two DP ranks per node,
and tensor parallelism uses four devices in each rank, so the resulting topology
is DP4/TP4/EP16. Prefill and Decode share the same engine and the same DP group,
and the combined context limit is 1048576 tokens.

Before starting the service:

- Set `MODEL_PATH` to the container-visible checkpoint directory recorded in
  [Section 3.1](#31-model-weight).
- Set `NODE0_IP`, `NODE1_IP`, and `NIC_NAME` to each node's service IP and the
  local network interface. Set `HCCL_IF_IP` to the communication IP owned by
  that interface; use the same value as the service IP when the two are equal.
  `--host` always binds the service IP.
- Start Node 0 first, then start Node 1 as soon as possible. On Node 1,
  `NODE0_IP` must equal `LOCAL_IP` on Node 0.
- Node 1 joins the DP group as a headless worker with
  `--data-parallel-start-rank 2`. The group runs two local ranks per node, so
  rank `2` is the first global rank owned by the second node.

Run the following script on each node. Select the tab for the node:

=== "Node 0"

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export NODE0_IP=<P0_IP>
    export NODE1_IP=<P1_IP>
    export NIC_NAME=<NIC_NAME>
    LOCAL_IP="${NODE0_IP}"
    export VLLM_USE_FASTOKENS=1

    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<NODE0_NIC_IP>
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 16384 \
        --max-model-len 1048576 \
        --tensor-parallel-size 4 \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-address "${NODE0_IP}" \
        --data-parallel-rpc-port 13389 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --prefix-cache-retention-interval 2048 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --enable-auto-tool-choice \
        --tool-call-parser step5 \
        --reasoning-parser step5
    ```

=== "Node 1"

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export NODE0_IP=<P0_IP>
    export NODE1_IP=<P1_IP>
    export NIC_NAME=<NIC_NAME>
    LOCAL_IP="${NODE1_IP}"
    export VLLM_USE_FASTOKENS=1

    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<NODE1_NIC_IP>
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 16384 \
        --max-model-len 1048576 \
        --tensor-parallel-size 4 \
        --data-parallel-size 4 \
        --data-parallel-size-local 2 \
        --data-parallel-address "${NODE0_IP}" \
        --headless \
        --data-parallel-start-rank 2 \
        --data-parallel-rpc-port 13389 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --prefix-cache-retention-interval 2048 \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --enable-auto-tool-choice \
        --tool-call-parser step5 \
        --reasoning-parser step5
    ```

The following values differ between the master and the worker node:

| Setting | Node 0 | Node 1 | Description |
| --- | --- | --- | --- |
| `LOCAL_IP` | Node 0 IP | Node 1 IP | Each node uses its own service IP. |
| `NODE0_IP` | Not required | Node 0 IP | The worker uses this address to join the DP group. |
| `--headless` | Omitted | Enabled | The worker does not expose an API endpoint. |
| `--data-parallel-address` | `${NODE0_IP}` | `${NODE0_IP}` | Always resolves to Node 0. |
| `--data-parallel-start-rank` | `0` (default) | `2` | First global rank owned by the node; every node needs a unique value. |

Key multi-node co-located deployment parameters:

| Parameter | Description |
| --- | --- |
| `--tensor-parallel-size 4` | Uses four NPUs per DP rank. |
| `--data-parallel-size 4` | Creates four global DP ranks across the two nodes. |
| `--data-parallel-size-local 2` | Runs two DP ranks per node, so all eight devices are in use with TP4. |
| `--enable-expert-parallel` | Shards MoE experts across the DP and TP domains, giving EP16. |
| `--max-model-len 1048576` | Combined input and output limit. |
| `--max-num-seqs 256`, `--max-num-batched-tokens 16384` | Per-engine scheduler limits. |
| `--compilation-config` | `FULL_DECODE_ONLY` graph replay for decode. |
| `--additional-config` | Enables FlashComm1, shared-expert data parallelism, and shared-expert overlap. |
| `--speculative-config` | Native MTP decoding with three speculative tokens. |
| `--enable-auto-tool-choice`, `--tool-call-parser step5` | Enables tool calling and parses Step-5's native tool-call format into standard `tool_calls`; without them `tools` requests are ignored. |
| `--reasoning-parser step5` | Splits the output at Step-5's thinking markers, returning the chain of thought in `reasoning_content` instead of mixing it into `content`. |
| `VLLM_USE_FASTOKENS=1` | Replaces the default Hugging Face tokenizer backend with the [fastokens](https://github.com/crusoecloud/fastokens) Rust implementation, which is faster on long prompts. Install `fastokens` (version 0.2.0 or later) in the same Python environment and check `python -c "import fastokens"`; with the variable set and the package missing, the engine stops at startup. Only the API server process on Node 0 builds a tokenizer, so the variable has no effect on the headless worker. |

Because the weights are large, the load time is related to the disk read speed, and the first load usually takes about 10 minutes. After both nodes have joined and node 0 reports `Application startup complete`, the service is ready; see [Section 6](#6-functional-verification) for verification.

### 5.2 Multi-Node PD Separation Deployment (2P4D) {: #52-multi-node-pd-separation-deployment-2p4d }

The quantized model `Step-5-Preview-w8a8-mxfp8` is deployed on six 950DT
products (96GB × 8 per node). Prefill runs as one DP1/TP4/PP4 engine across P0–P1,
and Decode runs as one DP8/TP4 group across D0–D3 with two DP ranks per node.
Both roles enable expert parallelism, and a Proxy forwards client requests to
both roles.

The parallelism layout, `--max-model-len`, and the scheduler limits in this
section come from the validated Prefill and Decode scripts. Both roles use
`--max-model-len 1048576` and `--max-num-seqs 256`; Prefill uses
`--max-num-batched-tokens 16384` and Decode uses `2048`.

Before starting the service:

- Set `MODEL_PATH` to the container-visible checkpoint directory recorded in
  [Section 3.1](#31-model-weight).
- Set the IP and NIC variables to each node's service IP, its communication IP,
  and the local interface that owns the communication IP.
- Start P0 before P1, and D0 before D1–D3. Every Prefill node uses the same
  master address and port; every Decode node uses the same DP address and DP RPC
  port.
- Only P0 and D0 expose the HTTP API. P1 and D1–D3 are headless workers, and
  every worker needs a unique node rank or DP start rank.

| Role | Nodes | DP | TP | PP | EP | Context limit | HTTP API |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Prefill | P0–P1 (2) | 1 | 4 | 4 | EP16 | 1048576 | P0 only |
| Decode | D0–D3 (4) | 8 | 4 | 1 | EP32 | 1048576 | D0 only |
| Proxy | One node (P0 in the reference run) | — | — | — | — | — | `<PROXY_PORT>` |
| MetaService | One node, co-located with P0 in the reference run ([Section 5.3](#53-pd-separation-with-kv-cache-pool) only) | — | — | — | — | — | `8000` (metrics) |

Use the same checkpoint revision, tokenizer, and quantization metadata on every
node. Two addresses matter on each node: `LOCAL_IP` is the service IP that
`--host` binds and that clients and the Proxy use, and `HCCL_IF_IP` is the
communication IP owned by `NIC_NAME`. Keep them separate when the cluster has a
service network and a compute network. Make sure every node can reach the
Prefill, Decode, and Proxy service IPs.

| Port or address | Purpose |
| --- | --- |
| `<P0_IP>`, `<P1_IP>` | Prefill node service IPs |
| `<D0_IP>`–`<D3_IP>` | Decode node service IPs |
| `<PROXY_IP>` | Proxy node IP |
| `7778` | Prefill and Decode HTTP API port; only P0 and D0 expose it |
| `7060` | Prefill pipeline master port |
| `13389` | Decode DP RPC port |
| `48100` | Prefill KV transfer base port (`kv_producer`) |
| `48200` | Decode KV transfer base port (`kv_consumer`) |
| `<PROXY_PORT>` | Proxy HTTP API port, chosen by the operator |

#### 5.2.1 Create the Engine Templates {: #521-create-the-engine-templates }

Save `run_p.sh` on both Prefill nodes and `run_d.sh` on all four Decode nodes.
Select the tab for the role:

=== "Prefill"

    Prefill is one DP1/TP4/PP4 engine spanning P0–P1 and the KV producer. P0 is the
    pipeline master and exposes the HTTP API; P1 joins as a headless worker.
    `kv_port` is a base address: each engine rank listens on `kv_port + rank`, so
    the 16 Prefill ranks need 16 consecutive ports and the 32 Decode ranks need
    32. Keep both base ports outside the Mooncake-reserved `20000–27999` range;
    the examples use `48100` and `48200`, which keeps the two ranges disjoint.

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export LOCAL_IP=<P0_IP>         # service IP bound by --host
    export NIC_NAME=<P0_NIC>
    export MASTER_ADDR=<P0_IP>
    export HCCL_IF_BASE_PORT=55000
    export VLLM_USE_FASTOKENS=1

    # Each Prefill node exports its own IP and NIC; do not copy one node's values to the other.
    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    # Layer distribution across the four pipeline stages; the first stage also holds the MTP draft.
    export VLLM_PP_LAYER_PARTITION="24,25,25,18"
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<P0_NIC_IP>   # communication IP owned by NIC_NAME
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 16384 \
        --max-model-len 1048576 \
        --nnodes 2 \
        --node-rank 0 \
        --master-addr "${MASTER_ADDR}" \
        --master-port 7060 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 4 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --prefix-cache-retention-interval 2048 \
        --enforce-eager \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_producer",
          "kv_port": "48100",
          "kv_connector_extra_config": {
            "prefill": {
              "dp_size": 1,
              "tp_size": 4,
              "pp_size": 4,
              "pp_layer_partition": "24,25,25,18"
            },
            "decode": {
              "dp_size": 8,
              "tp_size": 4,
              "pp_size": 1
            }
          }
        }'
    ```

    On P1, set `LOCAL_IP`, `HCCL_IF_IP`, and `NIC_NAME` to that node, change
    `--node-rank` to `1`, and add `--headless`. Keep `MASTER_ADDR` pointing at
    P0. Every other value, including `VLLM_PP_LAYER_PARTITION`, must stay
    identical.

=== "Decode"

    Decode is one DP8/TP4 group across D0–D3, with two local DP ranks per node and
    the KV consumer. D0 is the DP coordinator and exposes the HTTP API; D1–D3 join
    as headless workers.

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export LOCAL_IP=<D0_IP>         # service IP bound by --host
    export NIC_NAME=<D0_NIC>
    export DP_ADDRESS=<D0_IP>
    export HCCL_IF_BASE_PORT=56000
    export VLLM_USE_FASTOKENS=1

    # Each Decode node exports its own IP and NIC; only the DP address is shared.
    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<D0_NIC_IP>   # communication IP owned by NIC_NAME
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 2048 \
        --max-model-len 1048576 \
        --tensor-parallel-size 4 \
        --data-parallel-size 8 \
        --data-parallel-size-local 2 \
        --data-parallel-address "${DP_ADDRESS}" \
        --data-parallel-rpc-port 13389 \
        --prefix-cache-retention-interval 2048 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --kv-transfer-config '{
          "kv_connector": "MooncakeConnectorV2",
          "kv_role": "kv_consumer",
          "kv_port": "48200",
          "kv_connector_extra_config": {
            "prefill": {
              "dp_size": 1,
              "tp_size": 4,
              "pp_size": 4,
              "pp_layer_partition": "24,25,25,18"
            },
            "decode": {
              "dp_size": 8,
              "tp_size": 4,
              "pp_size": 1
            }
          }
        }'
    ```

    On D1–D3, set `LOCAL_IP`, `HCCL_IF_IP`, and `NIC_NAME` to that node, add
    `--data-parallel-start-rank` `2`, `4`, or `6`, and add `--headless`. Keep
    `DP_ADDRESS`, the DP RPC port, and `kv_port` identical on every Decode node.

#### 5.2.2 Start the Engines {: #522-start-the-engines }

Start P0, then P1, then D0–D3, and finally the Proxy in
[Section 5.2.3](#523-proxy). Start each coordinator first and bring up the
remaining nodes immediately afterward, otherwise a worker can hit the
engine-ready timeout.

Wait for `Application startup complete` in the P0 and D0 logs, then check both
APIs before starting the Proxy:

```bash
curl --noproxy '*' --fail --silent --show-error --output /dev/null \
    --write-out '%{http_code}' "http://<P0_IP>:7778/health"; echo
curl --noproxy '*' --fail --silent --show-error --output /dev/null \
    --write-out '%{http_code}' "http://<D0_IP>:7778/health"; echo
```

Expected result: `200` from both. The Proxy has no `/health` route, so verify the
full chain with a request through it as described in
[Section 6.1](#61-text-requests); the KV path is working when Prefill logs the
block save for that request and Decode logs the matching load. If either check
fails, read the P0 and D0 logs and follow [Section 10](#10-faq).

#### 5.2.3 Proxy {: #523-proxy }

Start the Proxy after P0 and the Decode API endpoints are ready. The general multi-node PD workflow, including the Mooncake prerequisites and endpoint mapping, is described in [PD Disaggregation with Mooncake](../features/pd_disaggregation_mooncake_multi_node.md#start-the-service).

The Proxy script ships with the vLLM Ascend image; no separate download or checkout is needed. Run it by absolute path from any working directory. If you installed from source ([Section 4.2](#42-source-code-installation)), use the same relative path under your checkout.

```bash
python /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port "${PROXY_PORT}" \
    --workers 1 \
    --max-retries 1 \
    --prefiller-hosts "${P0_IP}" \
    --prefiller-ports 7778 \
    --decoder-hosts "${D0_IP}" \
    --decoder-ports 7778
```

Register only P0 as the Prefill backend and only D0 as the Decode backend; P1 and D1–D3 are headless and expose no HTTP API. Send client requests to the Proxy; it forwards Prefill work to P0 and generation work to the Decode group. The Proxy must forward `kv_transfer_params` unchanged, because the Decode role validates the KV lease duration from that field.

#### 5.2.4 Key PD Settings {: #524-key-pd-settings }

| Setting | Prefill | Decode |
| --- | --- | --- |
| DP / TP / PP | 1 / 4 / 4 | 8 / 4 / 1 |
| EP | Enabled (EP16) | Enabled (EP32) |
| `--max-model-len` | 1048576 | 1048576 |
| `--max-num-seqs` | 256 | 256 |
| `--max-num-batched-tokens` | 16384 | 2048 |
| `--gpu-memory-utilization` | 0.9 | 0.9 |
| Graph mode | Disabled (`--enforce-eager`) | `FULL_DECODE_ONLY` |
| MTP | 3 speculative tokens | 3 speculative tokens |
| `--kv-cache-dtype` | Default (not set) | Default (not set) |
| `VLLM_PP_LAYER_PARTITION` | `24,25,25,18` | Omitted |
| `VLLM_USE_FASTOKENS` | `1` | `1` |
| `enable_flashcomm1` | `true` | `true` |
| `enable_shared_expert_dp` | `true` | `true` |
| `multistream_overlap_shared_expert` | `true` | `true` |
| `HCCL_BUFFSIZE` | `1024` | `1024` |
| `HCCL_OP_EXPANSION_MODE` | `CCU_MS` | `CCU_MS` |
| KV connector | `MooncakeConnectorV2` (`kv_producer`) | `MooncakeConnectorV2` (`kv_consumer`) |
| KV transfer port | `48100` | `48200` |
| `kv_connector_extra_config` | `prefill`: DP1/TP4/PP4 with the layer partition, `decode`: DP8/TP4 | Same block as Prefill |

Both roles declare `kv_connector_extra_config` with the **global** Prefill and Decode topology, not the local node: Prefill is one DP1/TP4/PP4 engine and Decode is one DP8/TP4 group. The `prefill.pp_layer_partition` must match `VLLM_PP_LAYER_PARTITION`, because the consumer uses it to map each layer to the Prefill pipeline stage that holds it. Keep the same block on every node of a group.

Prefill prioritizes TTFT, so it uses a larger batch-token budget and no graph capture; Decode prioritizes TPOT, so it captures the decode graph and uses a smaller batch-token budget. Keep `HCCL_BUFFSIZE` and `HCCL_OP_EXPANSION_MODE` identical on every node of the group; a different expansion mode changes the communication path and fails the process group setup.

`VLLM_USE_FASTOKENS=1` switches the tokenizer to the `fastokens` Rust backend. It requires `pip install fastokens` (version 0.2.0 or later) in the Python environment of every node where it is exported, and `python -c "import fastokens"` must succeed in that environment; when the variable is set without the package, the engine stops at startup. The setting is process-scoped: it applies to the process that builds the tokenizer, which is the API server on P0 and D0, and it has no effect on the headless nodes. Decode also tokenizes the prompt — the Proxy forwards the original chat request, and the KV transfer carries block ids only, not token ids — so keep the setting identical on both roles.

Tool calling and reasoning output are parsed inside the API server that returns the response (D0 here; P0 for symmetry). Add `--enable-auto-tool-choice --tool-call-parser step5 --reasoning-parser step5` to those two commands as in [Section 5.1](#51-multi-node-co-located-deployment) when clients send `tools` or expect a separate `reasoning_content`.

### 5.3 PD Separation with KV Cache Pool {: #53-pd-separation-with-kv-cache-pool }

This section adds a KV cache pool to the 2P4D deployment in [Section 5.2](#52-multi-node-pd-separation-deployment-2p4d). MemCache backs `AscendStoreConnector`, and `MultiConnector` runs two connectors in every engine: `MooncakeConnectorV2` transfers KV cache directly from Prefill to Decode, while `AscendStoreConnector` reads and writes cached prefixes in the pool, so a repeated prefix is computed only once by Prefill.

Every pool-connected engine rank registers its own DRAM slice with the pool, so both roles contribute capacity: with `dram.size = 64GB` on all six nodes, the pool has 48 ranks × 64GB = 3TiB. The data flow is one-directional — Prefill writes new prefixes, Decode reads them — and the pool itself holds DRAM only: each engine keeps its local HBM prefix cache as the first tier and queries the pool only for the blocks that miss locally. For backend installation, sizing, and eviction options, refer to the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md).

Before starting the service:

- Start MetaService first, on the Prefill master node (P0 in the reference deployment). Every pool-connected engine needs it reachable.
- Give Prefill `"lookup_rpc_port": "0"` and Decode `"1"`, and point every node at its own `MMC_LOCAL_CONFIG_PATH`.
- Keep both roles on the parallelism and scheduler limits from [Section 5.2](#52-multi-node-pd-separation-deployment-2p4d); the pool adds the DRAM tier and changes the KV configuration, not the engine shape.

| Element | Configuration in this deployment |
| --- | --- |
| Pool backend | MemCache (`"backend": "memcache"`) |
| Pool transport | `device_urma` |
| MetaService ports | `5000` (metadata), `6000` (config store), `8000` (metrics) |
| Mooncake transfer port | `48100` (Prefill), `48200` (Decode) |
| AscendStore lookup port | `"0"` (Prefill), `"1"` (Decode) |
| AscendStore layerwise mode | Prefill only: `"use_layerwise": true` on Prefill and `false` on Decode.|
| DRAM contributed per rank | `64GB`, on every rank of both roles (48 ranks → `3TiB` total) |

!!! note

    The pool commands in [Section 5.3.4](#534-extend-the-engine-commands) keep `--enable-prefix-caching` on both roles: each engine serves prefix hits from its local HBM cache first and only queries the pool for the blocks that miss in HBM, so HBM and the DRAM pool form a two-tier cache instead of the pool replacing HBM.

#### 5.3.1 Prepare the Pool Components {: #531-prepare-the-pool-components }

The pool packages ship with the image. On every node that runs MetaService or a pool-connected engine, confirm that the packages import, then add the bundled native libraries to the loader path:

```bash
python -c "from memcache_hybrid import MetaService; import memfabric_hybrid"

PYLIB=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH=${PYLIB}/memfabric_hybrid/lib:${PYLIB}/memcache_hybrid/lib:${LD_LIBRARY_PATH}
```

Expected result: the import command exits with status `0`.

#### 5.3.2 Configure the Metadata Service {: #532-configure-the-metadata-service }

The pool uses two kinds of configuration files, read by different processes:

| File | Read by | Copies | Content |
| --- | --- | --- | --- |
| `mmc-meta.conf` | the MetaService process (`MMC_META_CONFIG_PATH`) | one per deployment | where the metadata, config-store, and metrics endpoints are |
| `mmc-local-<node>.conf` | each engine (`MMC_LOCAL_CONFIG_PATH`) | one per engine node: P0, P1, D0–D3 | the same endpoints plus the client-side pool parameters (`world_size`, `protocol`, `dram.size`, `hcom_url`) |

The standalone-provider topology described in the [KV Cache Pool Deployment Guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) adds a third file, `mmc-local-standalone.conf`, for the separate MemCache processes. This deployment starts the pool client inside each engine, so it needs only the two files above.

Create `mmc-meta.conf` on the node that runs MetaService — the Prefill master P0 in the reference deployment — pointing `<POOL_META_HOST>` at an address that every engine can reach over TCP (in the reference deployment, the node's communication IP). Put it in a directory that every engine container can read, that is, the host directory exported as `MEMCACHE_CONFIG_DIR` and mounted in [Section 4.1](#41-docker-image-installation); [Section 5.3.5](#535-start-order-and-verification) points `MMC_META_CONFIG_PATH` at this file.

```bash
ock.mmc.meta_service_url = tcp://<POOL_META_HOST>:5000
ock.mmc.meta_service.config_store_url = tcp://<POOL_META_HOST>:6000
ock.mmc.meta_service.metrics_url = http://<POOL_META_HOST>:8000
ock.mmc.log_level = info
```

#### 5.3.3 Configure the Pool Client on Every Engine Host {: #533-configure-the-pool-client-on-every-engine-host }

Create one `mmc-local-<node>.conf` per engine node — six files for this topology: `mmc-local-p0.conf`, `mmc-local-p1.conf`, `mmc-local-d0.conf`, `mmc-local-d1.conf`, `mmc-local-d2.conf`, `mmc-local-d3.conf`. Keep them next to `mmc-meta.conf` in the same `MEMCACHE_CONFIG_DIR` directory. The meta-service and config-store endpoints are the same as in `mmc-meta.conf` (the client does not set `metrics_url`), and the extra lines make the node a pool client; only `<HCCL_IF_IP>` differs between nodes:

```bash
ock.mmc.meta_service_url = tcp://<POOL_META_HOST>:5000
ock.mmc.local_service.config_store_url = tcp://<POOL_META_HOST>:6000
ock.mmc.log_level = info
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_urma
ock.mmc.local_service.dram.size = 64GB
ock.mmc.local_service.hcom_url = tcp://<HCCL_IF_IP>:27000
```

!!! note

    `hcom_url` must use the real IP of the node. The default `tcp://127.0.0.1:7000` makes cross-host ranks unreachable and fails with `hybm_transport_connect failed: -6`. The port base plus the local rank selects the port, so eight ranks on one host use `27000–27007`. A single generic file without `hcom_url`, or with one shared `127.0.0.1`, only works for a one-node pool — do not reuse it for a multi-node deployment.

Both roles start the pool client inside the engine through `MMC_LOCAL_CONFIG_PATH`; no separate pool worker process is started.

#### 5.3.4 Extend the Engine Commands {: #534-extend-the-engine-commands }

Starting from the engine templates in [Section 5.2.1](#521-create-the-engine-templates), the pool makes four changes:

- `--kv-transfer-config` becomes a `MultiConnector` with two children: `MooncakeConnectorV2` for direct Prefill-to-Decode transfer and `AscendStoreConnector` for the pool.
- `MMC_LOCAL_CONFIG_PATH` points at that node's pool-client file from [Section 5.3.3](#533-configure-the-pool-client-on-every-engine-host).
- `--enable-prefix-caching` stays on in both roles: each engine serves prefix hits from its local HBM cache first and only queries the pool for the blocks that miss in HBM, so HBM and the DRAM pool form a two-tier cache. `enable_lookup: true` on both `AscendStoreConnector` children turns that pool fallback on.
- The `LD_LIBRARY_PATH` from [Section 5.3.1](#531-prepare-the-pool-components) must be exported in the same shell that starts the engine, otherwise `AscendStoreConnector` cannot load the bundled MemCache libraries and pool initialization fails.

The `MooncakeConnectorV2` child keeps the `prefill`/`decode` topology block from [Section 5.2.4](#524-key-pd-settings), because it is a separate connector with its own extra configuration. The commands below are complete; copy one per node and change only the per-node values called out after each tab.

=== "Prefill"

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export LOCAL_IP=<P0_IP>         # service IP bound by --host
    export NIC_NAME=<P0_NIC>
    export MASTER_ADDR=<P0_IP>
    export MEMCACHE_CONFIG_DIR=<MEMCACHE_CONFIG_DIR>
    export HCCL_IF_BASE_PORT=55000
    export VLLM_USE_FASTOKENS=1
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    # Each Prefill node exports its own IP, NIC, and pool-client file; do not copy one node's values to the other.
    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    # Layer distribution across the four pipeline stages; the first stage also holds the MTP draft.
    export VLLM_PP_LAYER_PARTITION="24,25,25,18"
    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<P0_NIC_IP>   # communication IP owned by NIC_NAME
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    # AscendStoreConnector loads the bundled MemCache libraries from here.
    PYLIB=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export LD_LIBRARY_PATH=${PYLIB}/memfabric_hybrid/lib:${PYLIB}/memcache_hybrid/lib:${LD_LIBRARY_PATH}
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local-p0.conf"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 16384 \
        --max-model-len 1048576 \
        --nnodes 2 \
        --node-rank 0 \
        --master-addr "${MASTER_ADDR}" \
        --master-port 7060 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 4 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --enforce-eager \
        --prefix-cache-retention-interval 2048 \
        --enable-prefix-caching \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --kv-transfer-config '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer",
                "kv_port": "48100",
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 1,
                    "tp_size": 4,
                    "pp_size": 4,
                    "pp_layer_partition": "24,25,25,18"
                  },
                  "decode": {
                    "dp_size": 8,
                    "tp_size": 4,
                    "pp_size": 1
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "lookup_rpc_port": "0",
                  "enable_lookup": true,
                  "use_layerwise": true
                }
              }
            ]
          }
        }'
    ```

    On P1, set `LOCAL_IP`, `HCCL_IF_IP`, and `NIC_NAME` to that node, point
    `MMC_LOCAL_CONFIG_PATH` at `${MEMCACHE_CONFIG_DIR}/mmc-local-p1.conf`,
    change `--node-rank` to `1`, and add `--headless`. Keep `MASTER_ADDR`
    pointing at P0. Every other value, including `VLLM_PP_LAYER_PARTITION`, must
    stay identical.

=== "Decode"

    ```bash
    export MODEL_PATH=<MODEL_PATH>
    export LOCAL_IP=<D0_IP>         # service IP bound by --host
    export NIC_NAME=<D0_NIC>
    export DP_ADDRESS=<D0_IP>
    export MEMCACHE_CONFIG_DIR=<MEMCACHE_CONFIG_DIR>
    export HCCL_IF_BASE_PORT=56000
    export VLLM_USE_FASTOKENS=1
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    # Each Decode node exports its own IP, NIC, and pool-client file; only the DP address is shared.
    unset http_proxy https_proxy ftp_proxy all_proxy
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

    export HCCL_BUFFSIZE=1024
    export HCCL_IF_IP=<D0_NIC_IP>   # communication IP owned by NIC_NAME
    export HCCL_OP_EXPANSION_MODE=CCU_MS
    export HCCL_SOCKET_IFNAME="${NIC_NAME}"
    # AscendStoreConnector loads the bundled MemCache libraries from here.
    PYLIB=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    export LD_LIBRARY_PATH=${PYLIB}/memfabric_hybrid/lib:${PYLIB}/memcache_hybrid/lib:${LD_LIBRARY_PATH}
    export GLOO_SOCKET_IFNAME="${NIC_NAME}"
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local-d0.conf"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    vllm serve "${MODEL_PATH}" \
        --host "${LOCAL_IP}" \
        --port 7778 \
        --served-model-name step \
        --trust-remote-code \
        --max-num-seqs 256 \
        --max-num-batched-tokens 2048 \
        --max-model-len 1048576 \
        --tensor-parallel-size 4 \
        --data-parallel-size 8 \
        --data-parallel-size-local 2 \
        --data-parallel-address "${DP_ADDRESS}" \
        --data-parallel-rpc-port 13389 \
        --enable-expert-parallel \
        --gpu-memory-utilization 0.9 \
        --distributed-executor-backend "mp" \
        --allowed-local-media-path / \
        --prefix-cache-retention-interval 2048 \
        --enable-prefix-caching \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}' \
        --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
        --kv-transfer-config '{
          "kv_connector": "MultiConnector",
          "kv_role": "kv_consumer",
          "kv_connector_extra_config": {
            "connectors": [
              {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_consumer",
                "kv_port": "48200",
                "kv_connector_extra_config": {
                  "prefill": {
                    "dp_size": 1,
                    "tp_size": 4,
                    "pp_size": 4,
                    "pp_layer_partition": "24,25,25,18"
                  },
                  "decode": {
                    "dp_size": 8,
                    "tp_size": 4,
                    "pp_size": 1
                  }
                }
              },
              {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                  "backend": "memcache",
                  "lookup_rpc_port": "1",
                  "enable_lookup": true,
                  "use_layerwise": false
                }
              }
            ]
          }
        }'
    ```

    On D1–D3, set `LOCAL_IP`, `HCCL_IF_IP`, and `NIC_NAME` to that node, point
    `MMC_LOCAL_CONFIG_PATH` at the matching
    `${MEMCACHE_CONFIG_DIR}/mmc-local-d1.conf`…`mmc-local-d3.conf`,
    add `--data-parallel-start-rank` `2`, `4`, or `6`, and add `--headless`. Keep
    `DP_ADDRESS`, the DP RPC port, `kv_port`, and `lookup_rpc_port` identical on
    every Decode node. If a Decode node reports
    `No available memory for the cache blocks` with `--data-parallel-size-local 2`,
    keep the four-node DP8 × TP4 group and lower `--max-model-len` first, then
    `--max-num-batched-tokens`, until the KV cache allocation succeeds.

#### 5.3.5 Start Order and Verification {: #535-start-order-and-verification }

Start MetaService on the node that hosts it — P0 in the reference deployment. It has no supervisor, so keep it in the foreground or under your own process manager:

```bash
export MEMCACHE_CONFIG_DIR=<MEMCACHE_CONFIG_DIR>
export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy ftp_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY

python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

The metrics endpoint returns `200` once the metadata service is ready:

```bash
curl --noproxy '*' --fail --silent --show-error --output /dev/null --write-out '%{http_code}' "http://<POOL_META_HOST>:8000/metrics"; echo
```

Start the engines and the Proxy in the order given in [Section 5.2.2](#522-start-the-engines).

Once the engines are ready, verify that the pool registered its capacity. The metrics endpoint reports one `memcache_segment_capacity_bytes` line per registered rank (48 in this topology, since both roles register) and a total of 48 × 64GB = 3TiB:

```bash
curl --noproxy '*' -sS "http://<POOL_META_HOST>:8000/metrics" \
    | grep -E '^memcache_(segment_capacity_bytes|total_capacity_bytes)'
```

Next, send a workload whose requests share a prefix (for example, a benchmark dataset with a 90% prefix repeat rate), and read the per-request hit count from the Prefill log rather than from a cumulative counter:

```bash
grep 'kvpool hit tokens' <prefill-log> | tail -n 5
```

Expected: lines of the form below, where the hit count is close to the total for repeated prefixes:

```output
Total tokens 131087, kvpool hit tokens: 117888
```

The ratio of the two numbers is the pool hit rate for that request (`117888 / 131087` ≈ 90% here). A hit rate of `0.00%` means the requests never reach the pool: check the `lookup_rpc_port` values (`"0"` on Prefill, `"1"` on Decode), the shared MetaService address, and each node's `MMC_LOCAL_CONFIG_PATH`.

!!! note

    A faster second run is not proof of a pool hit — the engine-local prefix cache produces the same effect. `need to load: 0` in the same log line means the local HBM already held the block. To attribute a hit to the pool, look for the Prefill block-save event and the matching Decode load event, and restart the engines between two identical measurements.

**Common Parameter Descriptions**

The deployment commands above use the parameters below. Use the values of the
selected role script; hardware-specific and role-specific values are not
interchangeable between Prefill and Decode.

| Parameter | Meaning and constraints |
| --- | --- |
| `--tensor-parallel-size`, `--pipeline-parallel-size` | TP partitions each model engine across devices; PP splits it into pipeline stages. Prefill uses TP4 × PP4, and Decode uses TP4 with PP1. |
| `--data-parallel-size`, `--data-parallel-size-local` | Number of DP ranks in the group and the number of ranks started on the current node. The multi-node co-located deployment uses DP4 with two local ranks per node; the 2P4D Decode role uses DP8 with two local ranks per node. |
| `--data-parallel-address`, `--data-parallel-rpc-port`, `--data-parallel-start-rank` | DP coordinator address, RPC port, and the first rank of the current node. Every node in a group uses the same address and port; only headless nodes set the start rank. |
| `--enable-expert-parallel` | Shards MoE experts across devices. Combined with DP and TP, EP covers the full group: EP16 for the two-node multi-node co-located deployment and EP32 for the four Decode nodes. |
| `--max-model-len` | Maximum total sequence length in tokens, including input and generated output. Requests must fit this limit and the available KV cache capacity. |
| `--max-num-seqs`, `--max-num-batched-tokens` | Per-engine scheduler limits for concurrent sequences and for tokens processed in one iteration. Prefill uses a larger token budget to reduce time to first token; Decode uses a smaller one to bound per-step latency. |
| `--gpu-memory-utilization` | Fraction of device memory budgeted to the model executor. Despite the flag name, it also applies to the NPU backend. Leave room for communication buffers; the examples use `0.9`. |
| `--speculative-config` | Native MTP decoding. All examples use three speculative tokens with `"method": "mtp"`. |
| `--compilation-config` | Decode graph capture. The examples use `{"cudagraph_mode": "FULL_DECODE_ONLY"}` on Decode and eager execution on Prefill. |
| `--additional-config` | Enables FlashComm1, shared-expert data parallelism, and shared-expert overlap: `{"enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true}`. |
| `--kv-transfer-config` | PD KV transfer. [Section 5.2](#52-multi-node-pd-separation-deployment-2p4d) uses `MooncakeConnectorV2`; [Section 5.3](#53-pd-separation-with-kv-cache-pool) wraps it with `AscendStoreConnector` in `MultiConnector`. |
| `VLLM_PP_LAYER_PARTITION` | Number of model layers per pipeline stage on Prefill (`24,25,25,18`). Set it on every Prefill node and leave it unset on Decode. |
| `HCCL_BUFFSIZE`, `HCCL_OP_EXPANSION_MODE` | CANN/HCCL communication buffer size in MB and the collective expansion mode. Both change the communication path and device-memory footprint; keep the same values on every node of a group. |
| `VLLM_USE_FASTOKENS` | `1` selects the [fastokens](https://github.com/crusoecloud/fastokens) Rust backend for Hugging Face fast tokenizers, which is substantially faster on long prompts. Requires `pip install fastokens` (version 0.2.0 or later) in the same Python environment as vLLM; with the variable set and the package missing, the engine stops at startup. Only the process that builds a tokenizer is affected — the API server on P0 and D0, not the headless workers. Decode tokenizes the prompt as well, because the Proxy forwards the raw chat request and the KV transfer carries block ids only. |

## 6 Functional Verification {: #6-functional-verification }

### 6.1 Text Requests {: #61-text-requests }

Send an inference request to the endpoint of the deployment you started. For the co-located deployment in [Section 5.1](#51-multi-node-co-located-deployment), use `<NODE0_IP>:7778`; for the PD deployment in [Section 5.2](#52-multi-node-pd-separation-deployment-2p4d), use `<PROXY_IP>:<PROXY_PORT>`. The engine binds to the node's service IP (`LOCAL_IP`), so a request to `127.0.0.1` works only if the service also listens on loopback.

```bash
# Replace <ENDPOINT_HOST> and <ENDPOINT_PORT> with the endpoint selected above.
curl --noproxy '*' -sS -X POST "http://<ENDPOINT_HOST>:<ENDPOINT_PORT>/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "step",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "temperature": 0
    }'
```

The server prints the log of the received request, similar to:

```output
INFO:     [IP]:[PORT] - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

Expected result: HTTP 200 and a JSON body containing a non-empty `choices` field with generated text. If the request fails, read the service logs and follow [Section 10](#10-faq). The response below is an example; identifiers, generated text, and token counts vary between requests:

```json
{"id":"chatcmpl-a58ee35377273703","object":"chat.completion","created":1795570544,"model":"step","choices":[{"index":0,"message":{"role":"assistant","content":"Hmm, the user is asking a straightforward question about my identity. This is a common first interaction, so I should provide a clear and concise introduction. \n\nI need to state my name, my developer, and my primary function. It’s also good to mention my capabilities briefly to give the user a sense of what I can do. \n\nI’ll keep it friendly and open-ended by inviting further questions. No need for extra details unless the user asks follow-ups.</think>Hello! I’m Step, a large language model developed by **StepFun** (also known as 阶跃星辰). I’m designed to understand and generate natural language, perform logical and visual reasoning, and assist with a wide range of tasks—from answering questions and writing to analyzing images and solving problems.  \n\nIf you’d like, you can think of me as your helpful, knowledgeable assistant, here to make your tasks easier and more efficient. How can I help you today? 😊","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null,"weight_versions":null}],"service_tier":null,"system_fingerprint":"vllm-0.34.0-tp4-dp4-ep-05b203f1","usage":{"prompt_tokens":15,"total_tokens":215,"completion_tokens":200,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null,"ec_transfer_params":null,"metrics":null}
```

### 6.2 Image Requests {: #62-image-requests }

`Step-5-Preview` accepts image input through the chat completions API. Put the image in an `image_url` content part, add a text part with the question, and send the request to the same endpoint as the text example. The example below embeds the image as a base64 `data:` URL, so the serving nodes do not need outbound network access; this works for both the co-located and the PD deployment. `IMAGE_PATH` must point to a local file readable by the shell that runs `curl`, and the MIME type in the data URL must match the file (`image/jpeg` for JPEG, `image/png` for PNG).

```bash
# Replace <ENDPOINT_HOST> and <ENDPOINT_PORT> as in Section 6.1.
# Set IMAGE_PATH to a local JPEG/PNG file readable by this shell.
IMAGE_PATH=/path/to/tiger.jpeg
IMAGE_BASE64="$(base64 -w 0 "${IMAGE_PATH}")"

curl --noproxy '*' -sS -X POST "http://<ENDPOINT_HOST>:<ENDPOINT_PORT>/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @- <<EOF
{
    "model": "step",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}},
            {"type": "text", "text": "Describe this image."}
        ]
    }],
    "temperature": 0
}
EOF
```

Expected result: HTTP 200 and a non-empty `choices` field describing the image. With the sample tiger image, the response is similar to:

```json
{"id":"chatcmpl-89a5494e809e836d","object":"chat.completion","created":1795581431,"model":"step","choices":[{"index":0,"message":{"role":"assistant","content":"The user wants me to describe the image. Let me look at what I can see in the image.\n\nThe image shows a tiger lying down on green grass. The tiger has orange fur with black stripes, a white/cream colored face with markings around the eyes and cheeks, and appears to be resting with its front paws extended forward. The tiger is looking directly at the camera. The background is green grass, suggesting an outdoor setting like a field or enclosure. The tiger appears relaxed but alert.\n\nI should provide a clear, detailed description of the image.</think>The image shows a **tiger** lying down on a bed of **green grass**. The tiger has the characteristic **orange and black striped** coat, with a lighter, cream-colored underside and facial markings. It is resting in a relaxed pose with its **front paws stretched forward** and its body facing the camera, while its head is turned to look directly at the viewer. The tiger's eyes appear calm and focused. The background consists entirely of lush green grass, suggesting an outdoor setting such as a meadow, field, or wildlife enclosure. The overall mood of the image is serene, capturing the majestic animal in a moment of rest.","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null,"routed_experts":null,"weight_versions":null}],"service_tier":null,"system_fingerprint":"vllm-0.34.0-tp4-dp4-ep-05b203f1","usage":{"prompt_tokens":352,"total_tokens":593,"completion_tokens":241,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null,"ec_transfer_params":null,"metrics":null}
```

## 7 Accuracy Evaluation {: #7-accuracy-evaluation }

### 7.1 Using AISBench {: #71-using-aisbench }

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

### 7.2 Accuracy Results {: #72-accuracy-results }

The following table shows the results of `Step-5-Preview`. Requests were sent to the multi-node co-located deployment described in [Section 5.1](#51-multi-node-co-located-deployment).

| dataset | model | hardware | metric | mode | vllm-api-general-chat |
| ----- | ----- | ----- | ----- | ----- | ----- |
| GPQA Diamond | Step-5-Preview-w8a8-mxfp8 | 950DT products | accuracy | gen | 93.43 |
| MMMU-Pro | Step-5-Preview-w8a8-mxfp8 | 950DT products | accuracy | gen | 76.13 |
| DeepSWE | Step-5-Preview-w8a8-mxfp8 | 950DT products | accuracy | gen | 67.26 |

## 8 Performance Evaluation {: #8-performance-evaluation }

### 8.1 Using AISBench {: #81-using-aisbench }

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

Configure `vllm_api_stream_chat` for the service started in [Section 5.1](#51-multi-node-co-located-deployment) or the Proxy started in [Section 5.2](#52-multi-node-pd-separation-deployment-2p4d). Set `stream=True` and `ignore_eos=True` for streaming performance testing, and update the following fields for your environment:

| Field | Value |
| --- | --- |
| `path` | The checkpoint directory recorded in [Section 3.1](#31-model-weight), accessible to the benchmark client |
| `model` | `step`, matching `--served-model-name` |
| `host_ip` | The service IP of node 0 (`NODE0_IP`) for multi-node co-located deployment, or the Proxy IP for PD deployment |
| `host_port` | `7778` for multi-node co-located deployment, or the Proxy port for PD deployment |

**Run the performance evaluation using the GSM8K dataset as an example:**

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf
```

**Expected result:**

AISBench prints one line per finished request and then a summary table. Check the following fields; the exact values depend on hardware, concurrency, and input/output lengths:

- `Request throughput (req/s)`: completed requests per second.
- `Output token throughput (tok/s)`: generated tokens per second.
- `Mean TTFT (ms)` / `P99 TTFT (ms)`: time to first token, average and 99th percentile.
- `Mean TPOT (ms)` / `P99 TPOT (ms)`: time per output token, average and 99th percentile.

### 8.2 Using vLLM Benchmark {: #82-using-vllm-benchmark }

Refer to [vLLM Benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning {: #9-performance-tuning }

### 9.1 Recommended Configurations {: #91-recommended-configurations }

The recommended configurations are the same as those specified in [Chapter 5, "Online Service Deployment"](#5-online-service-deployment). [Section 5.2.4](#524-key-pd-settings) summarizes the per-role PD settings, and [Section 5.3](#53-pd-separation-with-kv-cache-pool) lists the pool additions.

### 9.2 Tuning Guide {: #92-tuning-guide }

Use the validated deployment values above as a baseline. Adjust `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and `--gpu-memory-utilization` together for the target workload, and keep `HCCL_BUFFSIZE` and `HCCL_OP_EXPANSION_MODE` identical across a group.

Refer to the [performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md) and the [feature matrix](../../user_guide/support_matrix/feature_matrix.md) for additional guidance.

## 10 FAQ {: #10-faq }

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md).

- **Q: The service fails to start, or reports an out-of-memory error soon after accepting requests.**
A: The scheduler limits, context length, or graph capture do not fit device memory. Lower `--max-num-seqs` or `--max-num-batched-tokens` first, then `--max-model-len` or `--gpu-memory-utilization`. On Decode, DP8 × TP4 can report `No available memory for the cache blocks`; keep the four-node group and cut the context or batch-token budget first, or fall back to TP8.
- **Q: The pool hit rate is low or TTFT does not improve.**
A: Shared prefixes must be block-aligned (the reference runs use multiples of 4096 tokens); a prefix that ends mid-block is never stored. Restart the engines between two identical measurements and read per-request metrics, not the cumulative counter. A 16k-token prefix occupies about 3.75 GiB per rank, so size `dram.size` per rank.
- **Q: With the pool enabled, are prefix hits still served from local HBM?**
A: Yes. `--enable-prefix-caching` stays on and `enable_lookup: true` makes each engine query the pool only for the blocks that miss in local HBM, so HBM is the first tier and the DRAM pool the second.
- **Q: The Proxy hangs or stops forwarding.**
A: A single-worker Proxy is blocked by one unfinished streaming handler. Restart the Proxy after any engine restart, and use non-streaming requests for load tests. A `JSONDecodeError` in the Proxy log means a backend returned an empty body or an HTML error page.
