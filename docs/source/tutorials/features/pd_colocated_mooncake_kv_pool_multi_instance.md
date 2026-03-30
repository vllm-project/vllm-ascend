# PD-Colocated with Mooncake KV Pool Multi-Instance

## Getting Started

vLLM-Ascend now supports PD-colocated deployment with Mooncake features.
This guide provides step-by-step instructions to test these features with
constrained resources.

Using the Qwen2.5-72B-Instruct model as an example, this guide demonstrates
how to use vllm-ascend v0.11.0 (with vLLM v0.11.0) on two Atlas 800T A2
nodes to deploy two vLLM instances. Each instance occupies 4 NPU cards and
uses PD-colocated deployment.

For a more complete KV Pool user guide, see
{doc}`../../user_guide/feature_guide/kv_pool`.

## Verify Multi-Node Communication Environment

### Physical Layer Requirements

- The two Atlas 800T A2 nodes must be physically interconnected via a RoCE
  network. Without RoCE interconnection, cross-node KV Cache access
  performance will be significantly degraded.
- All NPU cards must communicate properly. Intra-node communication uses HCCS,
  while inter-node communication uses the RoCE network.

### Verification Process

The following process serves as a reference example. Please modify parameters
such as IP addresses according to your actual environment.

1. Single Node Verification:

   Execute the following commands sequentially. The results must all be
   `success` and the status must be `UP`:

   ```bash
   # Check the remote switch ports
   for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
   # Get the link status of the Ethernet ports (UP or DOWN)
   for i in {0..7}; do hccn_tool -i $i -link -g ; done
   # Check the network health status
   for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
   # View the network detected IP configuration
   for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
   # View gateway configuration
   for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
   ```

2. Check NPU HCCN Configuration:

   Ensure that the hccn.conf file exists in the environment. If using Docker,
   mount it into the container.

   ```bash
   cat /etc/hccn.conf
   ```

3. Get NPU IP Addresses:

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g; done
   ```

4. Cross-Node PING Test:

   ```bash
   # Execute the following command on each node, replacing x.x.x.x
   # with the target node's NPU card address.
   for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x; done
   ```

5. Check NPU TLS Configuration

   ```bash
   # The tls settings should be consistent across all nodes.
   for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
   ```

## Run with Docker

Start a Docker container on each node.

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:v0.11.0
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## (Optional) Install Mooncake

Mooncake is pre-installed and functional in the v0.11.0 image.
The following installation steps are optional.

Mooncake is the serving platform for Kimi, a leading LLM service provided by
Moonshot AI. Installation and compilation guide:
<https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries>.

First, obtain the Mooncake project using the following command:

```bash
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git submodule update --init --recursive
```

Install MPI:

```bash
apt-get install mpich libmpich-dev -y
```

Install the relevant dependencies (Go installation is not required):

```bash
bash dependencies.sh -y
```

Compile and install:

```bash
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

After installation, verify that Mooncake is installed correctly:

```bash
python -c "import mooncake; print(mooncake.__file__)"
# Expected output path:
# /usr/local/Ascend/ascend-toolkit/latest/python/
# site-packages/mooncake/__init__.py
```

## Start Mooncake Master Service

To start the Mooncake master service in one of the node containers, use the
following command:

```bash
docker exec -it vllm-ascend bash
cd /vllm-workspace/Mooncake
mooncake_master --port 50088 \
  --eviction_high_watermark_ratio 0.95 \
  --eviction_ratio 0.05
```

| Parameter                     | Value | Explanation                           |
| ----------------------------- | ----- | ------------------------------------- |
| port                          | 50088 | Port for the master service           |
| eviction_high_watermark_ratio | 0.95  | High watermark ratio (95% threshold)  |
| eviction_ratio                | 0.05  | Percentage to evict when full (5%)    |

## Create a Mooncake Configuration File Named mooncake.json

The template for the mooncake.json file is as follows:

```json
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "<your_server_ip>:50088",
    "global_segment_size": 107374182400
}
```

| Parameter   | Value                  | Explanation                           |
| --------------| ------------------------| -----------------------------------|
| metadata_server | P2PHANDSHAKE              | Point-to-point handshake mode  |
| protocol              | ascend              | Ascend proprietary protocol    |
| master_server_address | 90.90.100.188:50088(for example) | Master server address|
| global_segment_size   | 107374182400    | Size per segment (100 GB)      |

## vLLM Instance Deployment

Create containers on both Node 1 and Node 2, and launch the
Qwen2.5-72B-Instruct model service in each to test the reusability and
performance of cross-node, cross-instance KV Cache. Instance 1 utilizes NPU
cards [0-3] on the first Atlas 800T A2 server, while Instance 2 utilizes
cards [0-3] on the second server.

### Deploy Instance 1

Replace file paths, host, and port parameters based on your actual environment
configuration.

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/\
latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
# Required on Atlas 800T A2 for direct transmission over RoCE
export HCCL_INTRA_ROCE_ENABLE=1
# Required on Atlas 800 I/T A3 with HDK >= 26.0.0 and CANN >= 9.0.0
export ASCEND_ENABLE_USE_FABRIC_MEM=1
# NPU buffer pool: quantity:size(MB)
# Allocates 4 buffers of 8MB each for KV transfer
export ASCEND_BUFFER_POOL=4:8

vllm serve <path_to_your_model>/Qwen2.5-72B-Instruct/ \
--served-model-name qwen \
--dtype bfloat16 \
--max-model-len 25600 \
--tensor-parallel-size 4 \
--host <your_server_ip> \
--port 8002 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.9 \
--kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "backend": "mooncake",
          "lookup_rpc_port": "0"
      }
  }'
```

### Deploy Instance 2

The deployment method for Instance 2 is identical to Instance 1. Simply
modify the `--host` and `--port` parameters according to your Instance 2
configuration.

### Configuration Parameters

| Parameter | Description |
| :--- | :--- |
| `kv_connector` | Set to `AscendStoreConnector`, which is the recommended connector for Mooncake-backed KV Pool in PD-colocated multi-instance deployment. |
| `kv_role` | Set to `kv_both` so each instance can both produce and consume KV Cache. |
| `lookup_rpc_port` | Port for RPC communication between the pooling scheduler process and worker process. Each instance requires a unique port configuration. |
| `backend` | Set the storage backend for kvpool to `mooncake`. |
| `use_layerwise` | Optional. Controls whether KV Cache is transferred layer by layer. The default behavior is `false`, which transfers the entire KV Cache and is more suitable for cross-node scenarios with sufficient bandwidth. |

**Note on `use_layerwise`:**

- `false`: Transfer the entire KV Cache.
- `true`: Transfer KV Cache layer by layer, which may help in memory-constrained scenarios.

## Benchmark

We recommend using the **AISBench** tool to assess performance. The test uses
**Dataset A**, consisting of fully random data, with the following
configuration:

- Input/output tokens: 1024/10
- Total requests: 100
- Concurrency: 25

The test procedure consists of three steps:

### Step 1: Baseline (No Cache)

Send Dataset A to Instance 1 on Node 1 and record the Time to First Token
(TTFT) as **TTFT1**.

### Preparation for Step 2

Before Step 2, send a fully random Dataset B to Instance 1. Due to the
unified HBM/DRAM KV Cache with LRU (Least Recently Used) eviction policy,
Dataset B's cache evicts Dataset A's cache from HBM, leaving Dataset A's
cache only in Node 1's DRAM.

### Step 2: Local DRAM Hit

Send Dataset A to Instance 1 again to measure the performance when hitting
the KV Cache in local DRAM. Record the TTFT as **TTFT2**.

### Step 3: Cross-Node DRAM Hit

Send Dataset A to Instance 2. With the Mooncake KV Cache pool, this results
in a cross-node KV Cache hit from Node 1's DRAM. Record the TTFT as
**TTFT3**.

**Model Configuration**:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="<path_to_your_model>/Qwen2.5-72B-Instruct",
        model="qwen",
        request_rate = 0,
        retry = 2,
        host_ip = "<your_server_ip>",
        host_port = 8002,
        max_out_len = 10,
        batch_size= 25,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            ignore_eos = True,
        ),
    )
]
```

**Performance Benchmarking Commands**:

```shell
ais_bench --models vllm_api_stream_chat \
  --datasets gsm8k_gen_0_shot_cot_str_perf \
  --debug --summarizer default_perf --mode perf
```

### Test Results

| Requests | Concur | TTFT1 (ms) | TTFT2 (ms) | TTFT3 (ms) |
| -------- | ------ | ---------- | ---------- | ---------- |
| 100      | 25     | 2322       | 739        | 948        |
