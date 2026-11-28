# PD-Colocated with Mooncake Multi-Instance

## Getting Started

vllm-ascend now supports PD-colocated deployment with Mooncake features.
This guide provides step-by-step instructions to test these features with
constrained resources.

Using the Qwen2.5-72B-Instruct model as an example, this guide demonstrates
how to use vllm-ascend {{vllm_ascend_version}} (with vLLM {{vllm_version}}) on two Atlas 800T A2
nodes to deploy two vLLM instances. Each instance occupies 4 NPU cards and
uses PD-colocated deployment.

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

5. Check NPU TLS Configuration:

    ```bash
    # The tls settings should be consistent across all nodes.
    for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
    ```

## Run with Docker

Start a Docker container on each node. The command needs [Ascend Docker Runtime](../../getting_started/installation.md#installation-ascend-docker-runtime) on the host and selects four NPUs with `ASCEND_VISIBLE_DEVICES=0-3`.

```bash
# Update the vllm-ascend image
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
# This test uses four NPU cards to create the container.
# Mount the hccn.conf file from the host node into the container.
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
-e ASCEND_VISIBLE_DEVICES=0-3 \
-v /usr/local/Ascend/driver/tools/hccn_tool:\
/usr/local/Ascend/driver/tools/hccn_tool \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

## (Optional) Install Mooncake

Mooncake is pre-installed and functional in the {{vllm_ascend_version}} image.
The following installation steps are optional.

Mooncake is the serving platform for Kimi, a leading LLM service provided by
Moonshot AI. Installation and compilation guide: [Installation and compilation guide](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries).

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
```

Expected output path:

```text
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/__init__.py
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
| master_server_address | `<your_server_ip>:50088` | Master server address |
| global_segment_size   | 107374182400    | Size per segment (100GB)      |

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
export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
# Enable RoCE path for A2 series
export HCCL_INTRA_ROCE_ENABLE=1

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
      "kv_connector": "MooncakeConnectorStoreV1",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "use_layerwise": false,
          "mooncake_rpc_port": "0",
          "load_async": true,
          "register_buffer": true
      }
  }'
```

### Deploy Instance 2

The deployment method for Instance 2 is identical to Instance 1. Simply
modify the `--host` and `--port` parameters according to your Instance 2
configuration.

### Configuration Parameters

| Parameter         | Value                 | Explanation                      |
| ----------------- | ----------------------| -------------------------------- |
| kv_connector      | MooncakeConnectorStoreV1 | Use StoreV1 version           |
| kv_role         | kv_both                | Enable both produce and consume  |
| use_layerwise     | false                | Transfer entire cache (see note) |
| mooncake_rpc_port | 0                    | Automatic port assignment        |
| load_async        | true                 | Enable asynchronous loading      |
| register_buffer   | true                 | Required for PD-colocated mode   |

**Note on use_layerwise**:

- `false`: Transfer entire KV Cache (suitable for cross-node with sufficient
  bandwidth).
- `true`: Layer-by-layer transfer (suitable for single-node memory
  constraints).

## Benchmark

This example uses AISBench 3.1.0 to measure time to first token (TTFT).
Locate its streaming chat model configuration with
`ais_bench --models vllm_api_stream_chat --search`.
Set `path` to the Qwen2.5-72B-Instruct model directory, `model` to `qwen`,
and `host_ip` and `host_port` to Instance 1. Set `batch_size=25`,
`max_out_len=10`, and `generation_kwargs=dict(temperature=0, ignore_eos=True)`.

On the benchmark client, generate two JSONL files once. With the default
AISBench QA prompt and Qwen2.5-72B-Instruct chat template, 988 random digits
give 1,024 input tokens including formatting. If you change either template,
adjust the digit count using the input token count reported by the service.
Keep `dataset_a.jsonl` for all three measurements so they use identical prompts.
Run the following block and the replay commands in the same Bash session.

```bash
python - <<'PY'
import json
import random

for name, count, seed in [("a", 100, 1), ("b", 200, 2)]:
    rng = random.Random(seed)
    with open(f"dataset_{name}.jsonl", "w", encoding="utf-8") as output:
        for _ in range(count):
            question = "".join(rng.choices("0123456789", k=988))
            print(json.dumps({"question": question, "answer": "", "max_tokens": 10}), file=output)
PY

run_dataset() {
    ais_bench --models vllm_api_stream_chat --custom-dataset-path "$PWD/$1" \
        --num-warmups 0 --summarizer default_perf --mode perf
}
```

### Step 1: Baseline (No Cache)

Start with fresh instances and an empty Mooncake store so that A has no entries
from an earlier run. Keep other traffic off the instances during the benchmark.
Run `run_dataset dataset_a.jsonl` against Instance 1 and record **TTFT1**.
Use the same TTFT statistic from the AISBench summary for each run.

### Preparation for Step 2

Run `run_dataset dataset_b.jsonl` against Instance 1. Size B so that its
distinct KV blocks fill the device KV cache while the Mooncake store still
has room for A and B. The 200 requests above are an example. Adjust the count
using the device KV cache capacity in tokens from the vLLM startup log.

### Step 2: Replay on Instance 1 {id="step-2-local-dram-hit"}

Run `run_dataset dataset_a.jsonl` again against Instance 1 and record **TTFT2**.
Compare the changes in `vllm:prefix_cache_hits_total` (device) and
`vllm:external_prefix_cache_hits_total` (Mooncake) on its `/metrics` endpoint
before and after this replay. Both counters count tokens. External hits should
account for most of A's cacheable input tokens. If most hits still come from
the device, repeat from the cold baseline with a larger B. If both counts are
low, check whether A's blocks were stored and retained in Mooncake.

### Step 3: Replay on Instance 2 {id="step-3-cross-node-dram-hit"}

Set `host_ip` and `host_port` in the AISBench model configuration to Instance 2
and run `run_dataset dataset_a.jsonl`. Record **TTFT3** and check the external
prefix cache hit counter before and after this replay on Instance 2. Compare
the increase with A's cacheable input tokens to see how much was reused
through the connector. To attribute reads to Node 1's DRAM,
verify that the Mooncake transfers read A's blocks from segments on Node 1.

### Test Results

The following TTFT values are from the original benchmark run.

| Requests | Concurrency | TTFT1 (ms) | TTFT2 (ms) | TTFT3 (ms) |
| -------- | ------ | ---------- | ---------- | ---------- |
| 100      | 25     | 2322       | 739        | 948        |
