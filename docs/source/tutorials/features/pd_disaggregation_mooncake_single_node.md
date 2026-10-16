# Prefill-Decode Disaggregation (Qwen2.5-VL)

## Getting Started

vLLM-Ascend now supports prefill-decode (PD) disaggregation. This guide provides step-by-step instructions to verify this feature in resource-constrained environments.

Using the Qwen2.5-VL-7B-Instruct model as an example, use vLLM-Ascend {{vllm_ascend_version}} (with vLLM {{vllm_version}}) on 1 Atlas 800T A2 server to deploy the "1P1D" architecture (one Prefiller and one Decoder on the same node). Assume the IP address is 192.0.0.1.

For P/D transfer priority settings, see [QoS Configuration](pd_disaggregation_mooncake_multi_node.md#qos-configuration).

## Verify Communication Environment

### Verification Process

1. Single Node Verification:

    Execute the following commands in sequence. The results must all be `success` and the status must be `UP`:

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

    Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

    ```bash
    cat /etc/hccn.conf
    ```

3. Get NPU IP Addresses

    ```bash
    for i in {0..7}; do hccn_tool -i $i -ip -g;done
    ```

4. Cross-Node PING Test

    ```bash
    # Execute on the target node (replace 'x.x.x.x' with actual npu ip address).
    for i in {0..7}; do hccn_tool -i $i -ping -g address x.x.x.x;done
    ```

5. Check NPU TLS Configuration

    ```bash
    # The tls settings should be consistent across all nodes
    for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
    ```

## Run with Docker

Start a Docker container.

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /mnt/sfs_turbo/.cache:/root/.cache \
-it $IMAGE bash
```

## Install Mooncake

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI. Installation and Compilation Guide: <https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries>.
First, we need to obtain the Mooncake project. Refer to the following command:

```shell
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

(Optional) Replace go install URL if the network is poor.

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

Install MPI.

```shell
apt-get install mpich libmpich-dev -y
```

Install the relevant dependencies. The installation of Go is not required.

```shell
bash dependencies.sh -y
```

Compile and install.

```shell
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

Set environment variables.

**Note:**

- Adjust the Python path according to your specific Python installation
- Ensure `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`

```shell
export LD_LIBRARY_PATH=/usr/local/lib64/python3.12/site-packages/mooncake:$LD_LIBRARY_PATH
```

## Prefiller/Decoder Deployment

!!! note "Timeout for long sequences or high concurrency"

    For long sequences or high concurrency, the default KV-cache release timeout of 480 seconds may be insufficient. Increase it as needed in the prefiller startup script. For versions that use `VLLM_MOONCAKE_CONNECTOR_TIMEOUT` for this timeout, add:

    ```shell
    export VLLM_MOONCAKE_CONNECTOR_TIMEOUT=1800
    ```

    The value is in seconds: `1800` allows 30 minutes before timeout. In current versions, this setting is named `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`; use `export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800` instead. Check the variable name supported by your installed version.

    Increasing this timeout generally does not affect inference accuracy or performance during normal operation. If a decoder (D) node fails while requests still have KV caches on the prefiller (P) node that the decoder has not received, a longer timeout delays releasing those caches. They continue to occupy HBM, reducing the memory available for other requests on the P node until they are released.

We can run the following scripts to launch a server on the prefiller/decoder NPU, respectively.

=== "Prefiller"

    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0
    export HCCL_IF_IP=192.0.0.1  # node ip
    export GLOO_SOCKET_IFNAME="eth0"  # network card name
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10

    vllm serve /model/Qwen2.5-VL-7B-Instruct  \
      --host 0.0.0.0 \
      --port 13700 \
      --no-enable-prefix-caching \
      --tensor-parallel-size 1 \
      --seed 1024 \
      --served-model-name qwen25vl \
      --max-model-len 40000  \
      --max-num-batched-tokens 40000  \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 1,
                        "tp_size": 1
                 },
                 "decode": {
                        "dp_size": 1,
                        "tp_size": 1
                 }
          }
      }'
    ```

=== "Decoder"

    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=1
    export HCCL_IF_IP=192.0.0.1  # node ip
    export GLOO_SOCKET_IFNAME="eth0"  # network card name
    export TP_SOCKET_IFNAME="eth0"
    export HCCL_SOCKET_IFNAME="eth0"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10

    vllm serve /model/Qwen2.5-VL-7B-Instruct  \
      --host 0.0.0.0 \
      --port 13701 \
      --no-enable-prefix-caching \
      --tensor-parallel-size 1 \
      --seed 1024 \
      --served-model-name qwen25vl \
      --max-model-len 40000  \
      --max-num-batched-tokens 40000  \
      --trust-remote-code \
      --gpu-memory-utilization 0.9  \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30100",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 1,
                        "tp_size": 1
                 },
                 "decode": {
                        "dp_size": 1,
                        "tp_size": 1
                 }
          }
      }'
    ```

If you want to run "2P1D" (two Prefiller and one Decoder), please set ASCEND_RT_VISIBLE_DEVICES and port to different values for each P process.

## Example Proxy for Deployment

Run a proxy server on the same node with the prefiller service instance. You can get the proxy program in the repository's examples: [load\_balance\_proxy\_server\_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

```shell
python load_balance_proxy_server_example.py \
    --host 192.0.0.1 \
    --port 8080 \
    --prefiller-hosts 192.0.0.1 \
    --prefiller-port 13700 \
    --decoder-hosts 192.0.0.1 \
    --decoder-ports 13701
```

|Parameter  | Meaning |
| --- | --- |
| --port | Proxy port |
| --prefiller-port | All prefiller ports |
| --decoder-ports | All decoder ports |

## Verification

Check service health using the proxy server endpoint.

```shell
curl http://192.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustration?"}
            ]}
            ],
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```

## MooncakeConnectorV2 (experimental)

!!! note "Experimental feature"

    MooncakeConnectorV2 is currently experimental. Its configuration and supported feature combinations may change as development continues.

MooncakeConnectorV2 is a refactored version of MooncakeConnectorV1, with design improvements and support for more features. It uses asynchronous, decoder-initiated KV-cache transfers. Its capabilities and restrictions are summarized below; support for an individual feature does not imply that every model and feature combination has been validated.

| Feature | Support and restrictions |
| --- | --- |
| Parallel deployment | TP, DP, PP, PCP, and DCP, including per-rank endpoint discovery and PP layer mapping. GQA CP is outside the validated scope below. |
| Different P/D TP sizes | Cache-layout-aware transfer for MLA, GQA, and Mamba state. Supported TP ratios and head/state shapes depend on the cache type; arbitrary P/D TP combinations are not supported. |
| KVPP | Supported for **pure MLA models only**. Do not enable KVPP and DCP together on the producer. A KVPP producer can transfer to a DCP decoder. |
| Hybrid cache groups | Attention KV caches and Mamba convolution/recurrent states, including hybrid model cache-group mapping. |
| GQA | **Experimental; only the `LBHNC` and `HND` KV-cache layouts are supported.** Validated with Qwen3 pure GQA models and Qwen3.5 GQA + Mamba hybrid models. **GQA CP (PCP/DCP) and GQA C8 (INT8 KV cache) are not included in this validation.** |
| P/D block sizes and scales | Full-attention/MLA paths can map different logical block sizes/scales when the underlying kernel block size and physical transfer layout are compatible. SWA requires equal P/D logical block sizes. |
| Prefix-cache reuse | Accounts for local prefix-cache hits when determining the KV blocks to receive, including hybrid cache groups. |
| Model Runner V2 cache layouts | Includes compressed attention/indexer and hybrid state handling used by GLM-5.3-Flash. This does not imply support for every model-specific cache specification. |
| Receive failure handling | Reports failed KV loads to the scheduler for the configured `kv_load_failure_policy`, and notifies the producer to release retained blocks. |
| Completion and cleanup | Decoder completion notifications release producer blocks; timeout cleanup handles requests whose completion notification is missing. |
| Transfer metrics | Transfer duration, transferred bytes, and failed-transfer metrics. |

### Layout and port configuration

GQA is supported only with the `LBHNC` or `HND` KV-cache layout. Set `VLLM_KV_CACHE_LAYOUT=LBHNC` or `VLLM_KV_CACHE_LAYOUT=HND` in **both** P and D startup scripts, using the same value on both sides. Select `MooncakeConnectorV2`, with `kv_producer` on P and `kv_consumer` on D. Use compatible vLLM/vLLM-Ascend versions and the same model weights on both sides.

If `kv_port` is omitted from `--kv-transfer-config`, MooncakeConnectorV2 derives its endpoints from vLLM's default `kv_port` (currently `14579`). It does not derive the KV endpoints from the HTTP `--port`.

- **Single-host PD or multiple instances on one host:** explicitly configure a different `kv_port` for each instance, and reserve non-overlapping port ranges. Different HTTP ports alone are insufficient.
- **Multiple hosts with one instance per host:** `kv_port` can be omitted. Each host can use the same default port, provided the required ports are free and reachable.
- Each instance reserves worker handshake ports starting at `kv_port`, followed by scheduler control ports. Reserve `DP * PP * PCP * TP + DP` consecutive ports per instance. For the TP=DP=PP=PCP=1 examples below, this is two ports per instance. Increase the spacing when scaling parallelism, and keep these ranges separate from API, proxy, and distributed communication ports.

### Complete GQA startup example

Complete the environment and Mooncake installation steps earlier in this guide first. Run the scripts from the vllm-ascend repository root in separate terminals. Replace the model path, host IPs, network interface, and NPU IDs with your deployment values. These examples use Qwen3-8B with one NPU per role; for Qwen3.5, replace the model path on both sides with the same GQA + Mamba model and choose enough NPUs and matching TP settings for that model.

The examples keep CP, KVPP, and KV-cache quantization disabled. To try `HND`, replace `LBHNC` with `HND` in both startup scripts.

This example deploys P and D on `192.168.0.1`, using NPU 0 and NPU 1. P reserves KV ports `30000`–`30001`; D reserves `30100`–`30101`. Their API ports are `8100` and `8200`.

#### Prefiller

```bash
#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_HOST_IP=192.168.0.1
export HCCL_IF_IP=192.168.0.1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800

vllm serve /models/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8100 \
    --served-model-name qwen3 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000}'
```

#### Decoder

```bash
#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_HOST_IP=192.168.0.1
export HCCL_IF_IP=192.168.0.1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800

vllm serve /models/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8200 \
    --served-model-name qwen3 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100}'
```

#### Proxy and verification

After both P and D are ready, start the proxy on `192.168.0.1`:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 0.0.0.0 \
    --port 8000 \
    --prefiller-hosts 192.168.0.1 \
    --prefiller-ports 8100 \
    --decoder-hosts 192.168.0.1 \
    --decoder-ports 8200
```

Send a request from another terminal:

```bash
curl http://192.168.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3","messages":[{"role":"user","content":"Explain prefill and decode in one paragraph."}],"max_tokens":128,"temperature":0}'
```
