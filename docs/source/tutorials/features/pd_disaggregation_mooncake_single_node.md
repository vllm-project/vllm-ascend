# Prefill-Decode Disaggregation (Qwen2.5-VL)

## Getting Started

vLLM-Ascend now supports prefill-decode (PD) disaggregation. This guide provides step-by-step instructions to verify this feature in resource-constrained environments.

Using the Qwen2.5-VL-7B-Instruct model as an example, use vLLM-Ascend {{vllm_ascend_version}} (with vLLM {{vllm_version}}) on 1 Atlas 800T A2 server to deploy the "1P1D" architecture (one Prefiller and one Decoder on the same node). Assume the IP address is 192.0.0.1.

For P/D transfer priority settings, see [QoS Configuration](pd_disaggregation_mooncake_multi_node.md#qos-configuration).

For disjoint P/D device assignments and the interaction between
`ASCEND_RT_VISIBLE_DEVICES` and `--device-ids`, see
[NPU Device Selection](../../user_guide/configuration/device_selection.md#prefill-decode-disaggregation).

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

4. (Optional) Cross-Node PING Test (skip for single-node deployment)

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

Start a Docker container. The command needs [Ascend Docker Runtime](../../getting_started/installation.md#installation-ascend-docker-runtime) on the host and selects the NPUs with `ASCEND_VISIBLE_DEVICES`.

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
export NAME=vllm-ascend

# Run the container using the defined variables
docker run --rm \
--name $NAME \
--net=host \
--shm-size=1g \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
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

    The fixed timeout below applies to `MooncakeConnectorV1`. For
    `MooncakeConnectorV2`, configure `kv_lease_duration` instead; see
    [MooncakeConnectorV2 KV Leases](pd_disaggregation_mooncake_multi_node.md#mooncakeconnectorv2-kv-leases).

    For long sequences or high concurrency, the default KV-cache release timeout of 480 seconds may be insufficient. Increase it as needed by setting `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` in the prefiller startup script:

    ```shell
    export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800
    ```

    The value is in seconds: `1800` allows 30 minutes before timeout.

    Increasing this timeout generally does not affect inference accuracy or performance during normal operation. If a decoder (D) node fails while requests still have KV caches on the prefiller (P) node that the decoder has not received, a longer timeout delays releasing those caches. They continue to occupy HBM, reducing the memory available for other requests on the P node until they are released.

We can run the following scripts to launch a server on the prefiller/decoder NPU, respectively.

=== "Prefiller"

    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0
    export HCCL_IF_IP=192.0.0.1  # node ip
    export GLOO_SOCKET_IFNAME="eth0"  # network card name
    export TP_SOCKET_IFNAME="eth0"

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
    --prefiller-ports 13700 \
    --decoder-hosts 192.0.0.1 \
    --decoder-ports 13701
```

|Parameter  | Meaning |
| --- | --- |
| --port | Proxy port |
| --prefiller-ports | All prefiller ports |
| --decoder-ports | All decoder ports |

## Verification

### Decode health probes without a prefill service

With `MooncakeConnectorV2`, send a request directly to D with
`do_virtual: true` and `do_remote_prefill: true` in `kv_transfer_params` to
exercise D-side scheduling and inference without transferring KV cache from P.
This works in both default pull mode and layerwise push mode
(`kv_connector_extra_config.use_layerwise=true`). The flag defaults to false.

For pull mode, retain the required remote metadata fields as placeholders, for example:

```json
{
  "do_remote_prefill": true,
  "do_virtual": true,
  "remote_block_ids": [[0]],
  "remote_host": "127.0.0.1",
  "remote_port": 1,
  "remote_engine_id": "health-probe",
  "remote_request_id": "unique-probe-id",
  "remote_num_prompt_tokens": 32
}
```

Set `remote_num_prompt_tokens` to the actual prompt token count and use the
appropriate number of KV cache groups. For layerwise push probes, only
`do_remote_prefill` and `do_virtual` are needed in `kv_transfer_params`; D does
not dispatch the request to P through the metaserver. Use a short nonempty prompt,
disable prefix caching for this check, and request only a small number of output tokens.

A successful probe confirms request completion on D. Generated text is not an
accuracy result because historical KV cache was not transferred.

### Normal PD inference

Check service health using the proxy server endpoint.

```shell
curl http://192.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25vl",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustration?"}
            ]}
            ],
        "max_completion_tokens": 100,
        "temperature": 0
    }'
```
