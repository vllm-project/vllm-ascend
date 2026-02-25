# Prefill-Decode Disaggregation (Qwen2.5-VL)

## Getting Start

vLLM-Ascend now supports prefill-decode (PD) disaggregation. This guide takes one-by-one steps to verify these features with constrained resources.

Using the Qwen2.5-VL-7B-Instruct model as an example, use vllm-ascend v0.11.0rc1 (with vLLM v0.11.0) on 1 Atlas 800T A2 server to deploy the "1P1D" architecture. Assume the IP address is 192.0.0.1.

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

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:|vllm_ascend_version|
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
git clone -b v0.3.8.post1 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

(Optional) Replace go install url if the network is poor.

```shell
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

Install mpi.

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
export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
```

## Prefiller/Decoder Deployment

We can run the following scripts to launch a server on the prefiller/decoder NPU, respectively.

:::::{tab-set}

::::{tab-item} Prefiller

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
  "engine_id": "0",
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

::::

::::{tab-item} Decoder

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
  "engine_id": "1",
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

::::

:::::

If you want to run "2P1D", please set ASCEND_RT_VISIBLE_DEVICES and port to different values for each P process.

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
| --port | Port of proxy |
| --prefiller-port | All ports of prefill |
| --decoder-ports | All ports of decoder |

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
