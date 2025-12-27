# pd_aggregation_mooncake_single_noide.md（Qwen2.5-72B）



## Getting Start

vLLM-Ascend now supports PD aggregation deployment with Mooncake features.This guide takes one-by-one steps to verify these features with constrained resources.

Using the Qwen2.5-72B-Instruct model as an example, use vllm-ascend v0.11.0rc2 (with vLLM v0.11.0) on 1 Atlas 800T A2 server to deploy two vllm service instances in one docker container.Each service instance occupies 4 NPU cards,both use PD aggregation deployment architecture.



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

2. Check NPU network configuration:

   Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

   ```bash
   cat /etc/hccn.conf
   ```

3. Get NPU IP Addresses

   ```bash
   for i in {0..7}; do hccn_tool -i $i -ip -g;done
   ```



## Run with Docker

Start a Docker container.

```bash
# Update the vllm-ascend image
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:v0.11.0rc2
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

Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.Installation and Compilation Guide: https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries. First, we need to obtain the Mooncake project. Refer to the following command:

```bash
git clone -b v0.3.7.post2 --depth 1 https://github.com/kvcache-ai/Mooncake.git
```

(Optional) Replace go install url if the network is poor

```bash
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
```

Install mpi

```bash
apt-get install mpich libmpich-dev -y
```

Install the relevant dependencies. The installation of Go is not required.

```bash
bash dependencies.sh -y
```

Compile and install

```bash
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
make -j
make install
```

After installation, check if Mooncake is installed using the following command.

```bash
root@800t:/vllm-workspace# python -c "import mooncake; print(mooncake.__file__)"
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake/__init__.py
```

## Start the Mooncake master service process in the container to manage kvpool

```bash
mooncake_master --port 50088
```

## Create a Mooncake pooling configuration file, named mooncake.json

```bash
{
    "local_hostname": "90.90.135.188",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "use_ascend_direct": true,
    "alloc_in_same_node": true,
    "master_server_address": "90.90.135.188:50088",
    "global_segment_size": 107374182400
  }

```

## Mooncake instance deployment

We verified the reusability and performance of cross-instance kvcache by deploying two Qwen2.5-72B-Instruct model service instances within the same container. Instance 1 uses NPU cards [0~3] on this Atlas 800T A2 server, and instance 2 uses cards [4~7]. This allows you to verify cross-instance kvcache cache hit tests using only a single Docker container.

### Deployment instance 1

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3	#Deploy instance 1 using the first 4 NPUs [0~3].
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

vllm serve /model/Qwen2.5-72B-Instruct/  \
    --served-model-name qwen \
    --dtype bfloat16 \
    --max-model-len 25600 \
    --tensor-parallel-size 4 \
    --host 90.90.135.188 \
    --port 8002 \
    --enforce-eager \
    --enable-prefix-caching \
    --block-size 128 \
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

### Deployment instance 2

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7	#Instance 2 is deployed on cards [4~7]
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:/vllm-workspace/vllm
export MOONCAKE_CONFIG_PATH="/vllm-workspace/mooncake.json"
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

vllm serve /model/Qwen2.5-72B-Instruct/  \
    --served-model-name qwen \
    --dtype bfloat16 \
    --max-model-len 25600 \
    --tensor-parallel-size 4 \
    --host 90.90.135.188 \
    --port 8003 \
    --enforce-eager \
    --enable-prefix-caching \
    --block-size 128 \
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

## Benchmark

We recommend use aisbench tool to assess performance。The test dataset A is as follows: input/output 1024/10, a total of 100 data entries, 25 concurrency.The dataset contains completely random characters.The test steps are as follows：

Test step 1: Send dataset A to the service of instance 1, and get TTFT of 2463ms;

Test step 2: Send dataset A to the service of instance 1 again, and the TTFT is 743ms（hit kvcache);

Test step3: Send dataset A to the service of instance 2 , and the TTFT is 867ms（cross-instance hit kvcache);