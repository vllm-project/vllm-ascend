# Ascend Store Deployment Guide

## Environmental Dependencies

* Software:
  * Python >= 3.10, < 3.12
  * CANN == 8.3.rc2
  * PyTorch == 2.8.0, torch-npu == 2.8.0
  * vLLM：main branch
  * vLLM-Ascend：main branch

### KV Pool Parameter Description
**kv_connector_extra_config**: Additional Configurable Parameters for Pooling.  
**lookup_rpc_port**: Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration.  
**load_async**: Whether to Enable Asynchronous Loading. The default value is false.  
**backend**: Set the storage backend for kvpool, with the default being mooncake.

### Environment Variable Configuration
To guarantee uniform hash generation, it is required to synchronize the PYTHONHASHSEED environment variable across all nodes upon enabling KV Pool.

```bash
export PYTHONHASHSEED=0
```

## Example of using Mooncake as a KV Pool backend
* Software:
    * Check NPU network configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        ```bash
        cat /etc/hccn.conf
        ```

    * Install Mooncake

        Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.
        Installation and Compilation Guide: https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries.
        First, we need to obtain the Mooncake project. Refer to the following command:

        ```shell
        git clone -b v0.3.7.post2 --depth 1 https://github.com/kvcache-ai/Mooncake.git
        ```

        (Optional) Replace go install url if the network is poor

        ```shell
        cd Mooncake
        sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
        ```

        Install mpi

        ```shell
        apt-get install mpich libmpich-dev -y
        ```

        Install the relevant dependencies. The installation of Go is not required.

        ```shell
        bash dependencies.sh -y
        ```

        Compile and install

        ```shell
        mkdir build
        cd build
        cmake .. -DUSE_ASCEND_DIRECT=ON
        make -j
        make install
        ```

        Set environment variables

        **Note:**

        - Adjust the Python path according to your specific Python installation
        - Ensure `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`

        ```shell
        export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
        ```

### Run Mooncake Master

#### 1.Configure mooncake.json

The environment variable **MOONCAKE_CONFIG_PATH** is configured to the full path where mooncake.json is located.

```
{
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": "1GB" (1024MB/1048576KB/1073741824B/1073741824)
}
```

**metadata_server**: Configured as **P2PHANDSHAKE**.  
**protocol:** Must be set to 'Ascend' on the NPU.
**device_name**: ""
**master_server_address**: Configured with the IP and port of the master service.  
**global_segment_size**: Registered memory size per card to the KV Pool.

#### 2.Start mooncake_master

Under the mooncake folder:

```
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1
```

`eviction_high_watermark_ratio` determines the watermark where Mooncake Store will perform eviction，and `eviction_ratio` determines the portion of stored objects that would be evicted.

### PD Disaggregation Scenario

#### 1.Run `prefill` Node and `decode` Node

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`. `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` serves as the prefix-cache node.

`prefill` Node：

```
bash multi_producer.sh
```

The content of the multi_producer.sh script:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0 
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_OP_INIT_MODE=1

# ASCEND_BUFFER_POOL is the environment variable for configuring the number and size of buffer on NPU Device for aggregation and KV transfer，the value 4:8 means we allocate 4 buffers of size 8MB.
export ASCEND_BUFFER_POOL=4:8

# Unit: ms. The timeout for one-sided communication connection establishment is set to 10 seconds by default (see PR: https://github.com/kvcache-ai/Mooncake/pull/1039). Users can adjust this value based on their specific setup.
# The recommended formula is: ASCEND_CONNECT_TIMEOUT = connection_time_per_card (typically within 500ms) × total_number_of_Decode_cards.
# This ensures that even in the worst-case scenario—where all Decode cards simultaneously attempt to connect to the same Prefill card the connection will not time out.
export ASCEND_CONNECT_TIMEOUT=10000

# Unit: ms. The timeout for one-sided communication transfer is set to 10 seconds by default (see PR: https://github.com/kvcache-ai/Mooncake/pull/1039).
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no_enable_prefix_caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 10000 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "20001",
                "kv_connector_extra_config": {
                    "use_ascend_direct": true,
                    "prefill": {
                        "dp_size": 1,
                        "tp_size": 1
                    },
                    "decode": {
                        "dp_size": 1,
                        "tp_size": 1
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake"
                }
            }  
        ]
    }
    }'
```

`decode` Node：

```
bash multi_consumer.sh
```

The content of multi_consumer.sh:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export PYTHONHASHSEED=0 
export MOONCAKE_CONFIG_PATH="/xxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8200 \
    --trust-remote-code \
    --enforce-eager \
    --no_enable_prefix_caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 10000 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "connectors": [
        {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "20002",
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
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake"
                }
            }
        ]
    }
    }'
```

#### 2.Start proxy_server.

```
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host localhost\
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost\
    --decoder-ports 8200 \
```

Change localhost to your actual IP address.

#### 3.Run Inference

Configure the localhost, port, and model weight path in the command to your own settings.

Short question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```

### Colocation Scenario

#### 1.Run Mixed Department Script

```
bash mixed_department.sh
```

Content of mixed_department.sh:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0 
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no_enable_prefix_caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 10000 \
    --block-size 128 \
    --max-num-batched-tokens 4096 \
    --kv-transfer-config \
    '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lookup_rpc_port":"1",
        "backend": "mooncake"
    }
}' > mix.log 2>&1
```

#### 2.Run Inference

Configure the localhost, port, and model weight path in the command to your own settings. The requests sent will only go to the port where the mixed deployment script is located, and there is no need to start a separate proxy.

Short question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```
