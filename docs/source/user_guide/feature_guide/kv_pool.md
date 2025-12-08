# Ascend Store Deployment Guide

## Environmental Dependencies

* Software:
  * Python >= 3.9, < 3.12
  * CANN >= 8.3.rc1
  * PyTorch >= 2.7.1, torch-npu >= 2.7.1.dev20250724
  * vLLM：main branch
  * vLLM-Ascend：main branch

### KV Pooling Parameter Description
**kv_connector_extra_config**: Additional Configurable Parameters for Pooling.  
**lookup_rpc_port**: Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration.  
**load_async**: Whether to Enable Asynchronous Loading. The default value is false.  
**backend**: Set the storage backend for kvpool, with the default being mooncake.

## Example of using Mooncake as a KVCache pooling backend
* Software:
    * Mooncake：main branch

        Installation and Compilation Guide：https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries

        Make sure to build with `-DUSE_ASCEND_DIRECT` to enable ADXL engine.

        An example command for compiling ADXL：

        `rm -rf build && mkdir -p build && cd build \ && cmake .. -DCMAKE_INSTALL_PREFIX=/opt/transfer-engine/ -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_ASCEND_DIRECT=ON -DBUILD_SHARED_LIBS=ON -DBUILD_UNIT_TESTS=OFF \ && make -j \ && make install`

        Also, you need to set environment variables to point to them `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64/python3.11/site-packages/mooncake`, or copy the .so files to the `/usr/local/lib64` directory after compilation

### Run Mooncake Master

#### 1.Configure mooncake.json

The environment variable **MOONCAKE_CONFIG_PATH** is configured to the full path where mooncake.json is located.

```
{
    "local_hostname": "xx.xx.xx.xx",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "alloc_in_same_node": true,
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": "1GB" (1024MB/1048576KB/1073741824B/1073741824)
}
```

**local_hostname**: Configured as the IP address of the current master node.  
**metadata_server**: Configured as **P2PHANDSHAKE**.  
**protocol:** Configured for Ascend to use Mooncake's HCCL communication.  
**device_name**: ""  
**alloc_in_same_node**: Indicator for preferring local buffer allocation strategy.  
**master_server_address**: Configured with the IP and port of the master service.  
**global_segment_size**: Expands the kvcache size registered by the PD node to the master.

#### 2. Start mooncake_master

Under the mooncake folder:

```
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.95 --eviction_ratio 0.05
```

`eviction_high_watermark_ratio` determines the watermark where Mooncake Store will perform eviction，and `eviction_ratio` determines the portion of stored objects that would be evicted.

### Pooling and Prefill Decode Disaggregate Scenario

#### 1.Run `prefill` Node and `decode` Node

Using MultiConnector to simultaneously utilize both p2p connectors and pooled connectors. P2P performs kv_transfer, while pooling creates a larger prefix-cache.

`prefill` Node：

```
bash multi_producer.sh
```

The content of the multi_producer.sh script:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
# ASCEND_BUFFER_POOL is the environment variable for configuring the number and size of buffer on NPU Device for aggregation and KV transfer，the value 4:8 means we allocate 4 buffers of size 8MB.
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
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "use_layerwise": false,
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "20001",
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
                "kv_role": "kv_producer",
                "lookup_rpc_port":"0",
                "backend": "mooncake"
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
        "use_layerwise": false,
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
                "lookup_rpc_port":"1",
                "backend": "mooncake"
            }
        ]
    }
    }'
```

#### 2、Start proxy_server.

```
bash proxy.sh
```

proxy.sh content:
Change localhost to your actual IP address.

```
python vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host localhost\
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost\
    --decoder-ports 8200 \
```

#### 3. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings.

Short question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```

### Pooling and Mixed Deployment Scenario

#### 1、Run Mixed Department Script

The mixed script is essentially a pure pooling scenario for the P node.

```
bash mixed_department.sh
```

Content of mixed_department.sh:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
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
        "use_layerwise": false,
        "lookup_rpc_port":"1",
        "backend": "mooncake"
    }
}' > mix.log 2>&1
```

#### 2. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings. The requests sent will only go to the port where the mixed deployment script is located, and there is no need to start a separate proxy.

Short question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```
