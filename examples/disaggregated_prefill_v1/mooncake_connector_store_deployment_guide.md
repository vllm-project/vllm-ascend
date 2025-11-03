# Mooncacke Store Deployment Guide

## Environmental Dependencies

* Software:
  * Python >= 3.9, < 3.12
  * CANN >= 8.2.rc1
  * PyTorch == 2.7.1, torch-npu == 2.7.1
  * vLLM：main branch
  * vLLM-Ascend：main branch
  * Mooncake：[AscendTransport/Mooncake at pooling-async-memcpy](https://github.com/AscendTransport/Mooncake/tree/pooling-async-memcpy)(Currently available branch code, continuously updated.)
    Installation and Compilation Guide：https://github.com/AscendTransport/Mooncake/tree/pooling-async-memcpy?tab=readme-ov-file#build-and-use-binaries

### KV Pooling Parameter Description
**kv_connector_extra_config**:Additional Configurable Parameters for Pooling
    **mooncake_rpc_port**:Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration.
    **load_async**:Whether to Enable Asynchronous Loading. The default value is false.
    **register_buffer**:Whether to Register Video Memory with the Backend. Registration is Not Required When Used with MooncakeConnectorV1; It is Required in All Other Cases. The Default Value is false.

## run mooncake master

### 1.Configure mooncake.json

The environment variable **MOONCAKE_CONFIG_PATH** is configured to the full path where mooncake.json is located.

```
{
    "local_hostname": "xx.xx.xx.xx",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xx.xx.xx.xx:50088",
    "global_segment_size": 30000000000
}
```

**local_hostname**: Configured as the IP address of the current master node,
**metadata_server**: Configured as **P2PHANDSHAKE**,
**protocol:** Configured for Ascend to use Mooncake's HCCL communication,
**device_name**: ""
**master_server_address**: Configured with the IP and port of the master service
**global_segment_size**: Expands the kvcache size registered by the PD node to the master

### 2. Start mooncake_master

Under the mooncake folder:

```
mooncake_master --port 50088
```

## Pooling and Prefill Decode Disaggregate Scenario

### 1.Run `prefill` Node and `decode` Node

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
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
# The upper boundary environment variable for memory swap logging is set to mooncake, where 1 indicates enabled and 0 indicates disabled.
export ASCEND_AGGREGATE_ENABLE=1
# The upper-level environment variable is the switch for enabling the mooncake aggregation function, where 1 means on and 0 means off.

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
				"kv_connector": "MooncakeConnectorStoreV1",
				"kv_role": "kv_producer",
                "mooncake_rpc_port":"0"
			}  
		]
	}
}' > p.log 2>&1
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
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export ACL_OP_INIT_MODE=1
export ASCEND_TRANSPORT_PRINT=1
# The upper boundary environment variable for memory swap logging is set to mooncake, where 1 indicates enabled and 0 indicates disabled.
export ASCEND_AGGREGATE_ENABLE=1
# The upper-level environment variable is the switch for enabling the mooncake aggregation function, where 1 means on and 0 means off.

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
				"kv_connector": "MooncakeConnectorStoreV1",
				"kv_role": "kv_consumer",
                "mooncake_rpc_port":"1"
			}
		]
	}
    }' > d.log 2>&1
```

### 2、Start proxy_server.

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

### 3. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings.

Short question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```

## Pooling and Mixed Deployment Scenario

### 1、Run Mixed Department Script

The mixed script is essentially a pure pooling scenario for the P node.

```
bash mixed_department.sh
```

Content of mixed_department.sh:

```
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_OP_INIT_MODE=1
export ASCEND_TRANSPORT_PRINT=1
# The upper boundary environment variable for memory swap logging is set to mooncake, where 1 indicates enabled and 0 indicates disabled.
export ASCEND_AGGREGATE_ENABLE=1
# The upper-level environment variable is the switch for enabling the mooncake aggregation function, where 1 means on and 0 means off.

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
	"kv_connector": "MooncakeConnectorStoreV1",
	"kv_role": "kv_both",
	"kv_connector_extra_config": {
		"use_layerwise": false,
        "mooncake_rpc_port":"0"
	}
}' > mix.log 2>&1
```

### 2. Run Inference

Configure the localhost, port, and model weight path in the command to your own settings. The requests sent will only go to the port where the mixed deployment script is located, and there is no need to start a separate proxy.

Short question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

Long question:

```
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```