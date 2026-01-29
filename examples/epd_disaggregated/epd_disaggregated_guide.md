# EPD disaggregated deployment Guide

## Environmental Dependencies
* Software:
    * Python >= 3.10, < 3.12
    * CANN == 8.5.0
    * PyTorch == 2.8.0, torch-npu == 2.8.0
    * vLLM (same version as vllm-ascend and >=0.13.0)
    * mooncake-transfer-engine reference documentation(pd disaggregated needed): <https://github.com/kvcache-ai/Mooncake/blob/main/doc/zh/ascend_transport.md>

## run
The EPD disaggregated technology accelerates model inference by decoupling the visual encoding computation and LLM computation stages. Currently, the EPD separation feature can achieve different data transmissions between E and P/PD nodes by configuring different connector backends. Vllm-ascend currently supports the ECexample-connector backend implemented on vllm, and will support Mooncake as well as shared memory(SHM) backend transmission methods in the future.

### ECexample-connector deployment guide
Using the Qwen3-VL-8B model inference as an example.
#### 1. run 1e1pd case
##### 1.1 run e node
```shell
bash run_e.sh
```

Content of the run_e.sh script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/data/ec_cache}"
rm /data/ec_cache -rf
mkdir -p /data/ec_cache

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve "/your/local/model/path/Qwen3-VL-8B-Instruct" \
    --gpu-memory-utilization 0.01 \
    --port "23001" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name qwenvl \
    --max-model-len 32768  \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }'
```
`--gpu-memory-utilization`:For LLM Model, It is usually used to control the kv cache allocation.For model architectures like vision encoder that do not require KV Cache, it is usually set to 0.01 to minimize HBM usage.<br>
`--ec-transfer-config`:Specify ec-transfer connector settings.For ECExampleConnector, you need to specify the role played by the current node(For e node, set it to 'ec_producer') and the local memory address for data transfer between nodes.<br>
##### 1.2 run pd node
```shell
bash run_pd.sh
```

Content of the run_pd.sh script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/data/ec_cache}"

export ASCEND_RT_VISIBLE_DEVICES=1

vllm serve "/your/local/model/path/Qwen3-VL-8B-Instruct" \
    --gpu-memory-utilization 0.7 \
    --port "33003" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name qwenvl \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }'
```
`--ec-transfer-config`:Same as e node,but ec_role is set to 'ec_consumer'.<br>
##### 1.3 run proxy node
```shell
bash run_proxy.sh
```
Content of the run_proxy.sh script
```bash
python3 epd_load_balance_proxy_layerwise_server_example.py \
    --encoder-hosts 127.0.0.1 \
    --encoder-ports 23001 \
    --pd-hosts 127.0.0.1 \
    --pd-ports 33005 \
    --host 127.0.0.1 \
    --port 8001
```
TODO: explain the param.<br>
`--encoder-hosts`: E node IP address.<br>
`--encoder-ports`: The E node port number. It needs to be consistent with the --port in the E node's startup script.<br>
`--pd-hosts`: PD node IP address.<br>
`--pd-ports`: The PD node port number. It needs to be consistent with the --port in the PD node's startup script.<br>
`--host`: Proxy node IP address.<br>
`--port`: Proxy node port number.<br>
##### 1.4 run inference
```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "qwenvl",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```
#### 2.run 1e1p1d case
##### 2.1 run e node
```shell
bash run_e.sh
```

Content of the run_e.sh script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/data/ec_cache}"
rm /data/ec_cache -rf
mkdir -p /data/ec_cache

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve "/home/p00929506/Qwen3-VL-8B-Instruct" \
    --gpu-memory-utilization 0.01 \
    --port "23001" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name qwenvl \
    --max-model-len 32768  \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }'
```

##### 2.2 run p node
```shell
bash run_p.sh
```

Content of the run_p.sh script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/data/ec_cache}"

export ASCEND_RT_VISIBLE_DEVICES=1

vllm serve "/home/p00929506/Qwen3-VL-8B-Instruct" \
    --gpu-memory-utilization 0.7 \
    --port "33003" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name qwenvl \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config  \
      '{"kv_connector": "MooncakeLayerwiseConnector",
      "kv_role": "kv_producer",
      "kv_port": "50001",
      "engine_id": "0",
      "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
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
      }'
```
##### 2.3 run d node
```shell
bash run_d.sh
```

Content of the run_d.sh script

```shell
unset ftp_proxy
unset https_proxy
unset http_proxy

export ASCEND_RT_VISIBLE_DEVICES=4

vllm serve "/your/local/model/path/Qwen3-VL-8B-Instruct" \
    --gpu-memory-utilization 0.7 \
    --port "33006" \
    --enforce-eager \
    --enable-request-id-headers \
    --served-model-name qwenvl \
    --max-model-len 32768  \
    --max-num-seqs 128 \
    --kv-transfer-config  \
      '{"kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_consumer",
        "kv_port": "50001",
        "engine_id": "1",
        "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
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
        }'
```
##### 2.4 run proxy node
```shell
bash run_proxy.sh
```

Content of the run_proxy.sh script

```shell
python3 epd_load_balance_proxy_layerwise_server_example.py \
    --encoder-hosts 127.0.0.1 \
    --encoder-ports 23001 23002 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 33003 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 33006 \
    --host 127.0.0.1 \
    --port 8001
```
`--prefiller-hosts`: Prefill node IP address.<br>
`--prefiller-ports`: The Prefill node port number. It needs to be consistent with the --port in the Prefill node's startup script.<br>
`--decoder-hosts`: Decode node IP address.<br>
`--decoder-ports`: The Decode node port number. It needs to be consistent with the --port in the Decode node's startup script.<br>
##### 2.5 run inference
```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "qwenvl",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "text", "text": "What is the text in the illustrate?"}
    ]}
    ]
    }'
```


