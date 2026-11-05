# EPD disaggregated deployment Guide

## run

The EPD disaggregated technology accelerates model inference by decoupling the visual encoding computation and LLM computation stages. The EPD separation feature transfers encoder cache (EC) entries between E and P/PD nodes through the connector configured in `--ec-transfer-config`; vllm-ascend supports `ECExampleConnector`, `ECMooncakeConnector`, `ECCPUConnector`, and `ECMemcacheConnector`. For the connector matrix and the deployment overview, see the [Disaggregated Encoder user guide](../../docs/source/user_guide/feature_guide/epd_disaggregation.md).

Before running the commands below, export the paths used by every instance and run the commands from this directory:

- `MODEL_PATH`: local Qwen3-VL-8B-Instruct directory, identical on every E/P/D node.
- `EC_SHARED_STORAGE_PATH`: directory that the E node and the P/PD nodes can read and write; with `ECExampleConnector` it must be the same absolute path on all nodes.

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export EC_SHARED_STORAGE_PATH=/path/visible/to/all/nodes/ec_cache
```

## Model Runner support

EPD runs on Model Runner V2 by default; set `VLLM_USE_V2_MODEL_RUNNER=0` to force Model Runner V1. The VL encoder graph (`--compilation-config '{"cudagraph_mm_encoder": true}'`) is not supported on Model Runner V2 and falls back to Model Runner V1 automatically.

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
rm -rf "${EC_SHARED_STORAGE_PATH}"
mkdir -p "${EC_SHARED_STORAGE_PATH}"

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve "${MODEL_PATH}" \
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

vllm serve "${MODEL_PATH}" \
    --gpu-memory-utilization 0.7 \
    --port "33005" \
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

The parameters are explained as follows:<br>
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
        {"type": "text", "text": "What is the text in the illustration?"}
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
rm -rf "${EC_SHARED_STORAGE_PATH}"
mkdir -p "${EC_SHARED_STORAGE_PATH}"

export ASCEND_RT_VISIBLE_DEVICES=0

vllm serve "${MODEL_PATH}" \
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

vllm serve "${MODEL_PATH}" \
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

vllm serve "${MODEL_PATH}" \
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
    --encoder-ports 23001 \
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
        {"type": "text", "text": "What is the text in the illustration?"}
    ]}
    ]
    }'
```

## Health checks and graceful shutdown

Every instance exposes `GET /health`, which returns HTTP 200 once the engine
is ready. Wait for the instance health endpoint before starting the next stage;
the startup order in this guide is E, then P and D, then the proxy.

```bash
curl -s http://127.0.0.1:23001/health   # encoder (E)
curl -s http://127.0.0.1:33003/health   # prefill (P, 1E1P1D)
curl -s http://127.0.0.1:33006/health   # decode (D, 1E1P1D)
curl -s http://127.0.0.1:33005/health   # prefill+decode (PD, 1E1PD)
curl -s http://127.0.0.1:8001/healthcheck   # proxy
```

The layerwise proxy exposes `GET /healthcheck`, which reports the proxy status
and the number of connected encoder, prefill, decode, and PD instances. It
always returns HTTP 200 and does not aggregate instance health; check each
instance's own `/health` endpoint (above) for readiness.

To stop the deployment, stop the proxy first so that no new request is
admitted, then stop the D and P instances, and stop the E instance last. Send
SIGTERM to each process (Ctrl+C in its terminal, or `kill <pid>`) and wait for
it to exit. The `ECExampleConnector` cache is not removed automatically; remove
it when it should not be reused:

```bash
rm -rf "${EC_SHARED_STORAGE_PATH}"
```

## Relationship to the E+PD quick start

This guide is the complete 1E1P1D example: it uses three vLLM instances with
`ECExampleConnector` encoder-cache transfer, `MooncakeLayerwiseConnector` KV
transfer between P and D, and the layerwise proxy. The two-instance E+PD quick
start uses `examples/disaggregated_encoder/disagg_1e1pd_example.sh` with
`examples/disaggregated_encoder/disagg_epd_proxy.py` (E+PD topology,
ExampleConnector only). The two examples are complementary and do not
duplicate each other.
