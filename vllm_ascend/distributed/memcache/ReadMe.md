## multiConnector+memcache基础场景验证指导
### 一、基础环境

vllm 主线main分支代码 代码仓地址：[vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)

vllm-ascend 主线main分支代码

python 3.11

torch 2.7.1

torch-npu 2.7.1.dev20250724

环境上需要安装好mooncake，mooncake代码路径：[AscendTransport/Mooncake at pooling-async-memcpy](https://github.com/AscendTransport/Mooncake/tree/pooling-async-memcpy)

mooncake安装完后需要cp编译产物到特定文件夹目录下：

```
cp mooncake-transfer-engine/src/transport/ascend_transport/hccl_transport/ascend_transport_c/libascend_transport_mem.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
cp mooncake-transfer-engine/src/libtransfer_engine.so /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
```

### 二、Memcache master拉起

##### 1. 启动metaservice独立进程

启动master前先：

```
source /usr/local/mxc/memfabric_hybrid/set_env.sh
```

启动命令：

```
export MMC_META_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-meta.conf
cd /usr/local/mxc/memfabric_hybrid/1.0.0/aarch64-linux/bin/
./mmc_meta_service
```

该/usr/local/mxc/memfabric_hybrid/latest/config/mmc-meta.conf文件为安装完memcache后配置，可以自行修改：

```
# meta service启动url，在K8S集群meta service主备高可用场景，在Pod启动时自动修改为PodIP
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
# 是否使能主备高可用部署
ock.mmc.meta.ha.enable = false
# 日志级别
ock.mmc.log_level = debug
# 日志目录，可配置为绝对路径或相对路径，系统会自动追加logs目录
# 默认配置下日志绝对路径为：/path/to/mmc_meta_service/../logs
# 假设mmc_meta_service所在路径为/usr/local/mxc/memfabric_hybrid/latest/aarch64-linux/bin
# 则日志路径为/usr/local/mxc/memfabric_hybrid/latest/aarch64-linux/logs
ock.mmc.log_path = .
# 日志文件大小单位MB[1,500]
ock.mmc.log_rotation_file_size = 20
# 日志文件个数[1,50]
ock.mmc.log_rotation_file_count = 50

# 触发淘汰的水位，单位为空间使用百分比，超过此水位后put操作触发淘汰
ock.mmc.evict_threshold_high = 70
# 淘汰的目标水位，单位为空间使用百分比
ock.mmc.evict_threshold_low = 60

# TLS安全通信证书相关配置
ock.mmc.tls.enable = false
ock.mmc.tls.top.path = /opt/ock/security/
ock.mmc.tls.ca.path = certs/ca.cert.pem
ock.mmc.tls.ca.crl.path = certs/ca.crl.pem
ock.mmc.tls.cert.path = certs/server.cert.pem
ock.mmc.tls.key.path = certs/server.private.key.pem
ock.mmc.tls.key.pass.path = certs/server.passphrase
ock.mmc.tls.package.path = /opt/ock/security/libs/

```

##### 2. 通过pymmc提供的接口初始化客户端并拉起localservice，执行数据写入、查询、获取、删除等

```
export MMC_LOCAL_CONFIG_PATH=/usr/local/mxc/memfabric_hybrid/latest/config/mmc-local.conf
python3 -m unittest test_mmc_demo.py
python3 -m unittest test_mmc_layer.py
```

### 三、拉起p节点与d节点

启动脚本前：

```
source /usr/local/mxc/memfabric_hybrid/set_env.sh
```

p节点：

```
bash multi_producer.sh
```

multi_producer.sh脚本内容：

```
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export PYTHONPATH=$PYTHONPATH://vllm-ascend
export MMC_LOCAL_CONFIG_PATH=/xxxxx/config_npu0.conf
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/DeepSeek-V2-Lite-Chat \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --enable-expert-parallel \
    --tensor-parallel-size 2\
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
				"kv_buffer_device": "npu",
				"kv_rank": 0,
				"kv_port": "20001",
				"kv_connector_extra_config": {
					"prefill": {
						"dp_size": 1,
						"tp_size": 2
					},
					"decode": {
						"dp_size": 1,
						"tp_size": 2
					}
				}
			},
			{
				"kv_connector": "MemcacheConnectorStoreV1",
				"kv_role": "kv_producer",
				"kv_connector_extra_config":{"use_layerwise":false,"memcache_rpc_port":"0"}
			}
		]
	}
}'    > p.log 2>&1
```

d节点：

```
bash multi_consumer.sh
```

multi_consumer.sh内容：

```
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm-ascend
export MMC_LOCAL_CONFIG_PATH=/xxxxxxx/config_npu0.conf
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/DeepSeek-V2-Lite-Chat \
    --port 8200 \
    --trust-remote-code \
    --enforce-eager \
    --enable-expert-parallel \
    --no_enable_prefix_caching \
    --tensor-parallel-size 2 \
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
				"kv_buffer_device": "npu",
				"kv_rank": 1,
				"kv_port": "20002",
				"kv_connector_extra_config": {
					"prefill": {
						"dp_size": 1,
						"tp_size": 2
					},
					"decode": {
						"dp_size": 1,
						"tp_size": 2
					}
				}
			},
			{
				"kv_connector": "MemcacheConnectorStoreV1",
				"kv_role": "kv_consumer",
				"kv_connector_extra_config":{"use_layerwise":false,"memcache_rpc_port":"1"}
			}
		]
	}
    }'   > d.log 2>&1
```

config_npu0.conf的配置：

```
# meta service启动url
# 在meta service非HA场景，请与mmc-meta.conf中的配置项保持一致
# 在K8S集群meta service主备高可用场景，请配置为ClusterIP地址
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
# 日志级别
ock.mmc.log_level = info

# TLS安全通信证书相关配置
ock.mmc.tls.enable = false
ock.mmc.tls.top.path = /opt/ock/security/
ock.mmc.tls.ca.path = certs/ca.cert.pem
ock.mmc.tls.ca.crl.path = certs/ca.crl.pem
ock.mmc.tls.cert.path = certs/client.cert.pem
ock.mmc.tls.key.path = certs/client.private.key.pem
ock.mmc.tls.key.pass.path = certs/client.passphrase
ock.mmc.tls.package.path = /opt/ock/security/libs/

# client的总数
ock.mmc.local_service.world_size = 16
# BM服务启动url，在K8S集群meta service主备高可用场景，在Pod启动时自动修改为PodIP
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
# ip需要设为RDAM网卡ip，可以使用show_gids命令查询
ock.mmc.local_service.hcom_url = tcp://127.0.0.1:7000
# 数据传输协议，DRAM池使用roce，HBM池使用sdma
ock.mmc.local_service.protocol = sdma
# DRAM空间使用量，单位字节，默认128MB，和HBM二选一，需要2M对齐
ock.mmc.local_service.dram.size = 1024MB
# HBM空间使用量，单位字节，和DRAM二选一
ock.mmc.local_service.hbm.size = 0

ock.mmc.local_service.dram_by_sdma = true
# client请求meta service连接不存在时，重试总时长（重试间隔为200ms）
# 默认值为0，表示不重式失败时直接返回，取值范围[0, 600000]
ock.mmc.client.retry_milliseconds = 0

ock.mmc.client.timeout.seconds = 60

```

mooncake.json配置：

```
{
    "local_hostname": "xxxxx",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "xxxxxx:50088"
}
```

### 四、拉起proxy

proxy启动：

```
bash proxy.sh
```

proxy.sh内容：
localhost改成自己的实际ip

```
python toy_proxy_server.py \
    --host localhost\
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost\
    --decoder-ports 8200 \
```

toy_proxy_server.py代码内容：

```
# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0

import argparse
import itertools
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    limit = httpx.Limits(max_connections=100000,
                         max_keepalive_connections=100000)

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        app.state.prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None,
                              base_url=prefiller_base_url,
                              limits=limit),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        app.state.decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None,
                              base_url=decoder_base_url,
                              limits=limit),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8100])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[8200])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')

        # Send request to prefill service
        response = await send_request_to_service(prefill_client_info, api,
                                                 req_data, request_id)

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(decode_client_info,
                                                       api,
                                                       req_data,
                                                       request_id=request_id):
                yield chunk

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
```

### 五、下发推理请求

命令中的localhost、端口还有模型权重的路径配置成自己的

短问题：

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/mnt/weight/Qwen3-8B", "prompt": "Hello. I have a question. The president of the United States is", "max_tokens": 200, "temperature":0.0 }'
```

长问题：

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/mnt/weight/Qwen3-8B", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_tokens": 256, "temperature":0.0 }'
```
