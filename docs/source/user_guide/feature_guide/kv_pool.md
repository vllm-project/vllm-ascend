# Ascend Store Deployment Guide

## Environmental Dependencies

* Software:
    * CANN >= 8.5.0
    * vLLM：main branch
    * vLLM-Ascend：main branch
    * mooncake：>= 0.3.9

### KV Pool Parameter Description

#### `kv_connector_extra_config`: Additional Configurable Parameters for Pooling

| Parameter | Description |
| :--- | :--- |
| `lookup_rpc_port` | Port for RPC Communication Between Pooling Scheduler Process and Worker Process: Each Instance Requires a Unique Port Configuration. |
| `load_async` | Whether to Enable Asynchronous Loading. The default value is false. |
| `backend` | Set the storage backend for kvpool (`mooncake`, `memcache`, `yuanrong`), with the default being `mooncake`. |
| `consumer_is_to_put` | Whether Decode node put KV Cache into KV Pool. The default value is false. |
| `consumer_is_to_load` | Whether Decode node load KV cache from KV Pool. The default value is false. |
| `prefill_pp_size` | Prefill PP size, needs to be set when Prefill node enables PP. |
| `prefill_pp_layer_partition` | Prefill PP layer partition, needs to be set when Prefill node enables PP. |

### Environment Variable Configuration

To guarantee uniform hash generation, it is required to synchronize the PYTHONHASHSEED environment variable across all nodes upon enabling KV Pool.

```bash
export PYTHONHASHSEED=0
```

## Example of using Mooncake as a KV Pool backend

* Software:
    * Check NPU HCCN Configuration:

        Ensure that the hccn.conf file exists in the environment. If using Docker, mount it into the container.

        ```bash
        cat /etc/hccn.conf
        ```

    * Install Mooncake

        Mooncake is the serving platform for Kimi, a leading LLM service provided by Moonshot AI.
        Installation and Compilation Guide: <https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#build-and-use-binaries>.
        First, we need to obtain the Mooncake project. Refer to the following command:

        ```shell
        git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
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

        * Adjust the Python path according to your specific Python installation
        * Ensure `/usr/local/lib` and `/usr/local/lib64` are in your `LD_LIBRARY_PATH`

        ```shell
        export LD_LIBRARY_PATH=/usr/local/lib64/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
        ```

### Environment Variables Description

| Hardware | HDK & CANN versions | Export Command | Description |
| :--- | :--- | :--- | :--- |
| 800 I/T A3 series | HDK >= 26.0.0<br>CANN >= 9.0.0 | `export ASCEND_ENABLE_USE_FABRIC_MEM=1` | **Recommended**. Enables unified memory address direct transmission scheme. |
| 800 I/T A3 series | 25.5.0<=HDK<26.0.0 | `export ASCEND_BUFFER_POOL=4:8` | Configures the number and size of buffers on the NPU Device for aggregation and KV transfer (e.g., `4:8` means 4 buffers of 8MB). |
| 800 I/T A2 series | N/A | `export HCCL_INTRA_ROCE_ENABLE=1` | Required by direct transmission cheme on 800 I/T A2 series|

### FAQ for HIXL (ascend_direct) backend

For common troubleshooting and issue localization guidance for HIXL (ascend_direct), see:
<https://gitcode.com/cann/hixl/wiki/HIXL%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%89%8B%E5%86%8C.md>

### Run Mooncake Master

#### 1.Configure mooncake.json

The environment variable **MOONCAKE_CONFIG_PATH** is configured to the full path where mooncake.json is located.

```shell
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
**global_segment_size**: Registered memory size per card to the KV Pool. **Needs to be aligned to 1GB.**

#### 2.Start mooncake_master

Under the mooncake folder:

```shell
mooncake_master --port 50088 --eviction_high_watermark_ratio 0.9 --eviction_ratio 0.1 --default_kv_lease_ttl 11000
```

`eviction_high_watermark_ratio` determines the watermark where Mooncake Store will perform eviction，and `eviction_ratio` determines the portion of stored objects that would be evicted.
`default_kv_lease_ttl` controls the default lease TTL for KV objects (milliseconds); configure it via `--default_kv_lease_ttl` and keep it larger than `ASCEND_CONNECT_TIMEOUT` and `ASCEND_TRANSFER_TIMEOUT`.

### PD Disaggregation Scenario

#### 1.Run `prefill` Node and `decode` Node

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`. `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` serves as the prefix-cache node.

`prefill` Node：

```shell
bash multi_producer.sh
```

The content of the multi_producer.sh script:

```shell
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0 
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ACL_OP_INIT_MODE=1
#A3
export ASCEND_ENABLE_USE_FABRIC_MEM=1
#A2
#export HCCL_INTRA_ROCE_ENABLE=1

#Minimum retransmission timeout of the RDMA，equals 4.096 μs * 2 ^ timeout.
#Needs to satisfy the equation: ASCEND_TRANSFER_TIMEOUT > RDMA_TIMEOUT * 7, where 7 is the default number of retry for RDMA transfer.
#HCCL_RDMA_TIMEOUT also affects collective communication behavior and should be configured carefully.
export HCCL_RDMA_TIMEOUT=17

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
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
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

```shell
bash multi_consumer.sh
```

The content of multi_consumer.sh:

```shell
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export PYTHONHASHSEED=0 
export MOONCAKE_CONFIG_PATH="/xxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export ACL_OP_INIT_MODE=1
#A3
export ASCEND_ENABLE_USE_FABRIC_MEM=1
#A2
#export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8200 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
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

Currently, the key-value pool in PD Disaggregate only stores the kv cache generated by the Prefill node by default. In models using MLA, it is now supported that the Decode node stores the kv cache for use by the Prefill node, enabled by adding `consumer_is_to_put: true` to the AscendStoreConnector. If the Prefill node enables PP, `prefill_pp_size` or `prefill_pp_layer_partition` also needs to be set. Example as follows:

```python
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
        "lookup_rpc_port": "0",
        "backend": "mooncake",
        "consumer_is_to_put": true,
        "prefill_pp_size": 2,
        "prefill_pp_layer_partition": "30,31"
    }
}
```

#### 2、Start proxy_server

```shell
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

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

### PD-Mixed Inference

#### 1.Run Mixed Department Script

```shell
bash pd_mix.sh
```

Content of pd_mix.sh:

```shell
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/xxxxx/vllm
export MOONCAKE_CONFIG_PATH="/xxxxxx/mooncake.json"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0 
export ACL_OP_INIT_MODE=1
#A3
export ASCEND_ENABLE_USE_FABRIC_MEM=1
#A2
#export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TIMEOUT=17
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000

python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
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

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Hello. I have a question. The president of the United States is", "max_completion_tokens": 200, "temperature":0.0 }'
```

Long question:

```shell
curl -s http://localhost:8100/v1/completions -H "Content-Type: application/json" -d '{ "model": "/xxxxx/Qwen2.5-7B-Instruct", "prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?", "max_completion_tokens": 256, "temperature":0.0 }'
```

Note: For MooncakeStore with `ASCEND_BUFFER_POOL` enabled, it is recommended to perform a warm-up phase before running actual performance benchmarks.

This is because HCCL one-sided communication connections are created lazily after the instance is launched when Device-to-Device communication is involved. Currently, full-mesh connections between all devices are required. Establishing these connections introduces a one-time time overhead and persistent device memory consumption (4 MB of device memory per connection).

**For warm-up, it is recommended to issue requests with an input sequence length of 8K and an output sequence length of 1, with the total number of requests being 2–3× the number of devices (cards/dies).**

## Example of using Memcache as a KV Pool backend

### Installing Memcache

**MemCache depends on MemFabric. Therefore, MemFabric must be installed.Installing the memcache after the memfabric is installed.**

* **memfabric_hybrid**: <https://gitcode.com/Ascend/memfabric_hybrid/tree/master/doc/build.md>

* **memcache**: <https://gitcode.com/Ascend/memcache/blob/master/doc/build.md>

### Configuring the memcache Config File

    config Path：/usr/local/memcache_hybrid/latest/config/
    **Config file parameters description**：<https://gitcode.com/Ascend/memcache/blob/develop/doc/memcache_config.md>

    Set TLS certificate configurations. If TLS is disabled, you do not need to upload a certificate. If TLS is enabled, you need to upload a certificate.

```shell
# mmc-meta.conf
ock.mmc.tls.enable = false
ock.mmc.config_store.tls.enable = false

# mmc-local.conf
ock.mmc.tls.enable = false
ock.mmc.config_store.tls.enable = false
ock.mmc.local_service.hcom.tls.enable = false
```

You are advised to copy mmc-local.conf and mmc-meta.conf to your own path and modify them, and set the MMC_META_CONFIG_PATH environment variable to the path of your own mmc-meta.conf file.

**mmc-meta.conf：**

```shell
# Meta service start-up url
# It will automatically modified to PodIP at Pod startup in K8s meta service cluster master-standby high availability scenario
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
# config store url, It will automatically modified to PodIP at Pod startup in K8s
ock.mmc.meta_service.config_store_url = tcp://xx.xx.xx.xx:6000
# Enable or disable high availability deployment
ock.mmc.meta.ha.enable = false
# Log level: debug, info, warn, error
ock.mmc.log_level = error
# Log directory path, supports both relative and absolute paths, the system will automatically append 'logs' directory.
# The absolute log path at default value is '/path/to/mmc_meta_service/../logs'
# If the path of mmc_meta_service is '/usr/local/mxc/memfabric_hybrid/latest/aarch64-linux/bin'
# Then the path of log is '/usr/local/mxc/memfabric_hybrid/latest/aarch64-linux/logs'
ock.mmc.log_path = .
# Log rotation file size, unit is MB, value range [1,500]
ock.mmc.log_rotation_file_size = 20
# Log rotation file count, value range [1,50]
ock.mmc.log_rotation_file_count = 50

# The threshold that triggers eviction, measured as a percentage of space usage
# 'put' operation will trigger eviction when the threshold is exceeded
ock.mmc.evict_threshold_high = 90
# The target threshold of eviction, measured as a percentage of space usage
ock.mmc.evict_threshold_low = 80

# TLS configuration for metaservice
ock.mmc.tls.enable = false
ock.mmc.tls.ca.path = /opt/ock/security/certs/ca.cert.pem
ock.mmc.tls.ca.crl.path = /opt/ock/security/certs/ca.crl.pem
ock.mmc.tls.cert.path = /opt/ock/security/certs/server.cert.pem
ock.mmc.tls.key.path = /opt/ock/security/certs/server.private.key.pem
ock.mmc.tls.key.pass.path = /opt/ock/security/certs/server.passphrase
ock.mmc.tls.package.path = /opt/ock/security/libs/
ock.mmc.tls.decrypter.path =

# TLS configuration for config store
ock.mmc.config_store.tls.enable = false
ock.mmc.config_store.tls.ca.path = /opt/ock/security/certs/ca.cert.pem
ock.mmc.config_store.tls.ca.crl.path = /opt/ock/security/certs/ca.crl.pem
ock.mmc.config_store.tls.cert.path = /opt/ock/security/certs/server.cert.pem
ock.mmc.config_store.tls.key.path = /opt/ock/security/certs/server.private.key.pem
ock.mmc.config_store.tls.key.pass.path = /opt/ock/security/certs/server.passphrase
ock.mmc.config_store.tls.package.path = /opt/ock/security/libs/
ock.mmc.config_store.tls.decrypter.path =
```

**Key Focuses：**

| Parameter | Description |
| :--- | :--- |
| `ock.mmc.meta_service_url` | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| `ock.mmc.meta_service.config_store_url` | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| `ock.mmc.meta.ha.enable` | Set to `false` to disable TLS authentication modification. |
| `ock.mmc.config_store.tls.enable` | Set to `false` to disable TLS authentication modification. |

**mmc-local.conf：**

```shell
# Meta service start-up url
# K8s meta service cluster master-standby high availability scenario: ClusterIP address
# Non-HA scenario: keep consistent with the same name configuration in mmc-meta.conf
ock.mmc.meta_service_url = tcp://xx.xx.xx.xx:5000
# Log level: debug, info, warn, error
ock.mmc.log_level = error

# TLS configurations for metaservice
ock.mmc.tls.enable = false
ock.mmc.tls.ca.path = /opt/ock/security/certs/ca.cert.pem
ock.mmc.tls.ca.crl.path = /opt/ock/security/certs/ca.crl.pem
ock.mmc.tls.cert.path = /opt/ock/security/certs/client.cert.pem
ock.mmc.tls.key.path = /opt/ock/security/certs/client.private.key.pem
ock.mmc.tls.key.pass.path = /opt/ock/security/certs/client.passphrase
ock.mmc.tls.package.path = /opt/ock/security/libs/
ock.mmc.tls.decrypter.path =

# Total count of local service, including services that will be add in the future
ock.mmc.local_service.world_size = 256
# config store url, it will automatically modified to PodIP at Pod startup in HA scenario
# keep consistent with the same name configuration in mmc-meta.conf
ock.mmc.local_service.config_store_url = tcp://xx.xx.xx.xx:6000
# TLS configurations for config_store
ock.mmc.config_store.tls.enable = false
ock.mmc.config_store.tls.ca.path = /opt/ock/security/certs/ca.cert.pem
ock.mmc.config_store.tls.ca.crl.path = /opt/ock/security/certs/ca.crl.pem
ock.mmc.config_store.tls.cert.path = /opt/ock/security/certs/client.cert.pem
ock.mmc.config_store.tls.key.path = /opt/ock/security/certs/client.private.key.pem
ock.mmc.config_store.tls.key.pass.path = /opt/ock/security/certs/client.passphrase
ock.mmc.config_store.tls.package.path = /opt/ock/security/libs/
ock.mmc.config_store.tls.decrypter.path =

# Data transfer protocol, 'host_rdma': rdma over host; 'host_tcp': tcp over host; 'device_rdma': rdma over device; 'device_sdma': sdma over device
ock.mmc.local_service.protocol = device_sdma
# HBM/DRAM space usage, configuration type supports 134217728, 2048KB/2048K, 200MB/200mb/200m, 2.5GB or 1TB, case-insensitive, the maximum value is 1TB
# The system automatically calculates and aligns downwards to 2MB (host_sdma or host_tcp) or 1GB (device_sdma or device_rdma)
# After alignment, the HBM size and DRAM size cannot both be 0 at the same time
ock.mmc.local_service.dram.size = 2GB
ock.mmc.local_service.hbm.size = 0

# If the protocol is host_rdma, the ip needs to be set as RDMA network card ip. Use 'show_gids' command to query it
ock.mmc.local_service.hcom_url = tcp://127.0.0.1:7000
# HCOM TLS config
ock.mmc.local_service.hcom.tls.enable = false
ock.mmc.local_service.hcom.tls.ca.path = /opt/ock/security/certs/ca.cert.pem
ock.mmc.local_service.hcom.tls.ca.crl.path = /opt/ock/security/certs/ca.crl.pem
ock.mmc.local_service.hcom.tls.cert.path = /opt/ock/security/certs/client.cert.pem
ock.mmc.local_service.hcom.tls.key.path = /opt/ock/security/certs/client.private.key.pem
ock.mmc.local_service.hcom.tls.key.pass.path = /opt/ock/security/certs/client.passphrase
ock.mmc.local_service.hcom.tls.decrypter.path =

# The total retry duration (retry interval is 200ms) when client requests meta service and the connection does not exist
# Default value is 0, means no-retry and return immediately, value range [0, 600000]
ock.mmc.client.retry_milliseconds = 0

ock.mmc.client.timeout.seconds = 60

# read/write thread pool size, value range [1, 64]
ock.mmc.client.read_thread_pool.size = 16
ock.mmc.client.write_thread_pool.size = 2
```

**Key Focuses：**

| Parameter | Description |
| :--- | :--- |
| `ock.mmc.meta_service_url` | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| `ock.mmc.local_service.config_store_url` | Configure the IP address and port number of the master node. The IP address and port number of the P node and D node can be the same. |
| `ock.mmc.local_service.world_size` | Total count of local service, including services that will be added in the future. |
| `ock.mmc.local_service.protocol` | `host_rdma` (default), `device_rdma` (supported for A2 and A3 when device ROCE available, recommended for A2), `device_sdma` (supported for A3 when HCCS available, recommended for A3). Currently does not support heterogeneous protocol setting.|
| `ock.mmc.local_service.dram.size` | Sets the size of the memory occupied by the master. The configured value is the size of the memory occupied by each card. |
| `ock.mmc.meta.ha.enable` | Set to `false` to disable TLS authentication modification. |
| `ock.mmc.config_store.tls.enable` | Set to `false` to disable TLS authentication modification. |

### Memcache environment variables

```shell
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# Configuring Environment Variables in the Configuration File
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
```

### Run Memcache Master

Starting the MetaService service.

```shell
1. Set environment variables for the configuration file.
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf

2. Access the Python console or compile the following Python script to start the process:
from memcache_hybrid import MetaService
MetaService.main()
```

Method 2 for starting the MetaService service.

```shell
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export MMC_META_CONFIG_PATH=/home/memcache/shell/mmc-meta.conf # Set it to the path of your own configuration file.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.10/lib/
/usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service
```

### PD Disaggregation Scenario

#### 1.Run `prefill` Node and `decode` Node

Using `MultiConnector` to simultaneously utilize both `MooncakeConnectorV1` and `AscendStoreConnector`. `MooncakeConnectorV1` performs kv_transfer, while `AscendStoreConnector` enables KV Cache Pool

#### 800I A2/800T A2 Series

`prefill` Node：

```shell
rm -rf /root/ascend/log/*

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/mmc-local.conf

# nic_name can be looked up in ifconfig
nic_name="xxxxxx"
local_ip="xx.xx.xx.xx"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

rm -rf ./connector.log
vllm serve xxxxxxx/Qwen3-32B \
  --host 0.0.0.0 \
  --port 30050 \
  --enforce-eager \
  --data-parallel-size 2 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name qwen3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-num_seqs 20 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
    '{
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "engine_id": "2",
            "kv_connector_extra_config": {
                "connectors": [
                {
                            "kv_connector": "MooncakeConnectorV1",
                            "kv_role": "kv_producer",
                            "kv_port": "20001",
                            "kv_connector_extra_config": {
                                    "prefill": {
                                            "dp_size": 2,
                                            "tp_size": 4
                                    },
                                    "decode": {
                                            "dp_size": 2,
                                            "tp_size": 4
                                    }
                            }
                    },
                    {
                            "kv_connector": "AscendStoreConnector",
                            "kv_role": "kv_producer",
                            "kv_connector_extra_config":{
                                    "backend": "memcache",
                                    "lookup_rpc_port":"0"
                            }
                    }  
                ]
            }
    }' > log_p.log 2>&1
```

`decode` Node：

```shell
rm -rf /root/ascend/log/*

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/mmc-local.conf

# nic_name can be looked up in ifconfig
nic_name="xxxxxx"
local_ip="xx.xx.xx.xx"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

rm -rf ./connector.log
vllm serve xxxxxxx/Qwen3-32B \
  --host 0.0.0.0 \
  --port 30060 \
  --enforce-eager \
  --data-parallel-size 2 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --served-model-name qwen3 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-num_seqs 20 \
  --no-enable-prefix-caching \
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
                                                "dp_size": 2,
                                                "tp_size": 4
                                        },
                                        "decode": {
                                                "dp_size": 2,
                                                "tp_size": 4
                                        }
                                }
                    } ,
            {  
                               "kv_connector": "AscendStoreConnector",
                               "kv_role": "kv_consumer",
                               "kv_connector_extra_config":{
                                    "backend": "memcache",
                                    "lookup_rpc_port":"1"
                               }
                       }  

                ]
        }
  }' > log_d.log 2>&1
```

#### 800I A3/800T A3 Series

`prefill` Node：

```shell
rm -rf /root/ascend/log/*

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/shell/mmc-local.conf

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024


python -m vllm.entrypoints.openai.api_server \
  --model=xxxxxxxxx/DeepSeek-R1 \
  --served-model-name dsv3 \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --port 30050 \
  --max-num_seqs 20 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --enable_expert_parallel \
  --quantization ascend \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
 '{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "engine_id": "2",
  "kv_connector_extra_config": {
   "connectors": [
   {
     "kv_connector": "MooncakeConnectorV1",
     "kv_role": "kv_producer",
     "kv_port": "20001",
     "kv_connector_extra_config": {
      "prefill": {
       "dp_size": 2,
       "tp_size": 8
      },
      "decode": {
       "dp_size": 2,
       "tp_size": 8
      }
     }
    },
    {
     "kv_connector": "AscendStoreConnector",
     "kv_role": "kv_producer",
     "kv_connector_extra_config":{
      "backend": "memcache",
      "lookup_rpc_port":"0"
     }
    }  
   ]
  }
 }' > log_p.log 2>&1 
```

`decode` Node：

```shell
rm -rf /root/ascend/log/*

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/shell/mmc-local.conf

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024

python -m vllm.entrypoints.openai.api_server \
  --model=xxxxxxxxxxxxxxxx/DeepSeek \
  --served-model-name dsv3 \
  --trust-remote-code \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --port 30060 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --enforce-eager\
  --quantization ascend \
  --no-enable-prefix-caching \
  --max-num_seqs 20 \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --enable_expert_parallel \
  --gpu-memory-utilization 0.9 \
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
      "dp_size": 2,
      "tp_size": 8
     },
     "decode": {
      "dp_size": 2,
      "tp_size": 8
     }
    }
   },
    {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config":{
                "backend": "memcache",
                "lookup_rpc_port":"1"
    }
   }  
  ]
 }
  }' > log_d.log 2>&1
```

#### [2、Start proxy_server](#2start-proxy_server)

#### [3、run-inference](#3run-inference)

### PD-Mixed Scenario

#### 1.Run Mixed Department Script

#### 800I A2/800T A2 Series

The deepseek model needs to be run in a two-node cluster.

**Run_pd_mix_1.sh:**

```shell
rm -rf /root/ascend/log/*

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/mmc-local.conf

# nic_name can be looked up in ifconfig
nic_name="xxxxxxx"
local_ip="xx.xx.xx.xx"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name


export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

rm -rf ./connector.log
vllm serve xxxxxxx/DeepSeek-R1 \
  --host 0.0.0.0 \
  --port 30050 \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --api-server-count 2 \
  --data-parallel-address 141.61.33.167 \
  --data-parallel-rpc-port 13348  \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --max-num_seqs 20 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
                "backend": "memcache",
                "lookup_rpc_port":"0"
           }
  }' > log_pd_mix_1.log 2>&1

```

**Run_pd_mix_2.sh:**

```shell
rm -rf /root/ascend/log/*

source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/mmc-local.conf

# nic_name can be looked up in ifconfig
nic_name="xxxxxxx"
local_ip="xx.xx.xx.xx"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
# export VLLM_TORCH_PROFILER_DIR="./vllm-profiling"
# export VLLM_TORCH_PROFILER_WITH_STACK=0

rm -rf ./connector.log
vllm serve xxxxxxx/DeepSeek-R1 \
  --host 0.0.0.0 \
  --port 30050 \
  --headless  \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address 141.61.33.167 \
  --data-parallel-rpc-port 13348  \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name deepseek \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --max-num_seqs 20 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --kv-transfer-config \
   '{
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
                "backend": "memcache",
                "lookup_rpc_port":"0"
           }
  }' > log_pd_mix_2.log 2>&1

```

#### 800I A3/800T A3 Series

```shell
bash pd_mix.sh
```

Content of pd_mix.sh:

```shell
rm -rf /root/ascend/log/*

# memcache:
echo 200000 > /proc/sys/vm/nr_hugepages
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export MMC_LOCAL_CONFIG_PATH=/home/memcache/shell/mmc-local.conf

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export PYTHONHASHSEED=0
export HCCL_BUFFSIZE=1024


python -m vllm.entrypoints.openai.api_server \
  --model=xxxxxxx/DeepSeek-R1 \
  --served-model-name dsv3 \
  --trust-remote-code \
  --enforce-eager \
  -dp 2 \
  -tp 8 \
  --port 30050 \
  --max-num_seqs 20 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp"}' \
  --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --enable_expert_parallel \
  --quantization ascend \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --kv-transfer-config \
  '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port":"0"
      }
  }' > log_pd_mix.log 2>&1 

```

#### [2.Run Inference](#2run-inference)

## Example of using Yuanrong as a KV Pool backend

* Software:
    * Install `openyuanrong-datasystem` on all nodes (`yr.datasystem` must be importable).

### Install Yuanrong Datasystem

```bash
pip install openyuanrong-datasystem
```

### Start etcd

Yuanrong Datasystem uses etcd for service discovery. The following example
starts a single-node etcd cluster:

```bash
ETCD_VERSION="v3.5.12"
ETCD_IP="127.0.0.1"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
sudo cp etcd etcdctl /usr/local/bin/

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://${ETCD_IP}:2379 \
  --listen-peer-urls http://0.0.0.0:2380 \
  --initial-advertise-peer-urls http://${ETCD_IP}:2380 \
  --initial-cluster etcd-single=http://${ETCD_IP}:2380 &

etcdctl --endpoints "${ETCD_IP}:2379" put key "value"
etcdctl --endpoints "${ETCD_IP}:2379" get key
```

For production environments, refer to the official etcd clustering
documentation: <https://etcd.io/docs/current/op-guide/clustering/>

### Start Datasystem Worker

Start a Datasystem worker on each node by using `dscli`:

```bash
dscli start -w \
  --worker_address "${WORKER_IP}:31501" \
  --etcd_address "${ETCD_IP}:2379" \
  --shared_memory_size_mb 20480
```

The `--worker_address` value is consumed later by `DS_WORKER_ADDR`, so keep
the host and port identical on the same node.

For more parameters, refer to the `dscli` usage documentation on the Yuanrong
Datasystem official site:
<https://atomgit.com/openeuler/yuanrong-datasystem>

To stop the worker:

```bash
dscli stop --worker_address "${WORKER_IP}:31501"
```

### Environment Variable Configuration

Set the following environment variables on each node before starting vLLM:

| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `PYTHONHASHSEED` | Yes | `0` | Must be consistent across all nodes to guarantee uniform hash generation. |
| `DS_WORKER_ADDR` | Yes | N/A | Datasystem worker address in `<host>:<port>` format. This must match the local `dscli start --worker_address` value. |
| `DS_ENABLE_EXCLUSIVE_CONNECTION` | No | `0` | Passed to Yuanrong `HeteroClient.enable_exclusive_connection`. Use `1` to enable the exclusive connection mode when required by your deployment. |
| `DS_ENABLE_REMOTE_H2D` | No | `0` | Passed to Yuanrong `HeteroClient.enable_remote_h2d`. Use `1` only when remote host-to-device transfer is enabled in your Datasystem deployment. |

```bash
export PYTHONHASHSEED=0
export DS_WORKER_ADDR="${WORKER_IP}:31501"
export DS_ENABLE_EXCLUSIVE_CONNECTION=0
export DS_ENABLE_REMOTE_H2D=0
```

### Run AscendStoreConnector with Yuanrong backend

Use `AscendStoreConnector` with `backend: "yuanrong"`:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /xxxxx/Qwen2.5-7B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --enforce-eager \
    --no-enable-prefix-caching \
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
        "lookup_rpc_port": "1",
        "backend": "yuanrong"
    }
}'
```

`lookup_rpc_port` is the RPC port used between the pooling scheduler process
and the worker process. Each instance must use a unique port value.

### Notes

* The Yuanrong backend normalizes KV keys before calling Datasystem. Keys longer
  than 255 characters or containing unsupported characters are rewritten, so do
  not rely on the raw key string when debugging backend storage.
* No extra buffer pre-registration step is required for Yuanrong. The backend
  uses device pointers directly when building blob lists.

#### [2.Run Inference](#2run-inference)
