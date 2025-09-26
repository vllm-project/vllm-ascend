NPU-optimized implementation of LMCache

# 安装etcd + lmcache + mooncacke

## 直接安装
在需要部署的机器上，执行 `bash script/install,sh` 即可，会依次安装etcd,lmcache,ascend_lmcache,mooncake

## 打包安装
在宿主机上，执行`bash script/build.sh`，会编译得到 lmcache,ascend_lmcache以及mooncake的wheel包以及所需的lib库，然后在需要部署的机器上，通过`bash script/install_wheel.sh`安装

# 部署启动lmcache + mooncake

## 手动部署

### etcd

```
nohup etcd --listen-client-urls http://0.0.0.0:2377 --advertise-client-urls http://0.0.0.0:2377 &> etcd.log &
```


### mooncake master
```
nohup mooncacke_master -rpc_address={机器ip} --rpc_port=51051 -metrics_port=52051 -v=1 &> master.log &
```

### 配置文件

```
{
    "metadata_server": "etcd://{{ ETCD_IP }}:{{ ETCD_PORT}}",
    "local_hostname": "{{ HOST_IP }}",
    "protocol": "tcp",
    "tansfer_timeout": 60,
    "global_segment_size": 17179869184,
    "master_server_address": "{{ MOONCAKE_MASTER_IP}}:{{ MOONCAKE_MASTER_PORT }}"
}
```
注：
1. {{ ETCD_IP }}:{{ ETCD_PORT}}替换为etcd节点ip以及etcd端口2377
2. {{ HOST_IP }} 替换为节点ip
3. "master_server_address": "{{ MOONCAKE_MASTER_IP}}:{{ MOONCAKE_MASTER_PORT }}"替换为mooncacke master节点的ip以及端口51051。

```
# 256 Tokens per KV Chunk
chunk_size: 256
# Enable CPU memory backend
local_cpu: false # default
# 5GB of Pinned CPU memory
max_local_cpu_size: 5.0 #default

remote_url: "mooncacketore://{{ MOONCAKE_MASTER_IP}}:{{ MOONCAKE_MASTER_PORT}}"
external_lookup_client: "mooncakestore://{{ MOONCAKE_MASTER_IP }}:{{ MOONCAKE_MASTER_PORT}}"
remote_serde: "naive"

extra_config:
    remote_enable_mal_worker_id_as0: true
```

注：
1. {{ MOONCAKE_MASTER_IP}} : {{ MOONCAKE_MASTER_PORT }}替换为mooncake master节点的IP和端口号

### 实例拉起
如果使用lmcache，需要定义一下环境变量：
```
export MOONCAKE_CONFIG_PATH=mooncake_config.json
export LMCACHE_CONFIG_FILE=lmcache_mooncake_config.yaml
```

实例拉起时，P节点增加配置(APC):
```
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

或者（关闭APC）
```
--kv-transfer-config '{"kv_connector:"LMCacheConnectorV1","kv_role":"kv_producer"}'
```

D节点增加配置
```
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_consumer"}'
```

## ansible部署
将 omniinfer/tools/ansible/template/omni_infer_server_template.yml中的 KV_CONNECTOR设置为"LMCacheConnectorV1"即可

