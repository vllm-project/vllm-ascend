# PD分离快速部署

本文档介绍如何快速拉起PD分离部署推理，支持3机1P1D、4机2P1D、8机4P1D、16机8P1D和EP144 36机18P1D。

## 硬件要求

**硬件：** CloudMatrix384推理卡

**操作系统：** Linux

**镜像版本：** swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。

**网络联通：** 使用[ssh命令](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%BD%91%E7%BB%9C%E8%BF%9E%E9%80%9A%E6%80%A7%E6%A3%80%E6%9F%A5)确认机器互连。

## 模型准备

基于开源的 DeepSeek-V3/R1 进行[权重转换](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)。目前提供一份转换好的权重供直接下载使用，[下载链接](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?v2token=KBL+tPW8sihb1DQcY03GYZZdWrdKg8E2xUI8XrCsr7jo72H22pg6bY8V89ZgmD4Zq4VEcQa802+q2nR4Bydrzm9jjAO1ohYLIIDMeRtttFZR+EpCA2PWmZaVxazPlkJ6qtADFZaESGpEHUxChlLlFQ2xeLp6sXP5qVsj6JEPRh7MA6SmfqK8mLdgi/rmBjY6A0CRJFEe1K5JrgONubynmJaescenf5t0h36szT23dHV46pjw0BCjCFtxJyXqgGWc4T7pv3tugR09oHNLFaVoPi4ZlElNciul9a90kZ6ZOoNJ3ufoRyHA9bTdwqeJGg8jsBFzRr+d+tU1GXd8UaswFHUo805A3MoPlqSRiYThAz+3aPorLkveex99xiEwCq+pajn6S9GzSeY8FLjEMlopGMKfHJ6Z1B5aoGpIBY8UsjG878ixsE/YiZmetkXDO/FPYr/r9sbHqg5pLVXCmaH7uHqVvDabx6Mx0a8GRITN+yjwg08LjS3C76gwCfEqD7FegGitodr7RLSDsuewjGqjgK/7ST2J320CoBBBw5vtAAsDYiDC6LJOEZCN2ht/eAZUHvy7ZrCeBMN1AmagvqsXVJrsn2tof/CL5LpRm7z5eVoFAhLjpyKIOgWjUksthT0MLmqIZCiMoslj9BfCKv780nEeDQZqO/eerh7zT4qojk8Xaxuj9Xvv1fKtJnId608QPtLXDireSzh6aa4tF1b5W747AhNAPzLoNdOzcLheYyw=_Vsd2i2lmljqrcbVJHDC8TZw7tQFHpoZ6ZS0O3b864QM=_DuHhemY++UqCJXeQyjgwdA==)（提取码为12345678）。

## 部署

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni-ci/daily_omniinfer:20250722_26
git clone https://gitee.com/omniai/omniinfer.git
git clone https://github.com/vllm-project/vllm.git omniinfer/infer_engines/vllm
```

### 使用ansible一键部署

详见[ansible部署文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md)。以下为快速部署示例。

#### 环境准备

安装ansible，参考[文档](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87-1)。

#### 修改配置文件

需要修改`omni_infer_inventory_used_for_xP1D.yml`和 `omni_infer_server_template` 两处配置文件，位于`omniinfer/tools/ansible/templete/`路径下。以2P1D为例:

1. **omni_infer_inventory_used_for_2P1D.yml**

   将`p0/p1/d0/d1/c0`下面的`ansible_host` 与 `host_ip` 值改为对应的IP。<span style="color:red; font-weight:bold">对于多机组 D 的场景，所有 D 节点的 `host_ip` 为主节点 d0 的 IP。</span>


   ```YAML
   children:
     P:
       hosts:
         p0:
           ansible_host: "127.0.0.1"  # P0节点的IP
           ...
           host_ip: "127.0.0.1"  # P0节点的IP
           ...

         p1:
           ansible_host: "127.0.0.2"  # P1节点的IP
           ...
           host_ip: "127.0.0.2"  # P1节点的IP
           ...

     D:
       hosts:
         d0:
           ansible_host: "127.0.0.3"  # D0 节点的IP
           ...
           host_ip: "127.0.0.3"       # D0 节点的IP
           ...

         d1:
           ansible_host: "127.0.0.4"  # D1 节点的IP
           ...
           host_ip: "127.0.0.3"       # D0 节点的IP, 即 D 节点的主节点 IP
           ...

     C:
       hosts:
         c0:
           ansible_host: "127.0.0.1"  # C0 节点的IP，即 Global Proxy 节点
           ...

   ```

   生成私钥文件，参考[文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md#%E5%AF%86%E9%92%A5%E6%96%87%E4%BB%B6%E7%9A%84%E5%87%86%E5%A4%87)。将`ansible_ssh_private_key_file:`修改为私钥文件路径：

   ```YAML
    all:
      vars:
        ...
        ansible_ssh_private_key_file: /path/to/key.pem  # 私钥文件路径
        ...
   ```

2. **omni_infer_server_template.yml**

    修改以下环境变量
    ```yaml
    environment:
        # Global Configuration
        LOG_PATH: "/data/log_path"
        MODEL_PATH: "/data/models/DeepSeek-R1-w8a8"  #模型文件路径
        LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"
        CODE_PATH: "/data/local_code_path"  # omniinfer本地代码路径
        HTTP_PROXY: ""  # 下载nginx的HTTP代理地址，如果不需要代理可以留空

        # Configuration for containers
        DOCKER_IMAGE_ID: "REPOSITORY:TAG" #镜像与标签
        DOCKER_NAME_P: "you_name_omni_infer_prefill" # P容器名称
        DOCKER_NAME_D: "you_name_omni_infer_decode"  # D容器名称
        DOCKER_NAME_C: "you_name_omni_infer_proxy"   # Proxy 容器名称
    ```

配置文件详细解释说明请参考[文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/template/README.md#%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6%E8%A7%A3%E9%87%8A%E8%AF%B4%E6%98%8E)。

#### 执行命令

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml
```

#### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions -H "Content-Type:application/json" -d '{"model": "deepseek","temperature":0,"max_tokens":50,"prompt": "how are you?", "stream":true,"stream_options": {"include_usage": true,"continuous_usage_stats": true}}'
```

#### 注意事项

- ansible 每次执行时，会对vllm代码进行`checkout -f`，若修改代码，请确保代码已提交或暂存。

## 更高性能

可以使能以下特性来获得更高的性能：

**1. 使用图缓存**

首次启动服务时，模型会从头编译。建议首次成功启动后，重新执行以下命令以启用图缓存，提升性能：

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server
```


**2. 调整 proxy batch size**

当前的默认值为 25。建议根据每个 die 的平均并发量大小(n)进行调整。推荐值为(n+1)。进入proxy容器，在 /usr/local/nginx/conf/nginx.conf 文件中修改该值，然后执行 `nginx -c /usr/local/nginx/conf/nginx.conf -s reload` 以应用更改。



**3. 增加 batch size**

ansible部署默认开启 MTP 以优化性能。如需调整 `--max-num-seqs`（batch size），需同步修改omniinfer本地代码路径下 omniinfer/tests/test_config/test_config_decode.json中`decode_gear_list` 的值：

```JSON
"decode_gear_list": [batch_size * (1+num_speculative_tokens)]
```

 以MTP 1为例，`--max-num-seqs`设置为32，`"decode_gear_list":[64]`。
