# PD分离快速部署

本文档介绍如何快速拉起PD分离部署推理，支持8机1P1D。

## 硬件要求

**硬件：** Atlas 800IA3

**操作系统：** Linux

**镜像版本：** swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a3-arm:release_v0.4.2

[**驱动检查**](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#ascend-npu%E5%9B%BA%E4%BB%B6%E5%92%8C%E9%A9%B1%E5%8A%A8%E6%A3%80%E6%9F%A5): `npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。

**驱动版本：** Ascend HDK 25.2.1 
https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/266220744?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743

**网络联通：** 
1、使用[ssh命令](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md#%E7%BD%91%E7%BB%9C%E8%BF%9E%E9%80%9A%E6%80%A7%E6%A3%80%E6%9F%A5)确认机器互连。
2、服务机器ROCE面联通

## 模型准备

待补充 --- Deepseek BF16

## 部署

### 镜像及源码准备

```bash
docker pull swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a3-arm:release_v0.4.2
git clone https://gitee.com/omniai/omniinfer.git
```


### 环境准备

用户执行机需要安装ansible
```bash
yum install ansible
yum install openssh-server
```

目标机安装libselinux-python3 （可选）
```bash
yum install libselinux-python3
```


更多ansible部署指导详见[ansible部署文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/README.md)

### 修改配置文件

需要修改`omni_infer_inventory_used_for_xx.yml`和 `omni_infer_server_template_xx.yml` 两处配置文件，位于`omniinfer/tools/ansible/templete/`路径下。以1P1D为例(4机组P,4机组D):

1. **tools/ansible/template/omni_infer_inventory_used_for_1P32_1D32.yml**

   将`p/d/c`下面的`ansible_host` 与 `host_ip` 值改为对应的IP。<span style="color:red; font-weight:bold">对于多机组 D 的场景，所有 D 节点的 `host_ip` 为主节点 d0 的 IP。</span>


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
           host_ip: "127.0.0.1"  # P0节点的IP, 即 P 节点的主节点 IP
           ...
        ...

     D:
       hosts:
         d0:
           ansible_host: "127.0.0.5"  # D0 节点的IP
           ...
           host_ip: "127.0.0.5"       # D0 节点的IP
           ...

         d1:
           ansible_host: "127.0.0.6"  # D1 节点的IP
           ...
           host_ip: "127.0.0.5"       # D0 节点的IP, 即 D 节点的主节点 IP
           ...
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

2. **tools/ansible/template/omni_infer_server_template_a3_ds.yml**

    修改以下环境变量
    ```yaml
    environment:
        # Global Configuration
        LOG_PATH: "/data/log_path"  # 服务日志路径
        MODEL_PATH: "/data/models/origin/bf16"  # 模型文件路径
        LOG_PATH_IN_EXECUTOR: "/data/log_path_in_executor"
        CODE_PATH: "/data/local_code_path"  # [可选配置]omniinfer本地代码路径
        HTTP_PROXY: ""  # 下载nginx的HTTP代理地址，如果不需要代理可以留空

        # Configuration for containers
        DOCKER_IMAGE_ID: "REPOSITORY:TAG" # 镜像与标签
        DOCKER_NAME_P: "you_name_omni_infer_prefill" # P容器名称
        DOCKER_NAME_D: "you_name_omni_infer_decode"  # D容器名称
        DOCKER_NAME_C: "you_name_omni_infer_proxy"   # Proxy 容器名称
    ```

配置文件详细解释说明请参考[文档](https://gitee.com/omniai/omniinfer/blob/master/tools/ansible/template/README.md#%E7%9B%B8%E5%85%B3%E6%96%87%E4%BB%B6%E8%A7%A3%E9%87%8A%E8%AF%B4%E6%98%8E)。

### 执行命令

```bash
cd omniinfer/tools/ansible/template
ansible-playbook -i omni_infer_inventory_1p32_1d32.yml omni_infer_server_template_a3_ds.yml
```

### 服务拉起成功
查看服务启动日志

### curl 测试

拉起成功后，可以通过curl命令进行测试：

```bash
curl -X POST http://127.0.0.1:7000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "temperature": 0,
    "max_tokens": 50,
    "prompt": "how are you?",
    "stream": true,
    "stream_options": {
      "include_usage": true,
      "continuous_usage_stats": true
    }
  }'
```