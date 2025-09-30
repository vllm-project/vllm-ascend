# PD分离快速部署

----

注意：此教程仍然处于**实验阶段**，尚未合并到主干分支。其接口、功能和性能可能会在最终集成到vLLM和vLLM Ascend过程中发生变化。
更多合入计划及细节请参考：[RFC#3165: Contribute Omni-Infer feature to vLLM Ascend upstream](https://github.com/vllm-project/vllm-ascend/issues/3165)

----

本文档介绍如何快速拉起PD分离部署推理，支持8机1P1D。

## 硬件要求

**硬件：** Atlas 800IA3

**操作系统：**  Linux aarch64

**镜像版本：** quay.io/ascend/vllm-ascend:v0.9.0-dsv3.2

**驱动版本：** Ascend HDK 25.2.1 
`npu-smi info` 检查Ascend NPU固件和驱动是否正确安装。
下载链接：https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/266220744?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743

**网络状态检查：** 
1、使用ssh命令确认机器互连。
2、服务机器ROCE面联通

## 模型准备

权重下载，https://gitee.com/omniai/omniinfer/releases/tag/v0.4.2 下载BF16类型的权重，本地权重路径与下文中 omni_infer_server_template_a3_ds.yml的配置模型文件路径保持一致，默认/data/models/origin/bf16

## 部署

### 环境准备

用户执行机需要安装ansible
```bash
yum install ansible
yum install openssh-server
```

目标机安装libselinux-python3
```bash
yum install libselinux-python3
```


### 修改配置文件

以1P1D为例(4机组P,4机组D)，需要修改`omni_infer_server_template_a3_ds.yml`和 `omni_infer_inventory_used_for_1P32_1D32.yml` 两处配置文件，
文件获取地址:
https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/omni_infer_server_template_a3_ds.yml
https://github.com/vllm-project/vllm-ascend/blob/v0.9.1-dev/examples/omni_infer_inventory_used_for_1P32_1D32.yml

1. **omni_infer_inventory_used_for_1P32_1D32.yml**

   将`p/d/c`下面的`ansible_host` 与 `host_ip` 值改为对应的IP。对于多机组 D 的场景，所有 D 节点的 `host_ip` 为主节点 d0 的 IP。


   ```YAML
    all:
      vars:
        ...
        ansible_ssh_private_key_file: /path/to/key.pem  # 私钥文件路径
        ...

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

  生成私钥文件。将ansible_ssh_private_key_file:修改为私钥文件路径：

  ```bash
  # 首先在执行机生成密钥对:
  ssh-keygen -t ed25519 -C "Your SSH key comment" -f ~/.ssh/my_key  # -t 指定密钥类型（推荐ed25519）， -f 指定文件名
  # 密钥文件默认存放位置为: 私钥：~/.ssh/id_ed25519 公钥：~/.ssh/id_ed25519.pub. 设置密钥文件权限:
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/my_key   # 私钥必须设为 600
  chmod 644 ~/.ssh/my_key.pub
  # 部署公钥到远程目标机:以下例子是通过密码去传输密钥文件到远程目标机
  ssh-copy-id -i ~/.ssh/my_key.pub user@remote-host
  ```


2. **omni_infer_server_template_a3_ds.yml**

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
ansible-playbook -i omni_infer_inventory_1p32_1d32.yml omni_infer_server_template_a3_ds.yml
```
提示：建议起服务前清理一下全部节点的环境，例如：
```bash
ps aux | grep "python" | grep -v "grep" | awk '{print $2}' | xargs kill -9
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