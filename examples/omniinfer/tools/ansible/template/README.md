[TOC]

# 一、脚本介绍
本目录 (tools/ansible/template) 下的脚本实现一键式部署多机PD分离服务和 Global Proxy 服务， 用户只需按要求进行部分配置的修改， 即可实现容器自动启动， 自动生成全局 ranktable 以及拉起 vllm 服务和 Proxy 服务。
脚本文件目录结构如下：
```bash
└── template
    ├── omni_infer_inventory_used_for_2P1D.yml
    ├── omni_infer_inventory_used_for_4P1D.yml
    ├── omni_infer_server_template.yml
    └── README.md
```

# 二、相关文件解释说明
## omni_infer_inventory_used_for_2P1D.yml 和 omni_infer_inventory_used_for_4P1D.yml
两个文件一个用于 2P1D 四机场景，一个用于 4P1D 八机场景，都是用于定义被管理目标机的配置信息；文件中参数配置说明如下:

* `ansible_user`: 远程目标机的用户名， 如 user 等。

* `ansible_ssh_private_key_file`: 连接目标机的私钥文件路径。也可以使用密码登录目标机的方式， 则使用 `ansible_password` 字段并将密码填入， 如：`ansible_password: "password"` 。

* `global_port_base`: 基础端口号， 用于作为 `master-port` 的基准。

* `base_api_port`:  API 服务的基础端口号， 用于作为 `base-api-port` 的基准。

* `proxy_port`: global proxy 的监听基础端口号， Global Proxy 实例的端口： `proxy_port + node_rank`

* `port_offset`: 区分 Prefill 实例端口范围和 Decode 实例的端口范围， 避免在同一节点拉起时端口冲突 （多机场景可以忽略冲突问题）。

* `ansible_host`: 目标机的 IP 。

* `node_rank`: 用于多机组 P/D 的场景，多机的节点排序，主 P/D 节点的 node_rank 为0，其他节点的顺序值在此基础上依次递增。

* `kv_rank`: 用于 prefill 实例 kv_rank 的区分，索引从0开始。如果是多台机器组 P，这些机器的 `kv_rank` 的值需保持一致。

* `node_port`: 即 Prefill 和 Decode 的实际 `master-port`。
    Prefill 实例的默认端口: `global_port_base + port_offset.P + node_rank`。
    Decode 实例的默认端口: `global_port_base + port_offset.D`。

* `api_port`: 多 API Server 的端口号。
    Prefill 实例的 API Server 默认端口: `base_api_port + port_offset.P + node_rank`。
    Decode 实例的 API Server 默认端口: `base_api_port + port_offset.D + node_rank`。

* `host_ip`: 组成 Prefill 和 Decode 实例的主节点 IP。对于多机组 P/D 的场景，该 IP 就是主 P 或主 D 的 IP；对于单机组 P 的场景，该 IP 和 `ansible_host` 的值保持一致。

* `ascend_rt_visible_devices`: 每个 Prefill 或 Decode 实例需要使用的卡号， 参数值需要严格按照以下格式: `"x,x,x,x"` (用英文逗号分隔的连续值) ， 不能有多余逗号和空格。


## omni_infer_server_template.yml
该文件作用是管理目标节点执行相应的任务；文件中参数配置说明如下:

* `LOG_PATH`: Decode/Prefill/Global Proxy 实例日志的存放的路径。

* `MODEL_PATH`: 加载的模型路径， 要求 Prefill 和 Decode 所有实例所在的节点提前拷贝好模型并且模型路径保持一致。

* `MODEL_LEN_MAX_PREFILL`: Prefill 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为30000。

* `MODEL_LEN_MAX_DECODE`: Decode 侧模型的最大生成长度， 包含 prompt 长度和 generated 长度， 默认值为16384。

* `LOG_PATH_IN_EXECUTOR`: 执行机上存放 Decode/Prefill/Global Proxy 实例日志的路径。

* `CODE_PATH`: 执行机上的 omniinfer 源码路径，即用户通过 `git clone` 拉取的代码存放路径；例如你在 `/workspace/local_code_path` 下 git clone 了代码，那路径就是 `/workspace/local_code_path`；脚本会将执行机上的源码同步到目标机内相同路径下，并且将路径挂载到容器中。

* `HTTP_PROXY`: 下载 nginx 的 HTTP 代理地址，例如 "http://your.proxy:port"，如果不需要代理可以留空。

* `DOCKER_IMAGE_ID`: omniai 服务实例均在容器里面运行， 用来指定运行的容器镜像， 如: `swr.cn-southwest-2.myhuaweicloud.com/omni-ai/omniinfer:202506272026`，其中 `swr.cn-southwest-2.myhuaweicloud.com/omni-ai/omniinfer` 表示镜像仓地址，`202506272026` 表示镜像版本号，如果远程目标机没有此容器镜像，脚本会自动下载。

* `DOCKER_NAME_P`: Prefill 节点的容器别名， 默认前缀 `omni_infer_prefill_`。 如 `omni_infer_prefill_p0` 表示 Prefill 第一个实例的容器名。

* `DOCKER_NAME_D`: Decode 节点的容器别名， 默认前缀 `omni_infer_decode_`。 如 `omni_infer_prefill_d0` 表示 Decode 第一个实例的容器名。

* `DOCKER_NAME_C`: Proxy 节点的容器别名， 默认前缀 `omni_infer_proxy_`。 如 `omni_infer_proxy_c0` 表示 Proxy 第一个实例的容器名。

* `SCRIPTS_PATH`: ansible 运行过程中自动生成的脚本文件存放路径， 用户可以不关注， ansible 执行过程中会自动生成。

* `DECODE_TENSOR_PARALLEL_SIZE`: Decode 实例 tensor parallel 参数 tp， 默认值为1。

* `ranktable_save_path`: ansible 运行过程中， Prefill 和 Decode 实例对应的 ranktable 文件以及它们合并生成的 ranktable 文件存放路径。

# 三、环境准备
## 在执行机安装 ansible-playbook
```bash
# 安装ansible-playbook
yum install ansible
```

## 在执行机安装 sshpass
执行 ansible 依赖 sshpass 链接各个目标机，做远程机器管理
```bash
yum install openssh-server
```

## 密钥文件的准备
请注意，这里的密钥文件仅仅是用于执行机通过 ansible 去登录目标机，如果你已经有登录目标机的密钥文件，放在 omni_infer_inventory.yml 的 ansible_ssh_private_key_file 指定路径即可。
1. 首先在执行机生成密钥对:
    ```bash
    ssh-keygen -t ed25519 -C "Your SSH key comment" -f ~/.ssh/my_key  # -t 指定密钥类型（推荐ed25519）， -f 指定文件名
    ```
2. 密钥文件默认存放位置为: 私钥：~/.ssh/id_ed25519 公钥：~/.ssh/id_ed25519.pub. 设置密钥文件权限:
    ```bash
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/id_ed25519   # 私钥必须设为 600
    chmod 644 ~/.ssh/id_ed25519.pub
    ```
3. 部署公钥到远程目标机:
    ```bash
    # 以下例子是通过密码去传输密钥文件到远程目标机
    ssh-copy-id -i ~/.ssh/id_ed25519.pub user@remote-host
    ```

# 四、操作步骤

## 修改配置
在 **omni_infer_inventory_used_for_2P1D.yml 和 omni_infer_inventory_used_for_4P1D.yml** 中， 重点修改以下配置项 `ansible_user / ansible_ssh_private_key_file / ansible_host / node_rank / kv_rank / host_ip`;
在 **omni_infer_server_template.yml** 中， 重点修改以下配置项 `MODEL_PATH / DOCKER_IMAGE_ID / CODE_PATH / DOCKER_NAME_P / DOCKER_NAME_D / DOCKER_NAME_C`， 就可拉起 omniai 服务。
此外，建议修改 omni_infer_server_template.yml 中的 `LOG_PATH`、`LOG_PATH_IN_EXECUTOR`、`SCRIPTS_PATH` 和 `ranktable_save_path`，防止路径下的文件被其他人覆盖。

## 执行命令
```bash
# 拉取库上最新代码，比如执行机的 /data/local_code_path 路径下 clone 源码，执行下列命令即可
cd /data/local_code_path
git clone https://gitee.com/omniai/omniinfer.git
cd omniinfer/infer_engines/
git clone https://github.com/vllm-project/vllm.git 或者 git clone https://gitee.com/mirrors/vllm.git
# 进入到 ansible 脚本目录下
cd omniinfer/tools/ansible/template
# 如果是部署四机 2P1D，就执行如下命令：
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --skip-tags fetch_log
# 如果是部署八机 4P1D，就执行如下命令：
ansible-playbook -i omni_infer_inventory_used_for_4P1D.yml omni_infer_server_template.yml --skip-tags fetch_log

# 后续若修改了执行机上的源码，执行下列命令即可通过修改后的源码拉起服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags sync_code
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server

# 基于指定镜像创建并启动新的容器实例
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_docker
# 将执行机代码同步到目标机
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags sync_code
# 安装 omniinfer 相关包
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags pip_install
# 生成ranktable文件
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags ranktable
# 停止vllm以及nginx服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags stop_server
# 拉起vllm服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_server
# 拉起nginx服务
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags run_proxy
# 将日志存放在执行机指定路径
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template.yml --tags fetch_log
# 清理残余配置，主要是 stop 容器并且删除容器，以及删除 ranktable 和临时脚本
ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags clean_up
```

# 五、动态添加节点
## 特性介绍
在已有部署推理实例的情况下，为该实例动态新增推理节点，例如在已有 1P1D 实例，新增一个 Prefill 节点使该实例升级为 2P1D，这个过程中无需实例所有节点重启，除 nginx 重新连接导致服务断开几秒之外，实例服务其他大部分时间下都是可用状态。

该特性可有效缓解在高并发场景下，由于节点性能达到瓶颈造成的推理实例请求阻塞问题。

使用该特性需要注意以下几点：
1. 新增节点和已有节点必须在同一超节点内；
2. 该特性可以使用的前提，是**原推理实例的启动配置中，KV_CONNECTOR 必须为 "LMCacheConnectorV1"**，即使用 LMCache + Mooncake Store 的组合来管理KVCache；

## 操作步骤
假设现在已有一个 1P1D 的推理实例，使用 **omni_infer_inventory_used_for_1P1D.yml** 和 **omni_infer_server_template.yml** 两个脚本来启动，且启动时已将 KV_CONNECTOR 参数指定为"LMCacheConnectorV1"。将展示如何为该实例动态添加一个 Prefill 节点，使其变为 2P1D 的形态。

### 第一步：节点准备
1. 请参考本文中“密钥文件的准备”章节，为新节点同样配置好密钥；
2. 在和其他节点 `MODEL_PATH` 相同路径中拷贝好模型权重；

### 第二步：添加要新增的节点信息
**omni_infer_inventory_used_for_1P1D.yml** 脚本中当前仅存在一个 Prefill 节点 **p0** ，将新增节点信息添加为 **p1**。增加后的效果参考 **omni_infer_inventory_used_for_2P1D.yml**，可直接对比两个脚本查看差异。

由于我们修改好的节点配置文件内容已变为 2P1D，为防止混淆，下文该文件名称直接使用 **omni_infer_inventory_used_for_2P1D.yml**。

### 第三步：修改启动配置
请参考原实例启动所使用的脚本 **omni_infer_server_template.yml** 所修改的参数，对应修改 **omni_infer_server_add_node_template.yml**，配置参数一致即可。

### 第四步：执行命令
```bash
ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_add_node_template.yml
```
执行以上命令后，可观察到实例中原有节点状态不会变化，新增节点上服务开始启动并加载权重。启动过程中，nginx会短暂断开重连，导致服务有几秒中断，正在推理的请求可能会失败。

启动后，由于当前的 2P1D 实例，和直接使 **omni_infer_inventory_used_for_2P1D.yml**、**omni_infer_server_template.yml** 启动的服务并没有差异。如果需要修改代码或重启服务，可直接使用 **omni_infer_inventory_used_for_2P1D.yml**、**omni_infer_server_template.yml** 两个脚本，以及第四章提到的`--tags`方法。

本章节中所使用的 **omni_infer_server_add_node_template.yml**，仅作为动态添加节点使用，不支持第四章中提到的`--tags`能力。