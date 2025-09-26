# omni_cli 使用文档

## 安装

```bash
cd omniinfer/tools; python -m build --wheel      # 无wheel包，源码安装
pip install omni_cli***.wheel --force-reinstall --no-build-isolation --no-deps
```

## 快速开始

omni_cli 会在命令执行路径创建配置文件，并依赖该配置文件。不同用户、模型请在不同路径下执行。add_node第一次执行时会创建该配置文件。以下内容以deepseek 1p1d 三机部署为例，使用镜像容器中已经安装的omniinfer和vllm，依赖于容器中/workspace/omniinfer的拉起脚本。

1. 添加节点

   ```bash
   omni_cli add_node --role C --name c0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   omni_cli add_node --role P --name p0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   omni_cli add_node --role D --name d0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   omni_cli add_node --role D --name d1 --host_ip 本机IP --master-node 主D节点 --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像
   ```

2. 修改配置

   omni_cli具有推断配置的功能。因为不同模型、不同场景参数差别巨大，需要先配置模型路径，omni_cli依据路径名称推断模型，并提供部分场景默认配置。

   ```bash
   # 配置所有节点，支持按照Role、节点名称分别配置，参见下文omni_cli cfg详细介绍
   omni_cli cfg --set all env --MODEL_PATH 模型路径
   omni_cli cfg --set all env --LOG_PATH 日志路径
   omni_cli cfg --set all --container_name_prefix  容器名称前缀 # 依据前缀生成所有容器名称，如 prefix_prefill_p0
   ```

3. 服务拉起

   ```bash
   omni_cli run_docker  # 如需使用指定容器，必须跳过此步
   omni_cli start       # 服务拉起，包括ranktable生成、global proxy拉起
   omni_cli stop        # 服务停止
   ```

## 命令列表

   ```bash
   # 节点管理
   omni_cli ls
   omni_cli add_node
   omni_cli rm_node

   # 配置管理
   omni_cli cfg
   omni_cli inspect

   # 服务管理
   omni_cli run_docker
   omni_cli start
   omni_cli stop

   # 开发工具
   omni_cli upgrade     # 开发中
   omni_cli collect_log # 开发中
   omni_cli install_code
   omni_cli sync_code
   ```

### 节点管理

1. omni_cli ls

   功能：输出当前的节点，包括role, name, IP。

   使用示例：

   ```bash
   omni_cli ls
   ```

   效果展示：

   ```
   Role  | Name  | IP Address
   ------------------------------
   C     | c0    | 127.0.0.1
   D     | d0    | 127.0.0.2
   D     | d1    | 127.0.0.3
   P     | p0    | 127.0.0.1
   ------------------------------

   ```
2. omni_cli add_node

   功能：在当前工作区初始化或修改server_profiles.yml文件并向其添加节点。后续支持动态扩缩容

   参数说明：

   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择，分别代表Prefill，Decode 和 Global Proxy
   >
   > --name：节点名称（必需）
   >
   > --host_ip：主机IP地址（必需）
   >
   > --ssh_private_key_file：SSH私钥文件路径（必需）
   >
   > --docker_image_id：Docker镜像ID（必需）
   >
   > --user：SSH用户名，默认为 "root"
   >
   > --ssh_common_args：SSH通用参数，默认为"-o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
   >
   >  --master-node：主节点名称，默认为当前添加的节点自己，用于多机组服务。主节点无需设置，从节点需要配置主节点，例如，d0.

   使用示例：

   ```bash
   omni_cli add_node --role P --name p0 --host_ip 127.0.0.1 \
   --ssh_private_key_file /path/to/my/key --docker_image_id myapp:latest
   ```

3. omni_cli rm_node

   功能：从配置文件中删除指定节点。后续支持动态扩缩容

   参数说明：

   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择
   >
   > --name：要删除的节点名称（必需）
   >
   > --deploy_path：配置文件路径，默认为 $pwd/server_profiles.yml

    使用示例：

   ```bash
   omni_cli rm_node --role P --name p0
   ```

### 配置管理

1. omni_cli cfg

   命令执行时会检测当前配置文件中MODEL_PATH是否已经配置，如果没有配置，则拒绝生效，要求先配置MODEL_PATH。

   功能：修改或者删除节点的环境变量和参数。可以对所有节点或者某组节点批量修改或删除配置，也可以只针对某个节点修改或删除配置。

   使用示例：
   ```bash

   # NAME 可以是 all(所有节点)， p/d/c(按照Role)，p0/d0/c0(按照名称)

   # 配置设置
   #
   # 环境变量
   omni_cli cfg --set NAME env --MODEL_PATH /data/models/DeepSeek-R1-w8a8-fusion
   # 一般 args
   omni_cli cfg --set NAME args --num-servers 32
   # extra-args
   omni_cli cfg --set NAME args --extra-args '--max-num-batched-tokens 30000 --max-num-seqs 32'
   # additional-config
   omni_cli cfg --set NAME args --additional-config '--graph_model_compile_config level:1'
   # 单独支持 容器名称、镜像、visible device 配置
   omni_cli cfg --set NAME --container_name_prefix docker_name # 设置容器名前缀
   omni_cli cfg --set NAME --container_name docker_name # 设置容器名
   omni_cli cfg --set NAME --DOCKER_IMAGE_ID swr.cn-east-4.myhuaweicloud.com/omni-ci/omni_infer-a3-arm:master-202508191159-daily # 设置镜像 ID
   omni_cli cfg --set NAME --ascend_rt_visible_devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 # 设置卡号

   # 配置删除
   #
   # 环境变量
   omni_cli cfg --delete NAME env --MODEL_PATH
   # 一般 args
   omni_cli cfg --delete NAME args --num-servers
   # extra-args
   omni_cli cfg --delete NAME args --extra-args '--max-num-batched-tokens --max-num-seqs'
   # additional-config
   omni_cli cfg --delete NAME args --additional-config '--graph_model_compile_config'
   # 单独支持 容器名称、镜像、visible device 配置
   omni_cli cfg --delete NAME --container_name
   omni_cli cfg --delete NAME --DOCKER_IMAGE_ID
   omni_cli cfg --delete NAME --ascend_rt_visible_devices

   ```

2. omni_cli inspect

   功能：根据名称，输出该节点所有配置信息

   参数说明：
   > name: 节点名称，可通过omni_cli ls查看

   使用示例:
   ``` bash
   omni_cli inspect p0
   ```

### 服务管理

1. omni_cli run_docker

   功能：根据 add_node、cfg配置，启动所有容器。如果已有同名容器，会将同名容器销毁，重新创建

   参数说明：

   > --help: 帮助信息
   >
   > --config_path：配置文件路径，默认为 $pwd/server_profiles.yml
   >
   > --dry-run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli run_docker --config_path test.yml
   ```

2. omni_cli start

   功能：依据模板配置和配置推断，在已配置容器中，将推理服务拉起，包括ranktable生成、global proxy拉起。(ranktable生成是临时方案，动态扩缩容会去掉ranktable机制)。

   参数说明：

   > --help: 帮助信息
   >
   > --skip-verify-config:  跳过配置校验，避免配置推断修改用户指定配置
   >
   > --config_path:  使用指定omni_cli配置文件拉起服务，默认为 $pwd/server_profiles.yml
   >
   > --run_dev: 开发者快速拉起，跳过ranktable生成和global proxy拉起。需确保已经进行normal拉起，完成ranktable生成和global proxy拉起。

   使用示例：

   ```bash
   omni_cli start --config_path server_profiles.yml
   ```

3. omni_cli stop

   功能：在已配置容器中，将服务停止。注：不会停止配置容器之外的服务

   参数说明:

   > --help: 帮助信息

   使用示例：

   ```bash
   omni_cli stop
   ```

### 开发工具

1. omni_cli upgrade

    功能：执行安装流程，将omniinfer和global proxy安装包安装进入指定容器中。

    参数说明：

   > --help: 帮助信息
   >
   > --omni_package:  需要安装的omniinfer包
   >
   > --vllm_package:  需要安装的vllm包
   >
   > --proxy_package: 需要安装的Global Proxy包

    使用示例：

   ``` bash
   omni_cli upgrade --omni_package XXXXX.whl  --vllm_package XXXXX.whl --proxy_package XXXXX.rpm
   ```

2. omni_cli collect_log

    功能：将日志从各个容器中收集到执行机路径。

    参数说明：

   > --name: 指定需要获取的日志，all, P, D, C分别代表所有日志、所有P日志、所有D日志、Proxy日志

    使用示例：

   ```bash
   omni_cli collect_log --name all
   ```

3. omni_cli install_code

    功能：执行安装流程，将容器中/workspace/omniinfer代码采用editable模式安装，替换容器中自带omni_infer和vllm。对于同一个容器，只需执行一次，后续仅同步代码 sync_code 即可

    参数说明：

   > --help: 帮助信息
   >
   > --deploy_path：配置文件路径，默认为 $pwd/server_profiles.yml
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli install_code
   ```

4. omni_cli sync_code --code_path

    功能：将指定路径源代码同步到所有节点对应的容器中的/workspace/omniinfer。

    参数说明：

   > --help: 帮助信息
   >
   > --code_path：要同步的代码路径（必需）。
   >
   > --deploy_path：配置文件路径，默认为 $pwd/server_profiles.yml
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli sync_code --code_path /path/to/my/code   # 路径会被补全为 /path/to/my/code/omniinfer
   ```

## 开发计划
1. cfg 支持按照特性级别配置，并检查单个服务内配置是否冲突、互斥
2. 当前设计所有模型、场景默认/最优配置在同一个文件tools/omni_cli/configs/default_profiles.yml，后续要将不同模型最优配置拆分为独立文件
3. 拉起服务调用start命令，需要校验多个服务间的配置是否有差异，防止不同服务特性配置不一致
4. start命令要支持单独拉P/D/C
5. ls命令需要返回容器状态
6. start命令要检查服务是否拉起，以进度条方式展示
7. 完成 omni_cli upgrade
8. 完成 omni_cli collect_log