# OmniPlacement Guideline

为解决MoE模型推理中的专家负载不均衡问题，本系统研究提出了一种创新的负载均衡算法OmniPlacement。该算法通过专家重排列、层间冗余部署以及近实时动态调度等技术手段，显著提升了MoE模型的推理效率。本文主要介绍如何部署、如何启用、如何配置专家均衡的基本操作。

---

## 1. 功能说明

### 1.1 功能目标

- 支持MoE专家部署排布优化
- 支持MoE专家冗余部署
- 提供动态负载均衡策略和部署优化
- 支持采集MoE专家激活数据并生成静态部署pattern

### 1.2 约束条件

- 推理引擎需部署在 NPU/GPU 节点（Ascend/CUDA）
- 支持 safetensors、HF Transformers 格式模型
- 负载均衡策略需要采用PD分离部署
- 冗余部署和AllGather通信模式不支持同时开启

## 2. How to Deploy OmniPlacement
   ### 2.1 Overview
   OmniPlacement随OmniInfer交付，交付件包含3个部分，分别是：
   - omni_infer-xxxxx.whl  (xxxxx与版本号、系统架构相关，例如omni_infer-0.4.0-cp311-cp311-linux_aarch64.whl)
   - Config.yaml (特性配置文件，通常prefill节点和Decode节点分别配置)
   - Placement pattern

   ### 2.2 Prerequisites
   同OmniInfer配置要求

   ### 2.3 Deployment Steps
   - **Step 1: Build OmniPlacement wheel package** （如果OmniPlacement已随[OmniInfer部署](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md)安装好，可跳过此步骤）
      - 登录任意推理节点，例如prefill0
      - 进入docker容器
      ```
      [root@devserver placement]# docker exec -itu root w_omni_infer_prefill_p0 bash

         Welcome to 5.10.0-182.0.0.95.r1941_123.hce2.aarch64

         System information as of time:  Tue Jul  8 10:19:30 CST 2025

         System load:    33.04
         Processes:      6
         Memory used:    .5%
         Swap used:      0.0%
         Usage On:       19%
         Users online:   0


         [root@devserver-hps-91b9def8-g00615224-00016 data]# 
      ```
      - 在omni_infer代码目录中，进入到Placement代码
      ```
         [root@devserver data]# cd omni_infer/omni/accelerators/placement/
         [root@devserver  placement]#  
      ``` 
      - 编译wheel包，wheel包生成结果在dist目录中
      ```
         [root@devserver placement]# python setup.py bdist_wheel
         [root@devserver placement]# ll dist/
         -rw-r--r-- 1 root root 1.3M Jul  8 10:27 omni_placement-0.7.2.2a3-cp311-cp311-linux_aarch64.whl
      ```       
   - **Step 2: Install wheel package** （如果OmniPlacement已随[OmniInfer部署](https://gitee.com/omniai/omniinfer/blob/master/docs/omni_infer_installation_guide.md)安装好，可跳过此步骤）
      - 修改ansible启动任务，如在omni_infer_server.yml中的docker_update_prefill_code_cmd和docker_update_decode_code_cmd任务项，添加omni_placement wheel的安装:
      
          docker_update_prefill_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_P /bin/bash -c '`pip uninstall omni_placement -y && cd /workspace/omni_infer/omni/accelerators/placement/dist/ && pip install omni_placement-0.7.2.2a3-cp311-cp311-linux_aarch64.whl` && cd /workspace/omni_infer/infer_engines && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . '" 

         docker_update_decode_code_cmd: "{{ docker_exec_cmd }} $DOCKER_NAME_D /bin/bash -c '`pip uninstall omni_placement -y && cd /workspace/omni_infer/omni/accelerators/placement/dist/ && pip install omni_placement-0.7.2.2a3-cp311-cp311-linux_aarch64.whl` && cd /workspace/omni_infer/infer_engines && cd vllm && SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . && cd ../../ && pip install -e . '"
      
   - **Step 3: Install config.yaml and pattern**
      - 从placement拷贝样例config到运行环境
      ```
         [root@devserver placement]# cp config.yaml ../../../tests/test_config/config_p.yaml
         [root@devserver placement]# cp config.yaml ../../../tests/test_config/config_d.yaml
      ```
      - 从placement拷贝basepattern到运行环境
      ```      
         [root@devserver placement]# cp -r patterns/base_patterns/ ../../../tests/test_config/
      ```


## 2. How to Enable OmniPlacement
   ### 2.1 Enabling Process
   - **Option 1: D侧配置omni infer Configuration File**
      - Locate and edit omni infer configuration file (e.g., `omni_infer/tests/test_config/test_config_decode.json`)
      - Set `use_omni_placement: true`
      - Set `omni_placement_config_path: "/workspace/omni_infer/tests/test_config/config_d.yaml"`
      - Locate and edit omni placement configuration file (e.g., `/workspace/omni_infer/tests/test_config/config_d.yaml`)
      - Set  `pattern_path: "/workspace/omni_infer/tests/test_config/base_patterns/DSV3_baseline_64_devices_58_MoE_Layers.npy"`
      ```  
       注意：
       1.pattern file需要根据实际的物理部署匹配，上面例子中的DSV3_baseline_64_devices_58_MoE_Layers.npy仅使用64die的D集群，更多pattern file参考base_patterns目录
       $ ll base_patterns/
      -rw-r--r-- 1 root 1049089  7602304 Jul  7 10:52 DSV3_baseline_128_devices_58_MoE_Layers.npy
      -rw-r--r-- 1 root 1049089  1900672 Jul  7 10:52 DSV3_baseline_16_devices_58_MoE_Layers.npy
      -rw-r--r-- 1 root 1049089 15204480 Jul  7 10:52 DSV3_baseline_256_devices_58_MoE_Layers.npy
      -rw-r--r-- 1 root 1049089  3801216 Jul  7 10:52 DSV3_baseline_32_devices_58_MoE_Layers.npy
      -rw-r--r-- 1 root 1049089  7602304 Jul  7 10:52 DSV3_baseline_64_devices_58_MoE_Layers.npy
      -rw-r--r-- 1 root 1049089   950400 Jul  7 10:52 DSV3_baseline_8_devices_58_MoE_Layers.npy
      2.pattern_path的地址使用绝对路径
      ``` 
      ```  
       推荐：
       不设置pattern_path，或设置pattern_path: null，将自动匹配模型结构和物理部署形态，生成默认的base_pattern。
      ```
   - **Option 2: P侧配置omni infer Configuration File**
      - Locate and edit omni infer configuration file (e.g., `omni_infer/tests/test_config/test_config_prefill.json`)
      - Set `use_omni_placement: true`
      - Set `omni_placement_config_path: "/workspace/omni_infer/tests/test_config/config_p.yaml"`
      - Locate and edit omni placement configuration file (e.g., `/workspace/omni_infer/tests/test_config/config_p.yaml`)
      - Set  `pattern_path: "/workspace/omni_infer/tests/test_config/base_patterns/DSV3_baseline_16_devices_58_MoE_Layers.npy"`
      ```  
       推荐：
       不设置pattern_path，或设置pattern_path: null，将自动匹配模型结构和物理部署形态，生成默认的base_pattern。
      ```

## 3. How to Configure Expert Load Balancing
   ### 3.1 Overview
   - 专家均衡支撑静态和动态两种模式
   - 静态专家均衡分为dump专家激活数据、生成部署文件(pattern file)和应用部署文件三个大的步骤

   ```  
      推荐：
      推荐使用专家动态均衡模式。
   ```
   ### 3.2 Prerequisites
   已完成OmniPlacement的安装和启用

   ### 3.3 静态专家均衡Configuration Steps
   - **Step 1: Dump expert activations**
      - Locate and edit omni placement configuration file (e.g., `/workspace/omni_infer/tests/test_config/config_d.yaml`)
      - Set `enable_dump: true`
      - Set `dump_dir: "/home/profiling/dump_data"`
      ```  
       注意：dump_dir可以按照实际情况设置，P侧config_p.yaml做相同设置
      ``` 
   - **Step 2: Run the inference service and Send request**
      - 等待请求处理完成后，立即收集dump数据
      - dump出的数据生成在各个服务器的dump_dir内，文件夹内部会按时间戳生成文件夹，每次dump的数据会存放在对应的时间戳文件夹中。
      - 把Decode节点0服务器的dump文件夹取出（仅在路由专家节点dump数据），每个时间戳文件夹内都是一系列txt文件，所有Decoder的数据在1个文件夹内，地址标记为Decode_path
      - 把Prefill节点服务器的prefill文件夹取出，把每个prefill文件夹修改名字，并排放到某个文件夹下面，地址我们记录为P0_path，P1_path,…
   - **Step 3: 生成静态部署文件**
      - 切换到pattern工具目录 (e.g., omni_infer/omni/accelerators/placement/utils/omni_pattern_tool/)
      - run_pipeline.sh提供了从统计数据到生成pattern到分析pattern收益的流水线脚本，具体设置请参考 https://gitee.com/omniai/omniinfer/blob/master/omni/accelerators/placement/utils/omni_pattern_tool/Readme.md
      - 一个简单的运行模式为：
      ```
         ./run_pipeline.sh --input_txt_folders "/data/expert_activation/decode_data"  --num_ranks_target_pattern 256 --collecting_modes decode
      ``` 
      ```           
         参数介绍：
         --input_txt_folders：dump数据存放的文件夹，支持多个文件夹，每个用空格分隔：“path/prfill0/prefill” “path/prfill1/prefill” 。
         --num_ranks_target_pattern:加载pattern服务器的die数，例如pattern运行在256 die的实例上，则设置为256。
         --collecting_modes：decode或者prefill。
      ```
      - 默认会生成两个pattern：文件在./placement_pattern文件夹下面：重排的pattern带有rearrange字段，冗余的pattern带有redundant字段。
      ``` 
      $ ll -t placement_pattern
      -rw-r--r-- 1 root 1049089  3801216 Jun 19 09:53 placement_pattern_20250619_095310_58_redundant_layers_58_layers_64_ranks_epmaxdeploy_301_decode.npy
      -rw-r--r-- 1 root 1049089  3801216 Jun 19 09:53 placement_pattern_20250619_095310_58_rearrange_layers_58_layers_64_ranks_decode.npy
      ``` 
   - **Step 4: 应用部署文件**
      - Locate and edit omni placement configuration file (e.g., /workspace/omni_infer/tests/test_config/config_p.yaml)
     - Set `enable_dump: false`      
      - Set `pattern_path: "/workspace/omni_infer/tests/test_config/placement_pattern_20250619_093815_58_rearrange_layers_58_layers_16_ranks_prefill.npy"`
      ```  
       推荐：使用step3生成的部署文件，D侧config_d.yaml做相同设置，推荐32die及以下使用rearrange pattern，32die以上使用redundant pattern
      ``` 
   - **Step 5: 重启推理服务**
      - 观察均衡表现并调优

   ### 3.4 动态专家重排Configuration Steps
   - Locate and edit omni placement configuration file (e.g., `/workspace/omni_infer/tests/test_config/config_d.yaml`)
   - Set `enable_dynamic: True`
   - Set `max_redundant_per_expert: 1`
   - Set `max_redundant_per_rank: 0`
   - 启动推理服务，观察均衡表现并调优

   ### 3.5  动态专家冗余Configuration Steps
   - Locate and edit omni placement configuration file (e.g., `/workspace/omni_infer/tests/test_config/config_d.yaml`)
   - Set `enable_dynamic: True`
   - Set `max_redundant_per_expert: 10`
   - Set `max_redundant_per_rank: 1`
   ```           
      参数介绍：
      --max_redundant_per_expert：每个专家的冗余次数上限，典型值10，最大不超过总die数
      --max_redundant_per_rank:每个die上可以部署的冗余专家个数，典型值1
   ```
   - Tune parameters based on performance
## 4. Additional Resources
   - Community forums or support channels