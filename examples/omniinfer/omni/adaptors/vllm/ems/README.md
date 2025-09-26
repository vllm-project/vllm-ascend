# ems
## 简介
弹性内存存储（Elastic Memory Service，EMS）是一种以DRAM内存（动态随机存取存储器）为主要存储介质的云基础设施服务，为LLM推理提供缓存和推理加速。EMS实现AI服务器的分布式内存池化管理，将LLM推理场景下多轮对话及公共前缀等历史KVCache缓存到EMS内存存储中，通过以存代算，减少了冗余计算，提升推理吞吐量，大幅节省AI推理算力资源，同时可降低推理首Token时延（Time To First Token，TTFT），提升LLM推理对话体验。
更多介绍请查看[华为云-弹性内存存储 EMS](https://support.huaweicloud.com/productdesc-ems/ems_01_0100.html)。

## 在omni-infer中使用
### 环境准备
1. 部署好ems集群（参照[华为云-弹性内存存储 EMS](https://support.huaweicloud.com/productdesc-ems/ems_01_0100.html)申请公测)。
2. 确保将宿主机EMS服务端容器共享的unix domain socket目录"/mnt/paas/kubernetes/kubelet/ems"，通过增加负载配置文件hostPath项，将目录映射到推理容器目录："/dev/shm/ems"；同时在推理容器内，运行服务的用户能够读写该文件夹及其文件。
3. 在推理容器内运行pip install ems-.*linux_aarch64.whl命令执行安装对应版本的python sdk。
设置环境变量
在`tools/ansible/template/omni_infer_server_template.yml` 中的`run_vllm_server_prefill_cmd`下添加EMS相关环境变量。

| 变量名称 | 变量类型 | 描述 |
|----|----|-----|
| ENABLE_VLLM_EMS |	int | 参数解释：<br>开启EMS以存带算。<br><br>约束限制：<br>必须为数字。<br><br>取值范围：<br>“1”：开启。<br>其他：关闭。<br><br>默认取值：<br>“0”。|
| EMS_STORE_LOCAL | int | 参数解释：<br>KV Cache在ems内存池中是本地存储还是远端存储。<br><br>约束限制：<br>必须为数字。<br><br>取值范围：<br>“1”：本地存储。<br>其他：远端存储。<br><br>默认取值：<br>“0”。
| MODEL_ID | string | 参数解释：<br>唯一标识当前推理服务使用的推理模型ID。<br><br>约束限制：<br>1. 1～512个字符，支持数字、小写字母、“.”、“-”、“_”。<br>2. 需要保证全局唯一。<br><br>默认取值：<br>“cc_kvstore@_@ds_default_ns_001”。
| ACCELERATE_ID | string | 参数解释：<br>业务访问内存池身份凭证，由用户自行指定并保证唯一性，在需要进行业务多租隔离场景使用。<br><br>约束限制：<br>1. 1～512个字符，支持数字、小写字母、“.”、“-”、“_”。<br>2. 需要保证全局唯一。<br><br>默认取值：<br>“access_id”。