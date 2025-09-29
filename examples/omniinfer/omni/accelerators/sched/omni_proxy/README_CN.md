# Omni Proxy：大模型高性能推理调度引擎

## 总体介绍：背景与目的

在大模型推理服务快速发展的背景下，高效智能的请求调度成为提升系统性能的关键。华为开源的Omni Infer推理加速套件将昇腾推理加速的最佳实践与开源推理框架深度结合，实现了业界领先的推理性能。作为该项目的重要组成部分，Omni Proxy承载着请求调度与资源优化的核心使命。
![输入图片说明](https://foruda.gitee.com/images/1758273466531869394/5ebedb26_14535041.png "屏幕截图")

Omni Proxy是Omni Infer开源项目的第二代请求调度引擎，基于Nginx构建并深度融合大模型推理特性。在Omni Infer 0.3.0版本中，Global Proxy带来了超过10%的推理性能提升，成为大模型推理部署的重要基础设施。Omni Proxy在继承第一代Global Proxy算法优势的基础上，针对大模型推理场景的特殊需求进行了全面优化，通过性能监控、智能调度和缓存优化等技术手段，实现了显著的性能提升。

## 大模型推理请求转发的特性与挑战

与传统Web服务相比，大模型推理请求具有独特的特征，这对调度引擎提出了全新的要求：

![输入图片说明](https://foruda.gitee.com/images/1758272945333186294/67f7c981_14535041.png "屏幕截图")
**周期性负载特征**
大模型推理过程呈现明显的周期性特征，模型在进行推理阶段一般都耗时较长，Prefill为秒级，Decode在几十毫秒级。在当前批次推理结束前，新进的请求无法进入推理batch。另一方面，模型的推理输出也表现出明显的周期性。从这些周期性的输出即可预测出模型实际的执行周期。

**性能感知缺失**
传统调度器缺乏对推理过程深层指标的感知能力，无法获取tokenize时间、推理引擎批次大小、调度周期、KVCache利用率等关键指标，导致调度决策缺乏数据支撑。

**KV Cache精准匹配难题**
字符串格式的prompt请求与推理节点的实际KV缓存状态之间存在映射鸿沟。调度器无法准确感知各节点的缓存内容，导致请求分配缺乏针对性，缓存命中率不及预期。

**冗余计算问题**
在多机PD分离部署架构下，Prefill和Decode节点分别进行相同的tokenizer处理，造成计算资源浪费和额外的延迟开销。

这些特性使得基于传统的调度机制在大模型推理场景下面临严峻挑战，迫切需要专为大模型推理设计的智能调度解决方案。

## Omni Proxy的创新方案与架构

### 全生命周期精细化监控
为了高效的对推理请求进行调度，达成Prefill资源池和Decode资源池高的利用和整体的负载均衡，我们从请求级调度引擎的视角对一个推理请求的生命周期进行了定义，以便于统计各个阶段的性能数据，作为调度的依据。

Omni Proxy将推理请求划分为10个精细化的生命周期阶段：
![Omni Proxy架构](https://foruda.gitee.com/images/1758272742382062862/5c8fea19_14535041.png )
1. **接收到请求**： 获得推理的请求体
2. **Tokenize阶段**： 请求prompt文本到token id列表的转换过程，包括基于给定模型的chat template进行的模板展开
3. **APC matching阶段**： KV Cache前缀缓存upstream寻优
4. **Prefill waiting阶段**： 预填充等待调度，基于各个upstream的调度周期卡点调度
5. **Prefill scheduled阶段**： 调度器已经完成对该请求的Prefill调度，等待Nginx的worker进程执行调度结果
6. **Prefill running阶段**： 预填充执行中
7. **Decode waiting阶段**： 调度器已经完成对该请求的Decode调度，等待Nginx的worker进程执行调度结果
8. **Decode scheduled阶段**： 解码已调度
9. **Decode running阶段**： 解码执行中
10. **请求完成**

每个阶段都设有精细的性能采集点，通过Nginx共享内存机制实现多worker间的数据同步，为智能调度提供全面数据支撑。

### 双模式调度策略支持
在PD分离的部署场景下，KV Cache需要从Prefill发送到Decode节点。Decode侧KV Cache所需要的KV Blocks空间的分配策略有不同的实现。Omni Proxy提供了sequential和parallel两种调度模式，分别适配vLLM和SGLang的两种现有调度模式。

#### Omni Proxy支持的sequential调度模式流程示意图
在先P后D的sequential模式中，Prefill先完成推理，随后请求发送给Decode，Decode按需分配接收用的blocks，在完成分配后从Prefill拉取KV Cache到目标blocks空间内。这种模式给了调度器延迟选择Decode节点的机会，可以根据最新的Decode节点的负载情况优化D节点的选择。但也存在拉取KV Cache的时间难以并行掩盖的不足。vLLM采用这种实现方式，Omni Infer中的Global Proxy目前也是基于这种方式工作的。

![输入图片说明](https://foruda.gitee.com/images/1758272995926540800/19ee79d4_14535041.png "屏幕截图")

#### Omni Proxy支持的parallel调度模式流程示意图
另一个方式是P和D同步选择的parallel模式，调度器基于P和D节点的负责同时选择Prefill和Decode节点，并在Decode提前预分配好KV传输用的目的blocks后才开始Prefill推理。这种模式下，目标KV Cache空间已经预分配，Prefill在推理的过程中可以按层把生成的KV推送到Decode侧，在完成Prefill后Decode节点可以立刻进入下一轮推理batch进行Decode推理。SGLang采用这种协同方式，Omni Proxy也适配了这种调度模式，并在nginx.conf文件中可以进行配置。

![输入图片说明](https://foruda.gitee.com/images/1758273012842686718/4fde89fc_14535041.png "屏幕截图")


### APC感知智能调度

![输入图片说明](https://foruda.gitee.com/images/1758273096387153975/446158d2_14535041.png "屏幕截图")
**KV缓存状态同步**
通过订阅vLLM推理引擎的KV Cache广播消息，Omni Proxy实时构建全局radix索引树，维护各推理节点的缓存状态视图。

**精准匹配机制**
集成tokenizer能力，基于配套模型的对话模板实现请求展开并生成token id后使用与目标推理框架相同的哈希算法生成block hash完成精准匹配，以达成高效的KV Cache复用。

**分布式缓存协同**
通过共享内存维护全局缓存状态，支持多worker协同工作，确保调度决策的一致性和准确性。

### Tokenizer结果复用优化

![输入图片说明](https://foruda.gitee.com/images/1758273066299850557/4c8df917_14535041.png "屏幕截图")
**预处理优化**
在请求转发前完成对话模板展开和tokenizer处理，生成模型推理所需的input_ids。

**结果复用机制**
将tokenizer结果附加到请求体中，避免了下游节点的重复计算。在Deepseek v3等多机PD分离部署场景下，可减少约30%的tokenizer开销。

### 基于负载与等待时间的批处理请求

Prefill 调度器的核心目标是，将等待队列中的新请求（Prefill-Waiting 阶段）智能且高效地分配给最合适的上游推理节点（Upstream）。调度策略旨在平衡多个目标：提升系统吞吐量、降低请求平均等待时间、避免节点过载，并对长请求（大 prompt）和长时间等待的请求做出动态响应。
![输入图片说明](https://foruda.gitee.com/images/1758283914521959454/61919d05_14535041.png "Prefill_sche (2).png")

**请求排序** 
在为请求分配节点之前，调度器首先会对等待队列中的所有请求进行加权排序，以决定优先为哪个请求服务，其权重由请求Token长度与等待时间相关，越小、等待时间越长的请求越先被分发。同时为了防止大 prompt 请求因为权重低而长时间得不到调度，引入了阈值机制使得等待时间过长的请求得以被处理。

**上游节点选择**
优先APC的匹配，基于此，进行第二层负载均衡的调度算法，即选取当前负载token数量最小的upstream进行分配，并添加过载保护机制，如果节点负载已满，并且当前请求并非因饥饿而获得高优先级，则会跳过本次分配，将请求保留在队列中等待下一个调度周期。这可以防止节点因接收过多请求而产生较长的不可控waiting时间，违背我们的调度目标。

**基于预测的精准调度**
在分发请求时，实时更新所有上游节点的预期调度时间，也即对于上游服务器的省域请求处理时间进行预测，只有当预期时间接近时才进行分发，与过载保护机制一起实现对于请求的精准分配，避免了服务侧请求堆积大量等待的情况。

### 分布式架构优化

**主从调度机制**
从多worker中选举产生主调度器，负责全局调度决策。调度结果通过共享内存同步，确保系统的高可用性和扩展性。

**性能数据聚合**
每个worker在本地采集性能指标，通过原子操作更新共享内存中的全局数据，避免锁竞争带来的性能开销。

![输入图片说明](https://foruda.gitee.com/images/1758273137521170136/dd5f3785_14535041.png "屏幕截图")

**无缝集成部署**
保持与标准Nginx的兼容性，支持动态模块加载，可无缝集成到现有的Nginx部署环境中。

## 总结与展望

Omni Proxy作为专为大模型推理设计的智能调度引擎，通过多项技术创新解决了传统调度系统在大模型场景下的局限性。其全生命周期监控、APC感知调度、Tokenizer复用和负载感知的PD协同调度等特性，显著提升了系统吞吐量和推理效率。

未来，Omni Proxy将继续在以下几个方向深化发展：首先，增强对多模型、混合负载、SLA保障等场景的支持，提供更细粒度的资源隔离和调度策略；其次，深化与硬件特性的结合，实现更极致的性能优化；最后，完善生态集成，提供更便捷的部署和运维体验。

随着大模型技术的快速演进，Omni Proxy将继续推动推理调度技术的发展，为人工智能基础设施的完善和创新做出贡献。其开源特性也将促进业界最佳实践的共享与协作，推动整个行业的技术进步。


## 如何快速使用Omni Proxy
```
# 编译模块
bash build.sh

# 设置PYTHONHASHSEED
export PYTHONHASHSEED=123

# 生成配置文件
bash omni_proxy.sh \
  --nginx-conf-file /usr/local/nginx/conf/nginx.conf \
  --start-core-index 0 \
  --core-num 4 \
  --listen-port 8080 \
  --prefill-endpoints 127.0.0.1:8001,127.0.0.1:8002 \
  --decode-endpoints 127.0.0.1:9001,127.0.0.1:9002 \
  --omni-proxy-pd-policy sequential \
  --omni-proxy-model-path /path/to/DeepSeek
```
如需启用 APC 感知调度功能，请按以下步骤操作：
* 在 `omniinfer/tools/scripts/pd_run.sh` 文件中设置 `ENABLE_APC_EVENT=1`，同时在 Ansible 模板文件中设置 `USE_OMNI_PROXY=1`
* 向 `omni_proxy.sh` 脚本提供模型路径参数：`--omni-proxy-model-path /path/to/DeepSeek`

