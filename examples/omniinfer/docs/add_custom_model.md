# 添加自定义模型到omniinfer中

本指南将帮助你把自己的模型添加到omniinfer中。我们会详细介绍添加模型的步骤以及模型能够使用的接口列表。

## 1. 前提条件
在开始添加自定义模型之前，请确保你已经满足以下条件：
- 已经安装了omniinfer所需的依赖环境。
- 拥有自己训练好的模型文件，支持的格式有：[*.pt, *.safetensors等]。

## 2. 添加自定义模型步骤

### 2.1. 准备模型文件
将你的模型文件放置在 `omni/models/yourmodel` 目录，如果不存在，请新建该目录。模型文件的实现需要遵循一定的规范：
- 应基于昇腾亲和优化的高性能模型在`models`目录下新增模型结构定义文件，不能直接在其它模型上修改适配
- 应通过推理框架的插件机制注册新模型，不能直接覆推理框架原有的模型
- 为实现最大程度的复用，模型文件定义中不要有基本模块的定义
- 模型结构定义所需要的基本模块、数据结构定义只能从`omni/models/common`目录引入

抽象的模型结构定义如下：

```python
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata

class YourAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.attn = Attention(prefix=f"{prefix}.attn")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Implement attention logic
        ...

class YourDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = CustomAttention(vllm_config, prefix=f"{prefix}.self_attn")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Implement decoder layer
        ...

class YourModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") 
            for i in range(vllm_config.model_config.hf_config.num_hidden_layers)
        ])

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ...

    def load_weights(self, 
                    weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ...

class YourModelForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = CustomModel(vllm_config, prefix=f"{prefix}.model")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ...

    def compute_logits(self,
                      hidden_states: torch.Tensor,
                      sampling_metadata: SamplingMetadata) -> torch.Tensor:
        ...

    def load_weights(self, 
                    weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ...
```
**注意** YourModelForCausalLM的__init__函数定义需要严格遵循上述定义，否则会导致推理框架兼容性检查不通过。

模型注册接口：`omni/models/__init__.py`,`register_model`第一个参数对应`config.json`中的architecture名称，
第二个参数为模型脚本中对应的类名。
```python
def register_model():

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "omni.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
```

## 3. 基本模块
基本模块位于`omni/models/common/layers`目录下，定义了模型能够使用的基本数据结构和模块，
包括`attention/fused_moe/linear/activation/layernorm/embedding/logits_processor/sampler`。
当前的实现是基于推理框架的基类新增模块，或修改推理框架的原始实现并以monkey patch的方式覆盖。模块应尽量使用omniinfer提供的公共基本模块，
如果为某个模型定制模块，应放置在模型对应的文件夹中。
基本模块的新增及修改应遵循：
- 如需修改或新增`omni/models/common/layers`中的基本模块，需要在社区发起RFC
- 新增公共基本模块应尽可能基于推理框架已有的基类实现

### 3.1 Attention
#### 3.1.1 使用Attention类
推理框架针对不同的Attention实现方式提供了不同的backend，omniinfer当前针对昇腾硬件实现了两类backend，
一是针对Deepseek MoE模型的`AscendMLABackend`，二是针对稠密模型的`AscendAttentionBackend`。
- class Attention
  `infer_engines/vllm/vllm/attention/layer.py`
    - method __init__
    ```python
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        use_mla: bool = False,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **extra_impl_args,
    ) -> None:
    ```
    在初始化时可以通过`use_mla`参数指定是否使用MLA。
- class AscendMLABackend
  `omni/models/common/layers/attention/mla.py`

  其调用方式如下：
  ```python
  def forward(
          self,
          positions: torch.Tensor,
          hidden_states: torch.Tensor,
          kv_cache: torch.Tensor,
          attn_metadata: AttentionMetadata,
  ) -> torch.Tensor:

      return self.attn_mla.impl.forward(
          positions=positions,
          hidden_states=hidden_states,
          kv_cache=kv_cache,
          attn_metadata=attn_metadata)
  ```

- class AscendAttentionBackend
  `omni/models/common/layers/attention/attention.py`

  其调用方式如下：
  ```python
  def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
  ```
  注意：Attention计算逻辑应全部位于Attention backend中，不要在模型脚本中包含Attention相关的计算逻辑。

#### 3.1.2 新增Attention Backend
新增实现应位于`omni/models/common/layers/attention/`中, 可参考当前已有的实现。主要包含以下四个基本类：
```python
class AscendMetadata
class AscendAttentionMetadataBuilder
class AscendAttentionBackendImpl(AttentionImpl)
class AscendAttentionBackend(AttentionBackend)
```

### 3.2 MoE
```python
class FusedMoE(torch.nn.Module):
    def __init__(
    self,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    params_dtype: Optional[torch.dtype] = None,
    reduce_results: bool = False,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    quant_config: Optional[QuantizationConfig] = None,
    tp_size: Optional[int] = None,
    prefix: str = "",
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    )
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None,
                       routed_scaling_factor: Optional[torch.Tensor] = None,
                       layer: torch.nn.Module = None
                       )
    def forward(self, hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                pertoken_scale: torch.Tensor,
                attn_metadata: AttentionMetadata
                )
```
### 3.3 Linear
Linear类应尽量从推理框架继承，omniinfer支持的Linear类如下：
```python
class AscendMergedColumnParallelLinear(LinearBase)
class AscendRowParallelLinear(LinearBase)
class ColumnParallelLinearQuantGather(ColumnParallelLinear)
class RowParallelLinear(RowParallelLinearGPU)
class RowParallelLinearWithReduceScatter(RowParallelLinear)
class MergedReplicatedLinear(ReplicatedLinear)
```
### 3.4 Activation
```python
class SiluAndMul(nn.Module)
```
### 3.5 Layernorm
```python
#支持a8w8动态量化，替换了torch_npu.npu_rms_norm昇腾算子
class RMSNorm(RMSNormGPU)
```
### 3.6 Embedding
包括Token Embedding及Positional Embedding
```python
class VocabParallelEmbedding(VocabParallelEmbeddingGPU)
class ParallelLMHead(VocabParallelEmbedding)

# Positional Embedding
class RotaryEmbeddingTorchNpu(torch.nn.Module)
class YaRNScalingRotaryEmbedding(RotaryEmbeddingTorchNpu)
class LinearScalingRotaryEmbedding(RotaryEmbeddingTorchNpu)
class ExtendedRotaryEmbedding(RotaryEmbeddingTorchNpu)
class DeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbeddingGPU)

# 根据rope_scaling选择相应的RotaryEmbedding类
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    dual_chunk_attention_config: Optional[dict[str, Any]] = None
)
```
### 3.7 logits_processor
```python
#支持logits的TP并行
class LogitsProcessor(LogitsProcessorGPU)
```
### 3.8 smapler
```python
#继承了 vllm v0的sampler
class AscendSampler(Sampler)
#投机推理使用
class RejectionSampler(RejectionSamplerGPU)
class SimpleSampler(RejectionSamplerV1)
```
## 4. 模型量化
当前支持加载A8W8的量化权重，量化后的权重`config.json`需要包含`quantization_config`字段，才能被omniinfer正常加载。
权重量化可以参考[omni_infer_installation_guide中的权重转换章节](omni_infer_installation_guide.md#权重转换)，生成的权重的config.json中的`quantization_config`应如下所示：
```json
"quantization_config" {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "actorder": null,
        "block_structure": null,
        "dynamic": true,
        "group_size": null,
        "num_bits": 8,
        "observer": "memoryless",
        "observer_kwargs": {},
        "strategy": "token",
        "symmetric": true,
        "type": "int"
      },
      "output_activations": null,
      "targets": [
        "Linear"
      ],
      "weights": {
        "actorder": null,
        "block_structure": null,
        "dynamic": true,
        "group_size": null,
        "num_bits": {
          "self_attn.kv_a_proj_with_mqa": 8,
          "self_attn.q_a_proj": 8,
          "self_attn.q_b_proj": 8,
          "self_attn.o_proj": 8,
          "mlp.down_proj": 8,
          "mlp.gate_up_proj": 8,
          "mlp.shared_experts": 8,
          "mlp.experts": 8
        },
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": "channel",
        "symmetric": true,
        "type": int
      }
    }
  },
  "format": "int-quantized",
  "global_compression_ratio": 1.5943962512751308,
  "ignore": [
  ],
  "kv_cache_scheme": null,
  "quant_method": "compressed-tensors",
  "quantization_status": "compressed"
}
```
新增量化方法的步骤:
- 参考已有的CompressedTensorsQuantizer，在`omni/adaptors/vllm/utils.py`中新增支持的量化方法。
```python
ASCEND_COMPRESSED_TENSORS = "ascend_compressed_tensors"
`SUPPORTED_QUANTIZATION_METHODS` = [ASCEND_COMPRESSED_TENSORS]
```
- 在`omni/quantization/__init__.py`中新增引入，如`from omni.quantization.compressed_tensors import compressed_tensors`
- 在`omni/quantization/`目录中新增对应量化方法的适配。
- 如果是基于compressed_tensors新增不同的量化类型，如w4a8, 基于已有的compressed_tensors实现扩展。
## 5. 模型并行
当前omniinfer支持常见的并行策略，如TP/EP/DP/PP等。
推理框架中如vllm针对PP/TP/DP/EP分别提供了命令行参数及环境变量，如下所示：
- 命令行参数
  - PP: `--pipeline-parallel-size`
  - TP: `--tensor-parallel-size`
  - EP: `--enable-expert-parallel`
  - DP: `--data-parallel-size`
- 环境变量
  - DP：`VLLM_DP_SIZE`
  在PD分离场景，omniinfer采用了多api server架构，1 DP对应1个api server及1个vllm enginecore，
  `--data-parallel-size=1`，并设置`VLLM_DP_SIZE`为实际的dp size。
  模型当前依赖的接口为例，主要有2类：
  - 通信操作 `omni/adaptors/vllm/distributed/communication_op.py`
  ```python
  def expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor
  def expert_parallel_all_gather(input_: torch.Tensor,  dim=-1) -> torch.Tensor
  def tensor_model_parallel_reduce_scatter(input_: torch.Tensor) -> torch.Tensor
  def reduce_scatter_two_stage(input_: torch.Tensor, idx: int, reverse=False) -> torch.Tensor
  def all_gather_two_stage(input_: torch.Tensor, idx: int, dim=-1, reverse=False) -> torch.Tensor
  def reduce_scatter_local(input_: torch.Tensor, idx: int) -> torch.Tensor
  def reduce_scatter_cross(input_: torch.Tensor, idx: int) -> torch.Tensor
  def all_gather_local(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor
  def all_gather_cross(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor
  def local_rank_all_gather(input_: torch.Tensor, dim=-1)
  def mlp_all_gather(input_: torch.Tensor, dim=-1)
  def mlp_reduce_scatter(input_: torch.Tensor) -> torch.Tensor
  ```
  - 状态信息获取 `omni/adaptors/vllm/distributed/parallel_state.py`
  ```python
  def get_cross_group_from_list(idx: int):
  def get_data_parallel_rank():
  def get_data_parallel_world_size():
  def get_ep_group():
  def get_expert_parallel_rank():
  def get_expert_parallel_world_size():
  def get_local_group_from_list(idx: int):
  def get_local_group_rank_from_list(idx: int):
  def get_local_group_world_size_from_list(idx: int):
  def get_local_world_group():
  def get_mlp_tp_rank():
  def get_mlp_tp_size():
  def get_mlp_world_group():
  def get_world_group():
  def get_world_group_from_list(idx: int):
  ```
  - 基本通信算子扩展
  如果需要新增通信算子，或为已有的通信算子替换昇腾算子，需要修改`omni/adaptors/vllm/distributed/communicator.py`
## 6. 图模式
> 在推理框架中可通过命令行参数启用图模式，以 vllm 为例，参数为`--additional_config='{"graph_model_compile_config": {"level":1}}'`，其中level为1表示图模式，level为0表示eager模式 。

新增模型如果想要使能GE图，需要在模型类名上添加@support_torch_compile装饰器，同时需要实现should_use_eager_mode(self, *args, **kwargs)函数，告知框架该模型什么情况下走图模式，什么情况下走eager模式，默认走eager模式。

比如deepseekV3模型，在模型类中添加@support_torch_compile装饰器，并实现should_use_eager_mode(self, *args, **kwargs)函数，如下所示：

```python
@support_torch_compile
class DeepseekV3ForCausalLM(nn.Module):
    
     def should_use_eager_mode(self, *args, **kwargs):
            attn_metadata = kwargs.get("attn_metadata", None)
            if not attn_metadata:
                return True
    
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
    
            if attn_metadata.prefill:
                return True
    
            return False
```

如果模型中需要使用静态shape（框架默认已对input、position、kv_cache、attn_metadata做静态shape标记），则需要继承`GraphCompileConfiguration`类，并实现`mark_static_for_graph`方法，该方法中需要对模型中的tensor进行静态标记，如下所示：
```python
class GraphCompileConfiguration:
    """
    When the graph mode is turned on
    you can set the gear or clarify the static shape by inheriting this class to speed up the model running
    """
    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(args[0])
        torch._dynamo.mark_static(args[1])
#模型中需要对特定的tensor做mark static操作时，继承GraphCompileConfiguration类
class DeepseekV3ForCausalLM(nn.Module, GraphCompileConfiguration)
```

## 7. 模型配置
模型配置独立于推理框架提供的命令行参数、环境变量，当前实现位于`omni/models/common/model_config.py`，
可以通过环境变量`MODEL_EXTRA_CFG_PATH`来指定json格式的模型配置文件，PD分离场景可以为P/D分别指定不同的配置文件。如下所示：
```json
{
    "model_parallel_config": {
        "dense_mlp_tp_size": 4,                 # dense mlp tp大小，默认为4
        "dp_size": 1,                           # dp大小，P节点为1，D节点设置为die数
        "o_proj_tp_size": 1,                    # attention out_proj tp大小，默认为1
        "redundancy_shared_expert_num": 0       # 冗余共享专家数，默认为0
    },
    "operator_optimizition_config": {
        "enable_kv_rmsnorm_rope_cache": true,   # 是否开启rmsnorm和rope融合，默认开启
        "prefill_moe_all_to_all": true,         # P的moe层是否使用all to all，默认开启。设置为false时，使用allgather+scatter
        "moe_multi_stream_tune": false,         # 是否开启多流，只能图模式使用，单算子会报错, 开启提升3ms性能
        "best_ep": false,                       # 是否开启强制负载均衡，测试精度时必须关闭
        "merge_qkv": false,                     # merge_qkv当前未使能，为false
        "two_stage_comm": false,                # 卡内卡间多级通信，A2使用，为false
        "gmm_nz": false,                        # 是否开启gmm_nz，测试性能时使用，P开启，D关闭
        "decode_moe_dispatch_combine": true,    # D的moe层是否使用dispatch+combine算子，默认开启。设置为false时，使用all to all
        "use_omni_placement": false,            # 是否使用omni placement
        "omni_placement_config_path": null,     # omni placement配置文件
        "use_super_kernel": false,              # 是否使用super kernel融合算子，仅图模式可用，开启提升2ms性能
        "use_mlaprolog": false,                 # 暂未使能，默认关闭
        "opt_w2_scale_cast": false,             # 是否将w2_scale权重转换为float32，默认false
        "enable_mc2_v2": false,                 # 是否使用npu_moe_distribute_dispatch_v2版本，默认关闭，0714后的主线CANN需要打开
        "decode_gear_list": [1],                # 图模式挡位，decode节点图模式下使用
        "control_accept_rate": -1,              # <0 or >1 不控制, >=0 and <=1 控制MTP开启时接受率为该值，几乎必然导致输出结果异常，仅保证只投机1个token时满足这一数值
        "enable_round_pipeline_comm": false,    # d节点使用，moe部分通信pipline，decode为4机时打开，a2相关
        "enable_pipeline_comm": false,          # d节点使用，moe部分通信pipline，decode为2机时打开，a2相关
        "pd_seperate_prefill": false,           # pd分离版本下prefill节点打开，a2相关
        "prefill_enable_long_seq": false,       # prefill使能长序列，64k以上打开，默认关闭，a2相关
        "prefill_moe_multi_stream": true,       # p节点使用，moe部分是否开启多流，a2相关
        "prefill_enable_mla_alltoall_local": true, # p节点使用，mla部分是否开启all2all，a2相关
        "prefill_enable_pipeline_comm": true,   # p节点使用，moe部分是否开启通信pipline，a2相关
        "prefill_mla_multi_stream": true,       # p节点使用，mla部分是否开启多流，a2相关
        "enable_dense_local_tp": 1              # 前三层dense层mlp的tp大小，a2相关
    }
}
```
## 8. 模型加速
## 9. 常见问题
