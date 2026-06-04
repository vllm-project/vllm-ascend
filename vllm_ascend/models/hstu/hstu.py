# mypy: ignore-errors
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
import torch_npu
from torch.autograd.profiler import record_function
from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
# from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.forward_context import (ForwardContext, get_forward_context,
                                  set_forward_context)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.hstu_config import (
    HSTUInferenceRankingConfig, InferenceHSTUConfig, RankingConfig)

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.hstu_attention_v1 import AscendHSTUAttentionMetadata
from vllm_ascend.models.hstu.hstu_embs_inputprocess import (
    InferenceEmbedding, InputPreprocessModule)

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


def init_mlp_weights_optional_bias(m: torch.nn.Module, ) -> None:
    """
    Initialize the weights of a linear layer and optionally the bias.

    Args:
        m: The module to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLP(torch.nn.Module):  # type: ignore
    """
    Multi-Layer Perceptron (MLP) module wrapper for processing jagged data.

    Args:
        in_size (int): The input size.
        layer_sizes (List[int]): The sizes of the layers.
        bias (bool, optional): Whether to include bias in the layers. Defaults to True.
        activation (Union[str, Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]], optional): The activation function. Defaults to torch.relu.
        device (Optional[torch.device], optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
    """

    def __init__(
        self,
        in_size: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} not supported")

        self.feed_forward = torch.nn.Linear(in_features=in_size,
                                            out_features=in_size)
        self.out_layer = torch.nn.Linear(in_features=in_size, out_features=1)
        self.layer_norm_1 = torch.nn.LayerNorm([in_size], eps=1e-7)
        self.layer_norm_2 = torch.nn.LayerNorm([in_size], eps=1e-7)
        self.act = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if len(x.shape) == 3:
            x = x.squeeze(0)
        assert x.dim() == 2, "Tensor must be 2-dimensional"
        x = x + self.act(self.feed_forward(self.layer_norm_1(x)))
        x = self.out_layer(self.layer_norm_2(x))
        x = torch.sigmoid(x)
        return x


class RMSNorm_npu(torch.nn.Module):

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, residual: Optional[torch.Tensor]):
        if residual is None:
            y = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        else:
            y, _, x = torch_npu.npu_add_rms_norm(residual, x, self.weight,
                                                 self.eps)
        return y, x


class FFN_npu_swiglu(torch.nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float, dtype):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim,
                                  bias=False).to(torch.float16)
        self.w2 = torch.nn.Linear(hidden_dim, dim,
                                  bias=False).to(torch.float16)
        self.w3 = torch.nn.Linear(dim, hidden_dim,
                                  bias=False).to(torch.float16)
        self.dropout = torch.nn.Dropout(dropout).to(dtype)
        self.dtype = dtype

    @property
    def W_1(self):
        return torch.cat([self.w3.weight, self.w1.weight],
                         dim=0).transpose(0, 1)

    @property
    def W_2(self):
        return self.w2.weight.transpose(0, 1)

    def forward(self, x):
        return self.dropout(
            (torch_npu.npu_ffn(x.to(torch.float16),
                               self.W_1,
                               self.W_2,
                               'swiglu',
                               inner_precise=1)).to(self.dtype))


@support_torch_compile
class HstuAttention(torch.nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: InferenceHSTUConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self._layer_name = prefix

        self.attn = Attention(
            num_heads=config.num_heads,
            head_size=config.head_dim,
            scale=1.0 / math.sqrt(config.head_dim),
            num_kv_heads=config.num_heads,
            prefix=self._layer_name,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        attn_output = self.attn(query, key, value)

        return attn_output


class PagedHSTUInferLayer(torch.nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        config: InferenceHSTUConfig = vllm_config.model_config.hf_config.hstu_config
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim

        self._num_heads: int = config.num_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0 / (self._attention_dim_per_head**0.5)
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]

        dtype = (torch.bfloat16 if config.bf16 else
                 torch.float16 if config.fp16 else torch.float32)
        device = torch_npu.npu.current_device()

        total_output_size = sum(self._split_arg_list)

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            total_output_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_uvqk.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        self._linear_uvqk_weight = self._linear_uvqk.weight.T.contiguous()

        # input norm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None

        self.self_attn = HstuAttention(vllm_config=vllm_config,
                                       config=config,
                                       prefix=prefix)

        # output norm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(self._num_heads * self._linear_dim_per_head,
                       dtype=dtype,
                       device=device),
            requires_grad=False,
        )
        self._output_layernorm_bias = None

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

        for param in self._linear_proj.parameters():
            param.requires_grad = False
            param.copy_(torch.randn_like(param))

        # ffn
        self.has_ffn = config.has_ffn
        if config.has_ffn:
            self.norm_ffn = RMSNorm_npu(self._embedding_dim, self._eps)
            ffn_expand = config.ffn_expand
            self.feed_forward = FFN_npu_swiglu(
                dim=self._embedding_dim,
                hidden_dim=self._embedding_dim * ffn_expand,
                dropout=config.dropout_ratio,
                dtype=dtype,
            )

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._uvqk_buffer_ = torch.zeros(
            (
                max_num_tokens,
                (self._linear_dim_per_head * 2 +
                 self._attention_dim_per_head * 2) * self._num_heads,
            ),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

        # self._kv_buffer_ = torch.zeros(
        #     (
        #         2,
        #         max_num_tokens,
        #         self._linear_dim_per_head * self._num_heads,
        #     ),
        #     dtype=dtype,
        #     device=device,
        #     requires_grad=False,
        # )
        self._user = torch.zeros(
            (max_num_tokens, self._linear_dim_per_head * self._num_heads),
            dtype=dtype,
            device=device,
            requires_grad=False)

        self._key = torch.zeros_like(self._user,
                                     dtype=dtype,
                                     device=device,
                                     requires_grad=False)

        self._value = torch.zeros_like(self._user,
                                       dtype=dtype,
                                       device=device,
                                       requires_grad=False)

        self._query = torch.zeros_like(self._user,
                                       dtype=dtype,
                                       device=device,
                                       requires_grad=False)

        self._output_buffer_ = torch.zeros(
            (max_num_tokens, self._embedding_dim),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

        self._attn_output_buffer = torch.zeros(
            (max_num_tokens, self._embedding_dim),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    def should_use_eager_mode(self, *args, **kwargs):
        return True

    def forward_input(
        self,
        num_tokens: int,
        input_buffer: torch.Tensor,
    ):
        input_tensor = input_buffer[:num_tokens]
        normed_input = F.layer_norm(
            input_tensor,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=None,
            eps=self._eps,
        )

        self._uvqk_buffer_[:num_tokens,
                           ...] = F.silu(self._linear_uvqk(normed_input))

        (self.user, self.value, self.query, self.key) = torch.split(
            self._uvqk_buffer_[:num_tokens],
            self._split_arg_list,
            dim=-1,
        )

        #self._kv_buffer_[:, :num_tokens, ...] = torch.stack([key, value], dim=0)

        return self._uvqk_buffer_[:num_tokens, ...]

    def forward_attn(
        self,
        attn_metadata: Union[AttentionMetadata, dict] = None,
    ):
        with set_forward_context(attn_metadata, self.vllm_config):
            attn_output = self.self_attn(self.query, self.key, self.value)
            self._attn_output_buffer[:self.query.shape[0], ...] = attn_output

    def forward_output(
        self,
        num_tokens: int,
        input_buffer: torch.Tensor,
        attn_metadata: Union[AttentionMetadata, dict] = None,
    ):
        input_tensor = input_buffer[:num_tokens]

        attn_output = self._attn_output_buffer[:num_tokens].view(
            -1, self._num_heads * self._linear_dim_per_head)

        parallel_input = self.user * F.layer_norm(
            attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )

        layer_output = self._linear_proj(parallel_input)

        if self.has_ffn:
            ffn_input, _ = self.norm_ffn(
                layer_output, input_tensor if self._residual else None)
            layer_output = self.feed_forward(ffn_input) + layer_output
        else:
            if self._residual:
                torch.add(layer_output, input_tensor, out=layer_output)

        self._output_buffer_[:num_tokens, ...] = layer_output
        return self._output_buffer_[:num_tokens, ...]

    def forward(
        self,
        layer_input: torch.Tensor,
        is_dump: Optional[bool] = False,
    ) -> torch.Tensor:
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=None,
            eps=self._eps,
        )

        mixed_uvqk = F.silu(self._linear_uvqk(normed_input))
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )

        #self._kv_buffer_[:, :key.shape[0], ...] = torch.stack([key, value], dim=0)

        attn_output = self.self_attn(query, key, value)
        attn_output = attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head)

        parallel_input = user * F.layer_norm(
            attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )

        layer_output = self._linear_proj(parallel_input)

        if self.has_ffn:
            ffn_input, _ = self.norm_ffn(
                layer_output, layer_input if self._residual else None)
            layer_output = self.feed_forward(ffn_input) + layer_output
        else:
            if self._residual:
                torch.add(layer_output, layer_input, out=layer_output)

        return layer_output


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        hstu_config (InferenceHSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        hstu_config: InferenceHSTUConfig,
        task_config: RankingConfig,
    ):
        super().__init__()
        self._device = torch_npu.npu.current_device()
        self.vllm_config = vllm_config
        self.use_aclgraph = vllm_config.additional_config.get(
            'graph_model_compile_config', {}).get('use_aclgraph', None)
        self._is_merged_table = vllm_config.model_config.hf_config.merged_table if vllm_config.model_config.hf_config.merged_table is not None else True
        self.features_cnt = vllm_config.model_config.hf_config.features_cnt if vllm_config.model_config.hf_config.features_cnt else 20
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_dim = hstu_config.hidden_size
        self._num_layers = hstu_config.num_layers
        if self._is_merged_table:
            for ebc_config in task_config.embedding_configs:
                assert (
                    ebc_config.dim == self._embedding_dim
                ), f"hstu layer hidden size = {ebc_config.dim} should equal to embedding dim = {self._embedding_dim}"

        self._logit_dim_list = [
            layer_sizes[-1] for layer_sizes in task_config.prediction_head_arch
        ]

        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._embedding_collection = InferenceEmbedding(
            self.vllm_config.model_config.hf_config,
            max_num_batched_tokens=self.max_num_tokens)
        self._embedding_collection.to_empty(
            device=torch_npu.npu.current_device())
        if not self._is_merged_table:
            self._inputprocess_module = InputPreprocessModule(vllm_config)

        self.layers = torch.nn.ModuleList([
            PagedHSTUInferLayer(vllm_config=vllm_config,
                                prefix=f"model.layers.{layer_idx}.attn")
            for layer_idx in range(self._hstu_config.num_layers)
        ])

        self._dense_module = MLP(
            self._embedding_dim,
            task_config.prediction_head_act_type,
        )

        self.layers.npu()
        self._dense_module = self._dense_module.npu()

        dtype = (torch.bfloat16 if hstu_config.bf16 else
                 torch.float16 if hstu_config.fp16 else torch.float32)

        self.device = torch_npu.npu.current_device()

        self.max_batch_size = getattr(
            vllm_config.model_config.hf_config, "max_batch_size",
            vllm_config.scheduler_config.max_num_seqs)
        self.max_seq_len = math.ceil(self.max_num_tokens / self.max_batch_size)

        if not self._is_merged_table:
            # 多表情况
            self._embedding_input_buffer = torch.zeros(self.features_cnt,
                                                       self.max_batch_size *
                                                       self.max_seq_len,
                                                       dtype=torch.int64,
                                                       device=self.device)
        else:
            self._embedding_input_buffer = torch.zeros(self.max_batch_size *
                                                       self.max_seq_len,
                                                       dtype=torch.int64,
                                                       device=self.device)

        self._input_buffer = torch.zeros(
            (self.max_batch_size * self.max_seq_len, self._embedding_dim),
            dtype=dtype,
            device=self.device)

        self._position_buffer = torch.zeros(
            (self.max_batch_size * self.max_seq_len),
            dtype=torch.int64,
            device=self.device)

        self._output_buffer = torch.zeros(
            (self.max_batch_size * self.max_seq_len, 1),
            dtype=dtype,
            device=self.device)

        self._embedding_graph: Optional[Dict[int, Any]] = None  # type: ignore
        self._prefill_hstu_graph: Optional[Dict[int,
                                                Any]] = None  # type: ignore
        self._decode_hstu_graph: Optional[Dict[int,
                                               Any]] = None  # type: ignore
        self._dense_module_graph: Optional[Dict[int,
                                                Any]] = None  # type: ignore
        self._attn_metadatas: Optional[Dict[int, Any]] = None  # type: ignore

        self._capture_done = False
        self._is_capturing = False

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.layers.bfloat16()
        self._dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self.layers.half()
        self._dense_module.half()
        return self

    def _get_position_ids(
        self,
        metadata: Optional[AscendHSTUAttentionMetadata],
        num_tokens: int,
    ) -> torch.Tensor:
        if metadata is not None:
            position_ids = metadata.additional_metadata.get("position_ids")
            if position_ids is not None:
                return position_ids.to(device=self.device, dtype=torch.int64)
        return torch.arange(num_tokens, dtype=torch.int64, device=self.device)

    def forward_aclgraph(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        is_pd_merge: bool,
        is_prefill: bool,
    ):
        num_tokens = input_ids.shape[0]
        forward_context: ForwardContext = get_forward_context()
        layer_names = list(forward_context.no_compile_layers.keys())
        bs = attn_metadata[layer_names[0]].query_lens.shape[0]
        # if bs not in self._decode_hstu_graph:
        #     batch_size_pow2 = 2 ** math.ceil(math.log2(bs))
        # else:
        #     batch_size_pow2 = bs
        batch_size_pow2 = bs

        max_seq_len = attn_metadata[layer_names[0]].max_query_len
        print(f" - Target bs: {bs}, seq_len: {max_seq_len}")
        if max_seq_len not in self._decode_hstu_graph[
                batch_size_pow2]:  # type: ignore
            all_seq_lens = sorted(
                list(self._prefill_hstu_graph[batch_size_pow2].keys()))

            def binary_search(sorted_list, target):
                left, right = 0, len(sorted_list)
                while left < right:
                    mid = (left + right) // 2
                    if sorted_list[mid] < target:
                        left = mid + 1
                    else:
                        right = mid

                if left < len(sorted_list):
                    return sorted_list[left]
                else:
                    return None

            if all_seq_lens[-1] >= max_seq_len:
                max_seq_len_padded = binary_search(all_seq_lens, max_seq_len)
            else:
                max_seq_len_padded = all_seq_lens[-1]
        else:
            max_seq_len_padded = max_seq_len
        num_tokens_pow2 = batch_size_pow2 * max_seq_len_padded

        if max_seq_len_padded is None:
            raise ValueError(
                f"forward_aclgraph max_seq_len = {max_seq_len} found None in captured max_seq_len list!"
            )

        print(
            f" - Select bs: {batch_size_pow2}, seq_len: {max_seq_len_padded}, num_tokens: {num_tokens_pow2}"
        )

        self._embedding_input_buffer[:num_tokens].copy_(input_ids)
        position_ids = self._get_position_ids(attn_metadata[layer_names[0]],
                                              num_tokens)
        self._position_buffer[:num_tokens_pow2].zero_()
        self._position_buffer[:position_ids.shape[0]].copy_(position_ids)
        if is_prefill:
            self._embedding_graph[num_tokens_pow2][0].replay()
        else:
            self._embedding_graph[num_tokens_pow2][1].replay()

        if is_prefill:
            self._copy_replay_data(
                attn_metadata[layer_names[0]],
                self._attn_metadatas[batch_size_pow2][max_seq_len_padded][0])
            # replay prefill graph
            with set_forward_context(
                    self._attn_metadatas[batch_size_pow2][max_seq_len_padded]
                [0], self.vllm_config):
                for idx in range(0, self._num_layers):
                    self._prefill_hstu_graph[batch_size_pow2][
                        max_seq_len_padded][idx].replay()
                    self.layers[idx].forward_attn(
                        self._attn_metadatas[batch_size_pow2]
                        [max_seq_len_padded][0])
                self._prefill_hstu_graph[batch_size_pow2][max_seq_len_padded][
                    -1].replay()
        else:
            self._copy_replay_data(
                attn_metadata[layer_names[0]],
                self._attn_metadatas[batch_size_pow2][max_seq_len_padded][1])
            # replay decode graph
            with set_forward_context(
                    self._attn_metadatas[batch_size_pow2][max_seq_len_padded]
                [1], self.vllm_config):
                for idx in range(0, self._num_layers):
                    self._decode_hstu_graph[batch_size_pow2][
                        max_seq_len_padded][idx].replay()
                    self.layers[idx].forward_attn(
                        self._attn_metadatas[batch_size_pow2]
                        [max_seq_len_padded][1])
                self._decode_hstu_graph[batch_size_pow2][max_seq_len_padded][
                    -1].replay()

        if (is_prefill and is_pd_merge) or (not is_prefill):
            self._dense_module_graph[num_tokens_pow2].replay()
            hstu_output = torch.zeros_like(self._output_buffer[:num_tokens])
            hstu_output.copy_(self._output_buffer[:num_tokens])
            return hstu_output
        else:
            return self._output_buffer[:num_tokens]

    def split_by_value(self, tensor, feature_del, is_prefill):
        split_indices = torch.where(tensor == feature_del)[0]
        split_indices = [-1] + split_indices.tolist()
        if is_prefill:
            features = [
                tensor[split_indices[i] + 1:split_indices[i + 1]]
                for i in range(len(split_indices) - 2)
            ]
            action = tensor[split_indices[-2] + 1:split_indices[-1]]
        else:
            features = [
                tensor[split_indices[i] + 1:split_indices[i + 1]]
                for i in range(len(split_indices) - 1)
            ]
            action = None
        # 物品特征拼接成一个大tensor，减少成图回放copy_次数
        embedding_stack_seq_cnt = self._task_config.embedding_stack_seq_cnt
        features_res = []
        index = 0
        for cnt in embedding_stack_seq_cnt:
            feature_i = torch.stack(features[index:index + cnt], dim=0)
            features_res.append(feature_i)
            index += cnt
        return features_res, action

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        is_dummy = False
        if attn_metadata is not None and self.use_aclgraph and not self._capture_done and not self._is_capturing:
            print(" - First in forward with metadata, start capture_model...")
            is_dummy = True
            self.capture_model(attn_metadata)
        forward_context: ForwardContext = get_forward_context()
        layer_names = list(forward_context.no_compile_layers.keys())

        is_pd_merge = attn_metadata[layer_names[0]].attn_state == AscendAttentionState.PdMergedNoCache or \
            attn_metadata[layer_names[0]].attn_state == AscendAttentionState.PdMergedCacheHit if attn_metadata is not None else False

        is_prefill = attn_metadata[layer_names[0]].attn_state == AscendAttentionState.PrefillNoCache or \
            attn_metadata[layer_names[0]].attn_state == AscendAttentionState.PdMergedNoCache if attn_metadata is not None else False
        if self.use_aclgraph and not is_dummy and attn_metadata is not None:
            with record_function(
                    f"## Graph replay with input_ids len: {input_ids.shape[0]} ##"
            ):
                hstu_output = self.forward_aclgraph(
                    input_ids,
                    attn_metadata,
                    is_pd_merge,
                    is_prefill,
                )
        else:
            with record_function("## embeddings ##"):
                if self._is_merged_table:
                    # 单表
                    if attn_metadata is not None:
                        metadata = attn_metadata[layer_names[0]]
                        position_ids = self._get_position_ids(
                            metadata, input_ids.size(0))
                    else:
                        position_ids = self._get_position_ids(
                            None, input_ids.size(0))
                    embeddings = self._embedding_collection(
                        input_ids,
                        position_ids,
                        is_prefill=is_prefill,
                    )
                else:
                    # 多表查表以及交叉
                    # TODO 多batch下需要组batch
                    import time
                    st = time.time()
                    split_merged_input_ids, action = self.split_by_value(
                        input_ids, -2, is_prefill)
                    et = time.time()
                    print(f" - Split_by_value cost: {(et - st) * 1000}ms")
                    embeddings = self._embedding_collection(
                        split_merged_input_ids)
                    if is_prefill:
                        embeddings, _ = self._inputprocess_module(
                            past_ids=split_merged_input_ids[0][0],
                            user_feature_embs=embeddings[1],
                            past_item_embeddings=embeddings[0],
                            past_ratings=action,
                            num_rerank=self.vllm_config.model_config.hf_config.
                            max_num_candidates)
                    else:
                        embeddings = self._inputprocess_module.process_rerank_embs(
                            rerank_embs=embeddings[0],
                            past_lengths=attn_metadata[layer_names[0]].seq_lens
                            + 1)

            with record_function("## hstu_block ##"):
                input = embeddings
                for index, hstu_layer in enumerate(self.layers):
                    input = hstu_layer(
                        input,
                        True if index == 0 and not is_prefill else False,
                    )
                hstu_output = input

            if (is_prefill and is_pd_merge) or (not is_prefill):
                with record_function("## dense module ##"):
                    hstu_output = self._dense_module(hstu_output)

        if hstu_output is not None and is_prefill:
            hstu_output = self.postprocess_pd(
                hstu_output,
                attn_metadata[
                    layer_names[0]].additional_metadata["num_candidates"],
                attn_metadata[layer_names[0]].query_start_loc,
            )
        return hstu_output

    def postprocess_pd(self, hstu_output, num_candidate, query_start_loc):
        if torch.allclose(num_candidate, torch.zeros_like(num_candidate)):
            return hstu_output
        processed_output = []
        for idx in range(num_candidate.shape[0]):
            processed_output.append(
                hstu_output[query_start_loc[idx + 1] -
                            num_candidate[idx]:query_start_loc[idx + 1]])
        return torch.cat(processed_output, dim=0)

    '''
    copy metadata到capture时使用的metadata
    '''

    def _copy_replay_data(self, metadata, target_metadata):
        bs = metadata.query_lens.shape[0]
        self.query_lens[:bs].copy_(metadata.query_lens)
        self.query_start[:bs + 1].copy_(metadata.query_start_loc)
        if metadata.attn_state == AscendAttentionState.DecodeOnly:
            self.seq_offset_k[:bs + 1].copy_(
                metadata.additional_metadata["seq_offset_k"])
            self.page_ids[:metadata.additional_metadata["page_ids"].
                          shape[0]].copy_(
                              metadata.additional_metadata["page_ids"])
            self.page_offsets[:bs + 1].copy_(
                metadata.additional_metadata["page_offsets"])
            self.last_page_len[:bs].copy_(
                metadata.additional_metadata["last_page_len"])
        if metadata.attn_state == AscendAttentionState.PrefillNoCache:
            self.slot_mapping[:metadata.slot_mapping.shape[0]].copy_(
                metadata.slot_mapping)

        target_metadata.max_query_len = metadata.max_query_len
        if metadata.attn_state == AscendAttentionState.DecodeOnly:
            target_metadata.additional_metadata[
                "max_seq_len_k"] = metadata.additional_metadata[
                    "max_seq_len_k"]

    def _build_metadata_by_bs_and_num_token(
        self,
        bs,
        num_tokens,
        attn_state,
    ):
        _attn_metadata = AscendHSTUAttentionMetadata(
            num_actual_tokens=num_tokens,
            block_tables=self.block_tables,
            query_start_loc=self.query_start[:bs + 1],
            query_lens=self.query_lens[:bs],
            max_query_len=8192
            if attn_state == AscendAttentionState.PrefillNoCache else 400,
            seq_lens=self.seq_lens[:bs],
            seq_lens_list=self.seq_lens[:bs].tolist(),
            slot_mapping=self.slot_mapping[:num_tokens],
            attn_state=attn_state,
            additional_metadata={
                "max_seq_len_k": 8592,
                "seq_offset_k": self.seq_offset_k[:bs + 1],
                "page_offsets": self.page_offsets[:bs + 1],
                "page_ids": self.page_ids,
                "last_page_len": self.last_page_len[:bs],
                "num_candidates": self.num_candidates[:bs],
                "model_type": "hstu_inference_ranking"
            })

        return _attn_metadata

    def _capture_model(
        self,
        bs,
        num_tokens,
        max_seq_len,
        prefill_max_graph,
        decode_max_graph,
        embedding_max_graph,
        dense_max_graph,
    ):
        if self._prefill_hstu_graph is None:
            self._prefill_hstu_graph = {}
            self._decode_hstu_graph = {}
            self._embedding_graph = {}
            self._dense_module_graph = {}
        if bs not in self._prefill_hstu_graph:
            self._prefill_hstu_graph[bs] = {}
            self._decode_hstu_graph[bs] = {}
        if max_seq_len in self._prefill_hstu_graph[
                bs] and max_seq_len in self._decode_hstu_graph[bs]:
            return self._prefill_hstu_graph[bs][
                max_seq_len], self._decode_hstu_graph[bs][max_seq_len]

        graph_capture_warmup_stream = torch_npu.npu.Stream()
        graph_capture_warmup_stream.wait_stream(torch_npu.npu.current_stream())

        with torch.npu.stream(graph_capture_warmup_stream):
            pass

        # embedding捕获
        if self._is_merged_table:
            prefill_embedding_graph = torch.npu.NPUGraph()
            with torch_npu.npu.graph(prefill_embedding_graph,
                                     pool=embedding_max_graph):
                self._input_buffer[:num_tokens] = self._embedding_collection(
                    self._embedding_input_buffer[:num_tokens],
                    self._position_buffer[:num_tokens],
                    is_prefill=True)

            decode_embedding_graph = torch.npu.NPUGraph()
            with torch_npu.npu.graph(decode_embedding_graph,
                                     pool=embedding_max_graph):
                self._input_buffer[:num_tokens] = self._embedding_collection(
                    self._embedding_input_buffer[:num_tokens],
                    self._position_buffer[:num_tokens],
                    is_prefill=False)

            if num_tokens not in self._embedding_graph:
                self._embedding_graph[num_tokens] = {}
            self._embedding_graph[num_tokens][0] = prefill_embedding_graph
            self._embedding_graph[num_tokens][1] = decode_embedding_graph
        else:
            prefill_embedding_graph = torch.npu.NPUGraph()
            with torch_npu.npu.graph(prefill_embedding_graph,
                                     pool=embedding_max_graph):
                _ = self._embedding_collection(
                    self._embedding_input_buffer[:num_tokens])
                self._input_buffer[:num_tokens], _ = self._inputprocess_module(
                )
            self._embedding_graph[num_tokens] = []
            self._embedding_graph[num_tokens].append(prefill_embedding_graph)

        # dense捕获
        dense_graph = torch.npu.NPUGraph()
        with torch_npu.npu.graph(dense_graph, pool=dense_max_graph):
            self._output_buffer[:num_tokens] = self._dense_module(
                self.layers[-1]._output_buffer_[:num_tokens])

        self._dense_module_graph[num_tokens] = dense_graph

        attn_metadata = self._build_metadata_by_bs_and_num_token(
            bs, num_tokens, AscendAttentionState.PrefillNoCache)
        if self._attn_metadatas is None:
            self._attn_metadatas = {}
        if bs not in self._attn_metadatas:
            self._attn_metadatas[bs] = {}
        if max_seq_len not in self._attn_metadatas[bs]:
            self._attn_metadatas[bs][max_seq_len] = {}
        self._attn_metadatas[bs][max_seq_len][0] = attn_metadata

        # 完整prefill切分成input/attn/output三部分
        # 第一张图为input，最后一张图为output，其它图为output+下一层input

        # Prefill捕获
        capture_input_buffer = [self._input_buffer] + [
            self.layers[layer_idx]._output_buffer_
            for layer_idx in range(self._num_layers)
        ]

        prefill_graphs = [
            torch_npu.npu.NPUGraph() for _ in range(self._num_layers + 1)
        ]
        with torch_npu.npu.graph(prefill_graphs[0], pool=prefill_max_graph):
            self.layers[0].forward_input(
                num_tokens,
                capture_input_buffer[0],
            )
        if prefill_max_graph is None:
            prefill_max_graph = prefill_graphs[0].pool()

        for idx in range(self._num_layers - 1):
            self.layers[idx].forward_attn(attn_metadata)
            with torch_npu.npu.graph(prefill_graphs[idx + 1],
                                     pool=prefill_max_graph):
                self.layers[idx].forward_output(num_tokens,
                                                capture_input_buffer[idx],
                                                attn_metadata)
                self.layers[idx + 1].forward_input(
                    num_tokens,
                    capture_input_buffer[idx + 1],
                )

        self.layers[-1].forward_attn(attn_metadata)
        with torch_npu.npu.graph(prefill_graphs[-1], pool=prefill_max_graph):
            self.layers[-1].forward_output(num_tokens,
                                           capture_input_buffer[-1],
                                           attn_metadata)

        self._prefill_hstu_graph[bs][max_seq_len] = prefill_graphs

        # Decoding捕获
        decode_attn_metadata = self._build_metadata_by_bs_and_num_token(
            bs, num_tokens, AscendAttentionState.DecodeOnly)
        self._attn_metadatas[bs][max_seq_len][1] = decode_attn_metadata

        decode_graphs = [
            torch_npu.npu.NPUGraph() for _ in range(self._num_layers + 1)
        ]
        with torch_npu.npu.graph(decode_graphs[0], pool=decode_max_graph):
            self.layers[0].forward_input(
                num_tokens,
                capture_input_buffer[0],
            )
        if decode_max_graph is None:
            decode_max_graph = decode_graphs[0].pool()

        for idx in range(0, self._num_layers - 1):
            self.layers[idx].forward_attn(attn_metadata)
            with torch_npu.npu.graph(decode_graphs[idx + 1],
                                     pool=decode_max_graph):
                static_output = self.layers[idx].forward_output(
                    num_tokens, capture_input_buffer[idx],
                    decode_attn_metadata)
                self.layers[idx + 1].forward_input(
                    num_tokens,
                    static_output,
                )

        self.layers[idx].forward_attn(attn_metadata)
        with torch_npu.npu.graph(decode_graphs[-1], pool=decode_max_graph):
            self.layers[-1].forward_output(
                num_tokens, capture_input_buffer[self._num_layers - 1],
                decode_attn_metadata)

        self._decode_hstu_graph[bs][max_seq_len] = decode_graphs

        print(
            "Capture graphs for batch_size = {0} and max_seq_len = {1} and num_tokens = {2}"
            .format(bs, max_seq_len, num_tokens))

        return prefill_graphs, decode_graphs, prefill_embedding_graph, dense_graph

    def capture_model(self, attn_metadata):
        print(" - Start to capture_model...")
        self._is_capturing = True
        first_metadata = next(iter(attn_metadata.values()))
        self.block_tables = first_metadata.block_tables
        self.query_lens = torch.ones(self.max_batch_size,
                                     dtype=torch.int64,
                                     device=self.device)
        self.seq_lens = torch.ones(self.max_batch_size,
                                   dtype=torch.int64,
                                   device=self.device)
        self.slot_mapping = first_metadata.slot_mapping
        self.query_start = torch.arange(self.max_batch_size + 1,
                                        dtype=torch.int64,
                                        device=self.device)
        self.seq_offset_k = torch.arange(self.max_batch_size + 1,
                                         dtype=torch.int64,
                                         device=self.device)
        self.seq_offset_t = torch.arange(self.max_batch_size + 1,
                                         dtype=torch.int64,
                                         device=self.device)
        self.page_offsets = torch.arange(self.max_batch_size + 1,
                                         dtype=torch.int64,
                                         device=self.device)
        self.page_ids = torch.zeros(self.max_seq_len,
                                    dtype=torch.int64,
                                    device=self.device)
        self.last_page_len = torch.zeros(self.max_batch_size,
                                         dtype=torch.int64,
                                         device=self.device)
        self.num_candidates = torch.zeros(self.max_batch_size,
                                          dtype=torch.int64,
                                          device=self.device)

        # bs_list = [2 ** i for i in range((self.max_batch_size - 1).bit_length() + 1)]
        bs_list = [i for i in range(1, self.max_batch_size + 1)]
        seqlen_list = self.vllm_config.model_config.hf_config.captured_tokens_list

        print("Graph setup configs:")
        print("  Max batch size:", self.max_batch_size)
        print("  Max num tokens:", self.max_num_tokens)
        print("  Max seq len:", self.max_seq_len)
        print("  Batch size:", bs_list)
        print("  Length per sequence", seqlen_list)

        p_max, d_max, e_max, dense_max = self._capture_model(
            bs=self.max_batch_size,
            num_tokens=self.max_batch_size * self.max_seq_len,
            max_seq_len=self.max_seq_len,
            prefill_max_graph=None,
            decode_max_graph=None,
            embedding_max_graph=None,
            dense_max_graph=None,
        )

        for bs in bs_list:
            for seqlen in seqlen_list:
                if seqlen > self.max_seq_len:
                    break
                num_tokens = seqlen * bs
                if num_tokens > self.max_num_tokens:
                    break
                self._capture_model(
                    bs,
                    num_tokens,
                    seqlen,
                    p_max[0].pool(),
                    d_max[0].pool(),
                    e_max.pool(),
                    dense_max.pool(),
                )
        self._capture_done = True
        print(" - Capture finished...")


class HSTUInferenceForCausalLM(torch.nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: HSTUInferenceRankingConfig = vllm_config.model_config.hf_config
        self.model = InferenceRankingGR(
            vllm_config=vllm_config,
            hstu_config=self.config.hstu_config,
            task_config=self.config.task_config,
        )

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor] = None,
                attn_metadata: Union[AttentionMetadata, dict] = None,
                selected_indices: Optional[torch.Tensor] = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds=None,
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        attn_metadata = get_forward_context().attn_metadata
        hidden_states = self.model(input_ids, attn_metadata)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        # sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # use random model
        if self.config.use_random_model:
            return set()
        # 权重加载逻辑保持不变
        params_dict = dict(self.model.named_parameters())
        loaded_weights = set()
        model_module_mapping = self.config.model_module_mapping
        embedding_idx = 0
        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[5:]
            mapped_name = model_module_mapping.get(
                name) if model_module_mapping else name
            if mapped_name == "":
                continue
            if mapped_name in params_dict:
                if "_linear_uvqk" in mapped_name and (tuple(
                        reversed(params_dict[mapped_name].shape))
                                                      == loaded_weight.shape):
                    params_dict[mapped_name].data.copy_(loaded_weight.T)
                else:
                    if mapped_name == "_embedding_collection._embedding_layer.weight":
                        embedding_cnt = loaded_weight.shape[0]
                        params_dict[mapped_name].data[
                            embedding_idx:embedding_idx +
                            embedding_cnt].copy_(loaded_weight)
                        embedding_idx += embedding_cnt
                    else:
                        params_dict[mapped_name].data.copy_(loaded_weight)
                loaded_weights.add("model." + mapped_name)
            else:
                print(
                    f"model_pth: {name}, mapped_name: {mapped_name}, no match")

        for name in params_dict:
            if any(keyword in name
                   for keyword in {"norm_ffn", "feed_forward"}):
                loaded_weights.add("model." + name)
        return loaded_weights
