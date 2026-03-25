import collections
import math
from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.nn as nn
import vllm.envs as envs
from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.config import ArcticSpeculativeConfig
from arctic_inference.vllm.spec_dec.logits_processor_opt import LogitsProcessorOpt
from arctic_inference.vllm.spec_dec.vocab_parallel_embedding import ParallelLMHead as ArcticParallelLMHead
from arctic_inference.vllm.spec_dec.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from arctic_inference.vllm.spec_dec.vocab_parallel_embedding import (
    VocabParallelEmbedding as ArcticVocabParallelEmbedding,
)
from vllm import ModelRegistry
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.distributed import divide
from vllm.logger import logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.mlp_speculator import MLPSpeculatorConfig
from vllm.v1.outputs import SamplerOutput

SQRT2 = 2**0.5
DEFAULT_VOCAB_PADDING_SIZE = 64

if not hasattr(envs, "VLLM_USE_V1"):
    envs.VLLM_USE_V1 = False


@contextmanager
def graph_capture(device):
    """NPU ACL Graph capture context manager."""
    stream = torch.npu.Stream(device=device)
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

        class _GraphCaptureContext:
            stream: torch.npu.Stream

        context = _GraphCaptureContext()
        context.stream = stream
        with torch.npu.stream(stream):
            yield context


def padding_size_func(size: int) -> int:
    """Round up a size to the nearest multiple of 4."""
    mult = (1 << (size - 1).bit_length()) // 4
    if mult < 1:
        return size
    return (size + mult - 1) // mult * mult


class SpeculatorTPInit:
    def __init__(self):
        self.init_tensor_parallelism()

    def init_tensor_parallelism(self):
        from vllm.distributed.parallel_state import _TP

        # Work around due to cuda graph capture failure using SP_TP_GROUP's AllGather
        self.tp_size = _TP.world_size
        self.tp_rank = _TP.rank % self.tp_size

        self.TP_GROUP = _TP


class VocabParallelEmbedding(SpeculatorTPInit, ArcticVocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig = None,
        prefix: str = "",
        skip_quantization: bool = True,
    ):
        nn.Module.__init__(self)
        SpeculatorTPInit.__init__(self)
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = self.pad_vocab_size(self.org_vocab_size, self.padding_size)
        self.num_embeddings_padded = self.pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim

        quant_method = None
        if quant_config is not None and not skip_quantization:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self.__class__) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(type(quant_method))
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
        assert self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    def pad_vocab_size(self, vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
        """Pad the vocab size to the given value."""
        return ((vocab_size + pad_to - 1) // pad_to) * pad_to


class ParallelLMHead(VocabParallelEmbedding, ArcticParallelLMHead):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig = None,
        prefix: str = "",
        skip_quantization: bool = True,
    ):
        super().__init__(
            num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )
        self.quant_config = quant_config
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)


class MLPSpeculatorLayerNorm(nn.Module, SpeculatorTPInit):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value
         fits in the range of your encoding scheme
         (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale_and_shift : bool
        Include a learned scaling and shift term after normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale_and_shift=True,
    ):
        super().__init__()
        self.elementwise_scale_and_shift = elementwise_scale_and_shift
        if self.elementwise_scale_and_shift:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale_and_shift:
            x = self.weight * x
            x = x + self.bias
        return x


def _generate_cg_key(padding_size: int, head_index: int):
    return (padding_size << 16) + head_index


class ArcticMLPSpeculator(nn.Module, SpeculatorTPInit):
    """
    An implementation of the speculative models introduced in
    "Accelerating Production LLMs with Combined Token/Embedding
    Speculators"
    https://arxiv.org/pdf/2404.19124
    Trained speculators of this type are available on HF hub at:
    https://huggingface.co/ibm-fms and https://huggingface.co/ibm-granite
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        SpeculatorTPInit.__init__(self)

        config = vllm_config.model_config.hf_config

        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 else config.emb_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.tie_weights = config.tie_weights
        self.scale_input = config.scale_input

        self.quantize_lm_head = False

        quant_config = None

        self.qhead = None
        if self.tie_weights:
            assert self.n_predict > 1, "You cannot tie weights between stages when only 1 exists"
            embedding = VocabParallelEmbedding(config.vocab_size, self.inner_dim, org_num_embeddings=config.vocab_size)
            self.emb = nn.ModuleList([embedding] * self.max_speculative_tokens)

            # the initial projection from the base model may
            # have a different size, so that stays separate.
            proj_first = nn.Linear(self.emb_dim, self.inner_dim, bias=False)
            proj_tied = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

            self.proj = nn.ModuleList([proj_first] + [proj_tied] * (self.max_speculative_tokens - 1))

            head = ParallelLMHead(
                self.vocab_size,
                self.inner_dim,
                bias=False,
                quant_config=quant_config,
                skip_quantization=True,
            )
            self.head = nn.ModuleList([head] * self.max_speculative_tokens)

            ln = MLPSpeculatorLayerNorm(self.inner_dim, elementwise_scale_and_shift=True)
            self.ln = nn.ModuleList([ln] * self.max_speculative_tokens)

        else:
            self.emb = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        self.inner_dim,
                        org_num_embeddings=config.vocab_size,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )

            self.proj = nn.ModuleList(
                [
                    nn.Linear(
                        (self.emb_dim if i == 0 else self.inner_dim),
                        self.inner_dim,
                        bias=False,
                    )
                    for i in range(self.max_speculative_tokens)
                ]
            )

            self.head = nn.ModuleList(
                [
                    ParallelLMHead(
                        self.vocab_size,
                        self.inner_dim,
                        bias=False,
                        quant_config=quant_config,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )
            self.ln = nn.ModuleList(
                [
                    MLPSpeculatorLayerNorm(self.inner_dim, elementwise_scale_and_shift=True)
                    for _ in range(self.max_speculative_tokens)
                ]
            )

        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(self.emb_dim, elementwise_scale_and_shift=False)

        self.state_weight = 0.5 ** (0.5 / config.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()
        self.config = config
        self.logits_processor = LogitsProcessorOpt(
            vocab_size=config.vocab_size,
            org_vocab_size=config.vocab_size,
            scale=1.0,
            skip_last_gather=True,
        )

        self.cuda_graph_max_batch_size = 0
        self.cuda_graph_mode = False
        if not vllm_config.model_config.enforce_eager:
            self.cuda_graph_mode = True
            self.cuda_graphs: dict[int, torch.npu.NPUGraph] = {}
            self.cuda_graph_max_batch_size = padding_size_func(vllm_config.scheduler_config.max_num_seqs)
            self.static_cuda_buffers = {
                "last_tokens": torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long),
                "previous_hidden_states": torch.empty(self.cuda_graph_max_batch_size, 1, self.emb_dim),
                "next_tokens": [
                    torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long) for _ in range(self.n_predict)
                ],
            }

    def _prepare_cuda_graph_ios(self, size, last_tokens, previous_hidden_states):
        self.static_cuda_buffers["last_tokens"][:size] = last_tokens
        if previous_hidden_states is not None:
            self.static_cuda_buffers["previous_hidden_states"][:size] = previous_hidden_states

        padded_size = padding_size_func(size)

        static_last_tokens = self.static_cuda_buffers["last_tokens"][:padded_size]
        static_hidden_states = self.static_cuda_buffers["previous_hidden_states"][:padded_size]
        return (padded_size, static_last_tokens, static_hidden_states)

    def generate_states(
        self,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        head_index: int,
    ) -> torch.Tensor:
        if head_index == 0 and self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2
        # Project and predict
        z = self.emb[head_index](last_tokens)  # b k d
        states = self.proj[head_index](previous_hidden_states)

        # Weighted add of state_weight*state and emb_weight*z
        # Let subsequent LN take care of denominator
        # state_weight is close to 1, so shouldn't be any precision issues
        states.add_(z, alpha=self.emb_weight / self.state_weight)

        states = self.activation(self.ln[head_index](states))  # b k d

        return states

    def generate_token_ids(
        self,
        batch_size: int,
        num_predict_tokens: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_tokens_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        for head_index in range(num_predict_tokens):
            states = self.generate_states(last_tokens, previous_hidden_states, head_index)
            previous_hidden_states = states
            states = states.flatten(0, 1)
            head_weight = (
                self.qhead[head_index] if self.qhead is not None and batch_size <= 32 else self.head[head_index]
            )
            logits = self.logits_processor(head_weight, states)

            if self.tp_size == 1:
                last_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, -1)
            else:
                vals, indices = torch.topk(logits, 1, dim=-1)
                indices = indices + self.tp_rank * logits.shape[-1]

                packed_data = torch.cat([vals.to(torch.float64).view(torch.int32), indices], dim=0)
                packed_data = self.TP_GROUP.all_gather(packed_data)
                vals, indices = packed_data.split(batch_size, dim=0)
                vals = vals.view(torch.float64)

                argidx = torch.argmax(vals, -1).reshape(batch_size, -1)
                last_tokens = torch.gather(indices, -1, argidx)

            if next_tokens_tensors[head_index] is None:
                next_tokens_tensors[head_index] = last_tokens
            else:
                next_tokens_tensors[head_index].copy_(last_tokens)

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> list[torch.tensor]:
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(
                f"Max speculative tokens for model is "
                f"{self.max_speculative_tokens}, but "
                f"{num_predict_tokens} were requested"
            )

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        batch_size = input_ids.size(0)

        static_next_tokens: list[torch.Tensor | None] = [None] * num_predict_tokens

        if self.cuda_graph_mode and batch_size <= self.cuda_graph_max_batch_size:
            padded_size, static_last_tokens, static_hidden_states = self._prepare_cuda_graph_ios(
                batch_size, last_tokens, previous_hidden_states
            )
            cg_key = _generate_cg_key(padded_size, 0)
            g = self.cuda_graphs.get(cg_key)

            for i in range(num_predict_tokens):
                static_next_tokens[i] = self.static_cuda_buffers["next_tokens"][i][:padded_size]

            if g is None:
                device = torch.npu.current_device()
                with graph_capture(device=device):
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g):
                        self.generate_token_ids(
                            padded_size,
                            num_predict_tokens,
                            static_last_tokens,
                            static_hidden_states,
                            static_next_tokens,
                        )

                self.cuda_graphs[cg_key] = g
            else:
                g.replay()
        else:
            self.generate_token_ids(
                batch_size,
                num_predict_tokens,
                last_tokens,
                previous_hidden_states,
                static_next_tokens,
            )

        next_tokens = []
        for i in range(num_predict_tokens):
            assert static_next_tokens[i] is not None
            next_tokens.append(static_next_tokens[i][:batch_size])

        return torch.cat(next_tokens, dim=-1)

    def maybe_load_weight(self, param, loaded_weight):
        if param is not None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = name.replace("speculator.", "")
            param = params_dict.get(name)
            self.maybe_load_weight(param, loaded_weight)

            if name.startswith("head"):
                param = params_dict.get(name.replace("head", "qhead"))
                self.maybe_load_weight(param, loaded_weight)


class ArcticLSTMSpeculator(nn.Module, SpeculatorTPInit):
    """
    An implementation of the speculative models introduced in
    "Accelerating Production LLMs with Combined Token/Embedding
    Speculators"
    https://arxiv.org/pdf/2404.19124
    Trained speculators of this type are available on HF hub at:
    https://huggingface.co/ibm-fms and https://huggingface.co/ibm-granite
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        SpeculatorTPInit.__init__(self)

        config = vllm_config.model_config.hf_config

        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.input_hidden_dim = config.input_hidden_dim
        config.inner_dim = [int(i) for i in config.inner_dim.split(".")]
        self.inner_dim = config.inner_dim
        self.emb_dim = config.emb_dim
        config.proj_dim = [int(i) for i in config.proj_dim.split(".")]
        self.proj_dim = config.proj_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.tie_weights = config.tie_weights
        self.tie_lstm_embs = config.tie_lstm_embs
        self.scale_input = config.scale_input
        self.quantize_lm_head = False

        quant_config = None
        self.method = getattr(config, "method", "sum_rnn")

        self.activation = nn.GELU()
        self.qhead = None
        if self.tie_weights:
            head = ParallelLMHead(
                self.vocab_size,
                self.inner_dim[-1],
                bias=False,
                quant_config=quant_config,
                skip_quantization=True,
            )
            self.head = nn.ModuleList([head] * self.max_speculative_tokens)
        else:
            self.head = nn.ModuleList(
                [
                    ParallelLMHead(
                        self.vocab_size,
                        self.inner_dim[-1],
                        bias=False,
                        quant_config=quant_config,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )

        if self.method == "sum_rnn":
            embs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [VocabParallelEmbedding(self.vocab_size, self.emb_dim[0])]
                    for i in range(1, len(self.emb_dim)):
                        seqs.append(MLPSpeculatorLayerNorm(self.emb_dim[i], elementwise_scale_and_shift=True))
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.emb_dim[i - 1], self.emb_dim[i], bias=False))
                    embs.append(nn.Sequential(*seqs))
            self.emb = nn.ModuleList(embs)

            projs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i <= 1:
                    seqs = [
                        nn.Linear(
                            (self.input_hidden_dim if n_i == 0 else self.inner_dim[-1]),
                            self.proj_dim[0],
                            bias=False,
                        )
                    ]
                    for i in range(1, len(self.proj_dim)):
                        seqs.append(MLPSpeculatorLayerNorm(self.proj_dim[i], elementwise_scale_and_shift=True))
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.proj_dim[i - 1], self.proj_dim[i], bias=False))
                    projs.append(nn.Sequential(*seqs))
            self.proj = nn.ModuleList(projs)

            lns = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [MLPSpeculatorLayerNorm(self.inner_dim[0], elementwise_scale_and_shift=True)]
                    for i in range(1, len(self.inner_dim)):
                        seqs.append(self.activation)
                        seqs.append(nn.Linear(self.inner_dim[i - 1], self.inner_dim[i], bias=False))
                        seqs.append(MLPSpeculatorLayerNorm(self.inner_dim[i], elementwise_scale_and_shift=True))
                    lns.append(nn.Sequential(*seqs))
            self.ln = nn.ModuleList(lns)

        elif self.method == "sum_lstm":
            assert self.tie_weights
            self.forget_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim)])
            if not self.tie_lstm_embs:
                self.input_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim)])
                self.cell_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim)])
                self.output_emb = nn.ModuleList([nn.Embedding(self.vocab_size, self.emb_dim)])
            self.projs = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0] * 4, bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0] * 4, bias=False),
                ]
            )
            self.cell_ln = nn.ModuleList([MLPSpeculatorLayerNorm(self.inner_dim[0], elementwise_scale_and_shift=True)])
            self.state_ln = nn.ModuleList([MLPSpeculatorLayerNorm(self.inner_dim[0], elementwise_scale_and_shift=True)])

        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(self.input_hidden_dim, elementwise_scale_and_shift=False)

        self.state_weight = 0.5 ** (0.5 / config.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim[0] / 2))
        self.config = config
        self.logits_processor = LogitsProcessorOpt(
            vocab_size=config.vocab_size,
            org_vocab_size=config.vocab_size,
            scale=1.0,
            skip_last_gather=True,
        )

        self.cuda_graph_max_batch_size = 0
        self.cuda_graph_mode = False

        self.cuda_graph_max_batch_size = padding_size_func(vllm_config.scheduler_config.max_num_seqs)
        self.static_cuda_buffers = {
            "last_tokens": torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long),
            "previous_hidden_states": torch.empty(self.cuda_graph_max_batch_size, 1, self.input_hidden_dim),
            "cell_states": torch.empty(self.cuda_graph_max_batch_size, 1, self.inner_dim[-1]),
            "next_tokens": [
                torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long) for _ in range(self.n_predict)
            ],
        }
        if self.inner_dim[-1] != self.input_hidden_dim:
            self.static_cuda_buffers["next_previous_hidden_states"] = torch.empty(
                self.cuda_graph_max_batch_size, 1, self.inner_dim[-1]
            )

        if not vllm_config.model_config.enforce_eager:
            self.cuda_graph_mode = True
            self.cuda_graphs: dict[int, torch.npu.NPUGraph] = {}

    def _prepare_cuda_graph_ios(
        self,
        size: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        hidden_state_buffers: torch.Tensor,
        cell_states: torch.Tensor | None = None,
        use_lstm: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor] | tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.static_cuda_buffers["last_tokens"][:size] = last_tokens
        if cell_states is not None:
            self.static_cuda_buffers["cell_states"][:size] = cell_states
        if previous_hidden_states is not None:
            hidden_state_buffers[:size] = previous_hidden_states

        padded_size = padding_size_func(size) if self.cuda_graph_mode else size

        static_last_tokens = self.static_cuda_buffers["last_tokens"][:padded_size]
        static_hidden_states = hidden_state_buffers[:padded_size]
        if use_lstm:
            static_cell_states = self.static_cuda_buffers["cell_states"][:padded_size]
            return (
                padded_size,
                static_last_tokens,
                static_hidden_states,
                static_cell_states,
            )
        else:
            return (padded_size, static_last_tokens, static_hidden_states)

    def generate_states(
        self,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        head_index: int,
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:
        if head_index == 0 and self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2

        actual_i = 0 if self.tie_weights else head_index
        actual_proj_i = 1 if self.tie_weights and head_index >= 2 else head_index

        if self.method == "sum_lstm":
            assert self.tie_lstm_embs

            prev_state = previous_hidden_states

            z4 = self.forget_emb[actual_i](last_tokens).repeat(1, 1, 4)  # b n d
            states = self.projs[actual_proj_i](prev_state)

            #   prev_state: [B, 1, D_eff] (e.g., 2880 in the first round and 4096 later)
            #   states:     [B, 1, 4*D_gate] (e.g., 4*4096)
            #   z4:         [B, 1, 4*D_gate]
            states_4d = states.flatten(0, 1).contiguous()  # [B, 4*D_gate]
            z4_4d = z4.flatten(0, 1).contiguous()  # [B, 4*D_gate]

            orig_cell_shape = cell_states.shape  # [B, 1, D_gate]
            pc_d = cell_states.flatten(0, 1).contiguous()  # [B, D_gate]

            assert states_4d.size(-1) % 4 == 0
            assert z4_4d.size(-1) == states_4d.size(-1)
            assert pc_d.size(-1) == states_4d.size(-1) // 4

            w_cell = self.cell_ln[actual_i].weight
            b_cell = self.cell_ln[actual_i].bias
            w_state = self.state_ln[actual_i].weight
            b_state = self.state_ln[actual_i].bias

            alpha = float(self.emb_weight / self.state_weight)
            eps_cell = float(self.cell_ln[actual_i].eps)
            eps_state = float(self.state_ln[actual_i].eps)
            use_fast_gelu = False

            state_d, cell_d = torch.ops._C_ascend.npu_sum_lstm(
                states_4d, z4_4d, pc_d, w_cell, b_cell, w_state, b_state, alpha, eps_cell, eps_state, use_fast_gelu
            )

            state = state_d.reshape(orig_cell_shape)  # [B, 1, D_gate]
            cell_states = cell_d.reshape(orig_cell_shape)  # [B, 1, D_gate]

            return state, cell_states
        else:
            # Project and predict
            z = self.emb[actual_i](last_tokens)  # b k d
            states = self.proj[actual_proj_i](previous_hidden_states)

            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            states.add_(z, alpha=self.emb_weight / self.state_weight)
            states = self.activation(self.ln[actual_i](states))  # b k d

            return states

    def generate_token_ids(
        self,
        batch_size: int,
        num_predict_tokens: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_tokens_tensors: list[torch.Tensor],
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:
        for head_index in range(num_predict_tokens):
            if self.method == "sum_lstm":
                states, cell_states = self.generate_states(last_tokens, previous_hidden_states, head_index, cell_states)
            else:
                states = self.generate_states(last_tokens, previous_hidden_states, head_index)
            previous_hidden_states = states
            states = states.flatten(0, 1)
            head_weight = (
                self.qhead[head_index] if self.qhead is not None and batch_size <= 32 else self.head[head_index]
            )
            logits = self.logits_processor(head_weight, states)

            if self.tp_size == 1:
                last_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, -1)
            else:
                vals, indices = torch.topk(logits, 1, dim=-1)
                indices = indices + self.tp_rank * logits.shape[-1]

                packed_data = torch.cat([vals.to(torch.float64).view(torch.int32), indices], dim=0)
                packed_data = self.TP_GROUP.all_gather(packed_data)
                vals, indices = packed_data.split(batch_size, dim=0)
                vals = vals.view(torch.float64)

                argidx = torch.argmax(vals, -1).reshape(batch_size, -1)
                last_tokens = torch.gather(indices, -1, argidx)

            if next_tokens_tensors[head_index] is None:
                next_tokens_tensors[head_index] = last_tokens
            else:
                next_tokens_tensors[head_index].copy_(last_tokens)

        return next_tokens_tensors

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> list[SamplerOutput]:
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(
                f"Max speculative tokens for model is "
                f"{self.max_speculative_tokens}, but "
                f"{num_predict_tokens} were requested"
            )

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        batch_size = input_ids.size(0)
        state_shapes = list(previous_hidden_states.shape)
        state_shapes[-1] = self.inner_dim[-1]

        static_next_tokens: list[torch.Tensor | None] = [None] * num_predict_tokens
        static_cell_states: torch.Tensor | None = None
        static_last_tokens: torch.Tensor | None = None
        static_hidden_states: torch.Tensor | None = None

        static_states = self.static_cuda_buffers["previous_hidden_states"]
        if self.method == "sum_lstm":
            previous_cell_states = torch.zeros(
                state_shapes,
                device=previous_hidden_states.device,
                dtype=previous_hidden_states.dtype,
            )
            (
                padded_size,
                static_last_tokens,
                static_hidden_states,
                static_cell_states,
            ) = self._prepare_cuda_graph_ios(
                batch_size,
                last_tokens,
                previous_hidden_states,
                static_states,
                previous_cell_states,
                use_lstm=True,
            )
        else:
            padded_size, static_last_tokens, static_hidden_states = self._prepare_cuda_graph_ios(
                batch_size, last_tokens, previous_hidden_states, static_states
            )

        if self.cuda_graph_mode and batch_size <= self.cuda_graph_max_batch_size:
            cg_key = _generate_cg_key(padded_size, 0)
            g = self.cuda_graphs.get(cg_key)

            static_states = (
                self.static_cuda_buffers["next_previous_hidden_states"]
                if self.inner_dim[-1] != self.input_hidden_dim
                else self.static_cuda_buffers["previous_hidden_states"]
            )

            for i in range(num_predict_tokens):
                static_next_tokens[i] = self.static_cuda_buffers["next_tokens"][i][:padded_size]

            if g is None:
                device = torch.npu.current_device()
                for i in range(num_predict_tokens):
                    self.static_cuda_buffers["next_tokens"][i][:padded_size] = torch.zeros(
                        (padded_size, 1), dtype=torch.long, device=device
                    )
                with graph_capture(device=device):
                    g = torch.npu.NPUGraph()
                    with torch.npu.graph(g):
                        if self.method == "sum_lstm":
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                                cell_states=static_cell_states,
                            )
                        else:
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                            )
                self.cuda_graphs[cg_key] = g
            else:
                g.replay()
        else:
            if self.method == "sum_lstm":
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                    cell_states=static_cell_states,
                )
            else:
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                )

        next_tokens = []
        for i in range(num_predict_tokens):
            token_tensor = static_next_tokens[i]
            assert token_tensor is not None
            next_tokens.append(token_tensor[:batch_size])

        return torch.cat(next_tokens, dim=-1)

    def maybe_load_weight(self, param, loaded_weight):
        if param is not None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_dict: dict[str, torch.Tensor] = collections.OrderedDict(weights)
        if self.method == "sum_lstm" and self.tie_lstm_embs:
            try:
                weights_dict.pop("input_emb.0.weight")
                weights_dict.pop("cell_emb.0.weight")
                weights_dict.pop("output_emb.0.weight")
            except KeyError:
                # If the weights are not present, it means they are not tied
                # and we should not try to pop them.
                logger.warning("No tied LSTM embeddings found, skipping.")
                pass
            for name, param in self.named_parameters():
                if "projs." in name:
                    logger.info(f"REPLACING {name}")
                    forget_proj = weights_dict.pop(name.replace("projs", "forget_proj"))
                    input_proj = weights_dict.pop(name.replace("projs", "input_proj"))
                    output_proj = weights_dict.pop(name.replace("projs", "output_proj"))
                    cell_proj = weights_dict.pop(name.replace("projs", "cell_proj"))
                    weights_dict[name] = torch.cat([forget_proj, input_proj, output_proj, cell_proj])

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights_dict.items():
            logger.info(f"LOADING {name}")
            name = name.replace("speculator.", "")
            param = params_dict.get(name)
            self.maybe_load_weight(param, loaded_weight)

            if name.startswith("head"):
                param = params_dict.get(name.replace("head", "qhead"))
                self.maybe_load_weight(param, loaded_weight)


class SpeculativeConfigPatch(ArcticPatch[SpeculativeConfig]):
    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an ArcticSpeculativeConfig instead of a
        # SpeculativeConfig when creating a new instance of the class.
        if cls is SpeculativeConfig:
            return ArcticSpeculativeConfig.__new__(ArcticSpeculativeConfig, *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        is_arctic_method = self.method in ("arctic", "mlp_speculator")
        use_suffix = (self.method == "suffix") or (self.method is None and self.enable_suffix_decoding)
        use_hybrid = self.method == "arctic" and self.enable_suffix_decoding

        if use_hybrid:
            self.suffix_speculative_tokens = self.suffix_cache_max_depth

        if use_suffix:
            self.method = "suffix"
            self.enable_suffix_decoding = True
            # Use suffix_speculative_tokens if explicitly set, otherwise
            # default to 16 (not suffix_cache_max_depth which can be very
            # large and makes every step process 1+N tokens even when the
            # suffix cache has no matches).
            # NOTE: num_speculative_tokens defaults to None (not 0).
            if self.suffix_speculative_tokens > 0:
                self.num_speculative_tokens = self.suffix_speculative_tokens
            elif self.num_speculative_tokens is None:
                self.num_speculative_tokens = 16
            self._verify_args()
            return

        if is_arctic_method:
            actual_draft_model = getattr(self, "draft_model", None)
            actual_method = "arctic"

            self.draft_model = None
            self.max_model_len = None
            self.draft_tensor_parallel_size = 1

            try:
                self._orig_post_init()
            finally:
                self.draft_model = actual_draft_model
                self.method = actual_method

            if self.num_speculative_tokens == 0:
                self.num_speculative_tokens = getattr(self, "num_lookahead_slots", 1)
        else:
            self._orig_post_init()


class MLPSpeculatorConfigPatch(ArcticPatch[MLPSpeculatorConfig]):
    _orig_init = MLPSpeculatorConfig.__init__

    def __init__(self, *args, **kwargs):
        self.base_model_arch = kwargs.pop("base_model_arch", "")
        self._orig_init(*args, **kwargs)

        # Inject dummy attributes required by vLLM's ModelArchConfigConvertor
        # The converter tries to calculate head_size = hidden_size // num_attention_heads
        if not hasattr(self, "num_attention_heads"):
            self.num_attention_heads = 1

        if not hasattr(self, "hidden_size"):
            # Fallback to n_embd if present, otherwise default to a safe dummy value
            self.hidden_size = getattr(self, "n_embd", 1024)

        # Ensure hidden_size is an integer to prevent TypeError during division
        if hasattr(self, "hidden_size"):
            self.hidden_size = int(self.hidden_size)


def apply_arctic_patches():
    from arctic_inference.vllm.config import VllmConfigPatch

    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel", ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel", ArcticLSTMSpeculator)

    VllmConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()
    MLPSpeculatorConfigPatch.apply_patch()


apply_arctic_patches()
