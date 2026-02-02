from collections import defaultdict
from vllm.model_executor.models.utils import extract_layer_index
import torch
from vllm.attention.layer import Attention
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional
import torch_npu

import numpy as np
import logging
logger = logging.getLogger(__name__)

# largely follow vllm.v1.worker.utils.bind_kv_cache
def bind_hashk_cache(
    hashk_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_hashk_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated hashk cache to both ModelRunner and forward context so
    that the hashk cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's hashk cache list (`runner_hashk_caches`) with
         hashk_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding hashk cache in hashk_caches.

    Args:
        hashk_caches: The allocated hashk_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_hashk_caches: The hashk cache declared by ModelRunner.
    """
    # Bind hashk_caches to ModelRunner; ensure it is empty before binding
    assert len(runner_hashk_caches) == 0

    # Convert hashk_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in hashk_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # (TODO: ldeng), support multiple hashk caches for the same layer index later, e.g., encoder-decoder models.
        layer_name = layer_names[0]
        runner_hashk_caches.append(hashk_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, hashk_cache in hashk_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].hashk_cache = [hashk_cache]

def bind_hashk_cache_nope(
    hashk_caches_nope: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_hashk_caches_nope: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated hashk cache for nope in MLA to both ModelRunner and forward context so
    that the hashk cache for nope can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's hashk cache list (`runner_hashk_caches_nope`) with
         hashk_caches_nope.
      2) Associates each attention layer in the `forward_context` with its
         corresponding hashk cache for nope in MLA in hashk_caches_nope.

    Args:
        hashk_caches_nope: The allocated hashk_caches_nope with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_hashk_caches_nope: The hashk cache for nope declared by ModelRunner.
    """
    # Bind hashk_caches_nope to ModelRunner; ensure it is empty before binding
    assert len(runner_hashk_caches_nope) == 0

    # Convert hashk_caches_nope dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in hashk_caches_nope:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # (TODO: ldeng), support multiple hashk caches for the same layer index later, e.g., encoder-decoder models.
        layer_name = layer_names[0]
        runner_hashk_caches_nope.append(hashk_caches_nope[layer_name])

    # Bind hashk_caches_nope to forward context
    for layer_name, hashk_cache_nope in hashk_caches_nope.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].hashk_cache_nope = [hashk_cache_nope]

def bind_hashk_cache_rope(
    hashk_caches_rope: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_hashk_caches_rope: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    """
    Bind the allocated hashk cache for rope in MLA to both ModelRunner and forward context so
    that the hashk cache for rope can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's hashk cache list (`runner_hashk_caches_rope`) with
         hashk_caches_rope.
      2) Associates each attention layer in the `forward_context` with its
         corresponding hashk cache for rope in MLA in hashk_caches_rope.

    Args:
        hashk_caches_rope: The allocated hashk_caches_rope with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_hashk_caches_rope: The hashk cache for rope declared by ModelRunner.
    """
    # Bind hashk_caches_rope to ModelRunner; ensure it is empty before binding
    assert len(runner_hashk_caches_rope) == 0

    # Convert hashk_caches_rope dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in hashk_caches_rope:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # (TODO: ldeng), support multiple hashk caches for the same layer index later, e.g., encoder-decoder models.
        layer_name = layer_names[0]
        runner_hashk_caches_rope.append(hashk_caches_rope[layer_name])

    # Bind hashk_caches_rope to forward context
    for layer_name, hashk_cache_rope in hashk_caches_rope.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].hashk_cache_rope = [hashk_cache_rope]



@dataclass
class KVCompConfig:
    """
    Dataclass representing the configuration for KVComp.
    """

    model_name: str = "DummyModel"
    is_mla: bool = False

    # either "random" or "fixed"
    hash_weight_type: str = "random"

    num_hidden_layers: int = 36

    # the minimal seq_len to trigger KVComp
    seq_len_threshhold: int = 2048

    # any value divisible by 128
    chunk_size: int = 128

    # either "max", "min" or "sum"
    chunk_repre_method: str = "max"

    head_dim: int = 128
    hash_bits: int = 128

    top_k_ratio_per_layer: List[float] = field(default_factory=lambda: [0.3] * 36)
    top_k_index_reuse: List[int] = field(default_factory=lambda: [-1] * 36)

    # nonnegative means slicing from the start, negative means slicing from the end
    must_select_blocks: List[int] = field(default_factory=lambda: [0, -2, -1])

    # used when is_mla=True and hash_weight_type="fixed"
    hash_weight: Optional[List[List[float]]] = None

    # Conditional fields if is_mla=True
    kv_lora_rank: Optional[int] = None  # we need to specify it if is_mla=True
    qk_rope_head_dim: Optional[int] = None  # we need to specify it if is_mla=True
    hash_bits_kv_lora: Optional[int] = None  # we need to specify it if is_mla=True
    hash_bits_qk_rope: Optional[int] = None  # we need to specify it if is_mla=True
    hash_weight_kv_lora: Optional[List[List[float]]] = (
        None  # used when is_mla=True and hash_weight_type="fixed"
    )
    hash_weight_qk_rope: Optional[List[List[float]]] = (
        None  # used when is_mla=True and hash_weight_type="fixed"
    )

    vllm_hash_attention_topk: Optional[int] = None
    vllm_hash_attention_reduction_head_num: Optional[int] = None
    vllm_hash_attention_rollback_layers: List[int] = field(
        default_factory=lambda: []
    )  # layers to rollback, empty means no rollback
    vllm_hash_attention_skip_layers: List[int] = field(
        default_factory=lambda: []
    )  # layers to skip, empty means no skip

    # generate non-MLA config data
    def generate_config_data(
        self,
        model_name: str,
        hash_weight_type: str,
        num_hidden_layers: int,
        seq_len_threshhold: int,
        chunk_size: int,
        chunk_repre_method: str,
        head_dim: int,
        hash_bits: int,
        top_k_ratio_per_layer: List[float],
        top_k_index_reuse: List[int],
        must_select_blocks: List[int],
    ) -> None:
        self.is_mla = False
        self.model_name = model_name

        if hash_weight_type not in ["uniform", "fixed"]:
            raise ValueError(
                f"hash_weight_type should be either 'random' or 'fixed', but got {hash_weight_type}"
            )
        self.hash_weight_type = hash_weight_type

        self.num_hidden_layers = num_hidden_layers
        self.seq_len_threshhold = seq_len_threshhold

        if chunk_size % 128 != 0:
            raise ValueError(
                f"chunk_size should be divisible by 128, but got {chunk_size}"
            )
        self.chunk_size = chunk_size

        if chunk_repre_method not in ["max", "min", "sum"]:
            raise ValueError(
                f"chunk_repre_method should be either 'max', 'min' or 'sum', but got {chunk_repre_method}"
            )
        self.chunk_repre_method = chunk_repre_method

        self.head_dim = head_dim
        self.hash_bits = hash_bits

        if len(top_k_ratio_per_layer) != num_hidden_layers:
            raise ValueError(
                f"top_k_ratio_per_layer length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_ratio_per_layer)}"
            )
        self.top_k_ratio_per_layer = top_k_ratio_per_layer
        if len(top_k_index_reuse) != num_hidden_layers:
            raise ValueError(
                f"top_k_index_reuse length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_index_reuse)}"
            )
        self.top_k_index_reuse = top_k_index_reuse

        self.must_select_blocks = must_select_blocks

        if hash_weight_type == "random":
            logger.info(
                "hash_weight_type is 'random', hash_weight will be generated automatically."
            )
            self.hash_weight = None
        else:
            logger.warning(
                "hash_weight_type is 'fixed', please manually set hash_weight in the config json file."
            )

    # generate MLA config data
    def generate_mla_config_data(
        self,
        model_name: str,
        hash_weight_type: str,
        num_hidden_layers: int,
        seq_len_threshhold: int,
        chunk_size: int,
        chunk_repre_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        hash_bits_kv_lora: int,
        hash_bits_qk_rope: int,
        top_k_ratio_per_layer: List[float],
        top_k_index_reuse: List[int],
        must_select_blocks: List[int],
    ) -> None:
        self.is_mla = True
        self.model_name = model_name
        if hash_weight_type not in ["random", "fixed"]:
            raise ValueError(
                f"hash_weight_type should be either 'random' or 'fixed', but got {hash_weight_type}"
            )
        self.hash_weight_type = hash_weight_type

        self.num_hidden_layers = num_hidden_layers
        self.seq_len_threshhold = seq_len_threshhold
        if chunk_size % 128 != 0:
            raise ValueError(
                f"chunk_size should be divisible by 128, but got {chunk_size}"
            )
        self.chunk_size = chunk_size
        if chunk_repre_method not in ["max", "min", "sum"]:
            raise ValueError(
                f"chunk_repre_method should be either 'max', 'min' or 'sum', but got {chunk_repre_method}"
            )
        self.chunk_repre_method = chunk_repre_method
        self.head_dim = qk_rope_head_dim + kv_lora_rank
        self.hash_bits = hash_bits_qk_rope + hash_bits_kv_lora
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.hash_bits_kv_lora = hash_bits_kv_lora
        self.hash_bits_qk_rope = hash_bits_qk_rope

        if len(top_k_ratio_per_layer) != num_hidden_layers:
            raise ValueError(
                f"top_k_ratio_per_layer length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_ratio_per_layer)}"
            )
        self.top_k_ratio_per_layer = top_k_ratio_per_layer
        if len(top_k_index_reuse) != num_hidden_layers:
            raise ValueError(
                f"top_k_index_reuse length should be equal to num_hidden_layers={num_hidden_layers}, but got {len(top_k_index_reuse)}"
            )
        self.top_k_index_reuse = top_k_index_reuse

        self.must_select_blocks = must_select_blocks

        if hash_weight_type == "random":
            logger.info(
                "hash_weight_type is 'random', hash_weight_kv_lora and hash_weight_qk_rope will be generated automatically."
            )
            self.hash_weight = None
            self.hash_weight_kv_lora = None
            self.hash_weight_qk_rope = None
        else:
            logger.warning(
                "hash_weight_type is 'fixed', please manually set hash_weight_kv_lora and hash_weight_qk_rope in the config json file."
            )

    # set hash_weight when hash_weight_type is "fixed" for non-MLA models
    def set_hash_weight(self, hash_weight: List[List[float]]) -> None:
        if self.hash_weight_type != "fixed":
            raise ValueError(
                "hash_weight can only be set when hash_weight_type is 'fixed'"
            )

        if len(hash_weight) != self.head_dim or len(hash_weight[0]) != self.hash_bits:
            raise ValueError(
                f"hash_weight shape should be ({self.head_dim}, {self.hash_bits}), but got ({len(hash_weight)}, {len(hash_weight[0])})"
            )

        self.hash_weight = hash_weight

    # set hash_weight when hash_weight_type is "fixed" for MLA models
    def set_mla_hash_weight(
        self,
        hash_weight_kv_lora: List[List[float]],
        hash_weight_qk_rope: List[List[float]],
    ) -> None:
        if self.hash_weight_type != "fixed":
            raise ValueError(
                "hash_weight can only be set when hash_weight_type is 'fixed'"
            )

        if (
            len(hash_weight_kv_lora) != self.kv_lora_rank
            or len(hash_weight_kv_lora[0]) != self.hash_bits_kv_lora
        ):
            raise ValueError(
                f"hash_weight_kv_lora shape should be ({self.kv_lora_rank}, {self.hash_bits_kv_lora}), but got ({len(hash_weight_kv_lora)}, {len(hash_weight_kv_lora[0])})"
            )

        if (
            len(hash_weight_qk_rope) != self.qk_rope_head_dim
            or len(hash_weight_qk_rope[0]) != self.hash_bits_qk_rope
        ):
            raise ValueError(
                f"hash_weight_qk_rope shape should be ({self.qk_rope_head_dim}, {self.hash_bits_qk_rope}), but got ({len(hash_weight_qk_rope)}, {len(hash_weight_qk_rope[0])})"
            )

        self.hash_weight_kv_lora = hash_weight_kv_lora
        self.hash_weight_qk_rope = hash_weight_qk_rope

    def to_json(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, file_path: str) -> "KVCompConfig":
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class HashEncoder:
    """
    HashEncoder converts a float tensor to a binary hash code tensor,
    and it packs every 8 bits into a uint8 number.
    """

    def __init__(
        self, input_dim: int, hash_bits: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        self.input_dim = input_dim

        if hash_bits % 8 != 0:
            raise ValueError("hash_bits must be a multiple of 8")

        self.hash_bits = hash_bits

        # number of uint8 numbers to store hash_bits bits
        self.hash_numbers = self.hash_bits // 8

        self.dtype = dtype
        self.device = device

        assert self.device.type == "npu"

        if dtype not in [torch.float16, torch.float32, torch.float64]:
            logger.warning(
                "NPU only supports float16, float32 and float64 for hash_weights"
            )
            logger.warning("automatically using float16 for hash_weights now")
            self.dtype = torch.float16

        self._init_hash_weights()


    def _init_hash_weights(self):
        # Step 1: 随机高斯矩阵
        random_weights = torch.normal(
            mean=0,
            std=2,
            size=(self.input_dim, self.hash_bits),
            dtype=self.dtype,
            device=self.device,
        )
        # Step 2: QR分解
        Q, R = torch.linalg.qr(random_weights)

        # Step 3: 调整符号，保证Haar 分布
        d = torch.sign(torch.diag(R))
        self.hash_weights = Q * d

    def set_hash_weight(self, hash_weights: torch.Tensor) -> None:
        if hash_weights.shape != (self.input_dim, self.hash_bits):
            raise ValueError(
                f"hash_weights shape {hash_weights.shape} does not match required shape {(self.input_dim, self.hash_bits)}"
            )
        if hash_weights.dtype != self.dtype:
            raise ValueError(
                f"hash_weights dtype {hash_weights.dtype} does not match required dtype {self.dtype}"
            )
        if hash_weights.device != self.device:
            raise ValueError(
                f"hash_weights device {hash_weights.device} does not match required device {self.device}"
            )

        self.hash_weights.copy_(hash_weights)



    def compute_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hash code for input tensor x.
        Args:
            x: input tensor of shape (..., input_dim)
        Returns:
            A tensor of shape (..., hash_numbers=hash_bits // 8) representing the hash codes.
            Each element is a uint8 number representing 8 bits of the hash code.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must be of shape (..., {self.input_dim}), but got {x.shape}"
            )
        if x.device != self.device:
            raise ValueError(
                f"x device {x.device} does not match required device {self.device}"
            )

        # original shape without the last dimension
        # e.g. x.shape=[s1,s2,s3,input_dim], orig_shape=[s1,s2,s3]
        orig_shape = x.shape[:-1]

        # [N, input_dim], e.g., N = s1*s2*s3
        x_flat = x.reshape(-1, self.input_dim)

        if x_flat.dtype != self.dtype:
            x_flat = x_flat.to(self.dtype)

        # [N, hash_bits]
        xW = torch.matmul(x_flat, self.hash_weights)
        # [N * hash_bits]
        xW_flat = xW.view(-1)
        # [N*hash_numbers], where hash_numbers = hash_bits // 8
        packed_codes_flat = torch_npu.npu_sign_bits_pack(xW_flat, size=1)

        # e.g., [s1, s2, s3, hash_numbers]
        out_shape = orig_shape + (self.hash_numbers,)
        packed_codes = packed_codes_flat.view(out_shape)

        return packed_codes

    def _unpack_hash(self, packed_codes: torch.Tensor) -> torch.Tensor:
        """
        Unpack the hash codes to +1 or -1 bits.
        Args:
            packed_codes: input tensor of shape (..., hash_numbers), dtype=torch.uint8
        Returns:
            A tensor of shape (..., hash_bits=hash_numbers*8) representing the unpacked bits.
            Each element is either -1 or 1.
        """
        if packed_codes.shape[-1] != self.hash_numbers:
            raise ValueError(
                f"packed_codes must be of shape (..., {self.hash_numbers}), but got {packed_codes.shape}"
            )
        if packed_codes.device != self.device:
            raise ValueError(
                f"packed_codes device {packed_codes.device} does not match required device {self.device}"
            )
        if packed_codes.dtype != torch.uint8:
            raise ValueError(
                f"packed_codes dtype {packed_codes.dtype} is not torch.uint8"
            )

        # e.g., packed_codes.shape=[s1, s2, s3, hash_numbers]
        # orig_shape = [s1, s2, s3]
        orig_shape = packed_codes.shape[:-1]

        # [N * hash_numbers], e.g., N = s1*s2*s3
        packed_codes_flat = packed_codes.view(-1)

        # [N * hash_bits]
        unpacked_bits_flat = torch_npu.npu_sign_bits_unpack(
            packed_codes_flat, size=1, dtype=torch.float16
        )
    

        out_shape = orig_shape + (self.hash_bits,)
        unpacked_bits = unpacked_bits_flat.view(out_shape)

        return unpacked_bits



@dataclass
class KVCompMetaData:
    # for both GQA and MLA
    kvcomp_config: KVCompConfig 
    chunk_sizes_for_hamming_full: torch.Tensor 
    topk_for_hamming_full: torch.Tensor 
    topk_for_hamming_full_cpu: torch.Tensor 
    seq_lens_for_hamming: torch.Tensor 
    hamming_output: torch.Tensor 

    # for GQA
    hash_encoder: Optional[HashEncoder] = None
    hashk_caches: Optional[list[torch.Tensor]] = None

    # for MLA
    hash_encoder_nope: Optional[HashEncoder] = None
    hash_encoder_rope: Optional[HashEncoder] = None
    hashk_cache_nope: Optional[list[torch.Tensor]] = None
    hashk_cache_rope: Optional[list[torch.Tensor]] = None