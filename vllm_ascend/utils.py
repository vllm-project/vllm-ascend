#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/worker.py
#

import atexit
import functools
import math
from contextlib import contextmanager
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch_npu  # noqa: F401  # noqa: F401
from packaging.version import InvalidVersion, Version
from torch_npu.npu.streams import Event
from vllm.logger import logger

import vllm_ascend.envs as envs
from vllm_ascend.ascend_config import get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

# NOTE: Currently, we can only capture 1920 graphs at most,
# due to the limitation of ACL graph. This number is bounded by
# the number of streams, which is 2048, we save 128 streams
# as a buffer.
# Maximum number of graphs that can be captured by ACL Graph
MAX_CAPTURE_SIZE = 1920

ASCEND_QUATIZATION_METHOD = "ascend"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

_CUSTOM_OP_ENABLED = None
_IS_310P = None
_SLEEP_MODE_ENABLED = None
_CURRENT_STREAM = None
_ASCEND_CUSTOMOP_IS_REIGISTERED = False


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        from vllm_ascend import _build_info  # type: ignore
        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P


def sleep_mode_enabled():
    global _SLEEP_MODE_ENABLED
    if _SLEEP_MODE_ENABLED is None:
        from vllm_ascend import _build_info  # type: ignore
        _SLEEP_MODE_ENABLED = _build_info.__sleep_mode_enabled__
    return _SLEEP_MODE_ENABLED


def _round_up(x: int, align: int):
    # round up x to align, for example, if align is 16, x will be rounded up to 16, 32, 48, etc.
    # input: 15, 16 -> output: 16
    # input: 17, 16 -> output: 32
    # input: 30, 16 -> output: 32
    # input: 33, 16 -> output: 48
    # ...
    return (x + align - 1) // align * align


def _custom_pad(x, pad_dims):
    # pad the input tensor to the shape of pad_dims
    # input: (13, 30), pad_dims: [0, 2, 0, 3]
    # output: (16, 32)
    return torch.nn.functional.pad(x, pad_dims)


def _custom_reshape(x, target_shape):
    # reshape the input tensor to the shape of target_shape
    # input: (16, 32), target_shape: [1, 16, 2, 16]
    # output: (1, 16, 2, 16)
    return x.reshape(target_shape)


def _custom_transpose(x, dim1, dim2):
    # transpose the input tensor
    # input: (1, 16, 2, 16), dim1: 1, dim2: 2
    # output: (1, 2, 16, 16)
    return x.transpose(dim1, dim2)


def nd_to_nz_2d(in_tensor: torch.Tensor) -> torch.Tensor:
    # in_tensor: (13, 30)
    aux_dims = [1, 0, 0, 16]
    # aux_dims[1]: 16
    aux_dims[1] = _round_up(in_tensor.size(0), 16)
    # aux_dims[2]: 2
    aux_dims[2] = _round_up(in_tensor.size(1), 16) // 16

    # after: aux_dims: [1, 16, 2, 16]

    pad_dims = [0, 0, 0, 0]
    # pad_dims[1]: 2
    pad_dims[1] = _round_up(in_tensor.size(1), 16) - in_tensor.size(1)
    # pad_dims[3]: 3
    pad_dims[3] = _round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    # after: pad_dims: [0, 2, 0, 3]

    # return: (1, 2, 16, 16)
    return _custom_transpose(
        _custom_reshape(_custom_pad(in_tensor, pad_dims), aux_dims), 1,
        2).contiguous()


def nd_to_nz_spec(mask_tensor: torch.Tensor) -> torch.Tensor:
    num_tokens = mask_tensor.shape[0]
    max_seq_len = mask_tensor.shape[1]

    tokens_pad = (num_tokens + 15) // 16 * 16
    max_seq_len_pad = (max_seq_len + 15) // 16 * 16

    mask_tensor_pad = \
        torch.zeros((1, tokens_pad, max_seq_len_pad), dtype=mask_tensor.dtype, device=mask_tensor.device)
    mask_tensor_pad[0][:num_tokens, :max_seq_len] = mask_tensor
    mask = mask_tensor_pad.reshape(
        (1, tokens_pad, max_seq_len_pad // 16, 16)).permute(0, 2, 1, 3)
    return mask


def aligned_16(tensor: torch.Tensor):
    """Aligned tensor for 310P"""

    # Get the size of the current 0th dimension
    n = tensor.size(0)

    # Calculate the aligned size
    n_aligned = ((n + 15) // 16) * 16

    # If already aligned, return the original tensor
    if n == n_aligned:
        return tensor

    # Create a new tensor with shape (n_aligned, H, W) and fill it with zeros
    new_tensor = torch.zeros(n_aligned,
                             *tensor.shape[1:],
                             dtype=tensor.dtype,
                             device=tensor.device)

    # Copy the original tensor to the first N positions of the new tensor
    new_tensor[:n] = tensor

    return new_tensor


def maybe_converting_weight_acl_format(model, format=ACL_FORMAT_FRACTAL_NZ):
    # currently, there are some operations which do not support ACL_FORMAT_FRACTAL_NZ
    # in eager mode but support it in torchair graph mode. since ACL_FORMAT_FRACTAL_NZ
    # is much more preferred than ACL_FORMAT_FRACTAL_ND on 300I Duo, we add this
    # conversion when using torchair graph mode on 300I Duo platform.
    # TODO: we will remove this conversion if npu_quant_grouped_matmul_dequant
    # accepts weight format of ACL_FORMAT_FRACTAL_NZ in eager mode.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    use_torchair = get_ascend_config().torchair_graph_config.enabled
    if not is_310p() or not use_torchair:
        return
    for module in model.modules():
        if isinstance(module, FusedMoE):
            if torch_npu.get_npu_format(module.w13_weight.data) == format:
                return
            module.w13_weight.data = torch_npu.npu_format_cast(
                module.w13_weight.data, format)
            module.w2_weight.data = torch_npu.npu_format_cast(
                module.w2_weight.data, format)


def try_register_lib(lib_name: str, lib_info: str = ""):
    import importlib
    import importlib.util
    try:
        module_spec = importlib.util.find_spec(lib_name)
        if module_spec is not None:
            importlib.import_module(lib_name)
            if lib_info:
                logger.info(lib_info)
    except Exception:
        pass


def enable_custom_op():
    """
    Enable lazy init for vllm_ascend_C to avoid early initialization of CANN's RTS component. 
    Ensure that ASCEND_RT_VISIBLE_DEVICES can be dynamically modified before torch.npu.set_device().
    """
    global _CUSTOM_OP_ENABLED
    if _CUSTOM_OP_ENABLED is not None:
        return _CUSTOM_OP_ENABLED
    try:
        # isort: off
        # register custom ops into torch_library here
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
        # register the meta implementation for custom kernel if necessary
        import vllm_ascend.meta_registration  # type: ignore  # noqa: F401
        # isort: on
        _CUSTOM_OP_ENABLED = True
    except ImportError:
        _CUSTOM_OP_ENABLED = False
        logger.warning(
            "Warning: Failed to register custom ops, all custom ops will be disabled"
        )
    return _CUSTOM_OP_ENABLED


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = envs.HCCL_SO_PATH

    # manually load the hccl library
    if so_file:
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s",
                    so_file)
    else:
        if torch.version.cann is not None:
            so_file = "libhccl.so"
        else:
            raise ValueError("HCCL only supports Ascend NPU backends.")
        logger.info("Found hccl from library %s", so_file)
    return so_file


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _CURRENT_STREAM
    if _CURRENT_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _CURRENT_STREAM = torch.npu.current_stream()
    return _CURRENT_STREAM


def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401


@functools.cache
def vllm_version_is(target_vllm_version: str):
    if envs.VLLM_VERSION is not None:
        vllm_version = envs.VLLM_VERSION
    else:
        import vllm
        vllm_version = vllm.__version__
    try:
        return Version(vllm_version) == Version(target_vllm_version)
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the value follows the "
            "format of x.y.z.")


def get_max_hidden_layers(hf_config) -> int:
    cfg_dict = hf_config.to_dict()
    layer_counts = []

    def _rec_find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "num_hidden_layers" and isinstance(v, int):
                    layer_counts.append(v)
                else:
                    _rec_find(v)

    _rec_find(cfg_dict)
    if not layer_counts:
        raise ValueError("Not found num_hidden_layers in model config.")
    return max(layer_counts)


def update_aclgraph_sizes(vllm_config: VllmConfig) -> None:
    """Update ACL graph capture sizes based on hardware limitations"""
    # Store original configuration and temporarily clear it
    compilation_config = vllm_config.compilation_config
    original_sizes, compilation_config.cudagraph_capture_sizes = \
        compilation_config.cudagraph_capture_sizes, None

    # Calculate parallel configuration factor
    hf_config = vllm_config.model_config.hf_config
    if hasattr(hf_config, 'num_hidden_layers'):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # TODO: Find out whether we need to take into account the pp_size
    parallel_factor = 1 + sum(size > 1 for size in [
        parallel_config.data_parallel_size_local,
        parallel_config.tensor_parallel_size,
    ])

    # Calculate maximum supported batch sizes considering model architecture
    max_num_batch_sizes = math.floor(MAX_CAPTURE_SIZE /
                                     (num_hidden_layers + 1) / parallel_factor)
    logger.info("Calculated maximum supported batch sizes for ACL graph: %s",
                max_num_batch_sizes)

    # If original sizes exceed maximum, sample a representative subset
    if max_num_batch_sizes < len(original_sizes):
        # Sample uniformly from original sizes
        step = (len(original_sizes) - 1) / (max_num_batch_sizes - 1)
        indices = [round(i * step) for i in range(max_num_batch_sizes)]

        # Ensure first and last elements are preserved
        indices[0], indices[-1] = 0, len(original_sizes) - 1

        sampled_sizes = [original_sizes[i] for i in indices]
        compilation_config.init_with_cudagraph_sizes(sampled_sizes)

        logger.info(
            "Adjusted ACL graph batch sizes for %s model (layers: %d): %d → %d sizes",
            vllm_config.model_config.architectures[0],
            num_hidden_layers,
            len(original_sizes),
            len(compilation_config.
                cudagraph_capture_sizes  # type: ignore[arg-type]
                ))
    else:
        # No adjustment needed
        compilation_config.cudagraph_capture_sizes = original_sizes
        logger.info(
            "No adjustment needed for ACL graph batch sizes: %s model (layers: %d) with %d sizes",
            vllm_config.model_config.architectures[0], num_hidden_layers,
            len(original_sizes))


# TODO(wxy): Move to ops module
def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0, ), device=x.device, dtype=x.dtype))


class ProfileExecuteDuration:
    _instance = None
    _observations: List[Tuple[str, Event, Event]] = []
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                atexit.register(cls._instance.destroy)
            return cls._instance

    def destroy(self):
        with self._lock:
            self._observations.clear()

    @contextmanager
    def capture_async(self, duration_tag: str):
        if not envs.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
            yield
            return

        observe_start = Event(enable_timing=True)
        observe_start.record()
        try:
            yield
        finally:
            observe_end = Event(enable_timing=True)
            observe_end.record()
            with self._lock:
                self._observations.append(
                    (duration_tag, observe_start, observe_end))

    def pop_captured_sync(self) -> dict:
        """Pop and synchronize all events in the observation list"""
        durations: dict[str, float] = {}
        if not envs.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
            return durations

        while self._observations:
            with self._lock:
                tag, observe_start, observe_end = self._observations.pop()
            observe_end.synchronize()
            durations[tag] = observe_start.elapsed_time(observe_end)

        return durations


# TODO(wxy): Move to ops module
def npu_prefetch(input: torch.Tensor,
                 dependency: torch.Tensor,
                 max_size: int = 0,
                 *,
                 enabled: bool = True):
    if not enabled:
        return
    input_size = input.element_size() * input.numel()
    if max_size <= 0 or max_size > input_size:
        max_size = input_size
    torch_npu.npu_prefetch(input, dependency, max_size)


# TODO(ttanzhiqiang): rm_router_logits
# dp>1 will trigger
# In theory, this solution is only applicable to AllGather and AllGatherEP, because in the dp scenario, the previous operation was gate + two communications, and now it is changed to one communication + gate operation, which can save some communication time. In theory, all moe AllGather and AllGatherEP solutions can follow this logic, but now other moe models (qwen3-235b) dp solutions are not adjusted, so use the switch to control it to prevent code errors.
def get_rm_router_logits_state(ep_size: int, dp_size: int,
                               is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if dp_size > 1:
        if (envs.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
                and is_deepseek_v3_r1):
            return True
        elif ep_size == 1 and is_deepseek_v3_r1:
            return True
    return False


# TODO(ttanzhiqiang): all_reduce merge
# When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
# Currently, all_reduce_merge is enabled by default in the AllGather, AllGatherEP and NaiveMulticast scenarios of the deepseek model.
def get_all_reduce_merge_state(ep_size: int, is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        return True
    elif ep_size == 1 and is_deepseek_v3_r1:
        return True
    return False


def register_ascend_customop():
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`, 
    and ensure this will execute after model config is initilazed.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    CustomOp.register_oot(_decorated_op_cls=AscendQuickGELU, name="QuickGELU")
    CustomOp.register_oot(_decorated_op_cls=AscendSiluAndMul,
                          name="SiluAndMul")

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True


# TODO(zzzzwwjj): Currently there is no clear SOC_VERSION policy for A2 and A3 in CANN.
# So we get the version dynamically. In the future, we should get the version info from _build_info like 310p does.
class AscendSocVersion(Enum):
    A2 = 0
    A3 = 1
    UNDEFINED = 2


_ascend_soc_version = None


def init_ascend_soc_version():
    soc_version = torch_npu.npu.get_soc_version()
    global _ascend_soc_version
    if 220 <= soc_version <= 225:
        _ascend_soc_version = AscendSocVersion.A2
    elif 250 <= soc_version <= 255:
        _ascend_soc_version = AscendSocVersion.A3
    else:
        _ascend_soc_version = AscendSocVersion.UNDEFINED


def get_ascend_soc_version():
    global _ascend_soc_version
    assert _ascend_soc_version is not None
    return _ascend_soc_version


from vllm_ascend.ops.attention import vanilla_chunked_prefill_mla
def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        print(">>>>>>>>>>>>>>>>>Entered New Forward Prefill>>>>>>>>>>>>>>>>>>>>")
        assert attn_metadata.prefill is not None
        assert len(kv_c_and_k_pe_cache) > 1
        query = torch.cat([q_nope, q_pe], dim=-1)
        num_tokens = q_nope.size(0)
        attn_output = torch.empty(num_tokens,
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=query.dtype,
                                  device=query.device)
        k_pe = k_pe.expand((*k_nope.shape[:-1], -1))
        # Here is only 2 possibility of input, ChunkedPrefill or PrefillNoCache
        ascend_config = get_ascend_config()

        if attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ] and not ascend_config.chunked_prefill_for_mla: # 
            
            attn_output_torch = torch.empty(num_tokens,
                                            self.num_heads * self.v_head_dim,
                                            dtype=query.dtype,
                                            device=query.device)
            # current requests is chunked in prefill, disable flash attention with chunked prefill
            vanilla_chunked_prefill_mla(
                output=attn_output_torch,
                query=query,
                kv_cache=kv_c_and_k_pe_cache,
                block_tables=attn_metadata.prefill.block_table,
                query_lens=attn_metadata.prefill.query_lens,
                context_lens=attn_metadata.prefill.context_lens,
                kv_b_proj=self.kv_b_proj,
                max_query_len=attn_metadata.prefill.max_query_len,
                max_context_len=attn_metadata.prefill.max_seq_lens,
                nope_dim=self.qk_nope_head_dim,
                rope_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                scale=self.scale,
                alibi_slopes=None,
                causal=True)
        elif attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ]:
            query = torch.cat([q_nope, q_pe], dim=-1)
            attn_lse = torch.empty(self.num_heads,
                                   num_tokens,
                                   dtype=torch.float32,
                                   device=q_nope.device)
            mask = torch.triu(
                torch.ones(512, 512, device=query.device, dtype=query.dtype),
                1)  # 512: mask only support 512
            if attn_metadata.num_prefills > 1:
                mask = mask.unsqueeze(0).repeat(attn_metadata.num_prefills, 1,
                                                1)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=value,
                mask=mask,
                seqlen=torch.tensor(attn_metadata.prefill.query_lens,
                                    dtype=torch.int32),
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=None,
                prev_lse=None,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="mask_type_triu",
                input_layout="type_bsnd",
                calc_type="calc_type_first_ring",
                output=attn_output,
                softmax_lse=attn_lse)
            attn_output, attn_lse = self._compute_prefill_context( \
                query, kv_c_and_k_pe_cache, self.qk_rope_head_dim, attn_metadata, attn_output, attn_lse)

        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            key = torch.cat((k_nope, k_pe), dim=-1)
            torch_npu._npu_flash_attention(
                query=query,
                key=key,
                value=value,
                mask=attn_metadata.attn_mask,
                seq_len=attn_metadata.prefill.context_lens,
                scale_value=self.scale,
                num_heads=self.num_heads,
                num_kv_heads=self.num_heads,
                out=attn_output)
            attn_output = attn_output.view(-1, self.num_heads, self.v_head_dim)
        else:
            raise RuntimeError(
                "Unexpected path reached, AscendMLAImpl should only have PrefillNoCache, PrefillCacheHit, ChunkedPrefill and SpecDecoding scenario in forward prefill, please file a bug to vllm-ascend !"
            )
        attn_output = attn_output.reshape(
            [num_tokens, self.num_heads * self.v_head_dim])
        if attn_metadata.attn_state in [
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.PrefillCacheHit
        ] and not ascend_config.chunked_prefill_for_mla:
            attn_output = attn_output_torch

        return attn_output


self.chunked_prefill_for_mla = additional_config.get(
            "chunked_prefill_for_mla", False)


    @patch("vllm_ascend.attention.mla_v1.npu_prefetch")
    def test_mla_preprocess(self, magic_npu_fetch):
        magic_npu_fetch.return_value = MagicMock()
        batch_size = 4
        seq_len = 8
        hidden_size = 1024
        hidden_states = torch.randn(batch_size * seq_len, hidden_size)

        qk_nope_head_dim = 64
        qk_rope_head_dim = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        kv_lora_rank = 512
        kv_cache = MagicMock()

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_prefills = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_actual_tokens = 4
        num_prefill_tokens = 2
        attn_metadata.slot_mapping = torch.arange(4)
        attn_metadata.decode.cos = torch.randn(2, 64)
        attn_metadata.decode.sin = torch.randn(2, 64)
        attn_metadata.prefill.cos = torch.randn(2, 64)
        attn_metadata.prefill.sin = torch.randn(2, 64)

        self.impl.q_a_proj = MagicMock()
        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.num_heads,
            self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_nope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_head_dim)
        ]
        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.v_head_dim + self.impl.qk_nope_head_dim)
        ]
        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]
        self.impl.exec_kv_prefill = MagicMock()
        self.impl.exec_kv_prefill.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_rope_head_dim),
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.kv_lora_rank)
        ]
        self.impl._q_proj_and_k_up_proj = MagicMock()
        self.impl._q_proj_and_k_up_proj.return_value = [
            MagicMock(), MagicMock()
        ]
        self.impl.num_kv_heads = self.impl.num_heads

        decode_res, prefill_res = self.impl._mla_preprocess(
            hidden_states, kv_cache, attn_metadata, need_gather_q_kv=False)

        self.assertIsNotNone(decode_res)
        self.assertIsNotNone(prefill_res)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            None, None,
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank)
        ]

        k_pe, k_nope = self.impl.exec_kv_prefill(kv_no_split, cos, sin,
                                                 kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_decode(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank), None, None
        ]

        k_pe, k_nope = self.impl.exec_kv_decode(kv_no_split, cos, sin,
                                                kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch.npu.stream")
    @patch("vllm_ascend.attention.mla_v1.get_multistream_comm_context")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode(self, mock_npu_fused_infer_attention_score,
                            mock_get_multistream_comm_context,
                            mock_npu_stream):
        B = 2
        N = self.impl.num_kv_heads
        BS = 100
        HD = self.impl.v_head_dim
        self.impl.kv_lora_rank = 256
        self.impl.spec_token_num = 1
        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(B, N, HD)
        q_nope = torch.randn(B, N, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(B, N, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, N, self.impl.kv_lora_rank)
        k_pe = torch.randn(BS, N, self.impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_lengths_q = MagicMock()
        attn_metadata.decode.seq_lens_list = MagicMock()
        self.impl.enable_kv_nz = True

        mock_npu_fused_infer_attention_score.return_value = [
            torch.randn(B, N, self.impl.kv_lora_rank), None
        ]
        mock_get_multistream_comm_context.return_value = None

        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS,
                                           attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)

        self.impl.enable_kv_nz = False
        attn_metadata.attn_state = None
        mock_return_value = MagicMock()
        mock_get_multistream_comm_context.return_value = mock_return_value
        mock_return_value.before_comm_event = MagicMock()
        mock_return_value.comm_stream = MagicMock()
        mock_npu_stream.return_value = MagicMock()

        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS,
                                           attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)


