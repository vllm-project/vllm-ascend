from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

logger = init_logger(__name__)
_original_initialize_cudagraph_keys = (
    CudagraphDispatcher.initialize_cudagraph_keys
)


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    # FULL mode should not be treated as uniform decode
    if (
        uniform_decode
        and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    ):
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
    else:
        uniform_decode = False
        num_reqs = min(num_tokens_padded, max_num_seqs)

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
    )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor


def _initialize_cudagraph_keys(
    self,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
):
    additional_config = getattr(self.vllm_config, "additional_config", None) or {}
    ascend_compilation_config = additional_config.get(
        "ascend_compilation_config",
        {},
    )
    explicit_portfolio = (
        isinstance(ascend_compilation_config, dict)
        and "dflash_full_and_piecewise_capture_config"
        in ascend_compilation_config
    )
    if not explicit_portfolio:
        return _original_initialize_cudagraph_keys(
            self,
            cudagraph_mode,
            uniform_decode_query_len,
        )

    from vllm_ascend._310p.dflash_full_and_piecewise import (
        initialize_dflash_full_and_piecewise_cudagraph_keys,
    )

    if initialize_dflash_full_and_piecewise_cudagraph_keys(
        self,
        cudagraph_mode,
        uniform_decode_query_len,
    ):
        inventory = {
            mode.name: sorted(
                descriptor.num_tokens
                for descriptor in descriptors
            )
            for mode, descriptors in self.cudagraph_keys.items()
        }
        logger.info(
            "[310p-dflash-full-and-piecewise/portfolio] mode=%s "
            "inventory=%s",
            cudagraph_mode.name,
            inventory,
        )
        return
    _original_initialize_cudagraph_keys(
        self,
        cudagraph_mode,
        uniform_decode_query_len,
    )


CudagraphDispatcher.initialize_cudagraph_keys = _initialize_cudagraph_keys
