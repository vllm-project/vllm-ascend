"""Ascend query projection replicated within each decode context group."""

import torch
import torch_npu
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank

from vllm_ascend import utils
from vllm_ascend.ops.linear import AscendColumnParallelLinear


def use_dcp_q_replicate(vllm_config, config, quant_config):
    parallel = vllm_config.parallel_config
    requested = (
        envs.VLLM_DCP_Q_REPLICATE
        if envs.is_set("VLLM_DCP_Q_REPLICATE")
        else bool(getattr(parallel, "dcp_q_replicate", False))
    )
    if not requested or parallel.decode_context_parallel_size <= 1 or parallel.prefill_context_parallel_size > 1:
        return False
    unsupported = []
    if hasattr(config, "index_topk"):
        # Quantized indexer-cache paths have not been validated with Q replication.
        if vllm_config.attention_config.indexer_kv_dtype not in ("auto", "bf16"):
            unsupported.append("quantized indexer KV cache")
    # Quantized Q weights need matching group-wise scale/packed-weight sharding.
    if quant_config is not None:
        unsupported.append("quantized weights")
    # Quantized main KV needs a separately adapted attention/scale path.
    if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        unsupported.append("quantized KV cache")
    # Existing LoRA wrappers do not select the replicated-Q subclass or
    # guarantee adapter output shards match its group-head layout.
    if vllm_config.lora_config is not None:
        unsupported.append("LoRA adapters")
    if unsupported:
        raise ValueError("Ascend dcp_q_replicate does not yet support: " + ", ".join(unsupported))
    return True


class AscendDCPGroupColumnParallelLinear(AscendColumnParallelLinear):
    """Keep Ascend loading/GEMM while sharding Q weights across DCP groups."""

    def __init__(self, input_size, output_size, *, bias=False, quant_config=None, prefix=""):
        self.group_size = get_current_vllm_config().parallel_config.decode_context_parallel_size
        self.qrep_active = self.group_size > 1
        self.rank_in_group = get_tensor_model_parallel_rank() % self.group_size
        super().__init__(input_size, output_size, bias=bias, quant_config=quant_config, prefix=prefix)

    def prepare_local_weight(self):
        """Prepare the original TP shard after the group weight has been loaded."""
        weight = torch_npu.npu_format_cast(self.weight.detach(), utils.ACL_FORMAT_FRACTAL_ND)
        local_weight = weight.chunk(self.group_size, dim=0)[self.rank_in_group].contiguous()
        self.register_buffer("local_weight", utils.maybe_trans_nz(local_weight), persistent=False)

    def forward_local(self, x):
        """Project only this TP rank's heads for MLA prefill."""
        local_bias = None if self.bias is None else self.bias.chunk(self.group_size, dim=0)[self.rank_in_group]
        bias = None if self.skip_bias_add else local_bias
        output = torch.ops.vllm.unquantized_gemm(x, self.local_weight, bias)
        if not self.return_bias:
            return output
        return output, local_bias if self.skip_bias_add else None
