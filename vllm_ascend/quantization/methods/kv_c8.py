import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from .registry import register_scheme


def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    """fa_q weight loader."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
        assert param.size() == loaded_weight.size(), (
            f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()}) when TP is ({tp_size})"
        )

        param.data.copy_(loaded_weight)


@register_scheme("FAKQuant", "attention")
class AscendFAQuantAttentionMethod:
    def __init__(self):
        self.transpose_weight = True
        self.printFlag = False
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        params_dict = {}
        dtype = torch.get_default_dtype()
        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1), dtype=dtype)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1), dtype=torch.int8)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)

        for name, weight in params_dict.items():
            module_name, weight_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            # When loading weights, segment them according to TP
            weight_param.weight_loader = weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fa_k_scale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fak_descale_float = torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        layer.fak_descale = torch.nn.Parameter(fa_k_scale, requires_grad=False)
        layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale, requires_grad=False)
        fa_k_offset = torch.squeeze(layer.fa_k.offset).unsqueeze(0)
        layer.fak_offset = torch.nn.Parameter(fa_k_offset.to(layer.fak_descale.dtype), requires_grad=False)

        repeated_quant_kscale = fa_k_scale.repeat(self.kv_lora_rank)
        layer.quant_kscale = repeated_quant_kscale.view(1, self.kv_lora_rank)
        layer.quant_kscale = 1.0 / torch.nn.Parameter(layer.quant_kscale.to(torch.float), requires_grad=False)


@register_scheme("INT8_DYNAMIC", "attention")
class AscendSFAQuantAttentionMethod:
    def __init__(self):
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.index_head_dim = config.index_head_dim

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["indexer"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        params_dict = {}
        params_dict["indexer.q_rot"] = torch.empty((self.index_head_dim, self.index_head_dim), dtype=torch.float32)
        params_dict["indexer.k_rot"] = torch.empty((self.index_head_dim, self.index_head_dim), dtype=torch.float32)
        for name, weight in params_dict.items():
            module_name, weight_name = name.split(".")
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass
