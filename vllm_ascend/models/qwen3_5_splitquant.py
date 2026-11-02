"""Qwen3.5 text-only SplitQuant adapter."""
import torch
import os
from torch import nn
from vllm_ascend.models.qwen3_next import CustomQwen3NextForCausalLM
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.config import HybridAttentionMambaModelConfig


class SplitQuantQwen3_5ForConditionalGeneration(CustomQwen3NextForCausalLM,
                                                IsHybrid):
    """Temporary registry adapter while the Ascend GDN backend is selected."""
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "in_proj": ["in_proj_qkvz", "in_proj_ba"],
    }

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return CustomQwen3NextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return CustomQwen3NextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config)

    @classmethod
    def verify_and_update_config(cls, vllm_config):
        return HybridAttentionMambaModelConfig.verify_and_update_config(
            vllm_config)

    def __init__(self, *, vllm_config, prefix=""):
        if os.getenv('SPLITQUANT_QWEN35_GRAPH') == '1':
            if (vllm_config.parallel_config.tensor_parallel_size != 1
                    or vllm_config.scheduler_config.max_num_seqs != 1):
                raise NotImplementedError('Qwen3.5 Graph adapter currently requires TP1 and max_num_seqs=1')
            from vllm_ascend.quantization.splitquant_qwen35_graph import install_graph_audit
            from vllm.compilation.decorators import ignore_torch_compile
            from vllm_ascend.models.qwen3_next import CustomQwen3NextModel
            install_graph_audit()
            ignore_torch_compile(CustomQwen3NextModel)
        # The 27B export is a dense Qwen3.5 text tower.  Qwen3Next's generic
        # config supplies MoE defaults when these fields are absent from the
        # multimodal config, so explicitly select the dense MLP path.
        config = vllm_config.model_config.hf_config
        self._hf_config = config
        config.num_experts = 0
        config._splitquant_dense_qwen35 = True
        config.decoder_sparse_step = 1
        config.model_type = "qwen3_next"
        config.moe_intermediate_size = config.intermediate_size
        config.shared_expert_intermediate_size = 0
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if os.getenv('SPLITQUANT_DENSE') == '1':
            from vllm_ascend.quantization.splitquant_dense import install_qwen35
            install_qwen35(self)
            from vllm_ascend.quantization.splitquant_state import install as install_state
            install_state(self)

    def load_weights(self, weights):
        mapped = []
        qkvz = {}
        ba = {}
        for name, value in weights:
            if name.startswith(("model.visual.", "mtp.")):
                continue
            if ".mlp._up_gate_trans." in name or ".mlp._down_trans." in name:
                # These are calibration/reparameterization helpers.  The
                # exported gate/up/down tensors already carry their runtime
                # group transforms and the Ascend operator ABI does not expose
                # the helper object itself.
                continue
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            # Qwen3.5 exports four independent GDN projections; vLLM's
            # Qwen3Next kernel stores qkvz and ba as two fused matrices.
            if ".linear_attn.in_proj_qkv.weight" in name:
                key = name.replace("in_proj_qkv", "in_proj_qkvz")
                qkvz.setdefault(key, {})["qkv"] = value
                continue
            if ".linear_attn.in_proj_z.weight" in name:
                key = name.replace("in_proj_z", "in_proj_qkvz")
                qkvz.setdefault(key, {})["z"] = value
                continue
            if ".linear_attn.in_proj_b.weight" in name:
                key = name.replace("in_proj_b", "in_proj_ba")
                ba.setdefault(key, {})["b"] = value
                continue
            if ".linear_attn.in_proj_a.weight" in name:
                key = name.replace("in_proj_a", "in_proj_ba")
                ba.setdefault(key, {})["a"] = value
                continue
            mapped.append((name, value))
        for name, parts in list(qkvz.items()):
            if set(parts) != {"qkv", "z"}:
                raise RuntimeError(f"Incomplete Qwen3.5 GDN qkvz parts for {name}: {set(parts)}")
            cfg = self._hf_config
            nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
            dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
            q, k, v = parts["qkv"].split((nk * dk, nk * dk, nv * dv), dim=0)
            # Qwen3Next shards rows by key head and expects [q,k,v,z]
            # within each key-head group, whereas Qwen3.5 exports [Q,K,V].
            qkvz[name] = torch.cat((q.reshape(nk, dk, -1),
                                    k.reshape(nk, dk, -1),
                                    v.reshape(nk, nv // nk * dv, -1),
                                    parts["z"].reshape(nk, nv // nk * dv, -1)),
                                   dim=1).reshape(-1, q.shape[-1])
        for name, parts in list(ba.items()):
            if set(parts) != {"b", "a"}:
                raise RuntimeError(f"Incomplete Qwen3.5 GDN ba parts for {name}: {set(parts)}")
            nk = self._hf_config.linear_num_key_heads
            ba[name] = torch.cat((parts["b"].reshape(nk, -1, parts["b"].shape[-1]),
                                  parts["a"].reshape(nk, -1, parts["a"].shape[-1])),
                                 dim=1).reshape(-1, parts["b"].shape[-1])
        mapped.extend(qkvz.items())
        mapped.extend(ba.items())
        return super().load_weights(mapped)
