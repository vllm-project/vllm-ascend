# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
class LogitsProbeExtension:
    def logits_probe_features(self):
        from vllm_ascend.ascend_config import get_ascend_config, is_mega_moe_supported
        from vllm_ascend.ascend_forward_context import use_cann_megamoe
        from vllm_ascend.utils import enable_sp

        c = get_ascend_config()
        vc = self.vllm_config
        return {
            "dp_rank": vc.parallel_config.data_parallel_rank,
            "worker_rank": self.rank,
            "runner": type(self.model_runner).__module__ + "." + type(self.model_runner).__name__,
            "megamoe_supported": is_mega_moe_supported(),
            "megamoe_effective": use_cann_megamoe(vc),
            "fused_mc2": c.enable_fused_mc2,
            "flashcomm1": enable_sp(vllm_config=vc),
            "li_c8": c.enable_sparse_li_c8,
            "sfa_c8": c.enable_sparse_sfa_c8,
            "pp": vc.parallel_config.pipeline_parallel_size,
            "pcp": vc.parallel_config.prefill_context_parallel_size,
            "tp": vc.parallel_config.tensor_parallel_size,
            "dp": vc.parallel_config.data_parallel_size,
        }
