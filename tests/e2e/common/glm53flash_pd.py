# SPDX-License-Identifier: Apache-2.0
"""Read-only worker probes for real-checkpoint Flash PD serving coverage."""

import torch


def all_dp_results(result):
    # The API broadcasts collective_rpc to DP engines but returns only the
    # first engine's result. Gather on Gloo so assertions see every DP rank.
    from vllm.distributed import get_dp_group

    group = get_dp_group()
    results = [None] * group.world_size
    torch.distributed.all_gather_object(results, result, group=group.cpu_group)
    return results


class FlashPDWorker:
    """Read actual parallel groups and observe Mamba copies in test workers."""

    def flash_pd_inspect(self):
        from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group, get_tp_group

        layers = [m for m in self.model_runner.model.modules() if type(m).__name__ == "Glm5NextLinearAttention"]
        return all_dp_results(
            {
                "quant_methods": sorted(
                    {
                        type(getattr(m.quant_method, "quant_method", m.quant_method)).__name__
                        for m in self.model_runner.model.modules()
                        if hasattr(m, "quant_method")
                    }
                ),
                "kv_dtype": self.vllm_config.cache_config.cache_dtype,
                "tp": get_tp_group().world_size,
                "pcp": get_pcp_group().world_size,
                "dp": get_dp_group().world_size,
                "ep": get_ep_group().world_size,
                "dp_rank": get_dp_group().rank_in_group,
                "kda_heads": [layer.local_num_heads for layer in layers],
                "replayssm": all(layer.use_recoverssm for layer in layers),
                "mtp_tokens": self.vllm_config.speculative_config.num_speculative_tokens,
                "has_speculator": self.model_runner.speculator is not None,
                "prefix_caching": self.vllm_config.cache_config.enable_prefix_caching,
                "graph_mode": self.vllm_config.compilation_config.cudagraph_mode.name,
            }
        )

    def flash_pd_probe(self):
        from tests.e2e.common.glm53flash_pd_checks import observe_state_copy
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread

        if not hasattr(self, "_flash_pd_original"):
            self._flash_pd_original = MooncakePullRecvingThread._append_mamba_transfer_addresses

            def observe(receiver, *args, **kwargs):
                result, record = observe_state_copy(self._flash_pd_original, receiver, *args, **kwargs)
                self._flash_pd_copies.append(record)
                return result

            MooncakePullRecvingThread._append_mamba_transfer_addresses = observe
        self._flash_pd_copies = []
        return {"observer": "MooncakePullRecvingThread._append_mamba_transfer_addresses"}

    def flash_pd_records(self):
        return all_dp_results({"copies": list(self._flash_pd_copies)})
