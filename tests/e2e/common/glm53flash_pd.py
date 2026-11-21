# SPDX-License-Identifier: Apache-2.0
"""Test-only loader and CLI for the four-device Flash PD regression."""

import hashlib
import sys

import torch
from vllm.model_executor.model_loader import register_model_loader

from tests.e2e.pull_request.four_card.test_glm5_3_flash import FlashDummyLoader

LOADER = "glm53flash_pd_dummy"


def all_dp_results(result):
    # The API broadcasts collective_rpc to DP engines but returns only the
    # first engine's result. Gather on Gloo so assertions see every DP rank.
    from vllm.distributed import get_dp_group

    group = get_dp_group()
    results = [None] * group.world_size
    torch.distributed.all_gather_object(results, result, group=group.cpu_group)
    return results


def global_weight(name, shape, dtype, device):
    """Generate a canonical tensor before slicing, independent of PCP rank."""
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    generator = torch.Generator(device=device).manual_seed(seed)
    value = torch.empty(shape, dtype=dtype, device=device)
    if dtype == torch.int8:
        return value.random_(-8, 9, generator=generator)
    return value.uniform_(-0.01, 0.01, generator=generator)


@register_model_loader(LOADER)
class FlashPDLoader(FlashDummyLoader):
    """Retain EP2 on both roles; make KDA PCP2 and PCP1 weights equivalent."""

    def load_weights(self, model, model_config):
        super().load_weights(model, model_config)
        with torch.no_grad():
            for name, layer in model.named_modules():
                if type(layer).__name__ != "Glm5NextLinearAttention":
                    continue
                size, rank = layer.tp_size, layer.tp_rank
                # The packed projection has four head-sharded and two replicated
                # components. Seeding the local packed tensor is not equivalent.
                projection = layer.in_proj_qkvbfg_a.weight
                widths = [layer.num_heads * layer.head_dim] * 3 + [layer.num_heads, layer.head_dim, layer.head_dim]
                parts = []
                for shard, width in enumerate(widths):
                    full = global_weight(
                        f"{name}.in_proj.{shard}", (width, layer.hidden_size), projection.dtype, projection.device
                    )
                    parts.append(full.chunk(size, dim=0)[rank] if shard < 4 else full)
                projection.copy_(torch.cat(parts, dim=0))
                for child_name, dim in (
                    ("f_b_proj", 0),
                    ("g_b_proj", 0),
                    ("q_conv1d", 0),
                    ("k_conv1d", 0),
                    ("v_conv1d", 0),
                    ("o_proj", 1),
                ):
                    weight = getattr(layer, child_name).weight
                    shape = list(weight.shape)
                    shape[dim] *= size
                    full = global_weight(f"{name}.{child_name}.weight", shape, weight.dtype, weight.device)
                    weight.copy_(full.chunk(size, dim=dim)[rank])


class FlashPDWorker:
    """Read actual parallel groups and observe Mamba copies in test workers."""

    def flash_pd_inspect(self):
        from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group, get_tp_group

        layers = [m for m in self.model_runner.model.modules() if type(m).__name__ == "Glm5NextLinearAttention"]
        return all_dp_results(
            {
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
        # Observe transfer descriptors without changing the copies performed.
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread

        if not hasattr(self, "_flash_pd_original"):
            self._flash_pd_original = KVCacheRecvingThread._append_mamba_transfer_meta

            def observe(receiver, src, dst, lengths, **kwargs):
                start = len(lengths)
                self._flash_pd_original(receiver, src, dst, lengths, **kwargs)
                self._flash_pd_copies.append(
                    {
                        "state": kwargs["src_layer_base_addr"][1],
                        "block": kwargs["local_block_id"],
                        "offset": kwargs["remote_tp_offset"],
                        "pulls": kwargs["tp_num_need_pulls"],
                        "bytes": sum(lengths[start:]),
                        "expected_bytes": sum(kwargs["block_len"][:2]) // kwargs["tp_num_need_pulls"],
                    }
                )

            KVCacheRecvingThread._append_mamba_transfer_meta = observe
        self._flash_pd_copies = []
        return True

    def flash_pd_records(self):
        return all_dp_results({"copies": list(self._flash_pd_copies)})


def main():
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv[0] = "vllm"
    vllm_main()


if __name__ == "__main__":
    main()
