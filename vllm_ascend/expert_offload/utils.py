"""Utility functions for expert offload initialization."""

import torch


def init_expert_offload_config(offload_config, num_experts: int):
    """Prepare pre-super().__init__() expert offload setup.

    Builds an expert_map_offload that the upstream layer.py hook reads
    to shrink the device weight and skip loading cold experts.

    Args:
        offload_config: ExpertOffloadConfig instance from AscendConfig.
        num_experts: Total routed expert count from model config.

    Returns:
        (enable: bool, expert_map_offload: torch.Tensor | None)
    """
    enable = (offload_config.expert_offload
              and offload_config.num_device_experts > 0
              and offload_config.num_device_experts < num_experts)
    if not enable:
        return False, None

    ndev = offload_config.num_device_experts
    emap = torch.full((num_experts,), -1, dtype=torch.int32)
    emap[:ndev] = torch.arange(ndev, dtype=torch.int32)
    return True, emap


def init_log2phy_for_offload(global_num_experts: int, num_device_experts: int):
    """Initialize the forward-pass log2phy mapping table on CPU."""
    log2phy = torch.full((global_num_experts,), -1, dtype=torch.int32, device='cpu')
    log2phy[:num_device_experts] = torch.arange(num_device_experts, dtype=torch.int32)
    return log2phy
