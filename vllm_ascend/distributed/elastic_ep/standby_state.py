import torch
from vllm.distributed.parallel_state import (
    _init_stateless_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import get_cached_tcp_store_client

from vllm_ascend.ascend_config import get_ascend_config

_STANDBY_MC2: StatelessGroupCoordinator | None = None
_STANDBY_DYNAMIC_EPLB: StatelessGroupCoordinator | None = None
_STANDBY_FC3_QUANT_X: StatelessGroupCoordinator | None = None


def get_standby_mc2_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_MC2


def get_standby_dynamic_eplb_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_DYNAMIC_EPLB


def get_standby_fc3_quant_x_group() -> StatelessGroupCoordinator | None:
    return _STANDBY_FC3_QUANT_X


def create_ascend_standby_groups(
    new_dp_size: int,
    new_world_size_across_dp: int,
    master_ip: str,
    coord_store_port: int,
    backend: str | None = None,
) -> None:
    global _STANDBY_MC2, _STANDBY_DYNAMIC_EPLB, _STANDBY_FC3_QUANT_X

    assert new_world_size_across_dp == torch.distributed.get_world_size() * new_dp_size
    world_group = get_world_group()
    assert isinstance(world_group, StatelessGroupCoordinator)
    backend = backend or world_group.backend

    coord_store = get_cached_tcp_store_client(master_ip, coord_store_port)

    tp_size = get_tp_group().world_size
    pp_size = get_pp_group().world_size

    all_ranks = torch.arange(new_world_size_across_dp).reshape(-1, new_dp_size * pp_size * tp_size)
    group_ranks = all_ranks.unbind(0)
    standby_ep_ranks = [x.tolist() for x in group_ranks]

    _STANDBY_MC2 = _init_stateless_group(
        standby_ep_ranks,
        "mc2",
        master_ip,
        backend,
        coord_store=coord_store,
        use_device_communicator=False,
    )

    if get_ascend_config().eplb_config.dynamic_eplb:
        _STANDBY_DYNAMIC_EPLB = _init_stateless_group(
            standby_ep_ranks,
            "dynamic_eplb",
            master_ip,
            backend,
            coord_store=coord_store,
            use_device_communicator=False,
        )

    if get_ascend_config().multistream_overlap_gate:
        _STANDBY_FC3_QUANT_X = _init_stateless_group(
            standby_ep_ranks,
            "fc3_quant_x",
            master_ip,
            backend,
            coord_store=coord_store,
            use_device_communicator=False,
        )


def pop_ascend_standby_groups() -> dict:
    """Return all standby groups and clear the standby state."""
    global _STANDBY_MC2, _STANDBY_DYNAMIC_EPLB, _STANDBY_FC3_QUANT_X
    result = dict(
        mc2=_STANDBY_MC2,
        dynamic_eplb=_STANDBY_DYNAMIC_EPLB,
        fc3_quant_x=_STANDBY_FC3_QUANT_X,
    )
    _STANDBY_MC2 = None
    _STANDBY_DYNAMIC_EPLB = None
    _STANDBY_FC3_QUANT_X = None
    return result
