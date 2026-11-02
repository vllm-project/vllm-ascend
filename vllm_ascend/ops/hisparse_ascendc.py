"""A5 C8 request-union resolver with instance-owned graph launch arguments."""

from pathlib import Path

import torch


class BoundResolver:
    def __init__(self, arguments, library):
        a = arguments
        self.owners = dict(a)
        rows, topk = a["global_indices"].shape
        hot = a["device_global_indices"].shape[1]
        assert 0 < topk <= 2048 and 0 < hot <= 10240
        assert a["region_stride"] == hot and a["max_union_rows"] <= 8192
        assert a["hot_cache"].dtype == torch.float8_e4m3fn and a["hot_cache"].shape[-1] == 656
        for name in ("global_indices", "device_global_indices", "hot_block_table", "hot_indices"):
            assert a[name].dtype == torch.int32 and a[name].device == a["hot_cache"].device
        assert a["lru_slots"].dtype == torch.int16
        resolved = a.get("resolved_global_indices")
        if resolved is None:
            resolved = torch.empty((rows, topk), dtype=torch.int32, device=a["hot_cache"].device)
            self.owners["_resolved"] = resolved
        values = [0] * 64
        names = [
            "global_indices",
            "request_ids",
            "request_state_indices",
            "source_block_table",
            "resident_block_table",
            "hot_block_table",
            "device_global_indices",
            "lru_slots",
            "hot_indices",
            "attention_indices",
            "resolved_global_indices",
            "valid_counts",
            "swap_host_physical_rows",
            "swap_device_physical_rows",
            "swap_counts",
            "miss_mask",
            "stats",
        ]
        for index, name in enumerate(names):
            tensor = a.get(name)
            if tensor is not None:
                assert tensor.device == a["hot_cache"].device
                values[index] = tensor.data_ptr()
        values[10] = resolved.data_ptr()
        self.errors = torch.zeros((rows, 8), dtype=torch.int32, device=a["hot_cache"].device)
        values[17] = self.errors.data_ptr()
        source, resident = a.get("source_block_table"), a.get("resident_block_table")
        state = a.get("request_state_indices")
        attention, counts = a.get("attention_indices"), a.get("valid_counts")
        values[32:55] = [
            rows,
            topk,
            hot,
            a["host_cache"].shape[0],
            a["global_indices"].stride(0),
            state.numel() if state is not None else 0,
            a["device_global_indices"].shape[0],
            source.shape[0] if source is not None else 0,
            source.shape[1] if source is not None else 0,
            source.stride(0) if source is not None else 0,
            a.get("source_block_size", 1),
            resident.shape[0] if resident is not None else 0,
            resident.shape[1] if resident is not None else 0,
            resident.stride(0) if resident is not None else 0,
            a.get("resident_block_size", 1),
            a.get("resident_null_block", 0),
            a["hot_cache"].shape[1],
            a["hot_block_table"].stride(0),
            attention.stride(0) if attention is not None else topk,
            a.get("attention_block_stride", a["hot_cache"].shape[1]),
            counts.stride(0) if counts is not None else 1,
            a["max_union_rows"],
            16384,
        ]
        self.geometry_arguments = values[32:55]
        self.omitted = torch.empty(0, dtype=torch.int32, device=a["hot_cache"].device)
        self.tensor_arguments = [a.get(name) if a.get(name) is not None else self.omitted for name in names]
        self.tensor_arguments[10] = resolved
        self.tensor_arguments.append(self.errors)
        self.tensor_arguments.append(a.get("num_valid_rows") if a.get("num_valid_rows") is not None else self.omitted)

    def __call__(self):
        torch.ops.hisparse_fused60103.resolve(self.tensor_arguments, self.geometry_arguments)


_ARGUMENT_NAMES = (
    "host_cache",
    "hot_cache",
    "hot_block_table",
    "global_indices",
    "hot_indices",
    "device_global_indices",
    "lru_slots",
    "request_state_indices",
    "region_stride",
    "max_union_rows",
    "miss_mask",
    "stats",
    "attention_indices",
    "attention_block_stride",
    "request_ids",
    "source_block_table",
    "source_block_size",
    "resolved_global_indices",
    "valid_counts",
    "swap_host_physical_rows",
    "swap_device_physical_rows",
    "swap_counts",
    "resident_block_table",
    "resident_block_size",
    "resident_null_block",
    "num_valid_rows",
)


def try_resolve(owner, scope):
    a = {name: scope[name] for name in _ARGUMENT_NAMES}
    hot = a["hot_cache"]
    if not (
        owner.use_fused_index_ops
        and hot.dtype == torch.float8_e4m3fn
        and hot.shape[-1] == 656
        and a["global_indices"].shape[1] <= 2048
        and a["device_global_indices"].shape[1] <= 10240
        and a["max_union_rows"] <= 8192
    ):
        return False
    # Bindings retain their tensor storage for graph replay. An eager mixed
    # batch can pass a tiny Decode view backed by the full Prefill TopK tensor.
    # Retaining each new address here pins that entire allocation indefinitely.
    # Keep eager calls on the existing resolver; only captured graphs need
    # immutable, persistent argument owners. Never evict a live graph binding.
    if not torch.npu.is_current_stream_capturing():
        return False
    bindings = getattr(owner, "_ascendc_resolvers", None)
    if bindings is None:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A5:
            return False
        library = str(Path(__file__).with_name("torch_binding.so"))
        if library not in torch.ops.loaded_libraries:
            torch.ops.load_library(library)
        owner._ascendc_resolvers = bindings = {}
    key = tuple(
        (name, value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        if isinstance(value, torch.Tensor)
        else (name, value)
        for name, value in a.items()
    )
    bound = bindings.get(key)
    if bound is None:
        if len(bindings) >= 32:
            return False
        bound = BoundResolver(a, None)
        bindings[key] = bound
    bound()
    return True
