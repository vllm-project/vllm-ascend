# SPDX-License-Identifier: Apache-2.0
"""Ascend 950 localTP8 KDA verification with request-local ReplaySSM state."""

import ctypes
import weakref
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.distributed.parallel_state import get_gproj_tp_group, get_otp_group, get_qkv_iproj_tp_group

_WORKSPACE_BYTES = 32 << 20


class LocalTP8Workspace:
    """One IPC allocation per worker, shared by the KDA layers in its projection group."""

    def __init__(self, group):
        self.runtime = ctypes.CDLL("libascendcl.so")
        self.local = ctypes.c_void_p()
        self.imports = []
        self.rank = group.rank_in_group
        self.cpu_group = group.cpu_group
        self._call("aclrtMalloc", ctypes.byref(self.local), ctypes.c_size_t(_WORKSPACE_BYTES), ctypes.c_int(0))
        try:
            self._call("aclrtMemset", self.local, ctypes.c_size_t(4096), ctypes.c_int(0), ctypes.c_size_t(4096))
            pid = ctypes.c_int()
            key = ctypes.create_string_buffer(256)
            self._call("aclrtDeviceGetBareTgid", ctypes.byref(pid))
            self._call(
                "aclrtIpcMemGetExportKey",
                self.local,
                ctypes.c_size_t(_WORKSPACE_BYTES),
                key,
                ctypes.c_size_t(len(key)),
                ctypes.c_uint64(0),
            )
            peers = [None] * 8
            dist.all_gather_object(peers, (pid.value, key.value), group=self.cpu_group)
            pids = [pid for i, (pid, _) in enumerate(peers) if i != self.rank]
            self._call("aclrtIpcMemSetImportPid", key, (ctypes.c_int * len(pids))(*pids), ctypes.c_size_t(len(pids)))
            dist.barrier(group=self.cpu_group)
            addresses = []
            for rank, (_, peer_key) in enumerate(peers):
                if rank == self.rank:
                    addresses.append(self.local.value)
                else:
                    ptr = ctypes.c_void_p()
                    self._call(
                        "aclrtIpcMemImportByKey", ctypes.byref(ptr), ctypes.c_char_p(peer_key), ctypes.c_uint64(1)
                    )
                    self.imports.append(peer_key)
                    addresses.append(ptr.value)
            self.addresses = torch.tensor(addresses, dtype=torch.int64, device="npu")
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        status = getattr(self.runtime, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with ACL status {status}")

    def close(self):
        for key in self.imports:
            self.runtime.aclrtIpcMemClose(ctypes.c_char_p(key))
        self.imports.clear()
        if self.local.value:
            self.runtime.aclrtFree(self.local)
            self.local = ctypes.c_void_p()


class LocalTP8Verify:
    def __init__(self, package_dir):
        group = get_qkv_iproj_tp_group()
        if group.world_size != 8 or any(
            other.ranks != group.ranks for other in (get_otp_group(), get_gproj_tp_group())
        ):
            raise ValueError("LocalTP8 MegaKDA requires identical eight-rank QKV, G and O groups")
        lib = ctypes.CDLL(str(Path(package_dir) / "libmega_kda_localtp8_verify.so"))
        self.launch = lib.run_MegaKdaLocalTp8Verify
        self.launch.restype = None
        self.launch.argtypes = (
            [ctypes.c_void_p] * 25 + [ctypes.c_uint32] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_float] * 2
        )
        if not hasattr(group, "_kda_verify_workspace"):
            group._kda_verify_workspace = LocalTP8Workspace(group)
            weakref.finalize(group, group._kda_verify_workspace.close)
        self.workspace = group._kda_verify_workspace

    def supports_batch(self, hidden, meta):
        # Graph capture and warmup enter this host code once per forward. Every
        # member must capture the same communication path, including idle owners.
        commit = getattr(meta, "recoverssm_context", None)
        local = bool(
            hidden.shape == (32, 7168)
            and hidden.dtype == torch.bfloat16
            and hidden.is_contiguous()
            and meta is not None
            and meta.num_spec_decodes
            and not meta.num_prefills
            and not meta.num_decodes
            and meta.spec_query_start_loc is not None
            and meta.spec_query_start_loc.shape == (5,)
            and meta.spec_state_indices_tensor is not None
            and meta.spec_state_indices_tensor.shape[0] == 4
            and meta.spec_decode_metadata is not None
            and getattr(commit, "normalized_key_base_addrs", None) is not None
            and getattr(commit, "decay_base_addrs", None) is not None
        )
        vote = torch.tensor(int(local), dtype=torch.int32, device="cpu")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.workspace.cpu_group)
        return bool(vote.item())

    def __call__(self, layer, hidden, meta, output):
        proj = layer.in_proj_qkvgfab
        conv, checkpoint = layer.kv_cache
        records = layer.recoverssm_records
        desc = meta.spec_decode_metadata.spec_causal_conv1d
        slots = meta.spec_state_indices_tensor[:, 0]
        conv_slots = desc.cache_indices
        counts = desc.num_accepted_tokens
        starts = meta.spec_query_start_loc
        if (
            conv.dtype != torch.bfloat16
            or conv.shape[-2:] != (10, 36864)
            or conv.stride(-1) != 1
            or conv.stride(-2) != 36864
            or checkpoint.shape[-3:] != (96, 128, 128)
            or checkpoint.dtype != torch.float32
            or checkpoint.stride()[1:] != (16384, 128, 1)
        ):
            raise ValueError("LocalTP8 MegaKDA requires SD convolution state and contiguous FP32 checkpoint heads")
        if not all(t.dtype == torch.int32 for t in (slots, conv_slots, counts, starts)) or not starts.is_contiguous():
            raise ValueError("LocalTP8 MegaKDA requires int32 slot, acceptance and query-offset metadata")
        if (
            slots.shape != (4,)
            or conv_slots.shape != (4,)
            or counts.shape != (4,)
            or starts.shape != (5,)
            or hidden.device != conv.device
        ):
            raise ValueError("LocalTP8 MegaKDA requires four request rows on the local device")
        for tensor, dtype, width in (
            (records.correction, torch.float32, 128),
            (layer.recoverssm_normalized_key_cache, torch.float32, 128),
            (layer.recoverssm_decay_cache, torch.float32, 128),
            (records.key_gate, torch.bfloat16, 256),
        ):
            if (
                tensor.dtype != dtype
                or tensor.shape[0] < 4
                or tensor.shape[1:] != (96, 8, width)
                or not tensor.is_contiguous()
                or tensor.device != hidden.device
            ):
                raise ValueError("Incompatible LocalTP8 MegaKDA ReplaySSM record layout")
        args = (
            hidden,
            output,
            proj._localtp8_qkv,
            proj._localtp8_scale,
            layer._localtp8_bg_weight,
            layer._localtp8_bg_weight,
            layer.f_a_proj.weight,
            layer.f_b_proj.weight,
            layer.o_proj.weight,
            layer._localtp8_gamma,
            layer.ascend_conv1d_weight,
            conv,
            checkpoint,
            records.correction,
            layer.recoverssm_normalized_key_cache,
            layer.recoverssm_decay_cache,
            records.key_gate,
            slots,
            conv_slots,
            counts,
            layer.A_log,
            layer.dt_bias,
            self.workspace.addresses,
            starts,
        )
        self.launch(
            int(torch.npu.current_stream().npu_stream),
            *(t.data_ptr() for t in args),
            self.workspace.rank,
            slots.stride(0),
            conv_slots.stride(0),
            counts.stride(0),
            conv.stride(0),
            checkpoint.stride(0),
            layer.gate_lower_bound,
            layer.o_norm.eps,
        )


def localtp8_kda_forward(hidden: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    ctx = get_forward_context()
    layer = ctx.no_compile_layers[layer_name]
    meta = ctx.attn_metadata.get(layer_name) if isinstance(ctx.attn_metadata, dict) else None
    if layer._localtp8_verify.supports_batch(hidden, meta):
        layer._run_localtp8_mega(hidden, meta, output)
    else:
        output.copy_(layer._forward_local_tp(hidden))


def _fake(hidden: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    pass


direct_register_custom_op(
    op_name="kimi_localtp8_kda_forward", op_func=localtp8_kda_forward, mutates_args=["output"], fake_impl=_fake
)
