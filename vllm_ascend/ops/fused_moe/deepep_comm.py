# SPDX-License-Identifier: Apache-2.0
"""Experimental eager DeepEP dispatch/combine; expert math stays in the quant scheme."""

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchOutput, build_token_dispatch_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, MoECommMethod
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalize
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatcher

# This wheel selects the high-throughput transport above 256 tokens. Its
# low-latency transport does not support the eight-rank topology used here.
DEEPEP_MIN_HIGH_THROUGHPUT_CAPACITY = 257


def _load_private_shmem():
    """Keep the pinned wheel's HYBM symbols separate from memfabric_hybrid."""
    import deep_ep
    from deep_ep._native import _loaded_shmem_paths

    directory = Path(deep_ep.__file__).resolve().parent
    bundle = directory / ".libs"
    names = ("libshmem_utils.so", "aclshmem_bootstrap_config_store.so", "aclshmem_bootstrap_uid.so", "libshmem.so")
    hashes = json.loads((directory / "_build_info.json").read_text())["shmem_runtime"]
    if set(hashes) != set(names):
        raise ImportError("DeepEP requires the pinned wheel's complete bundled SHMEM manifest")
    for path in _loaded_shmem_paths():
        if path != bundle / path.name:
            raise ImportError(f"Conflicting SHMEM runtime is already loaded: {path}")
    for name in names:
        path = bundle / name
        if path.resolve().parent != bundle or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ImportError(f"DeepEP bundled SHMEM integrity check failed: {name}")
    # memfabric_hybrid exports the same C++ HYBM names. Without DEEPBIND the
    # SDK's device getter binds to memfabric's uninitialized state (device -1).
    # Preload the verified closure before DeepEP's normal native loader runs.
    libraries = []
    for name in names:
        scope = ctypes.RTLD_LOCAL if name.startswith("aclshmem_bootstrap_") else ctypes.RTLD_GLOBAL
        libraries.append(ctypes.CDLL(str(bundle / name), mode=scope | os.RTLD_DEEPBIND))
    return libraries


@dataclass
class DeepEPCombineMetadata:
    handle: object
    weights: torch.Tensor


class PrepareAndFinalizeWithDeepEP(PrepareAndFinalize):
    def __init__(self, moe_config):
        super().__init__(moe_config)
        self.num_tokens = 0
        self.padded_tokens = 0

    def prepare(self, hidden_states, router_logits, replace_allreduce=False, quant_type=None):
        self.num_tokens = hidden_states.shape[0]
        self.padded_tokens = max(1, self.num_tokens, int(_EXTRA_CTX.max_tokens_across_dp or 0))
        padding = self.padded_tokens - self.num_tokens
        # The wheel requires uniform positive B across EP. Invalid routes remove padding
        # before expert computation; outputs are trimmed back to the local token count.
        if padding:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding))
            router_logits = F.pad(router_logits, (0, 0, 0, padding))
        return MoEPrepareOutput(hidden_states, router_logits, None, None)

    def pad_and_split_input_ids(self, input_ids):
        return F.pad(input_ids, (0, self.padded_tokens - input_ids.numel()))

    def finalize(self, hidden_states, reduce_results=False, padded_hidden_states_shape=None):
        return hidden_states[: self.num_tokens]


class TokenDispatcherWithDeepEP(MoETokenDispatcher[DeepEPCombineMetadata]):
    def __init__(self, moe_config, prepare_finalize):
        super().__init__(top_k=moe_config.experts_per_token, num_experts=moe_config.num_experts)
        self.hidden = moe_config.hidden_dim
        self.prepare_finalize = prepare_finalize
        self.buffer = None
        self.calls = 0
        # Initialize once during model construction, before profiling enters
        # operator-specific ACL contexts. Layers with this shape reuse the owner.
        self._initialize_buffer()

    def _initialize_buffer(self):
        if self.buffer is None:
            from deep_ep import ElasticBuffer

            self._shmem_libraries = _load_private_shmem()
            torch.npu.set_device(torch.npu.current_device())
            capacity = get_current_vllm_config().scheduler_config.max_num_batched_tokens
            self.buffer = ElasticBuffer(
                get_ep_group().cpu_group,
                num_max_tokens_per_rank=max(DEEPEP_MIN_HIGH_THROUGHPUT_CAPACITY, capacity),
                hidden=self.hidden,
                num_topk=self.top_k,
                explicitly_destroy=True,
                machine_type=get_ascend_config().deepep_machine_type,
            )
            logger.info(
                "DeepEP active: EP=%d backend=%s hidden=%d experts=%d topk=%d capacity=%d",
                self.ep_size,
                self.buffer.machine_type,
                self.hidden,
                self.num_experts,
                self.top_k,
                capacity,
            )

    def token_dispatch(self, token_dispatch_input):
        self._initialize_buffer()
        x = token_dispatch_input.hidden_states
        if x.dtype != torch.bfloat16:
            raise ValueError("DeepEP validation expects unquantized BF16 dispatch input")
        ids = token_dispatch_input.topk_ids.to(torch.int64).contiguous()
        weights = token_dispatch_input.topk_weights.float().contiguous()
        count = self.prepare_finalize.num_tokens
        if count < x.shape[0]:
            ids = ids.clone()
            weights = weights.clone()
            ids[count:] = -1
            weights[count:] = 0
        recv, _, recv_weights, handle, event = self.buffer.dispatch(
            x.contiguous(),
            topk_idx=ids,
            topk_weights=weights,
            num_experts=self.num_experts,
            do_expand=True,
            expert_alignment=1,
            async_with_compute_stream=True,
        )
        event.current_stream_wait()
        group_list = torch.tensor(handle.num_recv_tokens_per_expert_list, dtype=torch.int64, device=x.device)
        self.calls += 1
        if self.calls in (1, 40, 80):
            logger.info(
                "DeepEP dispatch #%d: local_tokens=%d padded_tokens=%d recv_rows=%d",
                self.calls,
                count,
                x.shape[0],
                recv.shape[0],
            )
        return MoETokenDispatchOutput(recv, group_list, 1, DeepEPCombineMetadata(handle, recv_weights))

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        if bias is not None:
            raise ValueError("DeepEP validation does not support combine bias")
        # DeepEP moves routing weights separately; apply them exactly once, after GMM2.
        weighted = (hidden_states.float() * combine_metadata.weights[:, None]).to(torch.bfloat16)
        output, _, event = self.buffer.combine(
            weighted.contiguous(), combine_metadata.handle, async_with_compute_stream=True
        )
        event.current_stream_wait()
        if self.calls in (1, 40, 80):
            logger.info("DeepEP combine #%d: output_tokens=%d", self.calls, output.shape[0])
        return output

    def destroy(self):
        if self.buffer is not None:
            self.buffer.destroy()
            self.buffer = None


class DeepEPCommImpl(MoECommMethod):
    def __init__(self, moe_config):
        self.moe_config = moe_config
        self.prepare_finalize = self._get_prepare_finalize()
        self.token_dispatcher = self._get_token_dispatcher()
        self.lora_context = None

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithDeepEP(self.moe_config)

    def _get_token_dispatcher(self):
        return TokenDispatcherWithDeepEP(self.moe_config, self.prepare_finalize)

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)

    def fused_experts(self, fused_experts_input, quant_method=None):
        if quant_method is None:
            raise ValueError("DeepEP validation requires a quantization scheme")
        dispatched = self.token_dispatcher.token_dispatch(
            build_token_dispatch_input(fused_experts_input=fused_experts_input)
        )
        if dispatched.hidden_states.shape[0]:
            mlp_input = build_mlp_compute_input(
                fused_experts_input=fused_experts_input,
                token_dispatch_output=dispatched,
                moe_config=self.moe_config,
            )
            expert_output, _ = apply_moe_mlp(mlp_input, quant_method)
        else:
            # A rank with no selected experts still participates in combine.
            expert_output = dispatched.hidden_states
        output = self.token_dispatcher.token_combine(expert_output, dispatched.combine_metadata)
        return FusedExpertsResult(routed_out=output, expert_tokens=dispatched.group_list, group_list_type=1)
