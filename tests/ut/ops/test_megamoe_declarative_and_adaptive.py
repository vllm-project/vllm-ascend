# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Behavioral MegaMoE shared-expert fusion and reload tests.

Uses real vLLM MergedColumnParallelLinear / RowParallelLinear modules and their
real weight_loaders so TP de-interleaving and K-sharded reassembly are verified
against vLLM's actual sharding contract rather than a test-only projection stub.
"""

import sys
from contextlib import ExitStack, nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Spawned multiprocessing workers do not run pytest's conftest automatically.
if "torch_npu" not in sys.modules:
    import tests.ut.conftest  # noqa: F401

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.utils._pytree import tree_leaves, tree_map_only
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.model_executor.warmup import nz_warmup
from vllm_ascend.ops import linear as linear_mod
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_mod
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_mod
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe import routed_experts as routed_mod
from vllm_ascend.ops.fused_moe import shared_experts as shared_mod
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, SharedExpertParallelMode
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, vllm_version_is
from vllm_ascend.worker import worker
from vllm_ascend.worker.v2 import model_runner as runner_mod

_HIDDEN = 1024
_SHARED_INTERMEDIATE = 512
_ROUTED_INTERMEDIATE = 256
_GATE = "shared_mlp.gate_proj"
_UP = "shared_mlp.up_proj"
_GATE_UP = "shared_mlp.gate_up_proj"
_DOWN = "shared_mlp.down_proj"
_CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _register_cpu_megamoe_armed_op():
    library = torch.library.Library("vllm", "IMPL", "CPU")
    library.impl("ascend_moe_forward_megamoe_armed", fused_moe_mod._ascend_moe_forward_megamoe_armed)
    try:
        yield
    finally:
        library._destroy()


class _W8A8LinearQuantConfig:
    def get_quant_method(self, layer, prefix):
        return AscendLinearMethod(w8a8_dynamic.AscendW8A8DynamicLinearMethod())


def _shared_layer_checkpoint(prefix: str, seed: int, *, bf16: bool = False) -> dict[str, torch.Tensor]:
    """Generate an unsharded HuggingFace-style shared-expert checkpoint."""
    gen = torch.Generator().manual_seed(seed)
    if bf16:
        return {
            f"{prefix}.{_GATE}.weight": (torch.randn(_SHARED_INTERMEDIATE, _HIDDEN, generator=gen) * 0.1).bfloat16(),
            f"{prefix}.{_UP}.weight": (torch.randn(_SHARED_INTERMEDIATE, _HIDDEN, generator=gen) * 0.1).bfloat16(),
            f"{prefix}.{_DOWN}.weight": (torch.randn(_HIDDEN, _SHARED_INTERMEDIATE, generator=gen) * 0.1).bfloat16(),
        }

    def quant_linear(name: str, out_dim: int, in_dim: int) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}.{name}.weight": torch.randint(-8, 9, (out_dim, in_dim), dtype=torch.int8, generator=gen),
            f"{prefix}.{name}.weight_scale": (torch.rand(out_dim, 1, generator=gen) / 16 + 1 / 256).bfloat16(),
            f"{prefix}.{name}.weight_offset": torch.zeros(out_dim, 1, dtype=torch.bfloat16),
        }

    return (
        quant_linear(_GATE, _SHARED_INTERMEDIATE, _HIDDEN)
        | quant_linear(_UP, _SHARED_INTERMEDIATE, _HIDDEN)
        | quant_linear(_DOWN, _HIDDEN, _SHARED_INTERMEDIATE)
    )


def _full_model_checkpoint(model: nn.Module, seed: int, *, bf16: bool = False) -> dict[str, torch.Tensor]:
    """Generate an unsharded checkpoint covering dense, shared, and routed layers."""
    checkpoint: dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(seed)
    for idx in range(len(model.moe_layers)):
        checkpoint.update(_shared_layer_checkpoint(f"moe_layers.{idx}", seed + idx * 101, bf16=bf16))
        routed = model.moe_layers[idx].routed_experts
        for name, param in routed.named_parameters():
            if bf16:
                value = (torch.randn(param.shape, generator=gen) * 0.1).to(param.dtype)
            elif name.endswith("_weight"):
                value = torch.randint(-8, 8, param.shape, dtype=param.dtype, generator=gen)
            elif name.endswith("_weight_scale"):
                value = (torch.rand(param.shape, generator=gen) / 16 + 1 / 256).to(param.dtype)
            elif name.endswith("_scale_bias"):
                value = (torch.rand(param.shape, generator=gen) - 0.5).to(param.dtype)
            else:
                value = torch.zeros_like(param)
            checkpoint[f"moe_layers.{idx}.routed_experts.{name}"] = value
    if getattr(model, "dense_mlp", None) is not None:
        for name, param in model.dense_mlp.named_parameters():
            checkpoint[f"dense_mlp.{name}"] = (torch.randn(param.shape, generator=gen) * 0.1).to(param.dtype)
    return checkpoint


def _build_real_shared_mlp(*, shards: int, rank: int, bf16: bool) -> nn.Module:
    """Instantiate real vLLM MergedColumnParallelLinear and RowParallelLinear."""
    quant_config = None if bf16 else _W8A8LinearQuantConfig()
    fake_tp = SimpleNamespace(rank_in_group=rank, world_size=shards)
    shared_mlp = nn.Module()
    with patch.object(ps, "_TP", fake_tp):
        shared_mlp.gate_up_proj = MergedColumnParallelLinear(
            input_size=_HIDDEN,
            output_sizes=[_SHARED_INTERMEDIATE, _SHARED_INTERMEDIATE],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            return_bias=True,
        )
        shared_mlp.down_proj = RowParallelLinear(
            input_size=_SHARED_INTERMEDIATE,
            output_size=_HIDDEN,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            reduce_results=False,
            return_bias=True,
        )
    shared_mlp.act_fn = SiluAndMul.__new__(SiluAndMul)
    nn.Module.__init__(shared_mlp.act_fn)
    return shared_mlp


def _build_moe_runner(tp_group, shards: int, rank: int, routed_scheme_factory=None) -> AscendMoERunner:
    moe = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(moe)
    moe._fse_fuse_gate = False
    moe._shared_load_generation = 0
    moe._shared_completed_generation = 0
    routed_scheme = routed_scheme_factory() if routed_scheme_factory is not None else None
    bf16 = isinstance(routed_scheme, AscendUnquantizedFusedMoEMethod)
    moe.shared_mlp = _build_real_shared_mlp(shards=shards, rank=rank, bf16=bf16)
    moe.moe_config = SimpleNamespace(
        hidden_dim=_HIDDEN,
        intermediate_size_per_partition=_ROUTED_INTERMEDIATE,
        in_dtype=torch.bfloat16,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
        tp_group=tp_group,
    )
    routed = moe.routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(routed)
    routed.quant_method = None
    quant_type = QuantType.W4A8
    if routed_scheme is not None:
        quant_type = routed_scheme.quant_type
        routed.quant_method = routed_scheme if bf16 else AscendFusedMoEMethod(routed_scheme, moe.moe_config)
        routed.quant_method.create_weights(
            routed,
            num_experts=2,
            hidden_size=_HIDDEN,
            intermediate_size_per_partition=moe.moe_config.intermediate_size_per_partition,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
    shared = moe.ascend_shared_experts = AscendSharedExperts(
        moe.shared_mlp, moe.moe_config, quant_type, routed.quant_method
    )
    routed.ascend_megamoe_shared_fusion = shared.megamoe_shared_fusion()
    return moe


class _Model(nn.Module):
    """Realistic model with optional dense MLP layer and 1..N MoE layers."""

    def __init__(
        self,
        tp_group,
        shards: int,
        rank: int = 0,
        routed_scheme_factory=None,
        *,
        num_moe_layers: int = 1,
        with_dense_layer: bool = False,
    ):
        super().__init__()
        self._shards = shards
        self._rank = rank
        self.dense_mlp = nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16) if with_dense_layer else None
        if self.dense_mlp is not None:
            self.dense_mlp.weight.requires_grad_(False)
            self.dense_mlp.weight.weight_loader = default_weight_loader
        self.moe_layers = nn.ModuleList(
            [_build_moe_runner(tp_group, shards, rank, routed_scheme_factory) for _ in range(num_moe_layers)]
        )

    @property
    def moe(self) -> AscendMoERunner:
        return self.moe_layers[0]

    def load_weights(self, weights):
        """Load unsharded HuggingFace checkpoints through real vLLM weight_loaders."""
        loaded = set()
        params = dict(self.named_parameters())
        fake_tp = SimpleNamespace(rank_in_group=self._rank, world_size=self._shards)
        with patch.object(ps, "_TP", fake_tp):
            for name, weight in weights:
                if f".{_GATE}." in name:
                    target = name.replace(f".{_GATE}.", f".{_GATE_UP}.")
                    param = params[target]
                    param.weight_loader(param, weight, 0)
                    loaded.add(target)
                elif f".{_UP}." in name:
                    target = name.replace(f".{_UP}.", f".{_GATE_UP}.")
                    param = params[target]
                    param.weight_loader(param, weight, 1)
                    loaded.add(target)
                else:
                    param = params[name]
                    param.weight_loader(param, weight)
                    loaded.add(name)
        return loaded


def _make_runner(
    monkeypatch,
    world_size: int,
    shards: int,
    routed_scheme_factory=None,
    *,
    rank: int = 0,
    shared_fusion: bool = True,
    num_moe_layers: int = 1,
    with_dense_layer: bool = False,
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="ascend",
            word_embeddings_untied_by_checkpoint=False,
        ),
        load_config=SimpleNamespace(device="cpu"),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        lora_config=None,
    )
    ascend_config = SimpleNamespace(
        enable_fused_mc2=1,
        megamoe_shared_expert_fusion=shared_fusion,
        enable_shared_expert_dp=world_size > 1 and shards == 1,
        multistream_overlap_shared_expert=False,
    )
    monkeypatch.setattr(shared_mod, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(shared_mod, "shared_expert_tp_enable", lambda: False)
    monkeypatch.setattr(shared_mod, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(moe_utils, "supports_cann_mega_moe_mixed_int_shared", lambda: True)
    monkeypatch.setattr(
        shared_mod,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A3),
    )
    monkeypatch.setattr(w8a8_dynamic, "maybe_trans_nz", lambda weight: weight)
    monkeypatch.setattr(linear_mod, "maybe_trans_nz", lambda weight: weight)
    if routed_scheme_factory is not None:
        for quant_module in (routed_mod, w4a8, w8a8_dynamic):
            monkeypatch.setattr(quant_module, "use_cann_megamoe", lambda _: True)
        monkeypatch.setattr(routed_mod, "get_ascend_config", lambda: ascend_config)
        monkeypatch.setattr(routed_mod, "get_current_hardware_profile", shared_mod.get_current_hardware_profile)
        monkeypatch.setattr(w8a8_dynamic, "get_ascend_config", lambda: ascend_config)
        monkeypatch.setattr(torch.Tensor, "npu", lambda tensor: tensor, raising=False)
    group = SimpleNamespace(world_size=world_size, rank_in_group=rank, cpu_group=object(), all_gather=Mock())

    def agree(output, item, *, group):
        moe_config = runner.model.moe.moe_config
        assert group in (
            moe_config.tp_group.cpu_group,
            getattr(getattr(moe_config, "mc2_group", None), "cpu_group", None),
        )
        output[:] = [item] * len(output)

    monkeypatch.setattr(torch.distributed, "all_gather_object", agree)
    runner = runner_mod.NPUModelRunner.__new__(runner_mod.NPUModelRunner)
    runner.vllm_config = config
    runner.ascend_config = ascend_config
    runner.lora_config = None
    runner.speculator = None
    runner.encoder_cache = None
    runner.pooling_runner = None
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.model_state = SimpleNamespace()
    runner.prompt_logprobs_worker = None
    with set_current_vllm_config(config):
        runner.model = _Model(
            group,
            shards,
            rank=rank,
            routed_scheme_factory=routed_scheme_factory,
            num_moe_layers=num_moe_layers,
            with_dense_layer=with_dense_layer,
        )
        record_metadata_for_reloading(runner.model)
    return runner


def _load_model(monkeypatch, runner, checkpoint):
    def load_weights(owner, *args, **kwargs):
        with torch.inference_mode():
            owner.model.load_weights(checkpoint.items())
            process_weights_after_loading(owner.model, owner.vllm_config.model_config, _CPU)

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(GPUModelRunner, "load_model", load_weights)
        patch_ctx.setattr(nz_warmup, "start_nz_warm_thread", lambda _: None)
        patch_ctx.setattr(nz_warmup, "join_nz_warm_thread", lambda: None)
        patch_ctx.setattr(runner_mod, "get_pd_dspark_aux_layer_ids", lambda _: ())
        patch_ctx.setattr(runner_mod.expert_offload, "validate_loading", lambda *_: None)
        patch_ctx.setattr(runner_mod.expert_offload, "initialize", lambda _: None)
        with set_current_vllm_config(runner.vllm_config):
            runner.load_model()


def _materialize_peer_shards(
    monkeypatch,
    checkpoint: dict[str, torch.Tensor],
    shards: int,
    *,
    num_moe_layers: int = 1,
    bf16: bool = False,
    mutate_peer=None,
) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
    """Load the unsharded checkpoint into real per-rank vLLM linears to obtain true peer shards."""
    per_layer: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]] = [[] for _ in range(num_moe_layers)]
    for peer_rank in range(shards):
        peer_ckpt = {k: v.clone() for k, v in checkpoint.items()}
        if mutate_peer is not None:
            mutate_peer(peer_rank, peer_ckpt)
        fake_tp = SimpleNamespace(rank_in_group=peer_rank, world_size=shards)
        tmp = nn.Module()
        tmp.moe_layers = nn.ModuleList()
        for _ in range(num_moe_layers):
            holder = nn.Module()
            holder.shared_mlp = _build_real_shared_mlp(shards=shards, rank=peer_rank, bf16=bf16)
            tmp.moe_layers.append(holder)
        params = dict(tmp.named_parameters())
        with torch.inference_mode(), patch.object(ps, "_TP", fake_tp):
            for name, weight in peer_ckpt.items():
                if f".{_GATE}." in name:
                    target = name.replace(f".{_GATE}.", f".{_GATE_UP}.")
                    params[target].weight_loader(params[target], weight, 0)
                elif f".{_UP}." in name:
                    target = name.replace(f".{_UP}.", f".{_GATE_UP}.")
                    params[target].weight_loader(params[target], weight, 1)
                elif name in params:
                    params[name].weight_loader(params[name], weight)
            process_weights_after_loading(
                tmp,
                SimpleNamespace(
                    dtype=torch.bfloat16,
                    quantization="ascend",
                    word_embeddings_untied_by_checkpoint=False,
                ),
                _CPU,
            )
        for layer_idx, holder in enumerate(tmp.moe_layers):
            gu = holder.shared_mlp.gate_up_proj
            dn = holder.shared_mlp.down_proj
            scales = None if bf16 else torch.cat((gu.weight_scale.flatten().float(), dn.weight_scale.flatten().float()))
            gu_rows = gu.weight.detach() if bf16 else gu.weight.detach().T.contiguous()
            dn_rows = dn.weight.detach().T.contiguous() if bf16 else dn.weight.detach().contiguous()
            per_layer[layer_idx].append((gu_rows, dn_rows, scales))
    return per_layer


def _wire_simulated_tp_all_gather(
    monkeypatch,
    runner,
    checkpoint: dict[str, torch.Tensor],
    shards: int,
    rank: int,
    *,
    bf16: bool = False,
    mutate_peer=None,
):
    """Simulate TP all_gather using real vLLM-loaded peer shards (zero manual shard math)."""
    group = runner.model.moe.moe_config.tp_group
    group.all_gather.reset_mock()
    if shards == 1:
        group.all_gather.side_effect = AssertionError("Replicated weights must avoid TP weight gathers")
        return
    num_layers = len(runner.model.moe_layers)
    peer_layers = _materialize_peer_shards(
        monkeypatch, checkpoint, shards, num_moe_layers=num_layers, bf16=bf16, mutate_peer=mutate_peer
    )
    scale_calls = [[peer[2] for peer in layer_peers] for layer_peers in peer_layers if layer_peers[0][2] is not None]
    weight_calls = []
    for layer_peers in peer_layers:
        weight_calls.append([peer[0] for peer in layer_peers])
        weight_calls.append([peer[1] for peer in layer_peers])
    scale_iter = iter(scale_calls)
    weight_iter = iter(weight_calls)

    def gather(tensor, dim=0):
        slices = next(scale_iter) if tensor.ndim == 1 else next(weight_iter)
        torch.testing.assert_close(tensor, slices[rank], rtol=0, atol=0)
        return torch.cat(slices, dim=dim)

    group.all_gather.side_effect = gather


def _prepare_trainer_update(runner, weights):
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    model = runner.model
    config = runner.vllm_config.model_config
    start = partial(start_reload, model)
    finish = partial(finish_reload, model, config, _CPU)
    transfer_worker = worker.NPUWorker.__new__(worker.NPUWorker)
    transfer_worker.vllm_config = runner.vllm_config
    transfer_worker.model_runner = runner
    transfer_worker.weight_transfer_engine = SimpleNamespace(
        finish_weight_update=finish,
        reset_weight_update_target=lambda: None,
    )
    transfer_worker._weight_update_active = True
    transfer_worker._weight_update_is_draft = False
    transfer_worker._incomplete_weight_updates = set()
    transfer_worker._weight_update_kv_cache_loan = ExitStack()
    with torch.inference_mode():
        start()
        model.load_weights(weights)
    return transfer_worker


def _operands(runner, layer_idx: int = 0):
    weights = runner.model.moe_layers[layer_idx].ascend_shared_experts.megamoe_shared_fusion().weights
    assert weights is not None
    return weights.operands()


def _capture_shared(x, operands):
    bindings = tuple((tensor.data_ptr(), tensor.untyped_storage()) for tensor in operands)
    captured = tuple(
        torch.empty(0, dtype=tensor.dtype).set_(storage, tensor.storage_offset(), tensor.shape, tensor.stride())
        for tensor, (_, storage) in zip(operands, bindings, strict=True)
    )
    assert tuple(tensor.data_ptr() for tensor in captured) == tuple(address for address, _ in bindings)

    def replay():
        w1, w2, s1, s2 = captured
        gate, up = ((x @ w1.double()) * s1.to(torch.int32).view(torch.float32)).chunk(2, dim=-1)
        return ((F.silu(gate) * up) @ w2.double()) * s2.to(torch.int32).view(torch.float32)

    return replay, bindings


def _eager_shared(x, checkpoint: dict[str, torch.Tensor], *, prefix: str = "moe_layers.0"):
    gate = F.linear(x, checkpoint[f"{prefix}.{_GATE}.weight"].double())
    up = F.linear(x, checkpoint[f"{prefix}.{_UP}.weight"].double())
    gate = gate * checkpoint[f"{prefix}.{_GATE}.weight_scale"].flatten()
    up = up * checkpoint[f"{prefix}.{_UP}.weight_scale"].flatten()
    out = F.linear(F.silu(gate) * up, checkpoint[f"{prefix}.{_DOWN}.weight"].double())
    return out * checkpoint[f"{prefix}.{_DOWN}.weight_scale"].flatten()


def _assert_same_storage(runner, bindings, layer_idx: int = 0):
    assert tuple((tensor.data_ptr(), tensor.untyped_storage()) for tensor in _operands(runner, layer_idx)) == bindings


@pytest.mark.parametrize(("world_size", "shards"), [(4, 4), (1, 1), (4, 1)], ids=["tp4", "tp1", "tp4-replicated"])
@pytest.mark.parametrize("update", ["layerwise", "kernel", "gate-up-only", "trainer"])
def test_same_capture_reads_committed_shared_expert_reload(monkeypatch, world_size, shards, update):
    runner = _make_runner(monkeypatch, world_size, shards, rank=0)
    checkpoint_a = _shared_layer_checkpoint("moe_layers.0", seed=17)
    checkpoint_b = _shared_layer_checkpoint("moe_layers.0", seed=42)
    _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_a, shards, rank=0)
    _load_model(monkeypatch, runner, checkpoint_a)
    original_params = dict(runner.model.named_parameters())

    x = torch.linspace(-0.5, 0.5, steps=2 * _HIDDEN, dtype=torch.float64).view(2, _HIDDEN)
    before = _operands(runner)
    gate_up = runner.model.moe.shared_mlp.gate_up_proj
    standalone_scale = gate_up.weight_scale_fp32
    replay, bindings = _capture_shared(x, before)
    output_a = replay()
    torch.testing.assert_close(output_a, _eager_shared(x, checkpoint_a))
    if shards == 1:
        assert before[0].untyped_storage() == original_params[f"moe_layers.0.{_GATE_UP}.weight"].untyped_storage()
        assert before[1].untyped_storage() == original_params[f"moe_layers.0.{_DOWN}.weight"].untyped_storage()

    if update == "gate-up-only":
        checkpoint_b = {k: (checkpoint_a[k] if f".{_DOWN}." in k else v) for k, v in checkpoint_b.items()}

    # Build a fresh model for kernel-format parameter reference.
    with monkeypatch.context() as patch_ctx:
        fresh_b = _make_runner(patch_ctx, world_size, shards, rank=0)
        _wire_simulated_tp_all_gather(patch_ctx, fresh_b, checkpoint_b, shards, rank=0)
        _load_model(patch_ctx, fresh_b, checkpoint_b)

    _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_b, shards, rank=0)
    if update == "kernel":
        runner.reload_weights(fresh_b.model.named_parameters(), is_checkpoint_format=False)
    else:
        # Commit Down before Gate/Up, including the projection-only case.
        items = list(reversed(list(checkpoint_b.items())))
        if update == "gate-up-only":
            items = [(name, val) for name, val in items if f".{_DOWN}." not in name]
        if update == "trainer":
            _prepare_trainer_update(runner, items).finish_weight_update()
        else:
            runner.reload_weights(items)

    if shards == 1:
        runner.model.moe.moe_config.tp_group.all_gather.assert_not_called()
    _assert_same_storage(runner, bindings)
    assert gate_up.weight_scale_fp32 is standalone_scale
    torch.testing.assert_close(standalone_scale, fresh_b.model.moe.shared_mlp.gate_up_proj.weight_scale_fp32)
    output_b = replay()
    torch.testing.assert_close(output_b, _eager_shared(x, checkpoint_b))
    assert not torch.allclose(output_a, output_b)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("armed", [False, True], ids=["initial", "reload"])
def test_divergent_down_scale_rejects_fusion_and_allows_recovery(monkeypatch, rank, armed):
    runner = _make_runner(monkeypatch, 4, 4, rank=rank)
    checkpoint_a = _shared_layer_checkpoint("moe_layers.0", seed=17)
    checkpoint_b = _shared_layer_checkpoint("moe_layers.0", seed=42)
    checkpoint_c = _shared_layer_checkpoint("moe_layers.0", seed=73)
    scale_name = f"moe_layers.0.{_DOWN}.weight_scale"

    def skew_rank3(peer_rank, peer_ckpt):
        if peer_rank == 3:
            # A 1-ULP difference breaks the mathematical contract across TP shards.
            s = peer_ckpt[scale_name][:1]
            s.copy_(torch.nextafter(s, torch.ones_like(s)))

    x = torch.linspace(-0.5, 0.5, steps=2 * _HIDDEN, dtype=torch.float64).view(2, _HIDDEN)
    if armed:
        _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_a, 4, rank)
        _load_model(monkeypatch, runner, checkpoint_a)
        replay, bindings = _capture_shared(x, _operands(runner))
        frozen = tuple(tensor.clone() for tensor in _operands(runner))

        # Invalid reload B must fail without mutating captured storage.
        local_b = {k: v.clone() for k, v in checkpoint_b.items()}
        skew_rank3(rank, local_b)
        _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_b, 4, rank, mutate_peer=skew_rank3)
        with pytest.raises(RuntimeError, match="TP shards disagree on down_proj weight_scale"):
            runner.reload_weights(local_b.items())
        _assert_same_storage(runner, bindings)
        for tensor, values in zip(_operands(runner), frozen, strict=True):
            torch.testing.assert_close(tensor, values, rtol=0, atol=0)
        torch.testing.assert_close(replay(), _eager_shared(x, checkpoint_a))

        # Subsequent valid reload C must recover cleanly into the same captured storage.
        _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_c, 4, rank)
        runner.reload_weights(checkpoint_c.items())
        _assert_same_storage(runner, bindings)
        torch.testing.assert_close(replay(), _eager_shared(x, checkpoint_c))
    else:
        local_b = {k: v.clone() for k, v in checkpoint_b.items()}
        skew_rank3(rank, local_b)
        _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_b, 4, rank, mutate_peer=skew_rank3)
        _load_model(monkeypatch, runner, local_b)
        assert not runner.model.moe.ascend_shared_experts.megamoe_shared_weights_ready


@pytest.mark.parametrize("local_projection", [False, True], ids=["captured", "committed"])
def test_armed_reload_rejects_new_acl_format_before_refilling(monkeypatch, local_projection):
    runner = _make_runner(monkeypatch, 4, 4, rank=0)
    checkpoint_a = _shared_layer_checkpoint("moe_layers.0", seed=17)
    checkpoint_b = _shared_layer_checkpoint("moe_layers.0", seed=42)
    _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_a, 4, rank=0)
    _load_model(monkeypatch, runner, checkpoint_a)
    weights = _operands(runner)
    bindings = tuple((tensor.data_ptr(), tensor.untyped_storage()) for tensor in weights)
    frozen = tuple(tensor.clone() for tensor in weights)
    down = runner.model.moe.shared_mlp.down_proj.weight if local_projection else weights[1]
    get_format = shared_mod._get_tensor_acl_format
    monkeypatch.setattr(
        shared_mod,
        "_get_tensor_acl_format",
        lambda tensor: ACL_FORMAT_FRACTAL_NZ if tensor is down else get_format(tensor),
    )

    _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoint_b, 4, rank=0)
    layout_error = "registered shared projection layout changed" if local_projection else "w2 layout changed"
    with pytest.raises(RuntimeError, match=layout_error):
        runner.reload_weights(checkpoint_b.items())
    if local_projection:
        runner.model.moe.moe_config.tp_group.all_gather.assert_not_called()
    _assert_same_storage(runner, bindings)
    for tensor, values in zip(_operands(runner), frozen, strict=True):
        torch.testing.assert_close(tensor, values, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("tp_size", "shards", "shared_fusion"),
    [(1, 1, True), (4, 1, True), (4, 4, False)],
    ids=["armed-tp1", "armed-replicated", "fusion-disabled"],
)
def test_reload_uses_no_collectives_without_shared_weight_gathers(monkeypatch, tp_size, shards, shared_fusion):
    runner = _make_runner(monkeypatch, tp_size, shards, rank=0, shared_fusion=shared_fusion)
    moe_config = runner.model.moe.moe_config
    moe_config.mc2_group = SimpleNamespace(world_size=2 * tp_size, cpu_group=object())
    checkpoint_a = _shared_layer_checkpoint("moe_layers.0", seed=17)
    checkpoint_b = _shared_layer_checkpoint("moe_layers.0", seed=42)
    _load_model(monkeypatch, runner, checkpoint_a)
    assert runner.model.moe.ascend_shared_experts.megamoe_shared_weights_ready is shared_fusion

    vote, agree = Mock(), Mock()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", vote)
    monkeypatch.setattr(dist, "all_gather_object", agree)
    runner.reload_weights(checkpoint_b.items())
    trainer = _prepare_trainer_update(runner, checkpoint_a.items())
    trainer.finish_weight_update()
    assert not trainer._weight_update_active
    vote.assert_not_called()
    agree.assert_not_called()
    moe_config.tp_group.all_gather.assert_not_called()


def _prepare_moe_forward(monkeypatch, moe):
    moe.layer_name = "ut.megamoe"
    moe.routed_input_transform = None
    moe.routed_output_transform = None
    moe._shared_experts = object()
    moe._gate = None
    moe.gate = None
    moe.routed_scaling_factor = 1.0
    moe.divide_shared_output_in_fp16 = False
    moe.moe_config.is_sequence_parallel = True
    moe._sequence_parallel_context = nullcontext
    moe._maybe_pad_hidden_states = lambda shared, hidden: (hidden, None, None)
    moe._maybe_add_zero_expert_output = lambda out: out
    moe._encode_layer_name = lambda: moe.layer_name
    if not vllm_version_is("0.34.0"):
        from vllm.model_executor.layers.fused_moe.config import RoutedScaleMode

        moe.routed_scale_mode = RoutedScaleMode.ROUTER
    ctx = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2, use_mega_moe=True)
    monkeypatch.setattr(fused_moe_mod, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(fused_moe_mod, "get_layer_from_name", lambda _: moe)
    return ctx


def _cpu_megamoe(
    x,
    topk_ids,
    topk_weights,
    w1,
    w2,
    _context,
    *,
    l1_weights_sf,
    l2_weights_sf,
    l1_bias,
    l2_bias,
    shared_l1_weights,
    shared_l2_weights,
    shared_l1_weights_sf,
    shared_l2_weights_sf,
    packed_int4,
    **_kwargs,
):
    """Run routed + shared projections on CPU using the exact operands passed to CANN."""

    def unpack(packed):
        low = ((packed & 15) ^ 8) - 8
        return torch.stack((low, packed >> 4), dim=-1).flatten(-2)

    def project(x_in, weight, scales, biases=None, expert=0):
        y = x_in @ weight.double()
        if scales is not None:
            y = y * scales[expert].to(torch.int32).view(torch.float32)
        return y if biases is None else y + biases[expert]

    routed = []
    for expert, (gate_up, down) in enumerate(zip(w1, w2, strict=True)):
        if packed_int4:
            gate_up, down = unpack(gate_up), unpack(down)
        gate, up = project(x, gate_up, l1_weights_sf, l1_bias, expert).chunk(2, -1)
        routed.append(project(F.silu(gate) * up, down, l2_weights_sf, l2_bias, expert))
    selected = torch.stack(routed, dim=1).gather(1, topk_ids.long().unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    gate, up = project(x, shared_l1_weights[0], shared_l1_weights_sf).chunk(2, -1)
    shared = project(F.silu(gate) * up, shared_l2_weights[0], shared_l2_weights_sf)
    return (selected * topk_weights.unsqueeze(-1)).sum(dim=1) + shared, None


def _wire_cpu_megamoe(monkeypatch, runner, scheme):
    moe = runner.model.moe
    ctx = _prepare_moe_forward(monkeypatch, moe)
    for quant_module in (routed_mod, w4a8, w8a8_dynamic):
        monkeypatch.setattr(quant_module, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(comm_mod, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    comm = comm_mod.FusedMC2CommImpl.__new__(comm_mod.FusedMC2CommImpl)
    comm.token_dispatcher = comm_mod.TokenDispatcherWithMC2.__new__(comm_mod.TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 8
    comm.token_dispatcher.global_bs = 0
    comm.mega_moe_symm_buffer = SimpleNamespace()
    comm.mega_moe = Mock(wraps=partial(_cpu_megamoe, packed_int4=scheme.quant_type == QuantType.W4A8))
    comm._mega_moe_supported_kwargs = frozenset(
        {
            "shared_l1_weights",
            "shared_l2_weights",
            "shared_l1_weights_sf",
            "shared_l2_weights_sf",
            "shared_weight1_type",
            "shared_weight2_type",
            "shared_expert_quant_out_dtype",
        }
    )

    def forward_routed(*, hidden_states, router_logits, **_):
        topk_weights, topk_ids = router_logits.softmax(dim=-1).topk(2, dim=-1)
        layer = moe.routed_experts
        inputs = SimpleNamespace(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layer=layer,
            routing=SimpleNamespace(mc2_mask=None),
            quant=SimpleNamespace(quant_type=scheme.quant_type),
        )
        out, _ = comm._apply_cann_mega_moe(inputs, scheme.get_fused_mc2_weights(layer))
        return out, RoutedMoEMilestones()

    moe.routed_experts.forward_impl = forward_routed
    return comm.mega_moe


def _kernel_operands(call):
    args, kwargs = call
    return [tensor for tensor in tree_leaves((args[3:5], kwargs)) if isinstance(tensor, torch.Tensor)]


@pytest.mark.parametrize("use_expert_weight_list", [False, True], ids=["registered", "per-expert"])
@pytest.mark.parametrize(
    ("routed_quant", "quant_method"),
    [
        (QuantType.NONE, None),
        (QuantType.W8A8, None),
        (QuantType.W4A8, w4a8.ASCEND_QUANTIZATION_METHOD),
        (QuantType.W4A8, w4a8.COMPRESSED_TENSORS_METHOD),
    ],
    ids=["bf16", "w8a8", "w4a8-ascend", "w4a8-compressed-tensors"],
)
def test_full_moe_replay_reads_routed_and_tp_shared_reload(
    monkeypatch, routed_quant, quant_method, use_expert_weight_list
):
    bf16 = routed_quant == QuantType.NONE

    def make_scheme():
        if routed_quant == QuantType.W4A8:
            scheme = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
            scheme.quant_method = quant_method
            scheme.tp_size = 1
            scheme.use_expert_weight_list = use_expert_weight_list
            return scheme
        if routed_quant == QuantType.W8A8:
            scheme = w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
            scheme.use_expert_weight_list = use_expert_weight_list
            return scheme
        scheme = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        nn.Module.__init__(scheme)
        scheme.moe = SimpleNamespace(
            is_act_and_mul=True,
            has_bias=False,
            hidden_dim_unpadded=_HIDDEN,
            intermediate_size_per_partition_unpadded=_ROUTED_INTERMEDIATE,
        )
        scheme.moe_kernel = None
        scheme.dynamic_eplb = use_expert_weight_list
        return scheme

    def expert_tensors(owner, scheme):
        if not use_expert_weight_list:
            return []
        layer = owner.model.moe.routed_experts
        if bf16:
            return [*layer.w13_weight_list, *layer.w2_weight_list]
        return tree_leaves(scheme.get_eplb_weight_views(layer))

    runner = _make_runner(monkeypatch, 4, 4, make_scheme, rank=0)
    scheme = getattr(
        runner.model.moe.routed_experts.quant_method, "quant_method", runner.model.moe.routed_experts.quant_method
    )
    checkpoints = [_full_model_checkpoint(runner.model, seed, bf16=bf16) for seed in (17, 42, 73)]
    _wire_simulated_tp_all_gather(monkeypatch, runner, checkpoints[0], 4, rank=0, bf16=bf16)
    _load_model(monkeypatch, runner, checkpoints[0])
    kernel = _wire_cpu_megamoe(monkeypatch, runner, scheme)

    x = torch.linspace(-0.5, 0.5, steps=2 * _HIDDEN, dtype=torch.float64).view(2, _HIDDEN)
    logits = torch.tensor([[1.0, -1.0], [-0.5, 0.5]])
    previous = runner.model.moe(x, logits)
    addresses = [tensor.data_ptr() for tensor in _kernel_operands(kernel.call_args)]
    assert len(addresses) == {QuantType.NONE: 6, QuantType.W8A8: 12, QuantType.W4A8: 16}[routed_quant]
    expert_addresses = [tensor.data_ptr() for tensor in expert_tensors(runner, scheme)]
    if use_expert_weight_list:
        assert set(addresses) <= {*expert_addresses, *(t.data_ptr() for t in _operands(runner))}

    def bind(tensor):
        return torch.empty(0, dtype=tensor.dtype).set_(
            tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride()
        )

    args, kwargs = tree_map_only(torch.Tensor, bind, tuple(kernel.call_args))
    captured_operands = _kernel_operands((args, kwargs))
    assert [tensor.data_ptr() for tensor in captured_operands] == addresses
    replay = make_fx(lambda: _cpu_megamoe(*args, **kwargs, packed_int4=routed_quant == QuantType.W4A8)[0])()
    torch.testing.assert_close(replay(), previous, rtol=0, atol=0)

    updates = ("layerwise", "trainer" if use_expert_weight_list else "kernel")
    for update, ckpt in zip(updates, checkpoints[1:], strict=True):
        with monkeypatch.context() as patch_ctx:
            fresh = _make_runner(patch_ctx, 4, 4, make_scheme, rank=0)
            fresh_scheme = getattr(
                fresh.model.moe.routed_experts.quant_method,
                "quant_method",
                fresh.model.moe.routed_experts.quant_method,
            )
            _wire_simulated_tp_all_gather(patch_ctx, fresh, ckpt, 4, rank=0, bf16=bf16)
            _load_model(patch_ctx, fresh, ckpt)
            eager_kernel = _wire_cpu_megamoe(patch_ctx, fresh, fresh_scheme)
            eager = fresh.model.moe(x, logits)
            expected_operands = _kernel_operands(eager_kernel.call_args)
            expected_experts = expert_tensors(fresh, fresh_scheme)
        _wire_simulated_tp_all_gather(monkeypatch, runner, ckpt, 4, rank=0, bf16=bf16)
        if update == "trainer":
            _prepare_trainer_update(runner, ckpt.items()).finish_weight_update()
        elif update == "layerwise":
            runner.reload_weights(ckpt.items())
        else:
            runner.reload_weights(fresh.model.named_parameters(), is_checkpoint_format=False)
        torch.testing.assert_close(runner.model.moe(x, logits), eager, rtol=0, atol=0)
        assert [tensor.data_ptr() for tensor in _kernel_operands(kernel.call_args)] == addresses
        assert [tensor.data_ptr() for tensor in captured_operands] == addresses
        for captured, expected in zip(captured_operands, expected_operands, strict=True):
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        updated_experts = expert_tensors(runner, scheme)
        assert [tensor.data_ptr() for tensor in updated_experts] == expert_addresses
        for updated, expected in zip(updated_experts, expected_experts, strict=True):
            torch.testing.assert_close(updated, expected, rtol=0, atol=0)
        output = replay()
        assert output.shape == (2, _HIDDEN)
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        assert not torch.equal(output, previous)
        previous = output


def test_armed_fx_graph_switches_between_fused_and_standalone_shared_experts(monkeypatch):
    runner = _make_runner(monkeypatch, 1, 1, rank=0)
    ckpt_a = _shared_layer_checkpoint("moe_layers.0", seed=17)
    ckpt_b = _shared_layer_checkpoint("moe_layers.0", seed=42)
    _load_model(monkeypatch, runner, ckpt_a)
    moe = runner.model.moe
    ctx = _prepare_moe_forward(monkeypatch, moe)
    shared = moe.ascend_shared_experts
    shared.parallel_mode = lambda: SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY
    shared.prepare_input_before_routed = PreparedSharedExpertInput

    def standalone_shared(prepared, *_a, **_kw):
        # Compute directly from the live registered linear weights to confirm
        # fallback sees reloaded weights after reload_weights().
        w1 = shared.layer.gate_up_proj.weight.detach().double()
        s1 = shared.layer.gate_up_proj.weight_scale_fp32.detach().double()
        w2 = shared.layer.down_proj.weight.detach().double()
        s2 = shared.layer.down_proj.weight_scale_fp32.detach().double()
        gate, up = ((prepared.hidden_states.double() @ w1) * s1).chunk(2, dim=-1)
        return (((F.silu(gate) * up) @ w2) * s2).to(prepared.hidden_states.dtype)

    shared.forward = Mock(side_effect=standalone_shared)
    fusion_handle = shared.megamoe_shared_fusion()

    def routed_or_fused(*, hidden_states, skip_megamoe_shared=False, **_kw):
        if ctx.use_mega_moe and not skip_megamoe_shared:
            w1, w2, s1, s2 = fusion_handle.weights.operands()
            gate, up = ((hidden_states.double() @ w1.double()) * s1.to(torch.int32).view(torch.float32)).chunk(
                2, dim=-1
            )
            fused_shared = (((F.silu(gate) * up) @ w2.double()) * s2.to(torch.int32).view(torch.float32)).to(
                hidden_states.dtype
            )
            return hidden_states * 2 + fused_shared, RoutedMoEMilestones()
        return hidden_states * 2, RoutedMoEMilestones()

    moe.routed_experts.forward_impl = routed_or_fused

    x = torch.linspace(-0.25, 0.25, steps=2 * _HIDDEN, dtype=torch.float32).view(2, _HIDDEN)
    graph = make_fx(lambda states: moe(states, states), tracing_mode="fake")(x)
    assert torch.ops.vllm.ascend_moe_forward_megamoe_armed.default in [node.target for node in graph.graph.nodes]

    # Checkpoint A: FUSED_MC2 vs ALLTOALL fallback must produce identical outputs.
    fused_a = graph(x)
    shared.forward.assert_not_called()
    ctx.moe_comm_type = MoECommType.ALLTOALL
    ctx.use_mega_moe = False
    fallback_a = graph(x)
    shared.forward.assert_called_once()
    torch.testing.assert_close(fused_a, fallback_a, rtol=1e-4, atol=1e-4)

    # Reload to Checkpoint B: BOTH FUSED_MC2 and ALLTOALL fallback must reflect B.
    runner.reload_weights(ckpt_b.items())
    shared.forward.reset_mock()
    ctx.moe_comm_type = MoECommType.FUSED_MC2
    ctx.use_mega_moe = True
    fused_b = graph(x)
    shared.forward.assert_not_called()
    ctx.moe_comm_type = MoECommType.ALLTOALL
    ctx.use_mega_moe = False
    fallback_b = graph(x)
    shared.forward.assert_called_once()
    torch.testing.assert_close(fused_b, fallback_b, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(fused_a, fused_b)

    # Distinct shared_experts_input must bypass fused shared and run standalone.
    ctx.moe_comm_type = MoECommType.FUSED_MC2
    ctx.use_mega_moe = True
    shared.forward.reset_mock()
    distinct_out = moe(x, x, shared_experts_input=x * 0.5)
    shared.forward.assert_called_once()
    expected_distinct = x * 2 + _eager_shared((x * 0.5).double(), ckpt_b).to(x.dtype)
    torch.testing.assert_close(distinct_out, expected_distinct, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("expert_list", [False, True])
def test_w4a8_routed_weights_bind_the_int8_shared_descriptor(monkeypatch, expert_list):
    runner = _make_runner(monkeypatch, 1, 1, rank=0)
    _load_model(monkeypatch, runner, _shared_layer_checkpoint("moe_layers.0", seed=17))
    fusion = runner.model.moe.ascend_shared_experts.megamoe_shared_fusion()
    monkeypatch.setattr(w4a8, "_EXTRA_CTX", SimpleNamespace(use_mega_moe=True))
    method = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = expert_list
    weight = [torch.zeros(4, 4, dtype=torch.int8)]
    scale = [torch.ones(4, dtype=torch.int64)]
    bias = [torch.zeros(4)]
    layer = SimpleNamespace(
        ascend_megamoe_shared_fusion=fusion,
        ascend_megamoe_fold_scale=2.0,
        w13_weight_list=weight,
        w2_weight_list=weight,
        w13_weight_scale_list=scale,
        w2_weight_scale_list=scale,
        w13_scale_bias_list=bias,
        w2_scale_bias_list=bias,
        cann_mega_moe_w13_weight_list=weight,
        cann_mega_moe_w2_weight_list=weight,
        cann_mega_moe_w13_weight_scale_list=scale,
        cann_mega_moe_w2_weight_scale_list=scale,
        cann_mega_moe_w13_scale_bias_list=bias,
        cann_mega_moe_w2_scale_bias_list=bias,
    )
    weights = method.get_fused_mc2_weights(layer)
    comm = comm_mod.FusedMC2CommImpl.__new__(comm_mod.FusedMC2CommImpl)
    comm.token_dispatcher = comm_mod.TokenDispatcherWithMC2.__new__(comm_mod.TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 8
    comm.token_dispatcher.global_bs = 0
    owner = comm.mega_moe_symm_buffer = SimpleNamespace(context=object(), combine_comm_mode=1)
    comm.mega_moe = Mock(return_value=(torch.zeros(2, 4), torch.zeros(1, dtype=torch.int32)))
    comm._mega_moe_supported_kwargs = frozenset(
        {
            "shared_l1_weights",
            "shared_l2_weights",
            "shared_l1_weights_sf",
            "shared_l2_weights_sf",
            "shared_weight1_type",
            "shared_weight2_type",
        }
    )
    monkeypatch.setattr(
        comm_mod,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A3),
    )
    fused_input = SimpleNamespace(
        hidden_states=torch.ones(2, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        quant=SimpleNamespace(quant_type=QuantType.W4A8),
        routing=SimpleNamespace(mc2_mask=None),
        layer=layer,
    )
    comm._apply_cann_mega_moe(fused_input, weights)
    args, kwargs = comm.mega_moe.call_args
    assert args[5].context is owner.context
    assert args[5].combine_comm_mode == 0
    assert owner.combine_comm_mode == 1
    assert kwargs["shared_weight1_type"] == kwargs["shared_weight2_type"] == fusion.weight_type
    assert kwargs["shared_weight1_type"] != kwargs["weight1_type"]
    assert kwargs["shared_l1_weights"] is fusion.weights.w1
    assert kwargs["shared_l2_weights_sf"] is fusion.weights.w2_scale
    torch.testing.assert_close(args[2], fused_input.topk_weights * 2)

    comm._mega_moe_supported_kwargs -= {"shared_weight2_type"}
    comm.mega_moe.reset_mock()
    with pytest.raises(RuntimeError, match="lacks required shared-expert arguments.*shared_weight2_type"):
        comm._apply_cann_mega_moe(fused_input, weights)
    comm.mega_moe.assert_not_called()
