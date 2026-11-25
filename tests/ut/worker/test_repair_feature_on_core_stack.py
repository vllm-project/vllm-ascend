# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""kv_load_failure_policy='repair' against the linked vLLM repair stack.

test_repair_layout_gate.py injects a stand-in repair-executor module,
because the pinned vLLM commits (the verified main pin and the release
tag) carry no repair executor. A stand-in can only check the guard logic.
This module runs only when the vLLM on sys.path carries the real stack
(vllm-project/vllm#68377 and its prerequisites) and proves what the
stand-in cannot: a supported repair configuration survives real config
parsing and the Ascend platform validation, passes this runner's init-time
gate through the real Core resolver and geometry contract, and a
repair-only step is dispatched to the real executor with the binding this
runner supplies. On a vLLM without the stack every test here is skipped.
"""

import inspect
import logging
import os
from types import SimpleNamespace

import pytest
import torch
from vllm.config.kv_transfer import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_REPAIR_EXECUTOR = pytest.importorskip(
    "vllm.v1.worker.gpu.repair_executor",
    reason="the linked vLLM carries no repair executor stack (vllm-project/vllm#68377)",
)


@pytest.fixture(autouse=True)
def _no_dangling_ascend_file_handlers():
    """Drop Ascend file handlers whose directory no longer exists.

    An earlier test of the full suite (tests/ut/test_logger.py) leaves its
    RotatingAscendFileHandler on the vllm_ascend logger after the
    TemporaryDirectory is gone. Real config parsing logs through that handler
    and fails on the missing file, and the log level it raises then exposes
    later tests to the same handler. Such a handler can never write again, so
    it is removed, not restored.
    """
    for name in ("vllm", "vllm_ascend"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and not os.path.isdir(os.path.dirname(handler.baseFilename)):
                logger.removeHandler(handler)


# The supported Ascend repair configuration: the repair policy on the
# head-major KV cache layout, a 128-token block, and no scheduling mode
# that forks Scheduler.schedule().
_MODEL = "facebook/opt-125m"
_LAYOUT_ENV = "HND"
_BLOCK_SIZE = 128
_NUM_BLOCKS = 6
_NUM_LAYERS = 4
_LAYER_END = 2
# The authorized window: token offsets 16..79 of block 2. Rows 0..15 and
# 80..127 of the same block are unauthorized neighbors.
SPAN = (16, 80)
_WRITE_BLOCK = 2
_REQUEST_ID = "req-a"


def _adopt_kv_layout_like_engine_core(vllm_config):
    """The KV cache layout step the engine core runs before worker init:
    the worker reports the layouts its attention backends support, the
    engine resolves one for the model and records it on the cache config."""
    from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout

    backends = get_current_attn_backends(vllm_config)
    worker_support = [[layout.name for layout in get_supported_kv_cache_layouts(backends)]]
    layout = resolve_kv_cache_layout(vllm_config, worker_support)
    assert vllm_config.cache_config.kv_cache_layout == layout.name


def _parse_repair_config(monkeypatch, layout, additional_config=None):
    """Real EngineArgs -> VllmConfig parsing of a repair configuration.

    With vllm_ascend installed, create_engine_config() resolves the NPU
    platform and runs NPUPlatform.check_and_update_config itself, so the
    Ascend validation of the policy is the production path. The KV cache
    layout is then adopted the way the engine core adopts it.
    """
    from vllm.envs import disable_envs_cache

    disable_envs_cache()
    if layout is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
    engine_args = EngineArgs(
        model=_MODEL,
        block_size=_BLOCK_SIZE,
        enforce_eager=True,
        additional_config=additional_config or {},
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_load_failure_policy="repair",
        ),
    )
    vllm_config = engine_args.create_engine_config()
    _adopt_kv_layout_like_engine_core(vllm_config)
    return vllm_config


class _HeadMajorLayer:
    """A decoder layer binding this runner's per-K/V head-major views
    [num_blocks, num_kv_heads, block_size, head_dim]. On replay it writes
    its span through the forward-context metadata: the seam the repair
    fence validates. Every write adds 1.0, so the cache shows exactly how
    often each row was replayed."""

    def __init__(self, layer_name, kv_heads, head_dim, block_size):
        shape = (_NUM_BLOCKS, kv_heads, block_size, head_dim)
        attn = SimpleNamespace(kv_cache=[torch.zeros(shape), torch.zeros(shape)], layer_name=layer_name)
        self.self_attn = SimpleNamespace(attn=attn)

    def __call__(self, positions, hidden_states, residual):
        slots = get_forward_context().attn_metadata[self.self_attn.attn.layer_name].slot_mapping.long()
        for cache in self.self_attn.attn.kv_cache:
            cache[slots // cache.shape[2], :, slots % cache.shape[2], :] += 1.0
        return hidden_states + 1.0, residual


class _Stack:
    def __init__(self, layers):
        self.layers = layers

    def embed_input_ids(self, input_ids):
        return torch.zeros(input_ids.shape[0], 8)


def _model_for(vllm_config, block_size=None):
    """A model whose layers bind head-major views sized like the parsed
    configuration (the model's own KV heads and head size)."""
    block_size = vllm_config.cache_config.block_size if block_size is None else block_size
    kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_dim = vllm_config.model_config.get_head_size()
    layers = [_HeadMajorLayer(f"model.layers.{i}", kv_heads, head_dim, block_size) for i in range(_NUM_LAYERS)]
    return SimpleNamespace(model=_Stack(layers))


def _runner(vllm_config, model):
    """The runner state the gate and the binding read at the end of
    initialize_kv_cache; the methods under test are the real ones."""
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = vllm_config
    runner.kv_cache_config = None
    runner.model = model
    return runner


def _cpu_meta(block_rows, slot_mapping):
    """A real CPUAttentionMetadata with a padded 2D block table (the
    production constructor; the per-request row is block_table[row])."""
    width = max(len(row) for row in block_rows)
    table = torch.zeros(len(block_rows), width, dtype=torch.int32)
    for i, row in enumerate(block_rows):
        table[i, : len(row)] = row
    num_reqs = len(block_rows)
    return CPUAttentionMetadata(
        num_actual_tokens=slot_mapping.shape[0],
        max_query_len=slot_mapping.shape[0],
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        max_seq_len=slot_mapping.shape[0],
        seq_lens=torch.full((num_reqs,), slot_mapping.shape[0], dtype=torch.int32),
        block_table=table,
        slot_mapping=slot_mapping,
        scheduler_metadata=None,
    )


def _dispatch_repair_step(runner, binding):
    """One repair-only step exactly as GPUModelRunner.execute_model
    dispatches it: the model, the scheduler output, the runner's inputs
    and batch order, and a view binding."""
    from vllm.v1.core.sched.output import RepairWork

    model = runner.model
    work = RepairWork(
        request_id=_REQUEST_ID,
        repair_span=SPAN,
        layer_end=_LAYER_END,
        restored_extent=512,
        plan_id=1,
        epoch=7,
        expected_write_block_ids=(_WRITE_BLOCK,),
    )
    slots = torch.arange(_WRITE_BLOCK * _BLOCK_SIZE + SPAN[0], _WRITE_BLOCK * _BLOCK_SIZE + SPAN[1], dtype=torch.int64)
    positions = torch.arange(SPAN[0], SPAN[1], dtype=torch.int64)
    tokens = {_REQUEST_ID: SPAN[1] - SPAN[0]}
    scheduler_output = SchedulerOutput(
        repair_works={_REQUEST_ID: work},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    meta = _cpu_meta([torch.tensor([_WRITE_BLOCK])], slots)
    attn_metadata = {layer.self_attn.attn.layer_name: meta for layer in model.model.layers}
    with set_forward_context(attn_metadata, runner.vllm_config):
        return _REPAIR_EXECUTOR.execute_repair_only(
            model,
            scheduler_output,
            positions.clone(),
            positions,
            [_REQUEST_ID],
            view_binding=binding,
        )


def _span_written(model):
    """Per layer, how many span rows K and V hold, and whether anything
    outside the authorized rows was touched."""
    per_layer = []
    for layer in model.model.layers:
        caches = layer.self_attn.attn.kv_cache
        in_span = [cache[_WRITE_BLOCK, :, SPAN[0] : SPAN[1], :].sum().item() for cache in caches]
        total = [cache.sum().item() for cache in caches]
        per_layer.append((in_span, total))
    return per_layer


# ---------------------------------------------------------------------------
# 1. Config parsing: a supported repair configuration is admitted
# ---------------------------------------------------------------------------


def test_supported_repair_config_passes_real_config_parsing(monkeypatch):
    vllm_config = _parse_repair_config(monkeypatch, _LAYOUT_ENV)

    # Real parsing: the Core policy literal accepts 'repair' and resolves
    # the HND alias to its canonical layout name; the Ascend platform
    # validation inside create_engine_config() admitted the policy.
    assert vllm_config.kv_transfer_config.kv_load_failure_policy == "repair"
    assert vllm_config.cache_config.kv_cache_layout == "LBHNC"
    assert vllm_config.cache_config.block_size == _BLOCK_SIZE


def test_repair_config_is_refused_for_a_forking_scheduler_mode(monkeypatch):
    # A real additional_config: the recompute scheduler replaces
    # Scheduler.schedule() with a fork that never admits repair-only steps.
    with pytest.raises(
        ValueError, match="kv_load_failure_policy='repair' is not supported with recompute_scheduler_enable"
    ):
        _parse_repair_config(
            monkeypatch,
            _LAYOUT_ENV,
            additional_config={"scheduler_config": {"recompute_scheduler_enable": True}},
        )


# ---------------------------------------------------------------------------
# 2. Initialization: the runner's gate with the real Core stack
# ---------------------------------------------------------------------------


def test_init_gate_admits_supported_configuration_with_real_stack(monkeypatch):
    vllm_config = _parse_repair_config(monkeypatch, _LAYOUT_ENV)
    runner = _runner(vllm_config, _model_for(vllm_config))

    NPUModelRunner._check_repair_kv_layout_capability(runner)

    layer, layer_path = _REPAIR_EXECUTOR.find_repair_fence_layer(runner.model)
    assert layer is runner.model.model.layers[0]
    assert layer_path == "model.layers"
    # The binding handed to the executor is this runner's per-K/V
    # head-major contract for the parsed block size.
    assert NPUModelRunner._repair_view_binding(runner) == _REPAIR_EXECUTOR.per_kv_repair_binding(
        _BLOCK_SIZE, 2, "LBHNC"
    )


def test_init_gate_refuses_default_layout_with_real_stack(monkeypatch):
    vllm_config = _parse_repair_config(monkeypatch, None)
    # Without an explicit choice the engine resolves the token-major layout.
    assert vllm_config.cache_config.kv_cache_layout == "LBNHC"
    runner = _runner(vllm_config, _model_for(vllm_config))

    with pytest.raises(RuntimeError, match="head-major KV cache layouts"):
        NPUModelRunner._check_repair_kv_layout_capability(runner)


def test_init_gate_refuses_contradicting_layer_geometry_with_real_stack(monkeypatch):
    vllm_config = _parse_repair_config(monkeypatch, _LAYOUT_ENV)
    # The layer bound 64-token blocks while the parsed configuration says 128.
    runner = _runner(vllm_config, _model_for(vllm_config, block_size=64))

    with pytest.raises(RuntimeError, match="repair-fence compatible"):
        NPUModelRunner._check_repair_kv_layout_capability(runner)


# ---------------------------------------------------------------------------
# 3. Dispatch: a repair-only step through the real executor
# ---------------------------------------------------------------------------


def test_repair_steps_reach_the_inherited_dispatch():
    """The repair-only branch lives in GPUModelRunner.execute_model and
    calls the executor with self._repair_view_binding(). NPUModelRunner
    wraps execute_model but defers every step to it and has no repair
    branch of its own; if either stops holding, the dispatch check below
    no longer describes production."""
    wrapper = inspect.getsource(NPUModelRunner.execute_model)
    assert "super().execute_model(" in wrapper
    assert "repair_works" not in wrapper
    assert "execute_repair_only" not in wrapper
    dispatch = inspect.getsource(GPUModelRunner.execute_model)
    assert "execute_repair_only(" in dispatch
    assert "view_binding=self._repair_view_binding()" in dispatch
    assert "_repair_view_binding" in NPUModelRunner.__dict__
    assert "_repair_view_binding" in GPUModelRunner.__dict__


def test_repair_step_commits_with_the_runner_binding(monkeypatch):
    vllm_config = _parse_repair_config(monkeypatch, _LAYOUT_ENV)
    runner = _runner(vllm_config, _model_for(vllm_config))

    (report,) = _dispatch_repair_step(runner, NPUModelRunner._repair_view_binding(runner))

    assert (report.request_id, report.plan_id, report.epoch, report.span) == (_REQUEST_ID, 1, 7, SPAN)
    assert report.executed_layers == _LAYER_END
    assert not report.aborted
    assert report.structural_check
    # Layers [0, layer_end) replayed exactly the authorized span once, in
    # K and V; the neighbors in the block and the untouched layers are intact.
    span_rows = (SPAN[1] - SPAN[0]) * runner.model.model.layers[0].self_attn.attn.kv_cache[0].shape[1]
    span_rows *= runner.model.model.layers[0].self_attn.attn.kv_cache[0].shape[3]
    for index, (in_span, total) in enumerate(_span_written(runner.model)):
        if index < _LAYER_END:
            assert in_span == [span_rows, span_rows]
            assert total == [span_rows, span_rows]
        else:
            assert in_span == [0.0, 0.0]
            assert total == [0.0, 0.0]


def test_repair_step_is_refused_with_a_core_binding_on_per_kv_views(monkeypatch):
    """The base runner's core-logical binding contradicts the per-K/V
    views this runner binds: the real executor refuses before any layer
    runs, and both buffers stay intact."""
    vllm_config = _parse_repair_config(monkeypatch, _LAYOUT_ENV)
    runner = _runner(vllm_config, _model_for(vllm_config))

    (report,) = _dispatch_repair_step(runner, _REPAIR_EXECUTOR.core_logical_repair_binding(_BLOCK_SIZE, "LBHNC"))

    assert report.aborted
    assert report.executed_layers == 0
    assert not report.structural_check
    assert _span_written(runner.model) == [([0.0, 0.0], [0.0, 0.0])] * _NUM_LAYERS
