# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

import vllm_ascend.models.deepseek_v41.engram.elastic as offload
from vllm_ascend.models.deepseek_v41.engram.diagnostics import log_engram_runtime
from vllm_ascend.models.deepseek_v41.engram.elastic import ElasticEngramEmbedding, EngramTableState

WEIGHT_KEY = "model.layers.1.engram.embed.weight"
SCALE_KEY = "model.layers.1.engram.embed.scale"


def _table(*, storage=None, rows=6, width=4, rank=1):
    with patch.object(offload.dist, "get_rank", return_value=rank):
        table = ElasticEngramEmbedding(
            rows,
            width,
            2,
            group=object(),
            layer_id=1,
            minimum_rows=rows - 1,
            device="cpu",
            **({"storage_format": storage} if storage is not None else {}),
        )
    table._sync_phase_status = lambda succeeded, phase: succeeded
    return table


def _checkpoint(root, *, native_scale=False, invalid_scale=False):
    codes = torch.arange(1, 21, dtype=torch.float32).reshape(5, 4).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128], [126, 127], [0, 129], [128, 126], [129, 127]], dtype=torch.uint8)
    if invalid_scale:
        scales[0, 0] = 255  # A remote row for rank 1 must also be checked.
    save_file(
        {WEIGHT_KEY: codes, SCALE_KEY: scales.view(torch.float8_e8m0fnu) if native_scale else scales},
        root / "model.safetensors",
    )
    padded_codes = torch.zeros((6, 4), dtype=torch.uint8)
    padded_codes[:5].copy_(codes.view(torch.uint8))
    padded_scales = torch.full((6, 2), 127, dtype=torch.uint8)
    padded_scales[:5].copy_(scales)
    return padded_codes, padded_scales


def _install_buffer(monkeypatch, global_codes, *, corruption=None):
    class Buffer:
        instance = None

        @staticmethod
        def get_engram_storage_size_hint(rows, width, dtype):
            assert dtype == torch.float8_e4m3fn
            return rows * width + 64

        def __init__(self, group, **kwargs):
            type(self).instance = self
            self.waits = 0
            self.destroyed = False
            self.scale = None
            self.storage = None

        def engram_write(self, storage, sf=None):
            if corruption == "write":
                raise RuntimeError("OPP write failure")
            self.storage = storage.clone()
            self.scale = sf

        def engram_fetch(self, ids):
            codes = global_codes[ids.long()].clone()
            scales = self.scale.view(torch.uint8)[ids.long()].clone()
            if corruption == "codes":
                codes[0, 0] ^= 1
            if corruption == "scales":
                scales[0, 0] ^= 1

            def wait():
                self.waits += 1
                if corruption == "wait":
                    raise RuntimeError("OPP wait failure")
                return codes.view(torch.float8_e4m3fn), scales.view(torch.float8_e8m0fnu)

            return wait

        def destroy(self):
            self.destroyed = True
            self.scale = None

    package = ModuleType("cann_ops_transformer")
    ops = ModuleType("cann_ops_transformer.ops")
    ops.ElasticBuffer = Buffer
    package.ops = ops
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", package)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer.ops", ops)
    return Buffer


@pytest.mark.parametrize("storage", [None, "fp8", "mxfp8"])
@pytest.mark.parametrize("native_scale", [False, True])
def test_loader_preserves_local_fp8_and_full_e8m0_bytes(tmp_path, storage, native_scale):
    codes, scales = _checkpoint(tmp_path, native_scale=native_scale)
    table = _table(storage=storage)
    table.load_checkpoint(tmp_path, WEIGHT_KEY, chunk_rows=2)
    assert table.storage_format == "mxfp8"
    assert table.state is EngramTableState.STAGED
    assert table._host_weight.dtype == torch.float8_e4m3fn
    assert torch.equal(table._host_weight.view(torch.uint8), codes[3:6])
    assert torch.equal(table._host_scale, scales)
    assert table._scale_block_size == 2
    assert table._probe_ids.tolist() == [0, 2, 3, 5]


def test_loader_rejects_nan_scale_on_remote_row_and_releases_staging(tmp_path):
    _checkpoint(tmp_path, invalid_scale=True)
    table = _table()
    with pytest.raises(ValueError, match="invalid byte 255"):
        table.load_checkpoint(tmp_path, WEIGHT_KEY, chunk_rows=2)
    assert table.state is EngramTableState.POISONED
    assert table._host_weight is None and table._host_scale is None


def test_probe_staging_ignores_default_device(tmp_path):
    _checkpoint(tmp_path)
    table = _table()
    with torch.device("meta"):
        table.load_checkpoint(tmp_path, WEIGHT_KEY)
    assert table._host_weight.is_cpu and table._host_scale.is_cpu
    assert table._probe_ids.is_cpu and table._probe_codes.is_cpu and table._probe_scales.is_cpu


@pytest.mark.parametrize("storage", ["bf16", "fp16", "int8"])
def test_non_fp8_storage_is_rejected(storage):
    with pytest.raises(ValueError, match="only supports FP8 storage"):
        _table(storage=storage)


def test_fp8_offload_probe_and_inference_match_bf16(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY, chunk_rows=2)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    table.offload_weights()
    buffer = buffer_cls.instance
    assert table.state is EngramTableState.READY
    assert torch.equal(buffer.storage.view(torch.uint8), codes[3:6])
    assert table._host_weight is None and table._host_scale is None and table._probe_codes is None
    assert table._scale is buffer.scale
    assert buffer.scale.shape == (6, 2)
    assert buffer.waits == 1

    ids = torch.tensor([[5, 0, 3], [2, 3, 4]])
    golden = (
        (
            codes[ids.flatten()].view(torch.float8_e4m3fn).float().reshape(-1, 2, 2)
            * table._scale_to_float(buffer.scale.view(torch.uint8)[ids.flatten()]).unsqueeze(-1)
        )
        .reshape(2, 3, 4)
        .to(torch.bfloat16)
    )
    actual = table(ids)
    assert torch.equal(actual.view(torch.int16), golden.view(torch.int16))
    assert table(torch.empty((0, 3), dtype=torch.int64)).shape == (0, 3, 4)
    table(ids)
    assert caplog.text.count("FOR-ENGRAM inference fetch active:") == 1
    assert "FOR-ENGRAM FP8 runtime verified:" in caplog.text
    table.destroy()
    table.destroy()
    assert buffer.destroyed and table._scale is None


@pytest.mark.parametrize("corruption", ["write", "codes", "scales", "wait"])
def test_write_or_probe_failure_cleans_up(tmp_path, monkeypatch, corruption):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes, corruption=corruption)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    with pytest.raises(RuntimeError, match="OPP write failure|startup probe|fetch/wait failed"):
        table.offload_weights()
    assert buffer_cls.instance.destroyed
    assert table.state is EngramTableState.POISONED
    assert table._scale is None and table._host_scale is None and table._host_weight is None


def test_remote_probe_failure_poisons_local_table(tmp_path, monkeypatch):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    monkeypatch.setattr(table, "_sync_phase_status", lambda succeeded, phase: phase != "fp8-runtime-probe")
    with pytest.raises(RuntimeError, match="probe failed on another rank"):
        table.offload_weights()
    assert table.state is EngramTableState.POISONED
    assert buffer_cls.instance.destroyed and table._scale is None


def test_remote_write_failure_stops_before_runtime_probe(tmp_path, monkeypatch):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    phases = []

    def sync_status(succeeded, phase):
        phases.append((succeeded, phase))
        return phase != "elastic-buffer-write"

    monkeypatch.setattr(table, "_sync_phase_status", sync_status)
    probe = MagicMock()
    monkeypatch.setattr(table, "_verify_fp8_runtime", probe)

    with pytest.raises(RuntimeError, match="write failed on another rank"):
        table.offload_weights()

    assert phases[-1] == (True, "elastic-buffer-write")
    probe.assert_not_called()
    assert table.state is EngramTableState.POISONED
    assert buffer_cls.instance.destroyed and table._scale is None


def test_remote_format_failure_never_initializes_buffer(tmp_path, monkeypatch):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    monkeypatch.setattr(table, "_sync_phase_status", lambda succeeded, phase: False)
    with pytest.raises(RuntimeError, match="preparation failed on another rank"):
        table.offload_weights()
    assert buffer_cls.instance is None
    assert table.state is EngramTableState.POISONED and table._scale is None


def test_scale_capacity_failure_happens_before_elastic_init(tmp_path, monkeypatch):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    monkeypatch.setattr(table, "_target_device", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(mem_get_info=lambda device: (1, 1024)), raising=False)
    with pytest.raises(RuntimeError, match="HBM bytes; only 1 free"):
        table.offload_weights()
    assert buffer_cls.instance is None and table.state is EngramTableState.POISONED
    assert table._host_scale is None and table._host_weight is None


def test_dequant_all_finite_e4m3_and_e8m0_encodings():
    # Cross product: every finite FP8 bit pattern and every valid scale byte.
    code_bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    code_bits = code_bits[torch.isfinite(code_bits.view(torch.float8_e4m3fn).float())]
    codes = code_bits.repeat(255, 1).contiguous().view(torch.float8_e4m3fn)
    scales = torch.arange(255, dtype=torch.uint8).reshape(-1, 1)
    table = _table(rows=256, width=codes.shape[1])
    table._scale_block_size = codes.shape[1]
    actual = table._validate_fetch((codes, scales.view(torch.float8_e8m0fnu)), 255)
    expected = (codes.float() * torch.ldexp(torch.ones((255, 1)), scales.int() - 127)).to(torch.bfloat16)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("bad_output", ["single", "bf16", "float_scale", "scale_shape", "noncontiguous"])
def test_fp8_fetch_contract_is_checked(bad_output):
    table = _table()
    table._scale_block_size = 2
    codes = torch.ones((2, 4)).to(torch.float8_e4m3fn)
    scales = torch.full((2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    if bad_output == "single":
        output = codes
    elif bad_output == "bf16":
        output = (codes.to(torch.bfloat16), scales)
    elif bad_output == "float_scale":
        output = (codes, scales.float())
    elif bad_output == "scale_shape":
        output = (codes, scales[:, :1])
    else:
        output = (codes.t().contiguous().t(), scales)
    with pytest.raises(RuntimeError, match="MXFP8 fetch"):
        table._validate_fetch(output, 2)


@pytest.mark.parametrize("cause", ["EngramFetch Parameter fetched must be 2D", "transport failed"])
def test_fetch_failure_poisons_table_without_retry(cause):
    table = _table()
    calls = []

    def fetch(ids):
        calls.append(ids)
        raise RuntimeError(cause)

    table.engram_buffer = SimpleNamespace(engram_fetch=fetch)
    table._state = EngramTableState.READY
    with pytest.raises(RuntimeError, match="Only FP8 storage is supported") as error:
        table(torch.tensor([0]))
    assert str(error.value.__cause__) == cause
    assert len(calls) == 1 and table.state is EngramTableState.POISONED


@pytest.mark.parametrize(
    "schema,has_sf",
    [
        ("engram_fetch(Tensor context, int sf_table_addr) -> Tensor", True),
        ("engram_fetch(Tensor sf_table, Tensor fetched_sf) -> Tensor", False),
    ],
)
def test_preflight_rejects_old_python_abi(monkeypatch, schema, has_sf):
    from vllm_ascend.device import hardware_profile

    table = _table()
    monkeypatch.setattr(table, "_target_device", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        hardware_profile, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: True)
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "engram_fetch",
        SimpleNamespace(default=SimpleNamespace(_schema=schema)),
        raising=False,
    )
    buffer = SimpleNamespace(engram_write=(lambda storage, sf=None: None) if has_sf else (lambda storage: None))
    with pytest.raises(RuntimeError, match="tensor-scale ABI|engram_write"):
        table._check_fp8_runtime(buffer)


def test_runtime_diagnostics_survive_missing_metadata(caplog):
    caplog.set_level(logging.DEBUG)
    log_engram_runtime(1, 0, stage="test")
    assert "FOR-ENGRAM runtime:" in caplog.text
    assert '"opp_aclnn_version"' in caplog.text


def test_preflight_accepts_native_tensor_scale_abi(monkeypatch, caplog):
    from vllm_ascend.device import hardware_profile

    caplog.set_level(logging.DEBUG)
    table = _table()
    monkeypatch.setattr(table, "_target_device", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        hardware_profile, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: True)
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "engram_fetch",
        SimpleNamespace(default=SimpleNamespace(_schema="engram_fetch(Tensor sf_table, Tensor fetched_sf) -> Tensor")),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "engram_fetch_wait",
        SimpleNamespace(
            default=SimpleNamespace(
                _schema="cann_ops_transformer::engram_fetch_wait(Tensor context, Tensor fetched) -> Tensor"
            )
        ),
        raising=False,
    )
    table._check_fp8_runtime(SimpleNamespace(engram_write=lambda storage, sf=None: None))
    assert "opp_validation=pending-runtime-probe" in caplog.text


def test_preflight_rejects_missing_fetch_wait(monkeypatch):
    from vllm_ascend.device import hardware_profile

    table = _table()
    monkeypatch.setattr(table, "_target_device", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        hardware_profile, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: True)
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "engram_fetch",
        SimpleNamespace(default=SimpleNamespace(_schema="engram_fetch(Tensor sf_table, Tensor fetched_sf) -> Tensor")),
        raising=False,
    )
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "engram_fetch_wait", None, raising=False)
    with pytest.raises(RuntimeError, match="registered.*EngramFetchWait"):
        table._check_fp8_runtime(SimpleNamespace(engram_write=lambda storage, sf=None: None))


def test_preflight_rejects_cpu_runtime():
    table = _table()
    with pytest.raises(RuntimeError, match="requires an Ascend 950"):
        table._check_fp8_runtime(SimpleNamespace())


def test_fetch_handle_is_deferred_single_use_and_bound_to_its_table():
    table = _table()
    table._state, table._scale_block_size = EngramTableState.READY, 2
    calls = []

    def fetch(ids):
        def wait():
            calls.append("wait")
            return (
                torch.ones((ids.numel(), 4)).to(torch.float8_e4m3fn),
                torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            )

        return wait

    table.engram_buffer = SimpleNamespace(engram_fetch=fetch)
    handle = table.start_fetch(torch.tensor([[0, 1], [2, 3]]))
    assert calls == []
    with pytest.raises(RuntimeError, match="overlapping fetches"):
        table.start_fetch(torch.tensor([0]))
    with pytest.raises(ValueError, match="another table"):
        _table().finish_fetch(handle)
    result = table.finish_fetch(handle)
    assert calls == ["wait"] and result.shape == (2, 2, 4)
    torch.testing.assert_close(result, torch.ones((2, 2, 4), dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="already been completed"):
        table.finish_fetch(handle)


def test_failed_buffer_destroy_keeps_ownership_for_shutdown_retry():
    table = _table()
    buffer = SimpleNamespace(destroy=MagicMock(side_effect=[RuntimeError("busy"), None]))
    table._destroy_buffer_after_failure(buffer)
    assert table.engram_buffer is buffer and table.state is EngramTableState.POISONED
    table.destroy()
    assert buffer.destroy.call_count == 2 and table.state is EngramTableState.DESTROYED
