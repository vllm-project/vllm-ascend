# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

import vllm_ascend.models.deepseek_v41.engram.elastic as offload
from vllm_ascend.models.deepseek_v41.engram import diagnostics
from vllm_ascend.models.deepseek_v41.engram.diagnostics import log_engram_runtime
from vllm_ascend.models.deepseek_v41.engram.elastic import ElasticEngramEmbedding, EngramTableState

WEIGHT_KEY = "model.layers.1.engram.embed.weight"
SCALE_KEY = "model.layers.1.engram.embed.scale"
FETCH_SCHEMA = (
    "cann_ops_transformer::engram_fetch(Tensor context, Tensor indices, int hidden_size, "
    "int num_entries, int dtype, Tensor sf_table, Tensor fetched_sf) -> Tensor"
)
WAIT_SCHEMA = "cann_ops_transformer::engram_fetch_wait(Tensor context, Tensor fetched) -> Tensor"


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
            alignment = 2 * 1024 * 1024
            return ((rows * width + alignment - 1) // alignment) * alignment

        def __init__(self, group, *, num_cpu_bytes=0, with_grad=False, explicitly_destroy=False, netlayer="auto"):
            type(self).instance = self
            self.arguments = dict(
                group=group,
                num_cpu_bytes=num_cpu_bytes,
                with_grad=with_grad,
                explicitly_destroy=explicitly_destroy,
                netlayer=netlayer,
            )
            self.waits = 0
            self.destroyed = False
            self.scale = None
            self.storage = None
            self.zero_copy = None

        def engram_write(self, storage, sf=None, *, zero_copy=None):
            self.zero_copy = zero_copy
            assert zero_copy is False
            if corruption == "write":
                raise RuntimeError("OPP write failure")
            self.storage = storage.clone()
            self.scale = sf

        def engram_fetch(self, ids):
            if corruption == "submit":
                raise RuntimeError("Check executor != nullptr failed")
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


def _preflight(monkeypatch, *, fetch_schema=FETCH_SCHEMA, wait_schema=WAIT_SCHEMA):
    from vllm_ascend.device import hardware_profile

    table = _table()
    monkeypatch.setattr(table, "_target_device", lambda: SimpleNamespace(type="npu"))
    monkeypatch.setattr(
        hardware_profile, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda capability: True)
    )
    buffer = _install_buffer(monkeypatch, torch.empty((0, 4), dtype=torch.uint8))
    for name, schema in (("engram_fetch", fetch_schema), ("engram_fetch_wait", wait_schema)):
        op = None if schema is None else SimpleNamespace(default=SimpleNamespace(_schema=torch._C.parse_schema(schema)))
        monkeypatch.setattr(torch.ops.cann_ops_transformer, name, op, raising=False)
    return table, buffer


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
    assert buffer.arguments == dict(
        group=table.group,
        num_cpu_bytes=2 * 1024 * 1024,
        with_grad=False,
        explicitly_destroy=True,
        netlayer="auto",
    )
    assert buffer.zero_copy is False
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


@pytest.mark.parametrize("corruption", ["write", "codes", "scales", "submit", "wait"])
def test_write_or_probe_failure_cleans_up(tmp_path, monkeypatch, corruption):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes, corruption=corruption)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    with pytest.raises(
        RuntimeError, match="OPP write failure|startup probe|fetch submission failed|fetch completion failed"
    ):
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


@pytest.mark.parametrize("hint", [0, 1, 2 * 1024 * 1024 + 1, True, float(2 * 1024 * 1024)])
def test_invalid_storage_size_hint_is_rejected_before_buffer_creation(tmp_path, monkeypatch, hint):
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes)
    monkeypatch.setattr(buffer_cls, "get_engram_storage_size_hint", staticmethod(lambda *args: hint))
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    with pytest.raises(RuntimeError, match="storage size hint"):
        table.offload_weights()
    assert buffer_cls.instance is None
    assert table.state is EngramTableState.POISONED
    assert table._host_weight is None and table._scale is None


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
    with pytest.raises(RuntimeError, match="stage=fetch-submit") as error:
        table(torch.tensor([0]))
    assert str(error.value.__cause__) == cause
    assert "Install matching" not in str(error.value)
    assert len(calls) == 1 and table.state is EngramTableState.POISONED


@pytest.mark.parametrize(
    "schema,has_sf",
    [
        ("engram_fetch(Tensor context, int sf_table_addr) -> Tensor", True),
        ("engram_fetch(Tensor sf_table, Tensor fetched_sf) -> Tensor", False),
    ],
)
def test_preflight_rejects_old_python_abi(monkeypatch, schema, has_sf):
    table, buffer = _preflight(monkeypatch, fetch_schema=schema)
    if not has_sf:
        monkeypatch.setattr(buffer, "engram_write", lambda self, storage: None)
    with pytest.raises(RuntimeError, match="tensor-scale ABI|engram_write"):
        table._check_fp8_runtime(buffer)


def test_runtime_diagnostics_survive_missing_metadata(caplog):
    caplog.set_level(logging.DEBUG)
    log_engram_runtime(1, 0, stage="test")
    assert "FOR-ENGRAM runtime:" in caplog.text
    assert '"opp_aclnn_version"' in caplog.text


@pytest.mark.parametrize(
    "corruption,stage,cause",
    [("submit", "fetch-submit", "Check executor != nullptr failed"), ("wait", "fetch-wait", "OPP wait failure")],
)
def test_native_failure_logs_once_at_info_and_preserves_cause(tmp_path, monkeypatch, caplog, corruption, stage, cause):
    caplog.set_level(logging.INFO)
    codes, _ = _checkpoint(tmp_path)
    buffer_cls = _install_buffer(monkeypatch, codes, corruption=corruption)
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)

    with pytest.raises(RuntimeError, match=f"stage={stage}") as error:
        table.offload_weights()

    assert str(error.value.__cause__) == cause
    records = [record for record in caplog.records if record.msg.startswith("FOR-ENGRAM runtime:")]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert records[0].args[2] == stage
    details = json.loads(records[0].args[-1])
    assert details["tensors"]["indices"]["shape"] == [4]
    assert details["tensors"]["indices"]["dtype"] == "torch.int32"
    assert details["tensors"]["sf_table"]["shape"] == [6, 2]
    assert details["parameters"]["hidden_size"] == 4
    assert details["parameters"]["num_entries_per_rank"] == 3
    assert details["parameters"]["expected_fetch_shape"] == [4, 4]
    assert "not distinguished" in details["native_call"]["stage"]
    assert "not determined" in details["native_call"]["function_providers"]
    assert table.state is EngramTableState.POISONED and buffer_cls.instance.destroyed


def test_byte_validation_failure_logs_actual_outputs_at_info(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    codes, _ = _checkpoint(tmp_path)
    _install_buffer(monkeypatch, codes, corruption="codes")
    table = _table()
    table.load_checkpoint(tmp_path, WEIGHT_KEY)
    monkeypatch.setattr(table, "_check_fp8_runtime", lambda cls: None)
    with pytest.raises(RuntimeError, match="fetched E4M3 bytes differ"):
        table.offload_weights()
    records = [record for record in caplog.records if record.msg.startswith("FOR-ENGRAM runtime:")]
    assert len(records) == 1 and records[0].args[2] == "fp8-runtime-probe"
    details = json.loads(records[0].args[-1])
    assert details["tensors"]["fetched"]["dtype"] == "torch.float8_e4m3fn"
    assert details["tensors"]["fetched_sf"]["dtype"] == "torch.float8_e8m0fnu"


def test_failure_diagnostics_read_descriptors_without_tensor_values(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    tensor = torch.empty((4, 3), device="meta").t()
    forbidden = MagicMock(side_effect=AssertionError("diagnostics must not read or copy tensor data"))
    for name in ("item", "cpu", "numpy", "tolist", "to"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    log_engram_runtime(1, 0, stage="fetch-submit", failure=True, tensors={"indices": tensor})
    forbidden.assert_not_called()
    record = next(record for record in caplog.records if record.msg.startswith("FOR-ENGRAM runtime:"))
    details = json.loads(record.args[-1])
    assert details["tensors"]["indices"]["shape"] == [3, 4]
    assert details["tensors"]["indices"]["stride"] == [1, 3]
    assert details["tensors"]["indices"]["contiguous"] is False


def test_device_metadata_failure_preserves_other_diagnostics(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_available=lambda: True, current_device=MagicMock(side_effect=RuntimeError("device lost"))),
        raising=False,
    )
    monkeypatch.setattr(diagnostics, "_loaded_libraries", lambda: ["/test/libcust_opapi.so"])
    log_engram_runtime(1, 0, stage="fetch-submit", buffer=object(), failure=True, tensors={"indices": torch.empty(2)})
    record = next(record for record in caplog.records if record.msg.startswith("FOR-ENGRAM runtime:"))
    details = json.loads(record.args[-1])
    assert details["device_metadata_error"] == "device lost"
    assert details["loaded_libraries"] == ["/test/libcust_opapi.so"]
    assert details["tensors"]["indices"]["shape"] == [2]
    assert "write_signature_error" in details


def test_debug_diagnostics_skip_collection_at_info(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    collect = MagicMock(side_effect=AssertionError("DEBUG collection should be skipped"))
    monkeypatch.setattr(diagnostics, "_loaded_libraries", collect)
    log_engram_runtime(1, 0, stage="post-probe")
    collect.assert_not_called()


def test_broken_diagnostics_do_not_replace_native_error(caplog):
    caplog.set_level(logging.INFO)

    class Buffer:
        @property
        def _engram_context_tensor(self):
            raise RuntimeError("context metadata unavailable")

        def engram_fetch(self, ids):
            raise RuntimeError("original native failure")

    table = _table()
    table.engram_buffer = Buffer()
    table._state = EngramTableState.READY
    with pytest.raises(RuntimeError, match="stage=fetch-submit") as error:
        table(torch.tensor([0]))
    assert str(error.value.__cause__) == "original native failure"
    assert "context metadata unavailable" in caplog.text
    assert table.state is EngramTableState.POISONED


def test_inference_validation_failure_logs_actual_dtype(caplog):
    caplog.set_level(logging.INFO)
    table = _table()
    table._scale_block_size = 2
    table._state = EngramTableState.READY
    table.engram_buffer = SimpleNamespace(
        engram_fetch=lambda ids: (
            lambda: (
                torch.ones((ids.numel(), 4), dtype=torch.bfloat16),
                torch.full((ids.numel(), 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            )
        )
    )
    with pytest.raises(RuntimeError, match="fetch expects contiguous E4M3"):
        table(torch.tensor([0]))
    record = next(record for record in caplog.records if record.msg.startswith("FOR-ENGRAM runtime:"))
    assert record.args[2] == "fetch-validation"
    details = json.loads(record.args[-1])
    assert details["tensors"]["fetched"]["dtype"] == "torch.bfloat16"
    assert table.state is EngramTableState.POISONED


def test_loaded_library_paths_preserve_spaces_and_deleted_marker(monkeypatch):
    monkeypatch.setattr(
        diagnostics.Path,
        "read_text",
        lambda _: (
            "1000-2000 r-xp 00000000 08:01 123 /opt/vendor with spaces/libcust_opapi.so (deleted)\n"
            "2000-3000 r-xp 00000000 08:01 124 /opt/libnpu_elastic_buffer.so\n"
            "3000-4000 r-xp 00000000 08:01 125 /opt/other.so\n"
        ),
    )
    assert diagnostics._loaded_libraries() == [
        "/opt/libnpu_elastic_buffer.so",
        "/opt/vendor with spaces/libcust_opapi.so (deleted)",
    ]


def test_preflight_accepts_native_tensor_scale_abi(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    table, buffer = _preflight(monkeypatch)
    table._check_fp8_runtime(buffer)
    assert "opp_validation=pending-runtime-probe" in caplog.text
    assert "with_grad=false zero_copy=false netlayer=auto" in caplog.text


def test_preflight_rejects_missing_fetch_wait(monkeypatch):
    table, buffer = _preflight(monkeypatch, wait_schema=None)
    with pytest.raises(RuntimeError, match="registered.*engram_fetch_wait"):
        table._check_fp8_runtime(buffer)


def test_preflight_rejects_constructor_without_netlayer(monkeypatch):
    table, buffer = _preflight(monkeypatch)

    class LegacyBuffer(buffer):
        def __init__(self, group, *, num_cpu_bytes=0, with_grad=False, explicitly_destroy=False):
            pass

    with pytest.raises(RuntimeError, match="constructor.*netlayer"):
        table._check_fp8_runtime(LegacyBuffer)


def test_preflight_rejects_write_without_zero_copy(monkeypatch):
    table, buffer = _preflight(monkeypatch)
    monkeypatch.setattr(buffer, "engram_write", lambda self, storage, sf=None: None)
    with pytest.raises(RuntimeError, match="engram_write.*zero_copy=False"):
        table._check_fp8_runtime(buffer)


@pytest.mark.parametrize(
    "fetch_schema,wait_schema",
    [
        (FETCH_SCHEMA.replace("Tensor sf_table, Tensor fetched_sf", "Tensor fetched_sf, Tensor sf_table"), WAIT_SCHEMA),
        (FETCH_SCHEMA.replace("int hidden_size", "float hidden_size"), WAIT_SCHEMA),
        (FETCH_SCHEMA.replace(" -> Tensor", " -> (Tensor, Tensor)"), WAIT_SCHEMA),
        (FETCH_SCHEMA, WAIT_SCHEMA.replace("Tensor fetched)", "Tensor fetched, Tensor fetched_sf)")),
        (FETCH_SCHEMA, WAIT_SCHEMA.replace(" -> Tensor", " -> (Tensor, Tensor)")),
    ],
)
def test_preflight_checks_complete_fetch_and_wait_signatures(monkeypatch, fetch_schema, wait_schema):
    table, buffer = _preflight(monkeypatch, fetch_schema=fetch_schema, wait_schema=wait_schema)
    with pytest.raises(RuntimeError, match="inference tensor-scale ABI"):
        table._check_fp8_runtime(buffer)


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
