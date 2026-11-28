# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


def load_source(name, relative):
    path = Path(__file__).resolve().parents[3] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifact = load_source("ssd_artifact_under_test", "vllm_ascend/model_loader/engram_ssd_artifact.py")
manifest_module = load_source("ssd_manifest_under_test", "vllm_ascend/models/deepseek_v41/engram/ssd/manifest.py")
prepare_artifact = artifact.prepare_artifact
SSDManifest, SSDManifestError = manifest_module.SSDManifest, manifest_module.SSDManifestError


def make_checkpoint(root, dtype):
    root.mkdir()
    codes = np.arange(256, dtype=np.uint8)[None, :].repeat(3, axis=0)
    scales = np.array([[0, 1, 126, 127, 128, 129, 254, 255]] * 3, dtype=np.uint8)
    header = {
        "layers.1.engram.embed.weight": dict(dtype=dtype, shape=[3, 256], data_offsets=[0, 768]),
        "layers.1.engram.embed.scale": dict(dtype="F8_E8M0", shape=[3, 8], data_offsets=[768, 792]),
    }
    data = json.dumps(header).encode()
    (root / "table.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + codes.tobytes() + scales.tobytes())
    return codes, scales


@pytest.mark.parametrize("dtype", ["I8", "F8_E4M3", "F8_E4M3FN"])
def test_pack_preserves_codes_decodes_scales_and_reuses(tmp_path, dtype):
    model = tmp_path / "model"
    codes, scales = make_checkpoint(model, dtype)
    result = prepare_artifact(model, tmp_path / "ssd", {1: 3})
    table = Path(result["table_paths"][0])
    row_bytes = 288 if dtype == "I8" else 264
    packed = np.fromfile(table, dtype=np.uint8).reshape(-1, row_bytes)
    np.testing.assert_array_equal(packed[:3, :256], codes)
    if dtype == "I8":
        bits = packed[:3, 256:].copy().view("<u4")
        expected = scales.astype("<u4") << 23
        expected[scales == 0] = 0x00400000
        expected[scales == 255] = 0x7FC00000
        np.testing.assert_array_equal(bits, expected)
    else:
        np.testing.assert_array_equal(packed[:3, 256:], scales)
    assert not packed[3:].any()
    before = table.stat().st_mtime_ns
    assert prepare_artifact(model, tmp_path / "ssd", {1: 3}) == result
    assert table.stat().st_mtime_ns == before
    layout = {"version": result["artifact_version"], "layers": {1: {"rows": 3, "codes_dtype": dtype, "codes_dim": 256}}}
    manifest = SSDManifest.load(result["manifest"], layout, result["table_paths"])
    assert manifest.specs[0].codes_dtype == dtype
    assert manifest.specs[0].scale_dtype == ("F32" if dtype == "I8" else "F8_E8M0")
    assert manifest.specs[0].row_bytes == row_bytes
    layout["layers"][1]["codes_dtype"] = "F8_E4M3" if dtype == "I8" else "I8"
    with pytest.raises(SSDManifestError):
        SSDManifest.load(result["manifest"], layout, result["table_paths"])


def test_fp8_and_int8_artifacts_have_distinct_identity(tmp_path):
    make_checkpoint(tmp_path / "int8", "I8")
    make_checkpoint(tmp_path / "fp8", "F8_E4M3")
    a = prepare_artifact(tmp_path / "int8", tmp_path / "ssd", {1: 3})
    b = prepare_artifact(tmp_path / "fp8", tmp_path / "ssd", {1: 3})
    assert a["artifact_version"] != b["artifact_version"]
    assert a["table_paths"] != b["table_paths"]


def test_bfloat16_source_rejected_before_packing(tmp_path):
    make_checkpoint(tmp_path / "model", "BF16")
    with pytest.raises(ValueError, match="Unsupported Engram tensor"):
        prepare_artifact(tmp_path / "model", tmp_path / "ssd", {1: 3})


def test_manifest_rejects_invalid_scale_and_accepts_fp8_alias(tmp_path):
    make_checkpoint(tmp_path / "model", "F8_E4M3FN")
    result = prepare_artifact(tmp_path / "model", tmp_path / "ssd", {1: 3})
    layout = {"version": result["artifact_version"], "layers": {1: {"rows": 3, "codes_dtype": "F8_E4M3"}}}
    manifest = SSDManifest.load(result["manifest"], layout, result["table_paths"])
    assert manifest.specs[0].codes_dtype == "F8_E4M3FN"
    path = Path(result["manifest"])
    raw = json.loads(path.read_text())
    raw["sources"][0]["scale"]["dtype"] = "BF16"
    path.write_text(json.dumps(raw))
    with pytest.raises(SSDManifestError, match="scale dtype"):
        SSDManifest.load(path, layout, result["table_paths"])


def test_int8_keeps_existing_artifact_identity(tmp_path):
    make_checkpoint(tmp_path / "model", "I8")
    sources = artifact._sources(tmp_path / "model", {1: 3})
    identity = {"format": "engram-int8-fp32-aos288-v1", "sources": sources}
    expected_version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = prepare_artifact(tmp_path / "model", tmp_path / "ssd", {1: 3})
    assert result["artifact_version"] == expected_version


def test_legacy_expanded_fp8_manifest_rejected(tmp_path):
    make_checkpoint(tmp_path / "model", "F8_E4M3")
    result = prepare_artifact(tmp_path / "model", tmp_path / "ssd", {1: 3})
    path = Path(result["manifest"])
    raw = json.loads(path.read_text())
    raw["format"] = "engram-fp8-fp32-aos288-v1"
    path.write_text(json.dumps(raw))
    layout = {"version": result["artifact_version"], "layers": {1: {"rows": 3, "codes_dtype": "F8_E4M3"}}}
    with pytest.raises(SSDManifestError, match="format"):
        SSDManifest.load(path, layout, result["table_paths"])
