# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the OSCAR rotation contract (loader + math).

Rotation is REQUIRED for INT2 middle-tier quality (unrotated 2-bit K loses
retrieval outright). The contract under test has three legs that must stay
in sync:
  write  K/V stored rotated   (k @ R_k[l], v @ R_v[l])
  read   query rotated the same way (q @ R_k[l])
  output attention output inverse-rotated (out @ R_v[l].T)
"""

import pytest
import torch

from vllm_ascend.oscar.rotation import (
    OscarRotationLoader,
    load_oscar_rotations,
)


def _orthogonal(d, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


def _ckpt(tmp_path, name, layers, d=8, int_keys=False):
    state = {"layers": {(i if int_keys else str(i)): {"rotation": _orthogonal(d, 100 + i)} for i in range(layers)}}
    p = tmp_path / name
    torch.save(state, p)
    return str(p), state


class TestLoadOscarRotations:
    def test_load_shapes_and_indexing(self, tmp_path):
        p, state = _ckpt(tmp_path, "k.pt", layers=4, d=8)
        out = load_oscar_rotations(
            p, layer_num=4, start_layer=0, head_dim=8, device=torch.device("cpu"), dtype=torch.float32
        )
        assert out.shape == (4, 8, 8)
        assert torch.equal(out[2], state["layers"]["2"]["rotation"])

    def test_start_layer_offset(self, tmp_path):
        # checkpoint keyed by GLOBAL layer id; backend indexes locally
        p, state = _ckpt(tmp_path, "k.pt", layers=6, d=8, int_keys=True)
        out = load_oscar_rotations(
            p, layer_num=2, start_layer=4, head_dim=8, device=torch.device("cpu"), dtype=torch.float32
        )
        assert torch.equal(out[0], state["layers"][4]["rotation"])
        assert torch.equal(out[1], state["layers"][5]["rotation"])

    def test_str_and_int_keys_both_accepted(self, tmp_path):
        for ik in (True, False):
            p, _ = _ckpt(tmp_path, f"k{ik}.pt", layers=2, d=8, int_keys=ik)
            out = load_oscar_rotations(p, layer_num=2, start_layer=0, head_dim=8, device=torch.device("cpu"))
            assert out.shape == (2, 8, 8)

    def test_default_dtype_is_engine_bf16(self, tmp_path):
        p, _ = _ckpt(tmp_path, "k.pt", layers=2, d=8)
        out = load_oscar_rotations(p, layer_num=2, start_layer=0, head_dim=8, device=torch.device("cpu"))
        assert out.dtype == torch.bfloat16

    def test_missing_layer_fail_closed(self, tmp_path):
        p, _ = _ckpt(tmp_path, "k.pt", layers=3, d=8)
        with pytest.raises(ValueError, match="missing layer 3"):
            load_oscar_rotations(p, layer_num=4, start_layer=0, head_dim=8, device=torch.device("cpu"))

    def test_wrong_head_dim_fail_closed(self, tmp_path):
        p, _ = _ckpt(tmp_path, "k.pt", layers=2, d=8)
        with pytest.raises(ValueError, match="expected \\(16, 16\\)"):
            load_oscar_rotations(p, layer_num=2, start_layer=0, head_dim=16, device=torch.device("cpu"))

    def test_missing_layers_key_fail_closed(self, tmp_path):
        p = tmp_path / "bad.pt"
        torch.save({"unexpected": 1}, p)
        with pytest.raises(ValueError, match="missing 'layers'"):
            load_oscar_rotations(str(p), layer_num=2, start_layer=0, head_dim=8, device=torch.device("cpu"))


class TestRotationLoaderContract:
    def _loader(self, tmp_path, layers=3, d=8):
        kp, _ = _ckpt(tmp_path, "k.pt", layers, d)
        vp, _ = _ckpt(tmp_path, "v.pt", layers, d)
        return OscarRotationLoader(kp, vp, num_layers=layers, head_dim=d, device="cpu", dtype=torch.float32)

    def test_write_read_output_roundtrip(self, tmp_path):
        # the three-leg contract: rotate on write, same rotation on read,
        # inverse on output -> identity when applied in sequence
        L = self._loader(tmp_path)
        x = torch.randn(5, 8)  # one token-head row
        k_rot = L.rotate_k(x, 1)
        q_rot = L.rotate_q(x, 1)
        assert torch.allclose(k_rot, q_rot)  # same R_k both sides
        o = L.inverse_rotate_output(L.rotate_v(x, 1), 1)
        assert torch.allclose(o, x, atol=1e-5)  # R_v then R_v.T == I

    def test_orthogonality_preserves_norms(self, tmp_path):
        L = self._loader(tmp_path)
        x = torch.randn(4, 8)
        for li in range(3):
            assert torch.allclose(L.rotate_k(x, li).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
            assert torch.allclose(L.rotate_v(x, li).norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    def test_per_layer_matrices_differ(self, tmp_path):
        L = self._loader(tmp_path)
        assert not torch.allclose(L.get_k_rotation(0), L.get_k_rotation(1))

    def test_inverse_is_transpose(self, tmp_path):
        L = self._loader(tmp_path)
        assert torch.allclose(L.get_v_inverse_rotation(2), L.get_v_rotation(2).T)

    def test_shape_mismatch_fail_closed(self, tmp_path):
        kp, _ = _ckpt(tmp_path, "k.pt", layers=3, d=8)
        vp, _ = _ckpt(tmp_path, "v.pt", layers=3, d=16)  # head_dim differs
        with pytest.raises(AssertionError):
            OscarRotationLoader(kp, vp, num_layers=3, head_dim=8, device="cpu", dtype=torch.float32)

    def test_from_env_absent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VLLM_OSCAR_K_ROTATION_PATH", raising=False)
        monkeypatch.delenv("VLLM_OSCAR_V_ROTATION_PATH", raising=False)
        assert OscarRotationLoader.from_env(num_layers=3, head_dim=8, device="cpu") is None


class TestFromEnvDeviceSignature:
    def _env(self, monkeypatch, tmp_path):
        kp, _ = _ckpt(tmp_path, "k.t", 2, 8)
        vp, _ = _ckpt(tmp_path, "v.t", 2, 8)
        monkeypatch.setenv("VLLM_OSCAR_K_ROTATION_PATH", kp)
        monkeypatch.setenv("VLLM_OSCAR_V_ROTATION_PATH", vp)
        OscarRotationLoader._instance = None

    def test_equivalent_device_inputs_share_the_singleton(self, tmp_path, monkeypatch):
        # "cpu" (str) and torch.device("cpu") are the same device; the
        # singleton guard must not treat them as conflicting requests
        self._env(monkeypatch, tmp_path)
        try:
            first = OscarRotationLoader.from_env(2, 8, "cpu", torch.float32)
            second = OscarRotationLoader.from_env(2, 8, torch.device("cpu"), torch.float32)
            assert first is not None and first is second
        finally:
            OscarRotationLoader._instance = None

    def test_conflicting_device_still_fails_closed(self, tmp_path, monkeypatch):
        self._env(monkeypatch, tmp_path)
        try:
            assert OscarRotationLoader.from_env(2, 8, "cpu", torch.float32) is not None
            with pytest.raises(RuntimeError, match="refusing conflicting"):
                OscarRotationLoader.from_env(2, 8, torch.device("npu"), torch.float32)
        finally:
            OscarRotationLoader._instance = None
