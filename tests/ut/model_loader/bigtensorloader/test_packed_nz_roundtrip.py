# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip test for packed low-bit NZ snapshot save/restore."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.config import LoadConfig

from vllm_ascend.model_loader.bigtensorloader.bigtensorloader import BigTensorLoader

ACL_FORMAT_FRACTAL_NZ = 29


def _make_packed_int32_nz(shape_int8: tuple) -> torch.Tensor:
    """Build packed int8 NZ weight viewed as int32, as w4a8 does."""
    t = torch.randint(-127, 127, shape_int8, dtype=torch.int8)
    if t.dim() == 2:
        t = t.transpose(1, 0).contiguous().transpose(1, 0).contiguous()
    else:
        t = t.contiguous()
    t = t.to("npu")
    t = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
    return t.view(torch.int32).contiguous()


class _PackedModel(torch.nn.Module):
    def __init__(self, packed: torch.Tensor):
        super().__init__()
        self.register_parameter("w_packed", torch.nn.Parameter(packed, requires_grad=False))
        self.register_parameter(
            "w_plain", torch.nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        )


class TestPackedNzSnapshotRoundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if not torch.npu.is_available():
                raise unittest.SkipTest("NPU not available")
            torch.npu.config.allow_internal_format = True
            torch.npu.set_device(0)
            # is_available() can be a false positive on no-device CI
            # builds; probe with a real device allocation
            torch.zeros(1, device="npu")
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"NPU not available: {e}") from e

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="packed_nz_ut_")
        self._old_ckpt = os.environ.get("VLLM_ASCEND_CHECKPOINT_PATH")
        os.environ["VLLM_ASCEND_CHECKPOINT_PATH"] = self._tmpdir
        self._old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
        self.loader = BigTensorLoader(LoadConfig(load_format="bigtensorloader"))

    def tearDown(self):
        if self._old_ckpt is None:
            os.environ.pop("VLLM_ASCEND_CHECKPOINT_PATH", None)
        else:
            os.environ["VLLM_ASCEND_CHECKPOINT_PATH"] = self._old_ckpt
        if self._old_verify is None:
            os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
        else:
            os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = self._old_verify
        for fname in os.listdir(self._tmpdir):
            os.remove(os.path.join(self._tmpdir, fname))
        os.rmdir(self._tmpdir)

    def _roundtrip_once(self, shape_int8: tuple, bulk: bool) -> None:
        packed = _make_packed_int32_nz(shape_int8)
        self.assertEqual(int(torch_npu.get_npu_format(packed)), ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(packed.dtype, torch.int32)

        model = _PackedModel(packed).to("npu")
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest["tensors"]["w_packed"]
        self.assertEqual(entry["save_dtype"], "int8")
        self.assertEqual(entry["dtype"], "int32")
        self.assertEqual(entry["format"], ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(entry["nbytes"], packed.numel() * packed.element_size())
        # logical int32 shape: last dim divided by 4
        self.assertEqual(entry["shape"][-1], shape_int8[-1] // 4)
        self.assertNotIn("save_dtype", manifest["tensors"]["w_plain"])

        restored = _PackedModel(torch.empty(0, dtype=torch.int32)).to("npu")
        old_bulk = os.environ.get("VLLM_ASCEND_BIGTENSOR_BULK_H2D")
        old_chunk = os.environ.get("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB")
        os.environ["VLLM_ASCEND_BIGTENSOR_BULK_H2D"] = "1" if bulk else "0"
        if bulk:
            os.environ["VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB"] = "1"
        try:
            self.loader.restore_snapshot(restored, torch.device("npu"))
        finally:
            for key, old in (
                ("VLLM_ASCEND_BIGTENSOR_BULK_H2D", old_bulk),
                ("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB", old_chunk),
            ):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

        got = restored.w_packed.data
        self.assertEqual(got.dtype, torch.int32)
        self.assertEqual(tuple(got.shape), tuple(packed.shape))
        self.assertEqual(int(torch_npu.get_npu_format(got)), ACL_FORMAT_FRACTAL_NZ)
        self.assertTrue(torch.equal(got.view(torch.int8), packed.view(torch.int8)))
        self.assertTrue(torch.equal(restored.w_plain.data, model.w_plain.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_packed_int32_nz_roundtrip_per_tensor(self, _r, _w):
        for shape_int8 in ((64, 256), (48, 132), (16, 68), (4, 32, 128)):
            with self.subTest(shape_int8=shape_int8):
                self._roundtrip_once(shape_int8, bulk=False)

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_packed_int32_nz_roundtrip_bulk(self, _r, _w):
        for shape_int8 in ((64, 256), (4, 32, 128)):
            with self.subTest(shape_int8=shape_int8):
                self._roundtrip_once(shape_int8, bulk=True)

    def test_is_packed_nz_detection(self):
        packed = _make_packed_int32_nz((16, 64))
        self.assertTrue(BigTensorLoader._is_packed_nz(packed))
        plain = torch.randint(-127, 127, (16, 64), dtype=torch.int8, device="npu")
        plain = torch_npu.npu_format_cast(plain, ACL_FORMAT_FRACTAL_NZ)
        self.assertFalse(BigTensorLoader._is_packed_nz(plain))
        nd_tensor = torch.zeros(4, 8, dtype=torch.int32, device="npu")
        self.assertFalse(BigTensorLoader._is_packed_nz(nd_tensor))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_empty_placeholder_attr_not_saved(self, _r, _w):
        """Empty placeholders must not be saved (would crash frombuffer)."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.kv_cache = torch.empty(0, dtype=torch.float16, device="npu")

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertNotIn("kv_cache", manifest["tensors"])
        self.assertIn("w", manifest["tensors"])
        self.assertGreater(manifest["tensors"]["w"]["nbytes"], 0)

        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertEqual(restored.kv_cache.numel(), 0)
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_restore_legacy_zero_nbytes_entry(self, _r, _w):
        """Legacy snapshots with nbytes=0 entries must not crash on restore."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        w_entry = manifest["tensors"]["w"]
        manifest["tensors"]["legacy_empty_attr"] = {
            "dtype": "float16",
            "shape": [0],
            "format": 0,
            "offset": w_entry["offset"] + w_entry["nbytes"] + 1024,
            "nbytes": 0,
            "kind": "attr",
            "device": "npu",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        restored.legacy_empty_attr = torch.empty(0, dtype=torch.float16, device="npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertEqual(restored.legacy_empty_attr.numel(), 0)
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_corrupt_blob_sha256_fails_loud(self, _r, _w):
        """sha256 verification must catch byte-flipped blobs."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(8, 8), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        blob_path = os.path.join(self._tmpdir, "0.snapshot")
        with open(blob_path, "r+b") as f:
            f.seek(64)
            (b,) = f.read(1)
            f.seek(64)
            f.write(bytes([b ^ 0xFF]))

        restored = M().to("npu")
        old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = "sha256"
        try:
            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                self.loader.restore_snapshot(restored, torch.device("npu"))
        finally:
            if old_verify is None:
                os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
            else:
                os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = old_verify

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_corrupt_blob_size_mode_passes_silently(self, _r, _w):
        """size mode does not catch byte-flips within bounds (trade-off)."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(8, 8), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        blob_path = os.path.join(self._tmpdir, "0.snapshot")
        with open(blob_path, "r+b") as f:
            f.seek(64)
            (b,) = f.read(1)
            f.seek(64)
            f.write(bytes([b ^ 0xFF]))

        restored = M().to("npu")
        old_verify = os.environ.get("VLLM_ASCEND_BIGTENSOR_VERIFY")
        os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = "size"
        try:
            self.loader.restore_snapshot(restored, torch.device("npu"))
            self.assertFalse(torch.equal(restored.w.data, model.w.data))
        finally:
            if old_verify is None:
                os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
            else:
                os.environ["VLLM_ASCEND_BIGTENSOR_VERIFY"] = old_verify

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_truncated_blob_bounds_check_fails_loud(self, _r, _w):
        """Truncated blob must be rejected by bounds check."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(64, 64), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        blob_path = os.path.join(self._tmpdir, "0.snapshot")
        size = os.path.getsize(blob_path)
        with open(blob_path, "r+b") as f:
            f.truncate(size // 2)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "truncated or corrupt"):
            self.loader.restore_snapshot(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_shape_drift_same_numel_installs_silently(self, _r, _w):
        """Same-numel shape morph installs silently; fails at forward."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["shape"] = [2, 8]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertEqual(tuple(restored.w.shape), (2, 8))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_dtype_drift_same_bytes_installs_silently(self, _r, _w):
        """Same-bytes dtype morph installs silently; fails at forward."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w", torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float16), requires_grad=False)
                )

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["dtype"] = "float32"
        manifest["tensors"]["w"]["shape"] = [8]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertEqual(restored.w.dtype, torch.float32)
        self.assertEqual(tuple(restored.w.shape), (8,))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_stale_snapshot_entry_fails_loud(self, _r, _w):
        """Stale snapshot entry must be caught by reverse coverage check."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        # Inject stale param entry
        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["ghost_layer.weight"] = {
            "dtype": "float32",
            "shape": [2, 2],
            "format": 0,
            "offset": 0,
            "nbytes": 16,
            "kind": "param",
            "device": "npu",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "snapshot params not found"):
            self.loader.restore_snapshot(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_manifest_nbytes_inconsistency_fails_loud(self, _r, _w):
        """Descriptor self-consistency must catch nbytes/shape mismatch."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        entry = manifest["tensors"]["w"]
        entry["shape"] = [3, 8]  # 3*8*4=96 != nbytes=64
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        with self.assertRaisesRegex(RuntimeError, "descriptor inconsistency"):
            self.loader.restore_snapshot(restored, torch.device("npu"))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_ghost_entry_on_existing_module_registers_silently(self, _r, _w):
        """Ghost entry on existing module is registered (harmless)."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["ghost_w"] = {
            "dtype": "float32",
            "shape": [2, 2],
            "format": 0,
            "offset": 0,
            "nbytes": 16,
            "kind": "param",
            "device": "npu",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertTrue(hasattr(restored, "ghost_w"))
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_process_added_param_registers_and_fills(self, _r, _w):
        """Process-added param is dynamically registered and filled."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["scale_added"] = dict(manifest["tensors"]["w"])
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")  # fresh model: no scale_added attribute
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertTrue(hasattr(restored, "scale_added"))
        self.assertEqual(tuple(restored.scale_added.shape), (4, 4))
        self.assertTrue(torch.equal(restored.scale_added, restored.w))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_squeeze_transform_restores(self, _r, _w):
        """Squeeze transform (N,1)->(N,) restores correctly."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w", torch.nn.Parameter(torch.randn(64, 1, dtype=torch.bfloat16), requires_grad=False)
                )

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        # Simulate the process squeeze in the manifest (nbytes equal:
        # 64*1*2 == 64*2)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["tensors"]["w"]["shape"] = [64]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertEqual(tuple(restored.w.shape), (64,))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_stale_tmp_residue_cleaned_on_write(self, _r, _w):
        """Stale .tmp residue from interrupted convert must be cleaned."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))

        with open(os.path.join(self._tmpdir, "0.snapshot.tmp"), "wb") as bf:
            bf.write(b"junk-junk-junk")
        with open(os.path.join(self._tmpdir, "0.json.tmp"), "w") as jf:
            jf.write("{junk")

        model = M().to("npu")
        self.loader._save_snapshot_async(model)

        self.assertFalse(os.path.exists(os.path.join(self._tmpdir, "0.snapshot.tmp")))
        self.assertFalse(os.path.exists(os.path.join(self._tmpdir, "0.json.tmp")))
        self.assertTrue(os.path.exists(os.path.join(self._tmpdir, "0.json")))
        self.assertTrue(os.path.exists(os.path.join(self._tmpdir, "0.snapshot")))
        restored = M().to("npu")
        self.loader.restore_snapshot(restored, torch.device("npu"))
        self.assertTrue(torch.equal(restored.w, model.w))


if __name__ == "__main__":
    unittest.main()
