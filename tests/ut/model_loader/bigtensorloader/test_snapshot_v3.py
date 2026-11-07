# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v3 snapshot schema: container capture, alias dedup, schema gate, fallback.

These tests pin the reviewer-driven correctness upgrades of PR #16734:

* containers (list/tuple/dict, also nested / inside carrier objects) are
  captured with JSON-serializable access chains and recreated on restore --
  v2 silently skipped them;
* aliased tensors (tied weights, re-referenced derived tensors) are stored
  once and re-shared on restore -- v2 duplicated them and broke tying;
* manifests older than schema v3 are rejected (regenerated via fallback);
* per-tensor spot hashes catch in-bounds tampering of small tensors;
* restore failures make load_model delete the snapshot and fall back to a
  fresh load + process instead of crashing or serving garbage.
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.config import LoadConfig

from vllm_ascend.model_loader.bigtensorloader.bigtensorloader import BigTensorLoader


class _Carrier:
    """Plain-object tensor carrier (MLA impl-style)."""

    def __init__(self, w):
        self.w_uv = w
        self._scratch = torch.randn(1)


class _ContainerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
        self.caches = [torch.randn(2, 3), {"a.b": torch.randn(3, 2)}]
        self.frozen = (torch.randn(5),)


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.head.weight = self.emb.weight  # tied


class _CarrierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
        self.impl = _Carrier(torch.randn(6, 2))


class _BareModel(torch.nn.Module):
    """Restore target: only the param; all extra attrs must be recreated."""

    def __init__(self):
        super().__init__()
        self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))


class TestSnapshotV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if not torch.npu.is_available():
                raise unittest.SkipTest("NPU not available")
            torch.npu.config.allow_internal_format = True
            torch.npu.set_device(0)
            torch.zeros(1, device="npu")  # probe real device access
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"NPU not available: {e}") from e

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="snapshot_v3_ut_")
        self._env = {}
        for key in (
            "VLLM_ASCEND_CHECKPOINT_PATH",
            "VLLM_ASCEND_BIGTENSOR_VERIFY",
            "VLLM_ASCEND_BIGTENSOR_FAST_RESTORE",
            "VLLM_ASCEND_BIGTENSOR_BULK_H2D",
            "VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB",
        ):
            self._env[key] = os.environ.get(key)
        os.environ["VLLM_ASCEND_CHECKPOINT_PATH"] = self._tmpdir
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_VERIFY", None)
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_FAST_RESTORE", None)
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_BULK_H2D", None)
        os.environ.pop("VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB", None)
        self.loader = BigTensorLoader(LoadConfig(load_format="bigtensorloader"))

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for fname in os.listdir(self._tmpdir):
            os.remove(os.path.join(self._tmpdir, fname))
        os.rmdir(self._tmpdir)

    # -- helpers -----------------------------------------------------------

    def _save(self, model):
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            self.loader._save_snapshot_async(model)

    def _restore(self, model):
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            self.loader.restore_snapshot(model, torch.device("npu"))

    def _manifest(self) -> dict:
        with open(os.path.join(self._tmpdir, "0.json")) as f:
            return json.load(f)

    def _blob_size(self) -> int:
        return os.path.getsize(os.path.join(self._tmpdir, "0.snapshot"))

    # -- container capture -------------------------------------------------

    def test_containers_captured_with_paths_and_defs(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        manifest = self._manifest()

        self.assertEqual(manifest["schema_version"], 3)
        tensors = manifest["tensors"]
        self.assertIn("caches.0", tensors)
        self.assertIn("caches.1.a.b", tensors)
        self.assertIn("frozen.0", tensors)
        # dotted dict keys must survive the (dotted) name and the path
        self.assertEqual(tensors["caches.1.a.b"]["path"], [{"i": 1}, {"k": "a.b"}])
        self.assertEqual(tensors["caches.0"]["path"], [{"i": 0}])
        self.assertEqual(tensors["frozen.0"]["path"], [{"i": 0}])
        # container defs: type + length, keyed module|chain
        defs = manifest["containers"]
        self.assertIn('|[{"a": "caches"}]', defs)
        self.assertEqual(defs['|[{"a": "caches"}]'], {"type": "list", "len": 2})
        self.assertIn('|[{"a": "frozen"}]', defs)
        self.assertEqual(defs['|[{"a": "frozen"}]'], {"type": "tuple", "len": 1})
        self.assertEqual(defs['|[{"a": "caches"}, {"i": 1}]'], {"type": "dict"})

    def test_container_roundtrip_recreates_missing_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        # the (tiny) tensors make the spot-hash threshold irrelevant here
        restored = _BareModel().to("npu")
        self._restore(restored)

        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.caches[1]["a.b"], model.caches[1]["a.b"]))
        # tuples were recreated as lists and sealed back to tuples
        self.assertIsInstance(restored.frozen, tuple)
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    def test_container_restored_into_existing_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        restored = _ContainerModel().to("npu")  # containers already exist
        self._restore(restored)
        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.caches[1]["a.b"], model.caches[1]["a.b"]))
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertIsInstance(restored.frozen, tuple)  # existing tuple untouched

    def test_nested_container_recreation(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                self.grid = [[torch.randn(1), torch.randn(1)], torch.randn(1)]

        model = M().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertEqual(manifest["tensors"]["grid.0.1"]["path"], [{"i": 0}, {"i": 1}])
        self.assertIn('|[{"a": "grid"}, {"i": 0}]', manifest["containers"])

        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.grid[0][1], model.grid[0][1]))
        self.assertTrue(torch.equal(restored.grid[1], model.grid[1]))

    def test_nested_tuple_sealed_innermost_first(self):
        """回归: 嵌套 tuple 容器逐层密封，内层不能停留在 list。"""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.pair = ((torch.randn(1), torch.randn(1)),)

        model = M().to("npu")
        self._save(model)
        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertIsInstance(restored.pair, tuple)
        # 内层在保存时记录为 tuple，恢复后也必须是 tuple（而不是未密封的 list）
        self.assertIsInstance(restored.pair[0], tuple)
        self.assertTrue(torch.equal(restored.pair[0][0], model.pair[0][0]))
        self.assertTrue(torch.equal(restored.pair[0][1], model.pair[0][1]))

    def test_private_attrs_skipped_like_netloader(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                self._runtime_cache = [torch.randn(2)]

        self._save(M().to("npu"))
        manifest = self._manifest()
        self.assertNotIn("_runtime_cache.0", manifest["tensors"])

    # -- carrier objects ----------------------------------------------------

    def test_carrier_tensor_roundtrip(self):
        model = _CarrierModel().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertIn("impl.w_uv", manifest["tensors"])
        self.assertEqual(manifest["tensors"]["impl.w_uv"]["path"], [{"a": "w_uv"}])
        # netloader policy: private members of scanned objects are skipped
        self.assertNotIn("impl._scratch", manifest["tensors"])

        restored = _CarrierModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.impl.w_uv, model.impl.w_uv))
        # _scratch untouched by restore
        self.assertFalse(torch.equal(restored.impl._scratch, model.impl._scratch))

    def test_missing_carrier_fails_loud(self):
        model = _CarrierModel().to("npu")
        self._save(model)
        restored = _BareModel().to("npu")  # no impl carrier
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self._restore(restored)

    # -- alias dedup --------------------------------------------------------

    def test_tied_weights_dedup_and_restay_tied(self):
        model = _TiedModel().to("npu")
        self._save(model)
        manifest = self._manifest()

        emb_entry = manifest["tensors"]["emb.weight"]
        head_entry = manifest["tensors"]["head.weight"]
        self.assertGreater(emb_entry["nbytes"], 0)
        self.assertEqual(head_entry.get("alias_of"), "emb.weight")
        self.assertEqual(head_entry["nbytes"], 0)
        # the tied tensor is stored exactly once
        self.assertEqual(self._blob_size(), emb_entry["nbytes"] + emb_entry["offset"])

        restored = _TiedModel().to("npu")
        # break the tie so restore is what re-establishes it
        with torch.no_grad():
            restored.head.weight = torch.nn.Parameter(
                torch.randn_like(restored.head.weight.data),
                requires_grad=False,
            )
        self._restore(restored)
        self.assertTrue(torch.equal(restored.emb.weight.data, model.emb.weight.data))
        self.assertTrue(torch.equal(restored.head.weight.data, restored.emb.weight.data))
        self.assertEqual(restored.head.weight.data_ptr(), restored.emb.weight.data_ptr())

    def test_alias_tensor_in_two_containers_stored_once(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(2, 2), requires_grad=False))
                shared = torch.randn(3)
                self.left = [shared]
                self.right = {"key": shared}

        model = M().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertGreater(manifest["tensors"]["left.0"]["nbytes"], 0)
        self.assertEqual(manifest["tensors"]["right.key"]["alias_of"], "left.0")

        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.left[0], model.left[0]))
        self.assertTrue(torch.equal(restored.right["key"], model.left[0]))
        # aliasing (shared storage) survives the roundtrip
        self.assertEqual(restored.right["key"].data_ptr(), restored.left[0].data_ptr())

    # -- schema gate ----------------------------------------------------------

    def test_schema_gate_rejects_v2(self):
        model = _BareModel().to("npu")
        self._save(model)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        manifest = self._manifest()
        manifest["schema_version"] = 2
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaisesRegex(RuntimeError, "schema_version"):
            self._restore(_BareModel().to("npu"))

    def test_missing_schema_version_rejected(self):
        model = _BareModel().to("npu")
        self._save(model)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        manifest = self._manifest()
        del manifest["schema_version"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with self.assertRaisesRegex(RuntimeError, "schema_version"):
            self._restore(_BareModel().to("npu"))

    # -- drift detection --------------------------------------------------------

    def test_attr_drift_shape_mismatch_fails_loud(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.scale = torch.randn(8)

        model = M().to("npu")
        self._save(model)

        class Drifted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter("w", torch.nn.Parameter(torch.randn(4, 4), requires_grad=False))
                self.scale = torch.randn(16)  # shape drifted

        with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatch"):
            self._restore(Drifted().to("npu"))

    # -- spot hash ------------------------------------------------------------

    def test_spot_hash_small_tensor_only(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "small",
                    torch.nn.Parameter(torch.randn(4), requires_grad=False),
                )
                self.register_parameter(
                    "big",
                    torch.nn.Parameter(
                        torch.empty(BigTensorLoader._SPOT_HASH_MAX_BYTES + 1, dtype=torch.int8),
                        requires_grad=False,
                    ),
                )

        self._save(M().to("npu"))
        manifest = self._manifest()
        self.assertIn("sha256", manifest["tensors"]["small"])
        self.assertNotIn("sha256", manifest["tensors"]["big"])

    def test_alias_entries_are_not_spot_hashed_twice(self):
        model = _TiedModel().to("npu")
        self._save(model)
        manifest = self._manifest()
        self.assertNotIn("sha256", manifest["tensors"]["head.weight"])

    # -- bulk mode with containers -----------------------------------------------

    def test_bulk_restore_with_containers(self):
        model = _ContainerModel().to("npu")
        self._save(model)
        os.environ["VLLM_ASCEND_BIGTENSOR_BULK_H2D"] = "1"
        os.environ["VLLM_ASCEND_BIGTENSOR_BULK_CHUNK_MB"] = "1"
        restored = _BareModel().to("npu")
        self._restore(restored)
        self.assertTrue(torch.equal(restored.caches[0], model.caches[0]))
        self.assertTrue(torch.equal(restored.frozen[0], model.frozen[0]))
        self.assertTrue(torch.equal(restored.w.data, model.w.data))

    # -- delete + fallback ---------------------------------------------------------

    def test_delete_snapshot_removes_all_files(self):
        model = _BareModel().to("npu")
        self._save(model)
        manifest_path = os.path.join(self._tmpdir, "0.json")
        blob_path = os.path.join(self._tmpdir, "0.snapshot")
        open(blob_path + ".tmp", "w").close()  # stale tmp from an interrupted convert
        self.assertTrue(os.path.exists(manifest_path))
        self.assertTrue(os.path.exists(blob_path))

        with patch("torch.distributed.get_rank", return_value=0):
            self.loader._delete_snapshot()
        for path in (manifest_path, blob_path, blob_path + ".tmp"):
            self.assertFalse(os.path.exists(path), path)

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_model_falls_back_on_truncated_blob(self, _r, _w):
        model = _BareModel().to("npu").to("npu")
        self._save(model)
        blob_path = os.path.join(self._tmpdir, "0.snapshot")
        size = os.path.getsize(blob_path)
        with open(blob_path, "r+b") as f:
            f.truncate(size // 10)

        os.environ["VLLM_ASCEND_BIGTENSOR_FAST_RESTORE"] = "1"

        created = []
        processed = []

        def fake_process(model, model_config, target_device):
            processed.append(model)

        vllm_config = SimpleNamespace(
            device_config=SimpleNamespace(device="npu"),
            load_config=SimpleNamespace(device=None),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
                data_parallel_size=1,
            ),
            additional_config={},
        )
        # a registration owned by a live co-resident model: must survive the fallback
        foreign_module = object()
        forward_context = {"live.layers.0.self_attn.attn": foreign_module}
        moe_layers = ["live.layers.0.mlp.experts"]
        vllm_config.compilation_config = SimpleNamespace(
            static_forward_context=forward_context,
            static_all_moe_layers=moe_layers,
        )
        model_config = SimpleNamespace(model="test-model", dtype=torch.float32, quantization=None)

        def fake_initialize_model(**kwargs):
            created.append(_BareModel().to("npu").to("npu"))
            # simulate attention/MoE layers registering themselves as built
            forward_context["model.layers.0.self_attn.attn"] = created[-1]
            moe_layers.append("model.layers.0.mlp.experts")
            return created[-1]

        with (
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                side_effect=fake_initialize_model,
            ),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading",
                side_effect=fake_process,
            ),
            patch.object(BigTensorLoader, "load_weights", return_value=None),
        ):
            returned = self.loader.load_model(vllm_config, model_config)

        # restore attempt + fallback fresh load
        self.assertEqual(len(created), 2)
        self.assertIs(returned, created[1])
        # the abandoned model's registration was dropped (and the fresh model
        # re-registered the same prefix after the reset -- no "Duplicate layer
        # name"), while the live model's registration survived untouched
        self.assertIs(forward_context["model.layers.0.self_attn.attn"], created[1])
        self.assertIs(forward_context["live.layers.0.self_attn.attn"], foreign_module)
        # MoE registry rolled back to the pre-load baseline, then the fresh
        # model appended its own entry
        self.assertEqual(moe_layers, ["live.layers.0.mlp.experts", "model.layers.0.mlp.experts"])
        # fresh path ran process_weights exactly once, on the fresh model
        self.assertEqual(processed, [created[1]])
        # the corrupt snapshot was deleted and regenerated as v3
        manifest_path = os.path.join(self._tmpdir, "0.json")
        deadline = time.monotonic() + 10
        while not os.path.exists(manifest_path) and time.monotonic() < deadline:
            time.sleep(0.1)
        with open(manifest_path) as f:
            self.assertEqual(json.load(f)["schema_version"], 3)
        # fallback path returns fresh weights (not the corrupt snapshot's)
        self.assertTrue(torch.equal(returned.w.data, created[1].w.data))

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_model_restores_without_process_when_snapshot_valid(self, _r, _w):
        model = _BareModel().to("npu").to("npu")
        self._save(model)
        os.environ["VLLM_ASCEND_BIGTENSOR_FAST_RESTORE"] = "1"

        created = []
        processed = []

        def fake_initialize_model(**kwargs):
            created.append(_BareModel().to("npu").to("npu"))
            return created[-1]

        def fake_process(model, model_config, target_device):
            processed.append(model)

        vllm_config = SimpleNamespace(
            device_config=SimpleNamespace(device="npu"),
            load_config=SimpleNamespace(device=None),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
                data_parallel_size=1,
            ),
            additional_config={},
        )
        model_config = SimpleNamespace(model="test-model", dtype=torch.float32, quantization=None)

        with (
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.initialize_model",
                side_effect=fake_initialize_model,
            ),
            patch(
                "vllm_ascend.model_loader.bigtensorloader.bigtensorloader.process_weights_after_loading",
                side_effect=fake_process,
            ),
            patch.object(BigTensorLoader, "load_weights", return_value=None),
        ):
            returned = self.loader.load_model(vllm_config, model_config)

        # restore-only: one init, no process, snapshot weights installed
        self.assertEqual(len(created), 1)
        self.assertEqual(processed, [])
        self.assertTrue(torch.equal(returned.w.data, model.w.data))


if __name__ == "__main__":
    unittest.main()
