# SPDX-License-Identifier: Apache-2.0
"""End-to-end reduction tests over synthetic checkpoints.

Covers: prefix cropping with byte-identical tensor retention, MTP remap,
shared-indexer producer closure, sharded index regeneration and total_size,
tokenizer/aux preservation (including subdirectory safetensors such as
optional/quarot.safetensors), quant metadata filtering (fp8
modules_to_not_convert and ModelSlim quant_model_description.json with real
global keys), multimodal vision retention, tied embeddings, containment and
duplicate detection, source completeness, unknown-tensor and overwrite
protection, and manifest verification.
"""

import json
from pathlib import Path

import pytest

from tools.glm_reduced.errors import (
    ManifestError,
    ReductionError,
    SafetyError,
    UnknownTensorError,
    UnsupportedFormatError,
    UnsupportedQuantError,
)
from tools.glm_reduced.profiles import get_profile
from tools.glm_reduced.reducer import (
    SourceCheckpoint,
    _plan_quant_description,
    execute_plan,
    load_source,
    plan_reduction,
    verify_reduced,
)
from tools.glm_reduced.safetensors_io import read_safetensors, sha256_of_file, sha256_of_tensor

from .conftest import (
    f16_tensor,
    make_chatglm_checkpoint,
    make_dsa_checkpoint,
    make_flash_checkpoint,
    make_glm4v_checkpoint,
    write_shard_file,
)

KEEP = 8
NUM_LAYERS = 12


def _build(src: Path, dst: Path, profile_name: str, keep: int = KEEP, **kwargs):
    source = load_source(str(src))
    plan = plan_reduction(source, get_profile(profile_name), keep_layers=keep, **kwargs)
    manifest = execute_plan(plan, source, str(dst), max_shard_bytes=kwargs.get("max_shard_bytes", 1 << 30))
    return source, plan, manifest


def _out_tensors(dst: Path) -> dict[str, tuple[Path, object]]:
    result = {}
    for shard in dst.glob("*.safetensors"):
        parsed = read_safetensors(str(shard))
        for name, info in parsed.tensors.items():
            result[name] = (parsed, info)
    return result


def test_dsa_prefix_crop_keeps_bytes_and_remaps_mtp(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, mtp_layers=1)
    dst = tmp_path / "out"
    source, plan, manifest = _build(src, dst, "glm-moe-dsa")

    out = _out_tensors(dst)
    src_tensors = {}
    for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        parsed = read_safetensors(str(src / shard_name))
        src_tensors.update({name: (parsed, info) for name, info in parsed.tensors.items()})

    # Every kept tensor is byte-identical to its source counterpart.
    for name, (parsed, info) in out.items():
        src_name = name
        if name.startswith("model.layers.8."):
            src_name = name.replace("model.layers.8.", "model.layers.12.", 1)
        assert sha256_of_tensor(parsed, info) == sha256_of_tensor(*src_tensors[src_name]), src_name
        assert name in manifest["output"]["tensor_sha256"]

    # Dropped layers are gone; MTP layer 12 became layer 8.
    assert any(name.startswith("model.layers.8.") for name in out)  # remapped MTP
    assert not any(name.startswith(f"model.layers.{i}.") for i in (9, 10, 11, 12) for name in out)
    assert "model.embed_tokens.weight" in out
    assert "model.norm.weight" in out
    assert "lm_head.weight" in out

    # Config truncated, source untouched.
    config = json.loads((dst / "config.json").read_text())
    assert config["num_hidden_layers"] == KEEP
    assert len(config["indexer_types"]) == KEEP
    assert len(config["mlp_layer_types"]) == KEEP
    assert config["num_nextn_predict_layers"] == 1
    src_config = json.loads((src / "config.json").read_text())
    assert src_config["num_hidden_layers"] == NUM_LAYERS  # not mutated

    # Shared-indexer producers: kept shared layers (3,4,5,7) keep no indexer
    # weights, producers (0,1,2,6) do.
    assert "model.layers.6.self_attn.indexer.wq_b.weight" in out
    assert "model.layers.3.self_attn.indexer.wq_b.weight" not in out

    # Aux files preserved.
    assert (dst / "tokenizer.json").is_file()
    assert (dst / "tokenizer_config.json").is_file()

    report = verify_reduced(str(dst))
    assert report["ok"], report["problems"]
    assert report["tensor_count"] == len(out)


def test_sharded_output_index_and_total_size(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    dst = tmp_path / "out"
    source = load_source(str(src))
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)
    # Force multiple output shards.
    execute_plan(plan, source, str(dst), max_shard_bytes=256)

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    shards = sorted(dst.glob("model-*.safetensors"))
    assert len(shards) > 1
    total = 0
    for shard in shards:
        parsed = read_safetensors(str(shard))
        total += sum(t.nbytes for t in parsed.tensors.values())
    assert index["metadata"]["total_size"] == total
    assert set(index["weight_map"]) == set(_out_tensors(dst))
    assert verify_reduced(str(dst))["ok"]


def test_ascend_quant_index_filename_supported(tmp_path):
    src = make_dsa_checkpoint(
        tmp_path / "src", num_layers=NUM_LAYERS, index_name="quant_model_weights.safetensors.index.json"
    )
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    assert verify_reduced(str(dst))["ok"]
    # The stale source index name must not leak into the output as an aux file.
    assert not (dst / "quant_model_weights.safetensors.index.json").exists()


def test_quant_description_atomic_remap_and_globals(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, with_quant_description=True)
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")

    description = json.loads((dst / "quant_model_description.json").read_text())
    # Global metadata preserved untouched.
    assert description["group_size"] == 0
    assert description["version"] == "1.0.0"
    assert description["is_rot_used"] is True
    assert description["metadata"] == {}
    assert description["optional"] == {}
    assert description["model.embed_tokens.weight"] == "FLOAT"
    # Kept layers retain their own values; MTP layer 12 remapped to 8.
    for idx in range(KEEP):
        assert f"model.layers.{idx}.self_attn.q_proj.weight" in description
    assert f"model.layers.{KEEP}.self_attn.q_proj.weight" in description  # remapped MTP (was 12)
    for idx in (9, 10, 11, 12):
        assert f"model.layers.{idx}.self_attn.q_proj.weight" not in description
    src_description = json.loads((src / "quant_model_description.json").read_text())
    assert f"model.layers.{NUM_LAYERS}.self_attn.q_proj.weight" in src_description  # source not mutated


def test_quant_description_value_mapping_never_repaired():
    # Regression from independent review: {'layer0': 'A', 'dropped9': 'B',
    # 'lm_head': 'C'} must become {'layer0': 'A', 'lm_head': 'C'} — values must
    # never be re-paired to other keys.
    source = SourceCheckpoint(
        "", {}, "", None, {}, {}, [], {"model.layers.0.w": "A", "model.layers.9.w": "B", "lm_head.weight": "C"}, ""
    )
    result = _plan_quant_description(source, source_layers=10, keep_layers=8, mtp_layers=0)
    assert result == {"model.layers.0.w": "A", "lm_head.weight": "C"}


def test_optional_subdirectory_safetensors_copied(tmp_path):
    src = make_dsa_checkpoint(
        tmp_path / "src", num_layers=NUM_LAYERS, with_quant_description=True, with_optional_subdir=True
    )
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    assert (dst / "optional" / "quarot.safetensors").is_file()
    assert sha256_of_file(str(dst / "optional" / "quarot.safetensors")) == sha256_of_file(
        str(src / "optional" / "quarot.safetensors")
    )
    assert verify_reduced(str(dst))["ok"]


def test_flash_nested_config_and_fp8_metadata(tmp_path):
    src = make_flash_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    dst = tmp_path / "out"
    _build(src, dst, "glm5-next")

    config = json.loads((dst / "config.json").read_text())
    text = config["text_config"]
    assert text["num_hidden_layers"] == KEEP
    assert len(text["layer_types"]) == KEEP
    assert len(text["mlp_layer_types"]) == KEEP
    assert len(text["indexer_types"]) == KEEP
    assert text["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4, 5, 6]
    assert text["linear_attn_config"]["full_attn_layers"] == [3, 7]
    assert config["vision_config"]["depth"] == 2  # vision kept in full

    quant = config["quantization_config"]
    assert quant["quant_method"] == "fp8"
    not_convert = quant["modules_to_not_convert"]
    assert "lm_head" in not_convert and "model.visual.patch_embed.proj" in not_convert
    assert f"model.language_model.layers.{KEEP}.hc_attn_fn" in not_convert  # remapped MTP (was 12)
    assert not any(f"layers.{i}." in e for i in (9, 10, 11, 12) for e in not_convert)

    out = _out_tensors(dst)
    assert "model.visual.blocks.0.wqkv.weight" in out
    # Real-layout nested names, including the final norm the first pass missed.
    assert "model.language_model.norm.weight" in out
    assert "model.language_model.embed_tokens.weight" in out
    assert "model.language_model.layers.8.self_attn.indexer.wk.weight" in out  # MTP remap
    assert "model.language_model.layers.8.self_attn.indexer.wk.weight_scale_inv" in out
    assert "lm_head.weight" in out
    assert verify_reduced(str(dst))["ok"]


def test_chatglm_layout_and_layer_key_sync(tmp_path):
    src = make_chatglm_checkpoint(tmp_path / "src", num_layers=6, both_layer_keys=True)
    dst = tmp_path / "out"
    _build(src, dst, "chatglm", keep=4)
    config = json.loads((dst / "config.json").read_text())
    assert config["num_layers"] == 4
    assert config["num_hidden_layers"] == 4
    out = _out_tensors(dst)
    assert "transformer.encoder.layers.3.mlp.dense_4h_to_h.weight" in out
    assert "transformer.encoder.layers.4.mlp.dense_4h_to_h.weight" not in out
    assert "transformer.embedding.word_embeddings.weight" in out
    assert "transformer.encoder.final_layernorm.weight" in out
    assert "transformer.output_layer.weight" in out
    assert verify_reduced(str(dst))["ok"]


def test_chatglm_inconsistent_layer_keys_rejected(tmp_path):
    src = make_chatglm_checkpoint(tmp_path / "src", num_layers=6, consistent=False)
    source = load_source(str(src))
    with pytest.raises(ReductionError, match="inconsistent layer counts"):
        plan_reduction(source, get_profile("chatglm"), keep_layers=4)


def test_glm4v_layout(tmp_path):
    src = make_glm4v_checkpoint(tmp_path / "src", num_layers=6, vision_layers=2)
    dst = tmp_path / "out"
    _build(src, dst, "glm4v", keep=4)
    config = json.loads((dst / "config.json").read_text())
    assert config["text_config"]["num_hidden_layers"] == 4
    assert config["vision_config"]["depth"] == 2
    out = _out_tensors(dst)
    assert "model.language_model.layers.3.self_attn.q_proj.weight" in out
    assert "model.language_model.layers.4.self_attn.q_proj.weight" not in out
    assert "model.visual.blocks.1.attn.qkv.weight" in out
    assert "model.visual.merger.proj.weight" in out
    assert "model.language_model.norm.weight" in out
    assert verify_reduced(str(dst))["ok"]


def test_tied_embeddings_without_lm_head_ok(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, tied=True)
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    report = verify_reduced(str(dst))
    assert report["ok"], report["problems"]
    assert report["tie_word_embeddings"] is True


def test_missing_lm_head_untied_fails_build(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, tied=False)
    (src / "model.safetensors.index.json").unlink()  # rebuild single-shard without lm_head
    for shard in src.glob("model-*.safetensors"):
        shard.unlink()
    tensors = []
    for idx in range(NUM_LAYERS + 1):  # include the MTP layer, omit only lm_head
        for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight"):
            dtype, shape, data = f16_tensor((4, 4), seed=idx)
            tensors.append((f"model.layers.{idx}.{suffix}", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=1)
    tensors.append(("model.embed_tokens.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((16,), seed=2)
    tensors.append(("model.norm.weight", dtype, shape, data))
    write_shard_file(src / "model.safetensors", tensors)

    source = load_source(str(src))
    with pytest.raises(UnsupportedFormatError, match="lm_head"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)


def test_missing_decoder_layer_fails_build(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, with_index=False)
    # Remove every tensor of retained layer 5.
    parsed = read_safetensors(str(src / "model.safetensors"))
    tensors = [
        (name, info.dtype, info.shape, b"\x00" * info.nbytes)
        for name, info in parsed.tensors.items()
        if not name.startswith("model.layers.5.")
    ]
    (src / "model.safetensors").unlink()
    write_shard_file(src / "model.safetensors", tensors)
    source = load_source(str(src))
    with pytest.raises(UnsupportedFormatError, match="missing all tensors for retained decoder layers"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)


def test_index_containment_and_completeness(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.evil.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(UnsupportedFormatError, match="unsafe shard path"):
        load_source(str(src))

    src2 = make_dsa_checkpoint(tmp_path / "src2", num_layers=NUM_LAYERS)
    # Unindexed extra top-level shard next to an indexed checkpoint.
    dtype, shape, data = f16_tensor((2, 2), seed=9)
    write_shard_file(src2 / "extra.safetensors", [("model.layers.0.extra.weight", dtype, shape, data)])
    with pytest.raises(UnsupportedFormatError, match="not covered by"):
        load_source(str(src2))


def test_duplicate_tensor_names_across_unindexed_shards(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, with_index=False)
    dtype, shape, data = f16_tensor((2, 2), seed=9)
    write_shard_file(src / "dup.safetensors", [("model.norm.weight", dtype, shape, data)])
    with pytest.raises(UnsupportedFormatError, match="duplicate tensor names"):
        load_source(str(src))


def test_unknown_tensor_fails_by_default_and_is_auditable_when_allowed(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS, with_index=False)
    dtype, shape, data = f16_tensor((2, 2), seed=9)
    write_shard_file(src / "extra.safetensors", [("mystery_blob.weight", dtype, shape, data)])
    source = load_source(str(src))
    with pytest.raises(UnknownTensorError, match="allow-extra-tensors"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)

    dst = tmp_path / "out"
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP, allow_extra_tensors=True)
    execute_plan(plan, source, str(dst))
    assert "mystery_blob.weight" in _out_tensors(dst)
    manifest = json.loads((dst / "reduction_manifest.json").read_text())
    assert any("mystery_blob" in warning for warning in manifest["warnings"])


def test_overwrite_and_path_protection(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    source = load_source(str(src))
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)
    with pytest.raises(SafetyError, match="already exists"):
        execute_plan(plan, source, str(dst))
    with pytest.raises(SafetyError, match="inside the source"):
        execute_plan(plan, source, str(src / "nested"))
    with pytest.raises(SafetyError, match="equals the source"):
        execute_plan(plan, source, str(src))
    with pytest.raises(SafetyError, match="parent of the source"):
        execute_plan(plan, source, str(tmp_path))
    # Failed builds leave no staging dirs behind.
    assert not list(tmp_path.glob("*.glm-reduced-staging-*"))


def test_refused_formats_and_missing_tokenizer(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    (src / "pytorch_model.bin").write_bytes(b"fake")
    with pytest.raises(UnsupportedFormatError, match="non-safetensors"):
        load_source(str(src))
    (src / "pytorch_model.bin").unlink()
    (src / "tokenizer.json").unlink()
    (src / "tokenizer_config.json").unlink()
    with pytest.raises(UnsupportedFormatError, match="tokenizer"):
        load_source(str(src))


def test_unsupported_quant_scheme_fails_actionably(tmp_path):
    src = make_flash_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"]["quant_method"] = "awq"
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source = load_source(str(src))
    with pytest.raises(UnsupportedQuantError, match="awq"):
        plan_reduction(source, get_profile("glm5-next"), keep_layers=KEEP)


def test_broken_shared_indexer_closure_rejected(tmp_path):
    src = make_dsa_checkpoint(
        tmp_path / "src",
        num_layers=NUM_LAYERS,
        indexer_types=["shared"] + ["full"] * (NUM_LAYERS - 1),  # layer 0 shared without producer
    )
    source = load_source(str(src))
    with pytest.raises(ReductionError, match="producer"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)


def test_min_keep_layers_enforced(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    source = load_source(str(src))
    with pytest.raises(ReductionError, match="keep_layers >= 8"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=3)


def test_wrong_architecture_rejected(tmp_path):
    src = make_flash_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    source = load_source(str(src))
    with pytest.raises(ReductionError, match="not covered by profile"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)


def test_unknown_per_layer_config_array_requires_opt_in(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    config = json.loads((src / "config.json").read_text())
    config["mystery_per_layer"] = list(range(NUM_LAYERS))
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    source = load_source(str(src))
    with pytest.raises(ReductionError, match="truncate-unknown-arrays"):
        plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP, truncate_unknown_arrays=True)
    assert plan.new_config["mystery_per_layer"] == list(range(KEEP))


def test_verify_detects_tampered_output(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    (dst / "tokenizer.json").write_text("tampered", encoding="utf-8")
    report = verify_reduced(str(dst))
    assert not report["ok"]
    assert any("checksum mismatch" in problem for problem in report["problems"])
    (dst / "reduction_manifest.json").unlink()
    with pytest.raises(ManifestError):
        verify_reduced(str(dst))


def test_glm4_moe_flat_fixture(tmp_path):
    # glm4_moe layout: flat config, no indexer_types, MTP at num_hidden_layers.
    src = tmp_path / "src"
    src.mkdir()
    num_layers, mtp = NUM_LAYERS, 1
    config = {
        "architectures": ["Glm4MoeForCausalLM"],
        "model_type": "glm4_moe",
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": 3,
        "hidden_size": 16,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": mtp,
    }
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    tensors = []
    for idx in range(num_layers + mtp):
        for suffix in ("self_attn.q_proj.weight", "mlp.experts.0.gate_proj.weight", "mlp.gate.weight"):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 13 + len(suffix))
            tensors.append((f"model.layers.{idx}.{suffix}", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=1)
    tensors.append(("model.embed_tokens.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((16,), seed=2)
    tensors.append(("model.norm.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=3)
    tensors.append(("lm_head.weight", dtype, shape, data))
    write_shard_file(src / "model.safetensors", tensors)

    dst = tmp_path / "out"
    _build(src, dst, "glm4-moe")
    out = _out_tensors(dst)
    assert f"model.layers.{KEEP}.self_attn.q_proj.weight" in out  # MTP remap
    config = json.loads((dst / "config.json").read_text())
    assert config["num_hidden_layers"] == KEEP
    assert verify_reduced(str(dst))["ok"]


def test_sha256_of_source_files_recorded(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    manifest = json.loads((dst / "reduction_manifest.json").read_text())
    assert manifest["source"]["config_sha256"] == sha256_of_file(str(src / "config.json"))
    assert manifest["source"]["index_sha256"] == sha256_of_file(str(src / "model.safetensors.index.json"))
    assert {f["name"] for f in manifest["source"]["weight_files"]} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert manifest["reduction"]["layer_mapping"] == [{"src": i, "dst": i} for i in range(KEEP)] + [
        {"src": NUM_LAYERS, "dst": KEEP}
    ]


def test_hf_style_symlinked_aux_copied_readonly(tmp_path):
    """HF caches store snapshot files as symlinks into a blob store; aux files
    may therefore be symlinks. They must be copied dereferenced (read-only),
    never followed for writes."""
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    blob = blob_dir / "chat_template.blob"
    blob.write_text("{{ content }}", encoding="utf-8")
    link = src / "chat_template.jinja"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    copied = dst / "chat_template.jinja"
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_text(encoding="utf-8") == "{{ content }}"
    manifest = json.loads((dst / "reduction_manifest.json").read_text())
    assert any("symlink" in warning for warning in manifest["warnings"])
    assert verify_reduced(str(dst))["ok"]


def test_broken_symlink_rejected(tmp_path):
    src = make_dsa_checkpoint(tmp_path / "src", num_layers=NUM_LAYERS)
    link = src / "broken.jinja"
    try:
        link.symlink_to(tmp_path / "does-not-exist")
    except OSError:
        pytest.skip("symlink creation not permitted on this host")
    with pytest.raises(SafetyError, match="symlink"):
        load_source(str(src))


def _make_split_checkpoint(root: Path) -> Path:
    """Shard 1: retained layers 0-7 + globals + MTP layer 12. Shard 2: dropped
    layers 8-11 only. Used to test explicit selective-source loading."""
    root.mkdir()
    num_layers, mtp = 12, 1
    indexer = ["full", "full", "full", "shared", "shared", "shared"] + ["full", "shared", "shared", "shared"] * 2
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": 3,
        "hidden_size": 16,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": mtp,
        "indexer_types": indexer[:num_layers],
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (num_layers - 3),
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    def layer_tensors(idx):
        out = []
        for suffix in ("self_attn.q_proj.weight", "input_layernorm.weight"):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 7 + len(suffix))
            out.append((f"model.layers.{idx}.{suffix}", dtype, shape, data))
        return out

    shard1 = [t for idx in list(range(8)) + [12] for t in layer_tensors(idx)]
    for name, seed in (("model.embed_tokens.weight", 1), ("model.norm.weight", 2), ("lm_head.weight", 3)):
        dtype, shape, data = f16_tensor((64, 16), seed=seed)
        shard1.append((name, dtype, shape, data))
    shard2 = [t for idx in range(8, 12) for t in layer_tensors(idx)]
    write_shard_file(root / "model-00001-of-00002.safetensors", shard1)
    write_shard_file(root / "model-00002-of-00002.safetensors", shard2)
    weight_map = {name: "model-00001-of-00002.safetensors" for name, *_ in shard1}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name, *_ in shard2})
    total = sum(len(data) for *_x, data in shard1 + shard2)
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_selective_source_builds_with_missing_dropped_only_shard(tmp_path):
    from tools.glm_reduced.reducer import load_source_selective

    src = _make_split_checkpoint(tmp_path / "src")
    (src / "model-00002-of-00002.safetensors").unlink()  # dropped-only shard absent
    # Strict full-source loading must reject the missing shard.
    with pytest.raises(UnsupportedFormatError, match="missing shard"):
        load_source(str(src))
    # Explicit selective mode accepts it and records the absence.
    source = load_source_selective(str(src), get_profile("glm-moe-dsa"), keep_layers=KEEP)
    assert source.selective["missing_shards"] == ["model-00002-of-00002.safetensors"]
    plan = plan_reduction(source, get_profile("glm-moe-dsa"), keep_layers=KEEP)
    dst = tmp_path / "out"
    execute_plan(plan, source, str(dst))
    report = verify_reduced(str(dst))
    assert report["ok"], report["problems"]
    manifest = json.loads((dst / "reduction_manifest.json").read_text())
    assert manifest["source"]["selective"]["missing_shards"] == ["model-00002-of-00002.safetensors"]
    assert any("selective source" in w for w in manifest["warnings"])
    # Kept tensors byte-identical, MTP remapped.
    out = _out_tensors(dst)
    assert "model.layers.8.self_attn.q_proj.weight" in out
    assert not any(name.startswith("model.layers.9.") for name in out)


def test_selective_source_fails_on_missing_retained_shard(tmp_path):
    from tools.glm_reduced.reducer import load_source_selective

    src = _make_split_checkpoint(tmp_path / "src")
    (src / "model-00001-of-00002.safetensors").unlink()  # retained shard absent
    with pytest.raises(UnsupportedFormatError, match="REQUIRED shards"):
        load_source_selective(str(src), get_profile("glm-moe-dsa"), keep_layers=KEEP)


def test_required_shards_matches_selective_build(tmp_path):
    from tools.glm_reduced.reducer import required_shards

    src = _make_split_checkpoint(tmp_path / "src")
    result = required_shards(
        str(src / "config.json"), str(src / "model.safetensors.index.json"), get_profile("glm-moe-dsa"), KEEP
    )
    assert result["shards"] == ["model-00001-of-00002.safetensors"]
    assert result["total_shards"] == 2


def test_dangling_referenced_aux_rejected(tmp_path):
    # quant metadata references optional/quarot.safetensors but the file is absent.
    src = make_dsa_checkpoint(
        tmp_path / "src", num_layers=NUM_LAYERS, with_quant_description=True, with_optional_subdir=True
    )
    (src / "optional" / "quarot.safetensors").unlink()
    with pytest.raises(UnsupportedQuantError, match="optional/quarot.safetensors"):
        load_source(str(src))
    # With the referenced file restored the build succeeds and copies it.
    dtype, shape, data = f16_tensor((8, 8), seed=42)
    write_shard_file(src / "optional" / "quarot.safetensors", [("global_rotation", dtype, shape, data)])
    dst = tmp_path / "out"
    _build(src, dst, "glm-moe-dsa")
    assert (dst / "optional" / "quarot.safetensors").is_file()
    assert verify_reduced(str(dst))["ok"]
