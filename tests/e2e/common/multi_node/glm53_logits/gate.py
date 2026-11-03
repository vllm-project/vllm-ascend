# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Immutable, per-configuration golden checks; never generate a baseline in CI."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import compare

GOLDEN = Path(__file__).with_name("golden")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_golden(case, root=GOLDEN, artifact_root=None):
    root = Path(root)
    artifact_root = root if artifact_root is None else Path(artifact_root)
    manifest = json.loads((root / "manifest.json").read_text())
    if case not in manifest["cases"]:
        raise ValueError(f"No approved baseline for {case}")
    if sha256(root / "corpus.json") != manifest["corpus_sha256"]:
        raise ValueError("Golden corpus checksum mismatch")
    corpus = json.loads((root / "corpus.json").read_text())
    if [len(item["prompt_ids"]) for item in corpus] != manifest["input_lengths"]:
        raise ValueError("Golden input lengths differ from manifest")
    entries = manifest["cases"][case]
    expected = {(item["name"], role) for item in corpus for role in ("P", "D")}
    actual = {(entry["sample"], entry["role"]) for entry in entries}
    if actual != expected or len(entries) != len(expected):
        raise ValueError("Missing or duplicate golden sample/role")
    for entry in entries:
        path = artifact_root / entry["file"]
        if path.resolve().parent != (artifact_root / case).resolve():
            raise ValueError("Golden file must belong to the selected configuration")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Golden checksum mismatch: {path}")
    return manifest, corpus


def validate_model(model, manifest):
    model = Path(model)
    if not model.is_dir() or (model / "INCOMPLETE").exists():
        raise ValueError(f"Provision the completed 1D7M1MTP checkpoint at {model}")
    for record in manifest["model_hashes"]:
        name, expected = record["file"], record["sha256"]
        if sha256(model / name) != expected:
            raise ValueError(f"Model identity mismatch: {name}")


def validate_features(health, case, role):
    features = health["features"]
    expected = dict(
        pp=2 if role == "P" and case != "disable_pp" else 1,
        pcp=2 if role == "P" else 1,
        tp=2,
        dp=1 if role == "P" else 4,
        flashcomm1=role == "P" and case != "disable_flashcomm1",
        li_c8=case != "disable_lic8",
        sfa_c8=case != "disable_sfac8",
        megamoe_effective=case != "disable_megamoe",
    )
    workers = 4 if role == "P" and case == "disable_pp" else 8
    if len(features) != workers:
        raise ValueError(f"Expected {workers} worker probes, got {len(features)}")
    if len({(f["dp_rank"], f["worker_rank"]) for f in features}) != workers:
        raise ValueError("Duplicate worker probes")
    for feature in features:
        if ".v2." not in feature["runner"]:
            raise ValueError(f"MRV2 is not active: {feature}")
        for key, value in expected.items():
            if feature[key] != value:
                raise ValueError(f"{case}/{role}: {key}={feature[key]}, expected {value}")


def check_capture(candidate, entry, manifest, root=GOLDEN):
    with (
        np.load(Path(root) / entry["file"], allow_pickle=False) as reference,
        np.load(candidate, allow_pickle=False) as current,
    ):
        expected_shape = (entry["positions"], manifest["vocabulary_size"])
        if reference["logits"].shape != expected_shape or current["logits"].shape != expected_shape:
            raise ValueError(f"Incomplete logits: expected {expected_shape}")
        metrics = compare(reference, current)
    limits = manifest["limits"]
    passed = (
        metrics["max_centered_relative_l2"] <= limits["relative_l2"]
        and metrics["max_kl_ref_to_candidate"] <= limits["kl"]
        and metrics["min_centered_cosine"] >= limits["cosine"]
        and metrics["high_margin_top1_mismatches"] <= limits["high_margin_top1_mismatches"]
    )
    return {"sample": entry["sample"], "role": entry["role"], "passed": passed, **metrics}
