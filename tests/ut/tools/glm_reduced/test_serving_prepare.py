# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.glm_reduced.prepare import prepare_checkpoint
from tools.glm_reduced.safetensors_io import sha256_of_file

from .conftest import make_dsa_checkpoint


@pytest.fixture
def pinned_source(tmp_path):
    source = make_dsa_checkpoint(
        tmp_path / "source",
        index_name="quant_model_weights.safetensors.index.json",
        with_quant_description=True,
        with_optional_subdir=True,
    )
    files = {
        path.relative_to(source).as_posix(): {"sha256": sha256_of_file(str(path)), "bytes": path.stat().st_size}
        for path in source.rglob("*")
        if path.is_file()
    }
    descriptor = tmp_path / "source.json"
    descriptor.write_text(json.dumps({"model": "test/model", "revision": "fixed", "files": files}))
    return source, descriptor


def test_prepare_reuses_verified_cache_and_excludes_provider_metadata(pinned_source, tmp_path):
    source, descriptor = pinned_source
    (source / ".msc").write_text("machine-local cache metadata")
    cache = tmp_path / "cache"
    output, identity = prepare_checkpoint(descriptor, source, cache)
    assert not (output / ".msc").exists()
    assert (output / "optional/quarot.safetensors").is_file()
    again, again_identity = prepare_checkpoint(descriptor, source, cache)
    assert (again, again_identity) == (output, identity)
    (output / "tokenizer.json").write_text("corruption")
    with pytest.raises(ValueError, match="verification"):
        prepare_checkpoint(descriptor, source, cache)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_source_mismatch_never_publishes(pinned_source, tmp_path, mutation):
    source, descriptor = pinned_source
    path = source / "tokenizer.json"
    if mutation == "missing":
        path.unlink()
    else:
        path.write_text("[]")  # Same size as the fixture, different checksum.
    cache = tmp_path / "cache"
    with pytest.raises(ValueError):
        prepare_checkpoint(descriptor, source, cache)
    assert not list(cache.iterdir())
