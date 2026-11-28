# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Small synthetic fixtures for CPU tests; never used by nightly inference."""

import json

import numpy as np

from tests.e2e.common.multi_node.glm53_logits.config import CASES
from tests.e2e.common.multi_node.glm53_logits.gate import sha256


def make_fixture(root):
    corpus = [dict(name=f"len{length}", prompt_ids=[1] * length, trace=list(range(2, 10))) for length in [4, 8, 16, 32]]
    (root / "corpus.json").write_text(json.dumps(corpus))
    manifest = dict(
        input_lengths=[4, 8, 16, 32],
        vocabulary_size=20,
        corpus_sha256=sha256(root / "corpus.json"),
        cases={},
        limits=dict(relative_l2=0.02, kl=0.01, cosine=0.999, high_margin_top1_mismatches=0),
    )
    for index, case in enumerate(CASES):
        (root / case).mkdir()
        entries = []
        for item in corpus:
            for role in ["P", "D"]:
                positions = 1 if role == "P" else 8
                path = root / case / f"{item['name']}-{role}.npz"
                logits = np.arange(positions * 20, dtype=np.float32).reshape(positions, 20) * (index + 1)
                np.savez(path, logits=logits, prompt_ids=item["prompt_ids"], token_ids=item["trace"][:positions])
                entries.append(
                    dict(
                        sample=item["name"],
                        role=role,
                        positions=positions,
                        file=path.relative_to(root).as_posix(),
                        sha256=sha256(path),
                    )
                )
        manifest["cases"][case] = entries
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest, corpus
