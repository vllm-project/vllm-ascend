#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Let GLM-5.3-Flash grouping in vLLM accept platform MLA spec subclasses.

vLLM selects its GLM-5.3-Flash KV cache layout with ``type(spec) is
MLAAttentionSpec``. Hardware plugins subclass that spec to attach layout
metadata -- Ascend publishes ``AscendMLAAttentionSpec`` -- so the exact type
check rejects them and the model silently falls back to the generic grouping
path, which then refuses to pad GLM's unevenly sized MLA pages.

Relax the three checks to ``isinstance``. ``SlidingWindowMLASpec`` and
``KpoolTailSpec`` derive from ``SlidingWindowSpec`` rather than
``MLAAttentionSpec``, so widening the check keeps excluding them.

Re-run after re-installing or checking out vLLM; it is idempotent. Remove once
the equivalent fix lands upstream.
"""

import os
import sys
from pathlib import Path

# Running this as ``python tools/<script>.py`` puts ``tools/`` first on
# sys.path, where the ``bisect`` subcommand package shadows the standard
# library module that vLLM imports. Drop our own directory before importing it.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _SCRIPT_DIR]

HELPER = '''def _is_glm5_next_attn_spec(spec: KVCacheSpec) -> bool:
    """Whether ``spec`` is an MLA spec eligible for GLM-5.3-Flash grouping.

    Accepts subclasses so hardware plugins can attach their own layout
    metadata. Sliding-window and kpool-tail specs derive from
    ``SlidingWindowSpec`` and stay excluded.
    """
    return isinstance(spec, MLAAttentionSpec)


'''

EXACT_CHECK = "type(spec) is MLAAttentionSpec"
RELAXED_CHECK = "_is_glm5_next_attn_spec(spec)"
ANCHOR = "def _get_kv_cache_groups_glm5_next("
EXPECTED_CHECKS = 3


def main() -> int:
    try:
        import vllm.v1.core.kv_cache_utils as target
    except ImportError as exc:
        print(f"cannot import vllm.v1.core.kv_cache_utils: {exc}")
        return 1

    path = Path(target.__file__)
    source = path.read_text(encoding="utf-8")

    if RELAXED_CHECK in source:
        print(f"already patched: {path}")
        return 0

    found = source.count(EXACT_CHECK)
    if found != EXPECTED_CHECKS:
        print(
            f"expected {EXPECTED_CHECKS} occurrences of {EXACT_CHECK!r} in {path}, "
            f"found {found}. vLLM likely reworked this code -- check whether the "
            "exact-type gate still exists before patching."
        )
        return 1
    if ANCHOR not in source:
        print(f"cannot find {ANCHOR!r} in {path}")
        return 1

    source = source.replace(EXACT_CHECK, RELAXED_CHECK)
    source = source.replace(ANCHOR, HELPER + ANCHOR, 1)
    with path.open("w", encoding="utf-8", newline="\n") as patched:
        patched.write(source)
    print(f"patched {found} checks in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
