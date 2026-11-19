#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source for rare_chars feature / GB-input numeric ids.

Numeric values live in ``rare_feed_native.c`` (``FT_*`` / ``IN_*``). Python
string names map onto those C enumerators; CI parses the ``.c`` file so the
two sides cannot drift silently.
"""

from __future__ import annotations

import regex as re
from pathlib import Path

# Python rule-feature names → C FT_* enumerator.
FT_PY_TO_C: dict[str, str] = {
    "lp1": "FT_LP1",
    "lpc": "FT_LPC",
    "gap": "FT_GAP",
    "fs": "FT_FS",
    "fsw": "FT_FSW",
    "caseless_after_latin": "FT_CASELESS",
    "bad_utf8": "FT_BAD",
    "gb": "FT_GB",
    "answer": "FT_ANSWER",
}

# Python GB-input column names → C IN_* enumerator.
IN_PY_TO_C: dict[str, str] = {
    "lp1": "IN_LP1",
    "lpc": "IN_LPC",
    "gap": "IN_GAP",
    "cls": "IN_CLS",
    "pcls": "IN_PCLS",
    "ncls": "IN_NCLS",
    "first": "IN_FIRST",
    "last": "IN_LAST",
    "lead": "IN_LEAD",
    "pcat": "IN_PCAT",
    "ncat": "IN_NCAT",
    "fs": "IN_FS",
    "fsw": "IN_FSW",
    "answer": "IN_ANSWER",
    "bad_utf8": "IN_BAD",
}

_C_ENUM_RE = re.compile(
    r"\b((?:FT|IN)_[A-Z0-9_]+)\s*=\s*(-?\d+)\s*,?",
    re.MULTILINE,
)
# Size constants in the same .c file (``enum { FEAT_N = 9, IN_N = 15 }``), not ids.
_C_SIZE_CONSTANTS = frozenset({"FEAT_N", "IN_N"})

_C_PATH = Path(__file__).with_name("rare_feed_native.c")


def parse_c_enum_values(source: str) -> dict[str, int]:
    """Parse ``FT_*`` / ``IN_*`` assignments from ``rare_feed_native.c`` text."""
    found = {
        name: int(val)
        for name, val in _C_ENUM_RE.findall(source)
        if name not in _C_SIZE_CONSTANTS
    }
    if not found:
        raise ValueError("no FT_*/IN_* enumerators found in rare_feed_native.c")
    return found


def load_c_enum_values(c_path: Path | None = None) -> dict[str, int]:
    path = _C_PATH if c_path is None else Path(c_path)
    return parse_c_enum_values(path.read_text(encoding="utf-8"))


def build_feat_input(
    c_enums: dict[str, int] | None = None,
    *,
    c_path: Path | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(feat, input)`` dicts keyed by Python names, values from C."""
    enums = load_c_enum_values(c_path) if c_enums is None else c_enums
    missing = [c for c in (*FT_PY_TO_C.values(), *IN_PY_TO_C.values()) if c not in enums]
    if missing:
        raise KeyError(f"rare_feed_native.c missing enumerators: {missing}")
    extra_ft = sorted(n for n in enums if n.startswith("FT_") and n not in FT_PY_TO_C.values())
    extra_in = sorted(n for n in enums if n.startswith("IN_") and n not in IN_PY_TO_C.values())
    if extra_ft or extra_in:
        raise KeyError(
            "rare_enums.py name map missing C enumerators: "
            f"FT={extra_ft} IN={extra_in}"
        )
    feat = {py: enums[c] for py, c in FT_PY_TO_C.items()}
    inp = {py: enums[c] for py, c in IN_PY_TO_C.items()}
    return feat, inp


def assert_enums_match_c(
    feat: dict[str, int],
    inp: dict[str, int],
    *,
    c_path: Path | None = None,
) -> None:
    """Raise ``AssertionError`` if Python tables disagree with the ``.c`` file."""
    expected_feat, expected_inp = build_feat_input(c_path=c_path)
    if feat != expected_feat:
        raise AssertionError(f"FT feature id drift: got={feat} expected={expected_feat}")
    if inp != expected_inp:
        raise AssertionError(f"IN input id drift: got={inp} expected={expected_inp}")


# Import-time tables used by ``rare_native`` (C file next to this module).
FEAT, INPUT = build_feat_input()
