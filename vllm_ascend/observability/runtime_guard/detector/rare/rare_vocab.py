#!/usr/bin/env python3
"""Per-tokenizer tables of the rare_chars detector.

Built once at Guard warmup from the served tokenizer (or offline via ``main`` /
``build_from_token_bytes``). Online detection only indexes by token id.

  rcls        int8   script / category class (CLASSES)
  lead        bool   starts with whitespace
  first, last int8   kind of the first / last character
  u8          bool   not a complete, valid UTF-8 string on its own
  boff, bflat        the token's bytes: bflat[boff[i]:boff[i + 1]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata as U
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "tables")


def _byte_decoder():
    """inverse of GPT-2 bytes_to_unicode (byte-level BPE alphabet -> byte)"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


CLASSES = [
    "space_punct",
    "latin",
    "digit",
    "han",
    "kana",
    "hangul",
    "cyrillic",
    "greek",
    "arabic",
    "hebrew",
    "thai",
    "indic",
    "other_letter",
    "symbol",
    "emoji",
    "partial",
    "replacement",
    "control",
    "mixed",
    "combining",
    "special",
]
CI = {n: i for i, n in enumerate(CLASSES)}
LETTERS = [
    CI[c]
    for c in (
        "latin",
        "han",
        "kana",
        "hangul",
        "cyrillic",
        "greek",
        "arabic",
        "hebrew",
        "thai",
        "indic",
        "other_letter",
        "mixed",
    )
]
NONLATIN_LETTERS = [
    CI[c] for c in ("cyrillic", "arabic", "thai", "hangul", "hebrew", "indic", "kana", "other_letter")
]


def script(ch):
    o = ord(ch)
    if ch == "�":
        return "replacement"
    cat = U.category(ch)
    if cat in ("Cc", "Cf", "Co", "Cs", "Cn") and ch not in "\n\r\t":
        return "control"
    if cat.startswith("M"):
        return "combining"
    if cat == "Nd":
        return "digit" if o < 128 else "other_letter"
    if cat.startswith("L"):
        if o < 0x250 or 0x1E00 <= o < 0x1F00:
            return "latin"
        if 0x370 <= o < 0x400 or 0x1F00 <= o < 0x2000:
            return "greek"
        if 0x400 <= o < 0x530:
            return "cyrillic"
        if 0x590 <= o < 0x600:
            return "hebrew"
        if 0x600 <= o < 0x700 or 0x750 <= o < 0x780 or 0xFB50 <= o < 0xFF00:
            return "arabic"
        if 0x900 <= o < 0xE00:
            return "indic"
        if 0xE00 <= o < 0xE80:
            return "thai"
        if 0x3040 <= o < 0x3100 or 0x31F0 <= o < 0x3200 or 0xFF66 <= o < 0xFFA0:
            return "kana"
        if 0xAC00 <= o < 0xD7B0 or 0x1100 <= o < 0x1200 or 0x3130 <= o < 0x3190:
            return "hangul"
        if 0x3400 <= o < 0xA000 or 0xF900 <= o < 0xFB00 or 0x20000 <= o < 0x31350:
            return "han"
        if 0xFF21 <= o < 0xFF5B:
            return "latin"
        return "other_letter"
    if cat == "So" and (o >= 0x1F000 or 0x2600 <= o < 0x2800):
        return "emoji"
    if cat in ("So", "Sm", "Sc", "Sk") and o >= 128:
        return "symbol"
    return "space_punct"


def classify(s):
    kinds = {script(ch) for ch in s} - {"space_punct"}
    if not kinds:
        return "space_punct"
    for k in ("replacement", "control"):
        if k in kinds:
            return k
    letters = kinds - {"digit", "combining", "symbol", "emoji"}
    if len(letters) > 1:
        return "mixed"
    if letters:
        return letters.pop()
    for k in ("emoji", "symbol", "combining", "digit"):
        if k in kinds:
            return k
    return "space_punct"


def kind(ch):
    """0 none / space, 1 lower, 2 upper, 3 digit, 4 punctuation / symbol, 5 caseless letter"""
    if not ch or ch.isspace():
        return 0
    if ch.isdigit():
        return 3
    if ch.isalpha():
        return 1 if ch.islower() else 2 if ch.isupper() else 5
    return 4


def _assemble(chunks: list[bytes], special: set[int]) -> dict[str, Any]:
    V = len(chunks)
    rcls, first, last = np.zeros(V, np.int8), np.zeros(V, np.int8), np.zeros(V, np.int8)
    lead, u8 = np.zeros(V, bool), np.zeros(V, bool)
    for i, b in enumerate(chunks):
        lead[i] = b[:1] in (b" ", b"\n", b"\t", b"\r")
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            u8[i] = True
            rcls[i] = CI["special"] if i in special else CI["partial"]
            s = b.decode("utf-8", "ignore")
        else:
            rcls[i] = CI["special"] if i in special else CI[classify(s)]
        first[i], last[i] = kind(s[:1]), kind(s[-1:])
    boff = np.r_[0, np.cumsum([len(b) for b in chunks])].astype(np.int64)
    table = {
        "rcls": rcls,
        "lead": lead,
        "first": first,
        "last": last,
        "u8": u8,
        "boff": boff,
        "bflat": np.frombuffer(b"".join(chunks), np.uint8).copy(),
    }
    table["bytes"] = [table["bflat"][table["boff"][i] : table["boff"][i + 1]].tobytes() for i in range(V)]
    return table


def build_from_token_bytes(chunks: list[bytes], *, special_ids: set[int] | None = None) -> dict[str, Any]:
    """Build a rare table from raw per-id byte sequences (tests / synthetic vocabs)."""
    return _assemble(list(chunks), set(special_ids or ()))


def build(tokenizer_json: str) -> dict[str, Any]:
    """Offline build from a HuggingFace ``tokenizer.json`` path."""
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(tokenizer_json)
    with open(tokenizer_json, encoding="utf-8") as f:
        special = {a["id"] for a in json.load(f).get("added_tokens", []) if a.get("special")}
    bd = _byte_decoder()
    V = tk.get_vocab_size()
    chunks: list[bytes] = []
    for i in range(V):
        p = tk.id_to_token(i) or ""
        try:
            b = bytes(bd[c] for c in p)
        except KeyError:
            b = p.encode("utf-8")
        chunks.append(b)
    return _assemble(chunks, special)


def _token_piece(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "id_to_token"):
        piece = tokenizer.id_to_token(token_id)
        return "" if piece is None else str(piece)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(piece, (list, tuple)):
            return "" if not piece else str(piece[0])
        return "" if piece is None else str(piece)
    raise TypeError(f"tokenizer cannot map id→token: {type(tokenizer).__name__}")


def _vocab_size(tokenizer: Any) -> int:
    for attr in ("get_vocab_size",):
        fn = getattr(tokenizer, attr, None)
        if callable(fn):
            return int(fn())
    for attr in ("vocab_size", "n_vocab"):
        if hasattr(tokenizer, attr):
            return int(getattr(tokenizer, attr))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return int(len(get_vocab()))
    raise TypeError(f"tokenizer has no vocab size: {type(tokenizer).__name__}")


def _special_ids(tokenizer: Any) -> set[int]:
    out: set[int] = set()
    ids = getattr(tokenizer, "all_special_ids", None)
    if ids:
        out.update(int(x) for x in ids)
    added = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added, dict):
        for tid, tok in added.items():
            if getattr(tok, "special", False):
                out.add(int(tid))
    return out


def build_from_tokenizer(tokenizer: Any) -> dict[str, Any]:
    """Build a rare table from a loaded HF / tokenizers / vLLM tokenizer."""
    # Prefer the rust backend when present (same path as offline tokenizer.json).
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "id_to_token") and hasattr(backend, "get_vocab_size"):
        tokenizer = backend
    bd = _byte_decoder()
    V = _vocab_size(tokenizer)
    special = _special_ids(tokenizer)
    chunks: list[bytes] = []
    for i in range(V):
        p = _token_piece(tokenizer, i)
        try:
            b = bytes(bd[c] for c in p)
        except KeyError:
            b = p.encode("utf-8")
        chunks.append(b)
    return _assemble(chunks, special)


def resolve_think_end_id(tokenizer: Any | None, think_end: str) -> int | None:
    """Map config ``think_end`` to a token id (or None for chat / no boundary)."""
    raw = (think_end or "auto").strip().lower()
    if raw in ("none", "null", ""):
        return None
    if raw != "auto":
        return int(raw)
    if tokenizer is None:
        return None
    text = "</think>"
    # Prefer convert_tokens_to_ids when the piece is a single special token.
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        tid = convert(text)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is not None and tid != -1 and tid != unk:
            return int(tid)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
        if isinstance(ids, dict):
            ids = ids.get("input_ids", [])
        if hasattr(ids, "ids"):
            ids = ids.ids
        ids = list(ids) if ids is not None else []
        if len(ids) == 1:
            return int(ids[0])
    return None


_CACHE: dict[str, dict[str, Any]] = {}


def register(key: str, table: dict[str, Any]) -> str:
    """Cache an in-memory table under ``key`` for :func:`load` / RareDetector."""
    if "bytes" not in table:
        t = dict(table)
        rcls = t["rcls"]
        t["bytes"] = [t["bflat"][t["boff"][i] : t["boff"][i + 1]].tobytes() for i in range(len(rcls))]
        table = t
    _CACHE[str(key)] = table
    try:
        from . import rare_ref

        rare_ref.invalidate_cats(str(key))
    except Exception:
        pass
    return str(key)


def load(corpus: str) -> dict[str, Any]:
    """Load a warmup-registered in-memory table, or a ``*.npz`` from offline ``main``."""
    key = corpus if isinstance(corpus, str) else str(corpus)
    if key in _CACHE:
        return _CACHE[key]
    if key.endswith(".npz"):
        if not os.path.isfile(key):
            raise FileNotFoundError(f"rare_chars table not found: {key}")
        z = np.load(key)
        table = {k: z[k] for k in z.files}
        return register(key, table)
    raise FileNotFoundError(
        f"rare_chars table {key!r} is not registered. "
        "Enable rare_chars so model warmup builds it from the served tokenizer "
        "(register_rare_table), or load a .npz produced by rare_vocab.main."
    )


def main():
    ap = argparse.ArgumentParser(description="build the rare_chars table of one byte-level BPE tokenizer")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the served model")
    ap.add_argument("--out", required=True, help="output .npz")
    a = ap.parse_args()
    t = build(a.tokenizer)
    np.savez_compressed(a.out, **{k: t[k] for k in ("rcls", "lead", "first", "last", "u8", "boff", "bflat")})
    print(
        a.out,
        "vocab",
        len(t["rcls"]),
        "partial",
        int(t["u8"].sum()),
        {CLASSES[k]: int(v) for k, v in zip(*np.unique(t["rcls"], return_counts=True))},
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
