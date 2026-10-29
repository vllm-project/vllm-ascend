#!/usr/bin/env python3
"""Per-tokenizer tables of the rare_chars detector (built once offline; online only indexed by token id). Every token is
decoded on its own from its byte-level BPE bytes (special tokens: their literal text).
  rcls        int8   script / category class (CLASSES); a token whose letters span two scripts is 'mixed', an incomplete UTF-8
                     byte sequence is 'partial'
  lead        bool   starts with whitespace
  first, last int8   kind of the first / last character: 0 none or space, 1 lower, 2 upper, 3 digit, 4 punctuation / symbol,
                     5 caseless letter (CJK, Arabic, ...)
  u8          bool   not a complete, valid UTF-8 string on its own (the running decoder must look at it)
  boff, bflat        the token's bytes: bflat[boff[i]:boff[i + 1]]
Shipped tables: tables/{qwen3,glm,dsv4}_rare.npz (byte copies of the evaluated ones, MANIFEST.json). A new tokenizer:
  python3 -m v35_detector.rare_vocab --tokenizer /path/to/tokenizer.json --out my_model_rare.npz
then RareDetector('/path/to/my_model_rare.npz', config, think_end=<'</think>' id, or None in chat mode>); a vocabulary
without a reference table (rare_ref.py) gets neutral context features, and the learned score never fires.
"""
import argparse
import json
import os
import sys
import unicodedata as U

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, 'tables')
# '</think>' token id of the corpus's generation mode. DeepSeek chat mode: None.
THINK_END = {'qwen3': 151668, 'glm': 154842, 'dsv4': None}


def _byte_decoder():
    """inverse of GPT-2 bytes_to_unicode (byte-level BPE alphabet -> byte)"""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


CLASSES = ['space_punct', 'latin', 'digit', 'han', 'kana', 'hangul', 'cyrillic', 'greek', 'arabic', 'hebrew', 'thai', 'indic',
           'other_letter', 'symbol', 'emoji', 'partial', 'replacement', 'control', 'mixed', 'combining', 'special']
CI = {n: i for i, n in enumerate(CLASSES)}
LETTERS = [CI[c] for c in ('latin', 'han', 'kana', 'hangul', 'cyrillic', 'greek', 'arabic', 'hebrew', 'thai', 'indic',
                           'other_letter', 'mixed')]
NONLATIN_LETTERS = [CI[c] for c in ('cyrillic', 'arabic', 'thai', 'hangul', 'hebrew', 'indic', 'kana', 'other_letter')]


def script(ch):
    o = ord(ch)
    if ch == '�':
        return 'replacement'
    cat = U.category(ch)
    if cat in ('Cc', 'Cf', 'Co', 'Cs', 'Cn') and ch not in '\n\r\t':
        return 'control'
    if cat.startswith('M'):
        return 'combining'
    if cat == 'Nd':
        return 'digit' if o < 128 else 'other_letter'
    if cat.startswith('L'):
        if o < 0x250 or 0x1E00 <= o < 0x1F00:
            return 'latin'
        if 0x370 <= o < 0x400 or 0x1F00 <= o < 0x2000:
            return 'greek'
        if 0x400 <= o < 0x530:
            return 'cyrillic'
        if 0x590 <= o < 0x600:
            return 'hebrew'
        if 0x600 <= o < 0x700 or 0x750 <= o < 0x780 or 0xFB50 <= o < 0xFF00:
            return 'arabic'
        if 0x900 <= o < 0xE00:
            return 'indic'
        if 0xE00 <= o < 0xE80:
            return 'thai'
        if 0x3040 <= o < 0x3100 or 0x31F0 <= o < 0x3200 or 0xFF66 <= o < 0xFFA0:
            return 'kana'
        if 0xAC00 <= o < 0xD7B0 or 0x1100 <= o < 0x1200 or 0x3130 <= o < 0x3190:
            return 'hangul'
        if 0x3400 <= o < 0xA000 or 0xF900 <= o < 0xFB00 or 0x20000 <= o < 0x31350:
            return 'han'
        if 0xFF21 <= o < 0xFF5B:
            return 'latin'
        return 'other_letter'
    if cat == 'So' and (o >= 0x1F000 or 0x2600 <= o < 0x2800):
        return 'emoji'
    if cat in ('So', 'Sm', 'Sc', 'Sk') and o >= 128:
        return 'symbol'
    return 'space_punct'


def classify(s):
    kinds = {script(ch) for ch in s} - {'space_punct'}
    if not kinds:
        return 'space_punct'
    for k in ('replacement', 'control'):
        if k in kinds:
            return k
    letters = kinds - {'digit', 'combining', 'symbol', 'emoji'}
    if len(letters) > 1:
        return 'mixed'
    if letters:
        return letters.pop()
    for k in ('emoji', 'symbol', 'combining', 'digit'):
        if k in kinds:
            return k
    return 'space_punct'


def kind(ch):
    """0 none / space, 1 lower, 2 upper, 3 digit, 4 punctuation / symbol, 5 caseless letter"""
    if not ch or ch.isspace():
        return 0
    if ch.isdigit():
        return 3
    if ch.isalpha():
        return 1 if ch.islower() else 2 if ch.isupper() else 5
    return 4


def build(tokenizer_json):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer_json)
    special = {a['id'] for a in json.load(open(tokenizer_json)).get('added_tokens', []) if a.get('special')}
    bd = _byte_decoder()
    V = tk.get_vocab_size()
    rcls, first, last = np.zeros(V, np.int8), np.zeros(V, np.int8), np.zeros(V, np.int8)
    lead, u8 = np.zeros(V, bool), np.zeros(V, bool)
    chunks = []
    for i in range(V):
        p = tk.id_to_token(i) or ''
        try:
            b = bytes(bd[c] for c in p)
        except KeyError:
            b = p.encode('utf-8')                       # special / added tokens are written literally
        chunks.append(b)
        lead[i] = b[:1] in (b' ', b'\n', b'\t', b'\r')
        try:
            s = b.decode('utf-8')
        except UnicodeDecodeError:
            u8[i] = True
            rcls[i] = CI['special'] if i in special else CI['partial']
            s = b.decode('utf-8', 'ignore')
        else:
            rcls[i] = CI['special'] if (i in special or p is None) else CI[classify(s)]
        first[i], last[i] = kind(s[:1]), kind(s[-1:])
    boff = np.r_[0, np.cumsum([len(b) for b in chunks])].astype(np.int64)
    return {'rcls': rcls, 'lead': lead, 'first': first, 'last': last, 'u8': u8, 'boff': boff,
            'bflat': np.frombuffer(b''.join(chunks), np.uint8)}


_CACHE = {}


def load(corpus):
    """tables of a shipped tokenizer (qwen3 | glm | dsv4) or of a table file built by main() (a path ending in .npz)"""
    if corpus not in _CACHE:
        z = np.load(corpus if corpus.endswith('.npz') else os.path.join(TABLES, f'{corpus}_rare.npz'))
        _CACHE[corpus] = {k: z[k] for k in z.files}
        t = _CACHE[corpus]
        t['bytes'] = [t['bflat'][t['boff'][i]:t['boff'][i + 1]].tobytes() for i in range(len(t['rcls']))]
    return _CACHE[corpus]


def main():
    ap = argparse.ArgumentParser(description='build the rare_chars table of one byte-level BPE tokenizer')
    ap.add_argument('--tokenizer', required=True, help='tokenizer.json of the served model')
    ap.add_argument('--out', required=True, help='output .npz')
    a = ap.parse_args()
    t = build(a.tokenizer)
    np.savez_compressed(a.out, **t)
    print(a.out, 'vocab', len(t['rcls']), 'partial', int(t['u8'].sum()),
          {CLASSES[k]: int(v) for k, v in zip(*np.unique(t['rcls'], return_counts=True))}, file=sys.stderr)


if __name__ == '__main__':
    main()
