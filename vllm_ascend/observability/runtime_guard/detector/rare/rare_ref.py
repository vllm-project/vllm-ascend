#!/usr/bin/env python3
"""Reference tables of the rare_chars detector rc3 (runs/explore/rc3_report.md): how normal a token is, alone and next to the
character before / after it, in ordinary model output. Static per vocabulary; online the detector only indexes them.
  fcat, lcat   int8 [V]       category (CATS) of the token's first / last character (the token decoded on its own)
  n            int64 [V]      occurrences of the token in the reference stream
  npv, nnx     int64 [V, K]   occurrences right after a character of category k (the previous token's last character; 'none'
                              at the start) / right before one (the next token's first character; 'none' at the end)
  N            int            reference tokens
Derived per (token, category), log10, add-0.5 / add-one smoothed (rare.py features):
  u  = (n + 0.5) / N                    the token's frequency
  jp = (npv + 0.5) / N, jn = (nnx + 0.5) / N
  cp = (npv + 1) / (n + K), cn = (nnx + 1) / (n + K)     how unusual the context is for this token
Shipped tables: tables/{qwen3,glm}_rcref.npz (byte copies of the evaluated ones, MANIFEST.json), built in the research repo
(detector/rare_ref.py build_from_cache) from the negative records of dev (rare_chars whole-scope target 'negative', every arm
and fold). They need labelled model output, so this package only loads them. DeepSeek has no dev records, so no table: the
reference features are neutral there.
"""
import os
import unicodedata as U

import numpy as np

from .rare_vocab import TABLES, load as load_rare, script

CATS = ['none', 'space', 'newline', 'lower', 'upper', 'digit', 'punct', 'bracket', 'asym', 'han', 'cjkpunct', 'oletter', 'osym',
        'kana_hangul', 'backslash', 'under']
K = len(CATS)
GATE_C = -2.5                     # the learned score is also evaluated where cp or cn is below this (detector/rare.py gb_gate)


def charcat(ch):
    """category of one character (CATS)"""
    if not ch:
        return 0
    if ch in '\n\r':
        return 2
    if ch.isspace():
        return 1
    o = ord(ch)
    if o < 128:
        if 'a' <= ch <= 'z':
            return 3
        if 'A' <= ch <= 'Z':
            return 4
        if '0' <= ch <= '9':
            return 5
        if ch in '.,;:!?':
            return 6
        if ch in '()[]{}"\'`':
            return 7
        if ch == '\\':
            return 14
        if ch == '_':
            return 15
        return 8
    s = script(ch)
    if s == 'han':
        return 9
    if s in ('kana', 'hangul'):
        return 13
    if s == 'latin':
        return 4 if ch.isupper() else 3
    if s in ('greek', 'cyrillic', 'arabic', 'hebrew', 'thai', 'indic', 'other_letter'):
        return 11
    if 0x3000 <= o < 0x3040 or 0xFF00 <= o < 0xFF66 or U.category(ch).startswith('P'):
        return 10
    return 12


def cat_tables(corpus):
    """first / last character category of every token (decoded on its own; bytes of an incomplete sequence ignored)"""
    T = load_rare(corpus)
    V = len(T['rcls'])
    fc, lc = np.zeros(V, np.int8), np.zeros(V, np.int8)
    for i in range(V):
        s = T['bytes'][i].decode('utf-8', 'ignore')
        fc[i], lc[i] = charcat(s[:1]), charcat(s[-1:])
    return fc, lc


def contexts(tok, starts, ends, fcat, lcat):
    """prevcat / nextcat of a concatenation of records (starts / ends: bool, a record's first / last token)"""
    t = np.asarray(tok, np.int64)
    pc = np.r_[0, lcat[t[:-1]]].astype(np.int64)
    pc[starts] = 0
    nc = np.r_[fcat[t[1:]], 0].astype(np.int64)
    nc[ends] = 0
    return pc, nc


def counts(tok, pc, nc, V):
    """reference counts of a token stream with its contexts"""
    t = np.asarray(tok, np.int64)
    return {'n': np.bincount(t, minlength=V), 'npv': np.bincount(t * K + pc, minlength=V * K).reshape(V, K),
            'nnx': np.bincount(t * K + nc, minlength=V * K).reshape(V, K), 'N': len(t)}


def derive(c, fcat, lcat):
    """counts -> the tables the detector indexes (float32, log10)"""
    n, N = c['n'].astype(np.float64), float(c['N'])
    d = {'fcat': fcat, 'lcat': lcat, 'u': np.log10((n + 0.5) / N).astype(np.float32),
         'jp': np.log10((c['npv'] + 0.5) / N).astype(np.float32), 'jn': np.log10((c['nnx'] + 0.5) / N).astype(np.float32),
         'cp': np.log10((c['npv'] + 1) / (n[:, None] + K)).astype(np.float32),
         'cn': np.log10((c['nnx'] + 1) / (n[:, None] + K)).astype(np.float32)}
    d['lcat_s'] = np.r_[lcat, 0].astype(np.int64)       # index -1 (no neighbour) -> 'none'
    d['fcat_s'] = np.r_[fcat, 0].astype(np.int64)
    d['lowp'] = (d['cp'] < GATE_C).ravel()              # gate bits by token * K + category
    d['lown'] = (d['cn'] < GATE_C).ravel()
    return d


_CACHE = {}


def load(corpus):
    """derived tables of a corpus, or None if it has no reference table"""
    if corpus not in _CACHE:
        path = os.path.join(TABLES, f'{corpus}_rcref.npz')
        if not os.path.exists(path):
            _CACHE[corpus] = None
        else:
            z = np.load(path)
            V = len(z['fcat'])
            npv, nnx = np.zeros(V * K, np.int64), np.zeros(V * K, np.int64)
            npv[z['pv_idx']] = z['pv_cnt']
            nnx[z['nx_idx']] = z['nx_cnt']
            _CACHE[corpus] = derive({'n': z['n'].astype(np.int64), 'npv': npv.reshape(V, K), 'nnx': nnx.reshape(V, K),
                                     'N': int(z['N'])}, z['fcat'], z['lcat'])
    return _CACHE[corpus]


_CATS = {}


def load_cats(corpus):
    """character-category tables only (no counts), for a learned score without reference inputs (rc4, "ref": "cats"): built from
    the tokenizer's rare_vocab table at load (about 0.1 s), so they exist for every corpus, DeepSeek included"""
    if corpus not in _CATS:
        fcat, lcat = cat_tables(corpus)
        _CATS[corpus] = {'fcat': fcat, 'lcat': lcat, 'lcat_s': np.r_[lcat, 0].astype(np.int64),
                         'fcat_s': np.r_[fcat, 0].astype(np.int64)}
    return _CATS[corpus]
