"""Streaming rare_chars detector (DESIGN.md §17): token-level rules on the token id, the top-1 logprob and the generated token's
logprob (what vLLM returns with logprobs=1); no top-20.

    from v35_detector import RareDetector, CONFIG_RC3
    det = RareDetector('qwen3', CONFIG_RC3)                 # table name or a .npz built by rare_vocab.py; config: dict or JSON path
    st = det.start()
    for ids, top1_lp, chosen_lp in chunks:                  # natural-log logprobs, any chunk size
        res = det.feed(st, ids, top1_lp, chosen_lp)         # Result once a strict-tier rule fires, else None
    res = det.finish(st); mon = det.monitor(st)

Per-token features (tables: detector/rare_vocab.py):
  lp1, lpc      log10 of the top-1 / generated token's probability
  fs            share of the token's script class among the preceding letter tokens, (count + 1) / (letters + 1); 1 for non-letters
  cls           script / category class of the token
  bad_utf8      the running UTF-8 decode emits a U+FFFD here that the token does not contain literally (an invalid byte sequence;
                an incomplete sequence at the end of the stream is not one)
  caseless_after_latin   a caseless letter (CJK, Arabic, ...) glued to a Latin letter on its left (no space)
  letter_after_digit     a Latin letter glued to a digit on its left
  answer        at or after the first think-end token (chat mode: always)
rc2 (DESIGN.md §24) adds, computed only when a config's rules use them:
  fsw           share of the token's class among the last 64 letter tokens, (count + 1) / (min(letters, 64) + 1); 1 for non-letters
                (fsw and fse are evaluated lazily: only at the tokens where the rule's other atoms hold)
  ecls          class of the text the running UTF-8 decode emits at this token: the token's own class, except where it is or follows
                a partial byte sequence (a character completed from byte tokens gets that character's class; 'partial' while pending)
  fse           fs over ecls: share among the preceding emitted letters
rc3 (runs/explore/rc3_report.md) adds, computed only when a config's rules use them:
  gap           lp1 - lpc
  nospace       the token does not start with whitespace
  u             log10 frequency of the token in ordinary model output (static per-vocabulary tables, detector/rare_ref.py)
  jp, jn        log10 frequency of the token right after / right before the character category of its actual neighbour
  cp, cn        log10 P(that category | token)
  gb            learned token score: gradient-boosted trees over GB_INPUTS (detector/rare_gb.py), computed only at the tokens
                passing gb_gate (about 1% of tokens); -inf elsewhere
rc4 (runs/explore/rc4_report.md) adds two config keys, both optional (absent = rc3's behaviour):
  "gate": "fs"   the learned score is evaluated only where fs < 0.05 (a letter of a minority class) or bad_utf8
  "ref": "cats"  no reference count tables: only the character categories of each token's first / last character (pcat, ncat),
                 built from the tokenizer at load, so they exist for every corpus (DeepSeek included); needs "gate": "fs", a model
                 without u / cp / cn inputs and no rule on u / jp / jn / cp / cn
A config using any of u ... gb sees each token's right neighbour: it runs one token behind the stream, and every one of its
alerts (any rule) is raised when the token after the triggering one arrives (fired_at = position + 2), or at the end of the
stream for the last token (fired_at = n). A corpus without a reference table (DeepSeek: no dev data) gets neutral values and
gb never fires.
A rule is a conjunction of atoms; a record fires at the first token satisfying a rule (fired_at = tokens consumed). Rules with
"tier": "monitor" only record the monitoring tier's first alert; the monitoring tier includes every rule.
Bounded state: per-class letter counts (token classes and emitted classes), the last 64 letter classes, the previous token's last
character kind, the UTF-8 decoder's pending bytes (<= 3); rc3 adds the last decided token's id and the held-back token (id and
two logprobs).
"""
import codecs
import json
import math
import os

import numpy as np

from . import rare_ref
from .rare_gb import GB
from .rare_vocab import CI, LETTERS, THINK_END, classify, load as load_rare
from .result import Result

LN10 = math.log(10.0)
_LET = np.zeros(len(CI), bool)
_LET[LETTERS] = True
WIN = 64                                                 # fsw window, in letter tokens
_ECLS = {}                                               # emitted text -> class (classify is pure; shared, bounded cache)


LAZY = ('fsw', 'fse', 'u', 'jp', 'jn', 'cp', 'cn', 'gb')
REF = ('u', 'jp', 'jn', 'cp', 'cn')
GB_INPUTS = ['lp1', 'lpc', 'gap', 'cls', 'pcls', 'ncls', 'first', 'last', 'lead', 'pcat', 'ncat', 'fs', 'fsw', 'answer', 'bad_utf8',
             'u', 'cp', 'cn']
GB_CAT = ['cls', 'pcls', 'ncls', 'first', 'last', 'pcat', 'ncat']   # categorical inputs (integer codes; -1 -> 31)
GB_GATE = {'fs': 0.05, 'c': rare_ref.GATE_C}         # gb is scored where fs < 0.05 (a letter of a minority class), bad_utf8,
                                                      # cp < -2.5 or cn < -2.5
_NEG = {}                                             # chunk length -> read-only all -inf array (no learned score in the chunk)


def compile_rule(rule):
    """atoms in evaluation order (lazy features last): (memo key, feature, kind, argument)"""
    out = []
    for a in sorted(rule['and'], key=lambda a: a['f'] in LAZY):
        if 'lt' in a:
            out.append(((a['f'], 'lt', a['lt']), a['f'], 'lt', a['lt']))
        elif 'gt' in a:
            out.append(((a['f'], 'gt', a['gt']), a['f'], 'gt', a['gt']))
        elif 'in' in a:
            table = np.zeros(len(CI), bool)
            table[[CI[c] for c in a['in']]] = True
            out.append(((a['f'], 'in', tuple(a['in'])), a['f'], 'in', table))
        else:
            out.append(((a['f'],), a['f'], 'bool', None))
    return out


class Lazy:
    """a feature computed only at the positions where a rule's other atoms already hold: below(idx, thr) -> value < thr at idx;
    values(idx) -> the values (reference / tests)"""
    def __init__(self, values, n, below=None):
        self.values, self.n = values, n
        self.below = below or (lambda idx, thr: values(idx) < thr)

    def above(self, idx, thr):
        return self.values(idx) > thr

    def all(self):
        return self.values(np.arange(self.n))


def holds(rule, F, memo=None):
    """bool per token: every atom of the rule holds (F = feature arrays; memo shares atoms across the rules of one chunk;
    rule = a config rule or its compile_rule()). Lazy features are evaluated last, only where the other atoms hold."""
    memo = {} if memo is None else memo
    ok = None
    for key, f, kind, arg in (compile_rule(rule) if isinstance(rule, dict) else rule):
        v = F[f]
        if isinstance(v, Lazy):
            idx = np.flatnonzero(ok) if ok is not None else np.arange(v.n)
            t = v.below(idx, arg) if kind == 'lt' else v.above(idx, arg)
            if not t.any():
                return np.zeros(v.n, bool)               # lazy atoms come last: the rule holds nowhere in the chunk
            m = np.zeros(v.n, bool)
            m[idx[t]] = True
        else:
            m = memo.get(key)
            if m is None:
                m = memo[key] = (F[f] < arg if kind == 'lt' else F[f] > arg if kind == 'gt' else arg[F[f]] if kind == 'in'
                                 else F[f])
        ok = m if ok is None else ok & m                 # never written in place: a memo array may be returned as is
        if not ok.any():
            break
    return ok


def describe(rule):
    def one(a):
        if 'lt' in a:
            return f"{a['f']}<{a['lt']}"
        if 'gt' in a:
            return f"{a['f']}>{a['gt']}"
        if 'in' in a:
            return f"{a['f']}:{'|'.join(a['in'])}"
        return a['f']
    return ' & '.join(one(a) for a in rule['and'])


def _bad_utf8(dec, tok, T, ecls=None):
    """bool per token; dec carries the decoder state across calls. ecls (the token classes, int array) is overwritten in place
    with the emitted text's class wherever the decoder had to look (a partial token, or a token after pending bytes)."""
    out = np.zeros(len(tok), bool)
    u8, by = T['u8'], T['bytes']
    need, start = u8[tok], 0
    if not dec.getstate()[0]:
        if not need.any():
            return out                                   # complete, valid tokens with nothing pending decode cleanly
        start = int(np.argmax(need))
    for i, t in enumerate(tok[start:].tolist(), start):
        if not u8[t] and not dec.getstate()[0]:
            continue
        b = by[t]
        s = dec.decode(b)
        out[i] = s.count('�') > b.decode('utf-8', 'ignore').count('�')
        if ecls is not None:
            if s not in _ECLS:
                if len(_ECLS) >= 1 << 16:
                    _ECLS.clear()
                _ECLS[s] = CI[classify(s)] if s else CI['partial']
            ecls[i] = _ECLS[s]
    return out


def _share(cls, isl, counts, letters):
    """(count + 1) / (letters + 1) per letter token (1 elsewhere); updates counts in place, returns (share, new letter total)"""
    n = len(cls)
    fs = np.ones(n)
    li = np.flatnonzero(isl)
    L = len(li)
    if L:
        lc = cls[li]
        c0 = int(lc[0])
        if (lc == c0).all():                             # one class in the chunk (the common case): no per-class pass
            k = np.arange(1, L + 1)
            fs[li] = (counts[c0] + k) / (letters + k)    # (same-class letters before + 1) / (letters before + 1)
            counts[c0] += L
        else:
            tot = letters + np.arange(1, L + 1)          # letters before each letter token, + 1
            for c in np.unique(lc).tolist():
                m = lc == c
                cm = np.cumsum(m)
                fs[li[m]] = (counts[c] + cm[m]) / tot[m]
                counts[c] += int(cm[-1])
        letters += L
    return fs, letters


def _window_share(cls, isl, state):
    """Lazy: share of a letter token's class among the last WIN letter tokens before it (1 for non-letters); state['ring'] keeps
    the last WIN letter classes and is updated here, eagerly"""
    hist, lc = state['ring'], cls[isl]
    state['ring'] = lc[-WIN:] if len(lc) >= WIN else np.concatenate([hist, lc])[-WIN:]

    def values(idx):
        if len(idx) == 1:                                # one position (the learned score's input): count directly
            i = int(idx[0])
            if not isl[i]:
                return np.ones(1)
            nb = int(isl[:i].sum())                      # letters before it in the chunk
            j = len(hist) + nb
            lo = max(j - WIN, 0)
            c = cls[i]
            cnt = int((hist[lo:] == c).sum()) + int((lc[max(lo - len(hist), 0):nb] == c).sum())
            return np.array([(cnt + 1) / (j - lo + 1)])
        out = np.ones(len(idx))
        L = isl[idx]
        if L.any():
            seq = np.concatenate([hist, lc])
            j = (len(hist) + np.cumsum(isl) - isl)[idx]        # index in seq of the token's letter = letters before it
            lo = np.maximum(j - WIN, 0)
            c = cls[idx]
            for cc in np.unique(c[L]).tolist():
                C = np.r_[0, np.cumsum(seq == cc)]
                sel = L & (c == cc)
                out[sel] = (C[j[sel]] - C[lo[sel]] + 1) / (j[sel] - lo[sel] + 1)
        return out
    return Lazy(values, len(cls))


def _emitted_share(ecls, state):
    """Lazy: fs over the emitted classes; state counts are updated here, eagerly"""
    isle = _LET[ecls]
    c0, l0 = state['counts_e'].copy(), state['letters_e']
    nlet = int(isle.sum())
    if nlet:
        state['counts_e'] += np.bincount(ecls[isle], minlength=len(CI))
        state['letters_e'] += nlet

    def values(idx, thr=None):
        out = np.ones(len(idx))
        c, L = ecls[idx], isle[idx]
        if L.any():
            pre_l = None
            for cc in np.unique(c[L]).tolist():
                if thr is not None and (c0[cc] + 1) / (l0 + nlet + 1) >= thr:
                    continue                             # the share cannot drop below thr anywhere in this chunk
                if pre_l is None:
                    pre_l = l0 + (np.cumsum(isle) - isle)[idx]
                m = (ecls == cc) & isle
                pre = c0[cc] + (np.cumsum(m) - m)[idx]
                sel = L & (c == cc)
                out[sel] = (pre[sel] + 1) / (pre_l[sel] + 1)
        return out
    return Lazy(values, len(ecls), lambda idx, thr: values(idx, thr) < thr)


def _ctx_features(F, tok, prv, nxt, T, ref, gbm, need):
    """rc3 features of a chunk; prv / nxt: the previous / next token of each (-1 before the start / after the end of the stream).
    All lazy: a reference feature is looked up, and the learned score computed, only where a rule's other atoms hold."""
    m = len(tok)
    F['_ctx'] = (tok, prv, nxt, T, ref)
    if ref is None:                                      # no reference table for this corpus: neutral values, gb never fires
        zero = Lazy(lambda idx: np.zeros(len(idx)), m)
        for k in REF:
            F[k] = zero
        F['gb'] = Lazy(lambda idx: np.full(len(idx), -np.inf), m)
        return
    pcat, ncat = ref['lcat_s'][prv], ref['fcat_s'][nxt]  # index -1 = the 'none' sentinel
    F['pcat'], F['ncat'] = pcat, ncat
    F['_gate'] = getattr(gbm, 'gate', 'full')
    for k in need:
        if k.startswith('ref:'):
            k = k[4:]
            cat = None if k == 'u' else pcat if k in ('jp', 'cp') else ncat
            F[k] = Lazy(lambda idx, k=k, cat=cat: (ref[k][tok[idx]] if cat is None else ref[k][tok[idx], cat[idx]]).astype(np.float64), m)
    if gbm is not None:
        F['gb'] = _GBScore(F, gbm, m)


def _neg(m):
    """read-only all -inf array of length m (shared)"""
    if m not in _NEG:
        _NEG[m] = np.full(m, -np.inf)
        _NEG[m].flags.writeable = False
    return _NEG[m]


class _GBScore(Lazy):
    """the learned score of a chunk: the gate is computed on first use, each gated position is scored at most once"""
    def __init__(self, F, gbm, m):
        self.F, self.gbm, self.n, self.gate = F, gbm, m, None
        self.below = lambda idx, thr: self.values(idx) < thr

    def nothing(self):
        """no gated token in the chunk (about half the chunks): a rule using gb cannot hold"""
        if self.gate is None:
            self.gate = gb_gate_mask(self.F)
            self.empty = not self.gate.any()
            if not self.empty:
                self.score = np.full(self.n, -np.inf)
                self.todo = self.gate.copy()
        return self.empty

    def values(self, idx):
        if self.nothing():
            return _neg(self.n)[:len(idx)]
        new = idx[self.todo[idx]]
        if len(new):
            X = gb_inputs(self.F, new, self.gbm.quant, self.gbm.uses_fsw)
            self.score[new] = self.gbm.score(X if self.gbm.cols is None else X[:, self.gbm.cols])
            self.todo[new] = False
        return self.score[idx]


def gb_gate_mask(F):
    """where the learned score is evaluated: fs < 0.05 (a letter of a minority class), bad_utf8, or a context the token rarely
    has on its left / right (cp / cn < -2.5)"""
    if F.get('_gate') == 'fs':                           # rc4: minority letters and invalid UTF-8 only
        return (F['fs'] < GB_GATE['fs']) | F['bad_utf8']
    tok, prv, nxt, T, ref = F['_ctx']
    t16 = tok * rare_ref.K
    g = ref['lowp'][t16 + F['pcat']]
    g |= ref['lown'][t16 + F['ncat']]
    g |= F['fs'] < GB_GATE['fs']
    g |= F['bad_utf8']
    return g


def gb_gate(F):
    return np.flatnonzero(gb_gate_mask(F))


_TOKF = ('cls', 'first', 'last', 'lead', 'u')


def _tokf(T, ref):
    """per-token inputs in one table (cached in the reference dict): rcls, first, last, lead, u; the extra last row (index -1, no
    neighbour) has class code 31"""
    if '_tokf' not in ref:
        V = len(T['rcls'])
        tf = np.zeros((V + 1, len(_TOKF)))
        tf[:V] = np.column_stack([T['rcls'], T['first'], T['last'], T['lead'], ref['u'][:V] if 'u' in ref else np.zeros(V)])
        tf[V, 0] = 31
        ref['_tokf'] = tf
    return ref['_tokf']


def gb_inputs(F, idx, quant=0.0, fsw=True):
    """GB_INPUTS at positions idx (float64 [len(idx), len(GB_INPUTS)]; categorical codes -1 -> 31); quant > 0 rounds lp1, lpc and
    gap to that step (log10), so the trees cannot split on tiny logprob differences; fsw=False leaves that (costlier) column 1"""
    tok, prv, nxt, T, ref = F['_ctx']
    tf = _tokf(T, ref)
    if len(idx) == 1:                                    # one position (the common case online): scalar reads
        i = int(idx[0])
        t, pc, nc = int(tok[i]), int(F['pcat'][i]), int(F['ncat'][i])
        l1, lc = float(F['lp1'][i]), float(F['lpc'][i])
        x = np.empty((1, len(GB_INPUTS)))
        r = x[0]
        r[_GBI['tokf']] = tf[t]
        r[_GBI['pcls']], r[_GBI['ncls']] = tf[int(prv[i]), 0], tf[int(nxt[i]), 0]
        r[_GBI['pcat']], r[_GBI['ncat']] = pc, nc
        r[_GBI['cp']], r[_GBI['cn']] = (ref['cp'][t, pc], ref['cn'][t, nc]) if 'cp' in ref else (0.0, 0.0)
        p3 = np.array((l1, lc, l1 - lc))
        r[_GBI['prob']] = np.round(p3 / quant) * quant if quant else p3
        r[_GBI['fs']], r[_GBI['answer']], r[_GBI['bad_utf8']] = F['fs'][i], F['answer'][i], F['bad_utf8'][i]
        r[_GBI['fsw']] = F['fsw'].values(idx)[0] if fsw else 1.0
        return x
    t = tok[idx]
    X = np.empty((len(idx), len(GB_INPUTS)))
    X[:, _GBI['tokf']] = tf[t]
    X[:, _GBI['pcls']] = tf[prv[idx], 0]
    X[:, _GBI['ncls']] = tf[nxt[idx], 0]
    pc, nc = F['pcat'][idx], F['ncat'][idx]
    X[:, _GBI['pcat']], X[:, _GBI['ncat']] = pc, nc
    X[:, _GBI['cp']], X[:, _GBI['cn']] = (ref['cp'][t, pc], ref['cn'][t, nc]) if 'cp' in ref else (0.0, 0.0)
    l1, lc = F['lp1'][idx], F['lpc'][idx]
    P = np.column_stack((l1, lc, l1 - lc))
    X[:, _GBI['prob']] = np.round(P / quant) * quant if quant else P
    X[:, _GBI['fs']] = F['fs'][idx]
    X[:, _GBI['fsw']] = F['fsw'].values(idx) if fsw else 1.0
    X[:, _GBI['answer']] = F['answer'][idx]
    X[:, _GBI['bad_utf8']] = F['bad_utf8'][idx]
    return X


_GBI = {k: GB_INPUTS.index(k) for k in GB_INPUTS}
_GBI.update(tokf=[GB_INPUTS.index(k) for k in _TOKF], prob=[GB_INPUTS.index(k) for k in ('lp1', 'lpc', 'gap')])
assert set(GB_INPUTS) == set(_TOKF) | {'pcls', 'ncls', 'pcat', 'ncat', 'cp', 'cn', 'lp1', 'lpc', 'gap', 'fs', 'fsw', 'answer',
                                      'bad_utf8'}


def features(tok, top1_lp, chosen_lp, T, think_end, state=None, need=(), nxt=None, ref=None, gbm=None, prv=None):
    """per-token features of a chunk; state (dict) carries counts / previous token / decoder / answer across chunks;
    need: extra rc2 / rc3 features to compute ('fsw', 'ecls' -> ecls and fse, 'gap', 'nospace', 'ref' -> rc3 context features);
    prv / nxt: the previous / next token of each token (-1 outside the stream), for 'ref'; ref / gbm: rare_ref tables / GB model"""
    tok = np.asarray(tok, np.int64)
    n = len(tok)
    if state is None:
        state = new_state(need)
    cls = T['rcls'][tok]
    isl = _LET[cls]
    fs, state['letters'] = _share(cls, isl, state['counts'], state['letters'])
    first, last, lead = T['first'][tok], T['last'][tok], T['lead'][tok]
    prev_last = np.empty(n, last.dtype)
    if n:
        prev_last[0] = state['prev_last']
        prev_last[1:] = last[:-1]
    if n:
        state['prev_last'] = int(last[-1])
    let = (prev_last == 1) | (prev_last == 2)
    answer = np.full(n, state['answer'])
    if not state['answer'] and think_end is not None:
        hit = np.flatnonzero(tok == think_end)
        if len(hit):
            answer[hit[0]:] = True
            state['answer'] = True
    elif think_end is None:
        answer[:] = True
        state['answer'] = True
    ecls = cls.astype(np.int64) if 'ecls' in need else None
    F = {'lp1': np.asarray(top1_lp, np.float64) / LN10, 'lpc': np.asarray(chosen_lp, np.float64) / LN10, 'fs': fs, 'cls': cls,
         'bad_utf8': _bad_utf8(state['dec'], tok, T, ecls),
         'caseless_after_latin': (first == 5) & ~lead & let,
         'letter_after_digit': (((first == 1) | (first == 2)) & ~lead & (prev_last == 3)) if 'ref' not in need or 'lad' in need
                               else None,
         'answer': answer}
    if 'fsw' in need:
        F['fsw'] = _window_share(cls, isl, state)
    if ecls is not None:
        F['ecls'] = ecls
        F['fse'] = _emitted_share(ecls, state)
    if 'gap' in need:
        F['gap'] = F['lp1'] - F['lpc']
    if 'nospace' in need:
        F['nospace'] = ~lead
    if 'ref' in need:
        F['first'], F['last'], F['lead'] = first, last, lead
        _ctx_features(F, tok, np.asarray(prv, np.int64), np.asarray(nxt, np.int64), T, ref, gbm, need)
    return F


def new_state(need=()):
    st = {'counts': np.zeros(len(CI), np.int64), 'letters': 0, 'prev_last': 0, 'answer': False,
          'dec': codecs.getincrementaldecoder('utf-8')(errors='replace')}
    if 'fsw' in need:
        st['ring'] = np.zeros(0, np.int8)
    if 'ecls' in need:
        st.update(counts_e=np.zeros(len(CI), np.int64), letters_e=0)
    return st


def needs(rules):
    """extra features a rule set uses"""
    used = {a['f'] for r in rules for a in r['and']}
    if 'gb' in used:
        used |= {'fsw'}                                   # an input of the learned score
    return tuple(x for x, fs in (('fsw', {'fsw'}), ('ecls', {'ecls', 'fse'}), ('gap', {'gap'}), ('nospace', {'nospace'}),
                                 ('ref', set(REF) | {'gb'}), ('lad', {'letter_after_digit'})) if used & fs) + \
        tuple('ref:' + k for k in REF if k in used)


class _St:
    def __init__(self, need=()):
        self.t = 0
        self.f = new_state(need)
        self.result = None
        self.monitor = None
        self.done = False


class _StCtx(_St):
    """state of a config with context features (rc3)"""
    def __init__(self, need=()):
        super().__init__(need)
        self.prev = -1                                   # the last decided token, and the token not yet decided (id, lp1, lpc),
        self.hold = None                                 # which waits for its right neighbour


def _repo_path(path):
    # configs name their model as the research repo stores it ('detector/tables/<file>'); the package keeps it in tables/
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables', os.path.basename(path))


class RareDetector:
    def __init__(self, corpus, config, think_end='auto'):
        self.T = load_rare(corpus)
        if think_end == 'auto' and corpus not in THINK_END:
            raise ValueError(f"think_end='auto' needs a shipped table ({sorted(THINK_END)}); pass the '</think>' id or None")
        self.think_end = THINK_END[corpus] if think_end == 'auto' else think_end
        self.cfg = json.load(open(config)) if isinstance(config, str) else config
        self.rules = self.cfg['rules']
        self.need = needs(self.rules)
        self.compiled = [compile_rule(r) for r in self.rules]
        self.strict = [r.get('tier', 'intercept') != 'monitor' for r in self.rules]
        self.has_gb = [any(a['f'] == 'gb' for a in r['and']) for r in self.rules]
        self.look = 'ref' in self.need                   # context features: one token behind the stream
        self.gate = self.cfg.get('gate', 'full')
        cats = self.cfg.get('ref') == 'cats'
        if self.gate not in ('full', 'fs') or self.cfg.get('ref', 'counts') not in ('counts', 'cats'):
            raise ValueError('unknown "gate" / "ref"')
        if cats and (self.gate != 'fs' or any(a['f'] in REF for r in self.rules for a in r['and'])):
            raise ValueError('"ref": "cats" needs "gate": "fs" and no rule on a reference feature')
        self.ref = (rare_ref.load_cats(corpus) if cats else rare_ref.load(corpus)) if self.look else None
        uses_gb = any(a['f'] == 'gb' for r in self.rules for a in r['and'])
        if uses_gb and 'gb' not in self.cfg:
            raise ValueError('a rule uses gb but the config names no model ("gb": path)')
        self.gbm = GB.load(_repo_path(self.cfg['gb'])) if uses_gb and self.ref is not None else None
        if self.gbm is not None:                         # the model's inputs, a subset of GB_INPUTS in any order
            if cats and set(self.gbm.inputs) & {'u', 'cp', 'cn'}:
                raise ValueError('"ref": "cats": the learned score must not use u / cp / cn')
            cols = [GB_INPUTS.index(k) for k in self.gbm.inputs]
            self.gbm.cols = None if cols == list(range(len(GB_INPUTS))) else np.array(cols)
            self.gbm.uses_fsw = 'fsw' in self.gbm.inputs
            self.gbm.gate = self.gate

    def start(self):
        return _StCtx(self.need) if self.look else _St(self.need)

    def feed(self, st, tokens, top1_lp, chosen_lp):
        if st.done or not len(tokens):
            return st.result
        if not self.look:
            self._step(st, tokens, top1_lp, chosen_lp, None, None, 0)
            return st.result
        n = len(tokens)
        bt, b1, bc = np.empty(n + 2, np.int64), np.empty(n + 1), np.empty(n + 1)   # [previous, held, chunk...]
        h = 0 if st.hold is None else 1                  # first chunk: nothing held yet
        bt[0] = st.prev
        if h:
            bt[1], b1[0], bc[0] = st.hold
        bt[1 + h:1 + h + n], b1[h:h + n], bc[h:h + n] = tokens, top1_lp, chosen_lp
        m = n + h                                        # tokens in the buffer after `previous`
        st.hold = (int(bt[m]), float(b1[m - 1]), float(bc[m - 1]))
        if m > 1:                                        # decide bt[1:m]: previous tokens bt[0:m-1], next tokens bt[2:m+1]
            st.prev = int(bt[m - 1])
            self._step(st, bt[1:m], b1[:m - 1], bc[:m - 1], bt[:m - 1], bt[2:m + 1], 1)
        return st.result

    def _step(self, st, tokens, top1_lp, chosen_lp, prv, nxt, lag):
        """decide the tokens of a chunk (lag: their alerts are raised `lag` tokens later, when the next token is known)"""
        F = features(tokens, top1_lp, chosen_lp, self.T, self.think_end, st.f, self.need, nxt, self.ref, self.gbm, prv)
        first, memo = {}, {}
        for i, r in enumerate(self.compiled):
            if st.monitor is not None and not self.strict[i]:
                continue                                 # the monitoring tier has alerted: only strict rules still matter
            if self.has_gb[i] and self.gbm is not None and F['gb'].nothing():
                continue                                 # no gated token in the chunk
            h = holds(r, F, memo)
            if h.any():
                first[i] = int(np.argmax(h))
        if first:
            j = min(first.values())
            if st.monitor is None:
                i = min(k for k, v in first.items() if v == j)          # ties go to the earlier rule
                st.monitor = Result(True, st.t + j + 1 + lag, describe(self.rules[i]))
            strict = {k: v for k, v in first.items() if self.strict[k]}
            if strict:
                j = min(strict.values())
                i = min(k for k, v in strict.items() if v == j)
                st.result = Result(True, st.t + j + 1 + lag, describe(self.rules[i]))
                st.done = True
        st.t += len(tokens)

    def finish(self, st):
        if self.look and not st.done and st.hold is not None:   # the last token: decided at the end of the stream
            h, st.hold = st.hold, None
            self._step(st, np.array([h[0]], np.int64), np.array([h[1]]), np.array([h[2]]), np.array([st.prev], np.int64),
                       np.array([-1], np.int64), 0)
        st.done = True
        return st.result if st.result is not None else Result()

    @staticmethod
    def monitor(st):
        return st.monitor if st.monitor is not None else Result()


def record_features(det, tok, top1_lp, chosen_lp, ref='auto'):
    """whole-record features as the detector computes them (ref: override the reference tables, e.g. cross-fitted ones)"""
    tok = np.asarray(tok, np.int64)
    ctx = {'prv': np.r_[-1, tok[:-1]], 'nxt': np.r_[tok[1:], -1]} if det.look else {}
    return features(tok, top1_lp, chosen_lp, det.T, det.think_end, need=det.need, ref=det.ref if ref == 'auto' else ref,
                    gbm=det.gbm, **ctx)


def decide(tok, top1_lp, chosen_lp, corpus, config, think_end='auto', tier='intercept'):
    """whole-record reference: (fired_at, rule description) of the first alert of a tier, (-1, '') if none. The earliest
    deciding position wins, ties go to the earlier rule; with look-ahead an alert at position p is raised at min(p + 2, n)."""
    det = RareDetector(corpus, config, think_end)
    F = record_features(det, tok, top1_lp, chosen_lp)
    n = len(tok)
    best, pos = (-1, ''), None
    for r in det.rules:
        if tier == 'intercept' and r.get('tier', 'intercept') == 'monitor':
            continue
        h = np.flatnonzero(holds(r, F))
        if len(h) and (pos is None or h[0] < pos):
            pos = int(h[0])
            best = (min(pos + 2, n) if det.look else pos + 1, describe(r))
    return best
