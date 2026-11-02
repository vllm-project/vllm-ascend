/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * One rc4 step for many requests, with the GIL released for the whole step.
 *
 * The worker thread can launch NPU kernels while this runs on ActionQueue.
 * Pointers passed in are borrowed numpy buffers; no Python calls in feed_impl.
 *
 * Rule feature ids and GB input kinds must match rare_native.py.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

enum { WIN = 64, DEC_MAX = 4, FEAT_N = 9, IN_N = 15 };

enum {
    FT_LP1 = 0,
    FT_LPC = 1,
    FT_GAP = 2,
    FT_FS = 3,
    FT_FSW = 4,
    FT_CASELESS = 5,
    FT_BAD = 6,
    FT_GB = 7,
    FT_ANSWER = 8
};

enum {
    IN_LP1 = 0,
    IN_LPC = 1,
    IN_GAP = 2,
    IN_CLS = 3,
    IN_PCLS = 4,
    IN_NCLS = 5,
    IN_FIRST = 6,
    IN_LAST = 7,
    IN_LEAD = 8,
    IN_PCAT = 9,
    IN_NCAT = 10,
    IN_FS = 11,
    IN_FSW = 12,
    IN_ANSWER = 13,
    IN_BAD = 14
};

#define MAX_BUF 48

typedef struct {
    int V;
    int ncls;
    int n_in;
    int n_nodes;
    int n_trees;
    int n_atoms;
    int n_rules;
    int answer_always;
    int64_t think_end;
    double ln10;
    double quant;
    double gb_base;
    double gb_fs;
    const int8_t *rcls;
    const int8_t *first;
    const int8_t *last;
    const uint8_t *lead;
    const uint8_t *u8;
    const int64_t *boff;
    const uint8_t *bflat;
    const uint8_t *lit_fffd;
    const int64_t *lcat;
    const int64_t *fcat;
    const uint8_t *is_letter;
    const int8_t *in_kind;
    const int32_t *qfeat;
    const double *qthr;
    const uint8_t *qcat;
    const int64_t *qcmask;
    const int64_t *qrule;
    const int32_t *qstart;
    const double *qleaf;
    const int32_t *qleaf0;
    const int16_t *atom_feat;
    const int8_t *atom_kind;
    const double *atom_thr;
    const int32_t *rule_begin;
    const int32_t *rule_end;
    const uint8_t *rule_strict;
    Py_buffer bufs[MAX_BUF];
    int nbuf;
} Ctx;

static void ctx_release(Ctx *c) {
    int i;
    if (c == NULL) {
        return;
    }
    for (i = 0; i < c->nbuf; i++) {
        PyBuffer_Release(&c->bufs[i]);
    }
    PyMem_Free(c);
}

static void ctx_capsule_destroy(PyObject *capsule) {
    ctx_release((Ctx *)PyCapsule_GetPointer(capsule, "rare_feed_ctx"));
}

static int take(Ctx *c, PyObject *obj, Py_ssize_t nbytes, void **out) {
    Py_buffer *v;
    if (c->nbuf >= MAX_BUF) {
        PyErr_SetString(PyExc_RuntimeError, "rare feed ctx: too many buffers");
        return -1;
    }
    v = &c->bufs[c->nbuf];
    if (PyObject_GetBuffer(obj, v, PyBUF_SIMPLE) < 0) {
        return -1;
    }
    if (!PyBuffer_IsContiguous(v, 'C') || v->len < nbytes) {
        PyBuffer_Release(v);
        PyErr_SetString(PyExc_ValueError, "rare feed ctx: buffer is short or not contiguous");
        return -1;
    }
    c->nbuf++;
    *out = v->buf;
    return 0;
}

static PyObject *make_ctx(PyObject *self, PyObject *args) {
    PyObject *rcls, *first, *last, *lead, *u8, *boff, *bflat, *lit;
    PyObject *lcat, *fcat, *is_letter, *in_kind;
    PyObject *qfeat, *qthr, *qcat, *qcmask, *qrule, *qstart, *qleaf, *qleaf0;
    PyObject *atom_feat, *atom_kind, *atom_thr, *rule_begin, *rule_end, *rule_strict;
    int V, ncls, n_in, n_nodes, n_trees, n_atoms, n_rules, answer_always;
    int64_t think_end;
    double ln10, quant, gb_base, gb_fs;
    Ctx *c;
    PyObject *cap;

    if (!PyArg_ParseTuple(
            args, "iiiiiiiiLddddOOOOOOOOOOOOOOOOOOOOOOOOOO",
            &V, &ncls, &n_in, &n_nodes, &n_trees, &n_atoms, &n_rules, &answer_always,
            &think_end, &ln10, &quant, &gb_base, &gb_fs,
            &rcls, &first, &last, &lead, &u8, &boff, &bflat, &lit,
            &lcat, &fcat, &is_letter, &in_kind,
            &qfeat, &qthr, &qcat, &qcmask, &qrule, &qstart, &qleaf, &qleaf0,
            &atom_feat, &atom_kind, &atom_thr, &rule_begin, &rule_end, &rule_strict)) {
        return NULL;
    }
    if (V < 1 || ncls < 1 || n_in < 1 || n_in > IN_N || n_trees < 1 || n_rules < 1) {
        PyErr_SetString(PyExc_ValueError, "rare feed ctx: bad dimensions");
        return NULL;
    }
    c = (Ctx *)PyMem_Calloc(1, sizeof(Ctx));
    if (c == NULL) {
        return PyErr_NoMemory();
    }
    c->V = V;
    c->ncls = ncls;
    c->n_in = n_in;
    c->n_nodes = n_nodes;
    c->n_trees = n_trees;
    c->n_atoms = n_atoms;
    c->n_rules = n_rules;
    c->answer_always = answer_always;
    c->think_end = think_end;
    c->ln10 = ln10;
    c->quant = quant;
    c->gb_base = gb_base;
    c->gb_fs = gb_fs;

#define TAKE(obj, nbytes, field)                                                                 \
    do {                                                                                         \
        void *_p = NULL;                                                                         \
        if (take(c, (obj), (nbytes), &_p) < 0) {                                                 \
            ctx_release(c);                                                                      \
            return NULL;                                                                         \
        }                                                                                        \
        c->field = _p;                                                                           \
    } while (0)

    TAKE(rcls, (Py_ssize_t)V, rcls);
    TAKE(first, (Py_ssize_t)V, first);
    TAKE(last, (Py_ssize_t)V, last);
    TAKE(lead, (Py_ssize_t)V, lead);
    TAKE(u8, (Py_ssize_t)V, u8);
    TAKE(boff, (Py_ssize_t)(V + 1) * (Py_ssize_t)sizeof(int64_t), boff);
    TAKE(bflat, 0, bflat);
    TAKE(lit, (Py_ssize_t)V, lit_fffd);
    TAKE(lcat, (Py_ssize_t)(V + 1) * (Py_ssize_t)sizeof(int64_t), lcat);
    TAKE(fcat, (Py_ssize_t)(V + 1) * (Py_ssize_t)sizeof(int64_t), fcat);
    TAKE(is_letter, (Py_ssize_t)ncls, is_letter);
    TAKE(in_kind, (Py_ssize_t)n_in, in_kind);
    TAKE(qfeat, (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(int32_t), qfeat);
    TAKE(qthr, (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(double), qthr);
    TAKE(qcat, (Py_ssize_t)n_nodes, qcat);
    TAKE(qcmask, (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(int64_t), qcmask);
    TAKE(qrule, (Py_ssize_t)n_nodes * (Py_ssize_t)sizeof(int64_t), qrule);
    TAKE(qstart, (Py_ssize_t)(n_trees + 1) * (Py_ssize_t)sizeof(int32_t), qstart);
    TAKE(qleaf, 0, qleaf);
    TAKE(qleaf0, (Py_ssize_t)n_trees * (Py_ssize_t)sizeof(int32_t), qleaf0);
    TAKE(atom_feat, (Py_ssize_t)n_atoms * (Py_ssize_t)sizeof(int16_t), atom_feat);
    TAKE(atom_kind, (Py_ssize_t)n_atoms, atom_kind);
    TAKE(atom_thr, (Py_ssize_t)n_atoms * (Py_ssize_t)sizeof(double), atom_thr);
    TAKE(rule_begin, (Py_ssize_t)n_rules * (Py_ssize_t)sizeof(int32_t), rule_begin);
    TAKE(rule_end, (Py_ssize_t)n_rules * (Py_ssize_t)sizeof(int32_t), rule_end);
    TAKE(rule_strict, (Py_ssize_t)n_rules, rule_strict);
#undef TAKE

    cap = PyCapsule_New(c, "rare_feed_ctx", ctx_capsule_destroy);
    if (cap == NULL) {
        ctx_release(c);
        return NULL;
    }
    return cap;
}

static int is_cont(unsigned byte) { return (byte & 0xC0u) == 0x80u; }

static int lead_len(unsigned lead) {
    if (lead >= 0xC2 && lead <= 0xDF) {
        return 2;
    }
    if (lead >= 0xE0 && lead <= 0xEF) {
        return 3;
    }
    if (lead >= 0xF0 && lead <= 0xF4) {
        return 4;
    }
    return 0;
}

/* Second byte of an otherwise well-sized sequence. ED accepts every continuation;
 * surrogate code points are rejected when the three bytes are complete. */
static int second_ok(unsigned lead, unsigned byte) {
    if (lead == 0xE0) {
        return byte >= 0xA0;
    }
    if (lead == 0xF0) {
        return byte >= 0x90;
    }
    if (lead == 0xF4) {
        return byte <= 0x8F;
    }
    return 1;
}

static uint32_t assemble(const unsigned char *p, int n) {
    if (n == 2) {
        return ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
    }
    if (n == 3) {
        return ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
    }
    return ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
           ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
}

/* Incremental UTF-8, errors=replace, non-final. Returns U+FFFD count emitted. */
static int utf8_feed(unsigned char *pend, uint8_t *pn, const unsigned char *bytes, Py_ssize_t n) {
    int emitted = 0;
    Py_ssize_t i = 0;
    while (i < n) {
        unsigned byte = bytes[i];
        int have = (int)(*pn);
        if (have == 0) {
            int need = lead_len(byte);
            if (byte < 0x80) {
                i++;
                continue;
            }
            if (need == 0) {
                emitted++;
                i++;
                continue;
            }
            pend[0] = (unsigned char)byte;
            *pn = 1;
            i++;
            continue;
        }
        if (!is_cont(byte)) {
            /* One replacement for the whole pending prefix, then retry this byte. */
            emitted++;
            *pn = 0;
            continue;
        }
        if (have == 1 && !second_ok(pend[0], byte)) {
            emitted += have + 1;
            *pn = 0;
            i++;
            continue;
        }
        pend[have] = (unsigned char)byte;
        have++;
        *pn = (uint8_t)have;
        i++;
        if (have == lead_len(pend[0])) {
            uint32_t cp = assemble(pend, have);
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                emitted += have;
            } else if (cp == 0xFFFD) {
                emitted++;
            }
            *pn = 0;
        }
    }
    return emitted;
}

static int lowbit(int64_t mask) {
    int idx = 0;
    uint64_t u = (uint64_t)(mask & -mask);
    while (u > 1ull) {
        u >>= 1;
        idx++;
    }
    return idx;
}

static double quantize(double v, double q) {
    if (q == 0.0) {
        return v;
    }
    return nearbyint(v / q) * q;
}

static double score_gb(const Ctx *c, const double *x) {
    double sum = 0.0;
    int t;
    for (t = 0; t < c->n_trees; t++) {
        int64_t mask = -1;
        int j;
        for (j = c->qstart[t]; j < c->qstart[t + 1]; j++) {
            int feat = c->qfeat[j];
            int right;
            if (c->qcat[j]) {
                int code = (int)x[feat];
                if (code < 0) {
                    code = 0;
                }
                if (code > 31) {
                    code = 31;
                }
                right = ((c->qcmask[j] >> code) & 1) == 0;
            } else {
                right = x[feat] > c->qthr[j];
            }
            if (right) {
                mask &= c->qrule[j];
            }
        }
        sum += c->qleaf[c->qleaf0[t] + lowbit(mask)];
    }
    return c->gb_base + sum;
}

typedef struct {
    double lp1, lpc, gap, fs, fsw, gb;
    int caseless, bad, answer;
    int gb_ready;
} Feat;

static int atom_ok(const Feat *f, int feat, int kind, double thr) {
    double v = 0.0;
    switch (feat) {
    case FT_LP1:
        v = f->lp1;
        break;
    case FT_LPC:
        v = f->lpc;
        break;
    case FT_GAP:
        v = f->gap;
        break;
    case FT_FS:
        v = f->fs;
        break;
    case FT_FSW:
        v = f->fsw;
        break;
    case FT_CASELESS:
        v = f->caseless ? 1.0 : 0.0;
        break;
    case FT_BAD:
        v = f->bad ? 1.0 : 0.0;
        break;
    case FT_GB:
        v = f->gb;
        break;
    case FT_ANSWER:
        v = f->answer ? 1.0 : 0.0;
        break;
    default:
        return 0;
    }
    if (kind == 0) {
        return v < thr;
    }
    if (kind == 1) {
        return v > thr;
    }
    return v != 0.0;
}

static void fill_gb(const Ctx *c, Feat *f, int64_t tok, int64_t prv, int64_t nxt, double *x) {
    int i;
    double cls = (double)c->rcls[tok];
    double pcls = prv < 0 ? 31.0 : (double)c->rcls[prv];
    double ncls = nxt < 0 ? 31.0 : (double)c->rcls[nxt];
    double pcat = (double)c->lcat[prv < 0 ? c->V : prv];
    double ncat = (double)c->fcat[nxt < 0 ? c->V : nxt];
    double lp1q = quantize(f->lp1, c->quant);
    double lpcq = quantize(f->lpc, c->quant);
    double gapq = quantize(f->gap, c->quant);
    for (i = 0; i < c->n_in; i++) {
        switch (c->in_kind[i]) {
        case IN_LP1:
            x[i] = lp1q;
            break;
        case IN_LPC:
            x[i] = lpcq;
            break;
        case IN_GAP:
            x[i] = gapq;
            break;
        case IN_CLS:
            x[i] = cls;
            break;
        case IN_PCLS:
            x[i] = pcls;
            break;
        case IN_NCLS:
            x[i] = ncls;
            break;
        case IN_FIRST:
            x[i] = (double)c->first[tok];
            break;
        case IN_LAST:
            x[i] = (double)c->last[tok];
            break;
        case IN_LEAD:
            x[i] = c->lead[tok] ? 1.0 : 0.0;
            break;
        case IN_PCAT:
            x[i] = pcat;
            break;
        case IN_NCAT:
            x[i] = ncat;
            break;
        case IN_FS:
            x[i] = f->fs;
            break;
        case IN_FSW:
            x[i] = f->fsw;
            break;
        case IN_ANSWER:
            x[i] = f->answer ? 1.0 : 0.0;
            break;
        case IN_BAD:
            x[i] = f->bad ? 1.0 : 0.0;
            break;
        default:
            x[i] = 0.0;
            break;
        }
    }
}

typedef struct {
    int64_t *counts;
    int64_t *letters;
    int16_t *prev_last;
    uint8_t *answer;
    int16_t *ring;
    int16_t *ring_len;
    uint8_t *dec_n;
    uint8_t *dec_b;
    int64_t *prev;
    uint8_t *has_hold;
    int64_t *hold_id;
    double *hold_lp1;
    double *hold_lpc;
    int64_t *t;
    uint8_t *done;
    uint8_t *monitor_set;
    int64_t *tokens_seen;
    int16_t *recent_len;
    int64_t *recent_tok;
    double *recent_lp1;
    double *recent_lpc;
    int cap;
    int ncls;
} State;

enum { RECENT_CAP = 16 };

static void push_recent(State *s, int row, int64_t tok, double lp1, double lpc) {
    int n = (int)s->recent_len[row];
    int64_t *dst_tok = s->recent_tok + (size_t)row * RECENT_CAP;
    double *dst_lp1 = s->recent_lp1 + (size_t)row * RECENT_CAP;
    double *dst_lpc = s->recent_lpc + (size_t)row * RECENT_CAP;
    if (n < RECENT_CAP) {
        dst_tok[n] = tok;
        dst_lp1[n] = lp1;
        dst_lpc[n] = lpc;
        s->recent_len[row] = (int16_t)(n + 1);
    } else {
        memmove(dst_tok, dst_tok + 1, (size_t)(RECENT_CAP - 1) * sizeof(int64_t));
        memmove(dst_lp1, dst_lp1 + 1, (size_t)(RECENT_CAP - 1) * sizeof(double));
        memmove(dst_lpc, dst_lpc + 1, (size_t)(RECENT_CAP - 1) * sizeof(double));
        dst_tok[RECENT_CAP - 1] = tok;
        dst_lp1[RECENT_CAP - 1] = lp1;
        dst_lpc[RECENT_CAP - 1] = lpc;
    }
    s->tokens_seen[row] += 1;
}

static int bad_utf8(const Ctx *c, State *s, int row, int64_t tok) {
    Py_ssize_t begin, end, n;
    int emitted;
    if (s->dec_n[row] == 0 && !c->u8[tok]) {
        return 0;
    }
    begin = (Py_ssize_t)c->boff[tok];
    end = (Py_ssize_t)c->boff[tok + 1];
    n = end - begin;
    if (n < 0) {
        n = 0;
    }
    emitted = utf8_feed(s->dec_b + (size_t)row * DEC_MAX, &s->dec_n[row], c->bflat + begin, n);
    return emitted > (int)c->lit_fffd[tok];
}

static void decide(const Ctx *c, State *s, int row, int64_t nxt, int lag, int32_t *hit_row, int16_t *hit_mon,
                   int16_t *hit_strict, int64_t *hit_fired, int *n_hits, int max_hits) {
    int64_t tok = s->hold_id[row];
    int64_t prv = s->prev[row];
    int cls, isl, let, mon_before, mon_i, strict_i, ri;
    int64_t t0;
    Feat f;
    if (tok < 0 || tok >= c->V) {
        return;
    }
    cls = (int)c->rcls[tok];
    isl = (cls >= 0 && cls < c->ncls) ? c->is_letter[cls] : 0;
    f.fs = 1.0;
    if (isl) {
        int64_t prev_c = s->counts[(size_t)row * (size_t)c->ncls + (size_t)cls];
        f.fs = ((double)prev_c + 1.0) / ((double)s->letters[row] + 1.0);
        s->counts[(size_t)row * (size_t)c->ncls + (size_t)cls] = prev_c + 1;
        s->letters[row] += 1;
    }
    let = s->prev_last[row] == 1 || s->prev_last[row] == 2;
    f.caseless = (c->first[tok] == 5) && !c->lead[tok] && let;
    s->prev_last[row] = (int16_t)c->last[tok];
    if (c->answer_always) {
        f.answer = 1;
        s->answer[row] = 1;
    } else {
        f.answer = s->answer[row] ? 1 : 0;
        if (!f.answer && tok == c->think_end) {
            f.answer = 1;
            s->answer[row] = 1;
        }
    }
    f.bad = bad_utf8(c, s, row, tok);
    f.fsw = 1.0;
    if (isl) {
        int rl = (int)s->ring_len[row];
        int cnt = 0;
        int k;
        int16_t *ring = s->ring + (size_t)row * WIN;
        int16_t c16 = (int16_t)cls;
        if (rl > WIN) {
            rl = WIN;
        }
        for (k = 0; k < rl; k++) {
            if (ring[k] == c16) {
                cnt++;
            }
        }
        f.fsw = ((double)cnt + 1.0) / ((double)rl + 1.0);
        if (rl == WIN) {
            memmove(ring, ring + 1, (size_t)(WIN - 1) * sizeof(int16_t));
            ring[WIN - 1] = c16;
        } else {
            ring[rl] = c16;
            s->ring_len[row] = (int16_t)(rl + 1);
        }
    }
    f.lp1 = s->hold_lp1[row] / c->ln10;
    f.lpc = s->hold_lpc[row] / c->ln10;
    f.gap = f.lp1 - f.lpc;
    f.gb = -INFINITY;
    f.gb_ready = 0;
    mon_before = s->monitor_set[row] ? 1 : 0;
    mon_i = -1;
    strict_i = -1;
    for (ri = 0; ri < c->n_rules; ri++) {
        int a, ok = 1;
        if (!c->rule_strict[ri] && mon_before) {
            continue;
        }
        for (a = c->rule_begin[ri]; a < c->rule_end[ri]; a++) {
            int feat = c->atom_feat[a];
            if (feat == FT_GB && !f.gb_ready) {
                if (!(f.fs < c->gb_fs || f.bad)) {
                    f.gb = -INFINITY;
                } else {
                    double x[IN_N];
                    fill_gb(c, &f, tok, prv, nxt, x);
                    f.gb = score_gb(c, x);
                }
                f.gb_ready = 1;
            }
            if (!atom_ok(&f, feat, c->atom_kind[a], c->atom_thr[a])) {
                ok = 0;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        if (!mon_before && mon_i < 0) {
            mon_i = ri;
        }
        if (c->rule_strict[ri] && strict_i < 0) {
            strict_i = ri;
        }
    }
    t0 = s->t[row];
    s->t[row] = t0 + 1;
    if (mon_i >= 0) {
        s->monitor_set[row] = 1;
    }
    if (strict_i >= 0) {
        s->done[row] = 1;
    }
    if ((mon_i >= 0 || strict_i >= 0) && *n_hits < max_hits) {
        int h = *n_hits;
        hit_row[h] = row;
        hit_mon[h] = (int16_t)mon_i;
        hit_strict[h] = (int16_t)strict_i;
        hit_fired[h] = t0 + 1 + lag;
        *n_hits = h + 1;
    }
}

static void apply_token(const Ctx *c, State *s, int row, int64_t tok, double lp1, double lpc, int32_t *hit_row,
                        int16_t *hit_mon, int16_t *hit_strict, int64_t *hit_fired, int *n_hits, int max_hits) {
    if (tok < 0 || tok >= c->V || s->done[row]) {
        return;
    }
    if (!s->has_hold[row]) {
        s->hold_id[row] = tok;
        s->hold_lp1[row] = lp1;
        s->hold_lpc[row] = lpc;
        s->has_hold[row] = 1;
        push_recent(s, row, tok, lp1, lpc);
        return;
    }
    decide(c, s, row, tok, 1, hit_row, hit_mon, hit_strict, hit_fired, n_hits, max_hits);
    s->prev[row] = s->hold_id[row];
    s->hold_id[row] = tok;
    s->hold_lp1[row] = lp1;
    s->hold_lpc[row] = lpc;
    push_recent(s, row, tok, lp1, lpc);
}

static int feed_impl(const Ctx *c, State *s, const int64_t *rows, const int64_t *token_ids, const double *top1,
                     const double *chosen, int n, int finish, int32_t *hit_row, int16_t *hit_mon, int16_t *hit_strict,
                     int64_t *hit_fired) {
    int i, n_hits = 0;
    for (i = 0; i < n; i++) {
        int row = (int)rows[i];
        if (row < 0 || row >= s->cap || s->done[row]) {
            continue;
        }
        if (finish) {
            if (s->has_hold[row]) {
                decide(c, s, row, -1, 0, hit_row, hit_mon, hit_strict, hit_fired, &n_hits, n);
                s->has_hold[row] = 0;
            }
            s->done[row] = 1;
            continue;
        }
        apply_token(c, s, row, token_ids[i], top1[i], chosen[i], hit_row, hit_mon, hit_strict, hit_fired, &n_hits, n);
    }
    return n_hits;
}

/* Several buffered tokens per row, lengths differ, one GIL-free call.
 * tok/top1/chosen are row-major (buf_rows, width). lengths[s] tokens of rows[s]
 * are applied in order, the same as that many one-token feeds. */
static int feed_spans_impl(const Ctx *c, State *s, const int64_t *rows, const int32_t *lengths, int nspan, int width,
                           int buf_rows, const int64_t *tok, const double *top1, const double *chosen, int32_t *hit_row,
                           int16_t *hit_mon, int16_t *hit_strict, int64_t *hit_fired, int max_hits) {
    int s_i, n_hits = 0;
    for (s_i = 0; s_i < nspan; s_i++) {
        int row = (int)rows[s_i];
        int len = (int)lengths[s_i];
        int k;
        const int64_t *rt;
        const double *r1;
        const double *rc;
        if (row < 0 || row >= s->cap || row >= buf_rows || len <= 0) {
            continue;
        }
        if (len > width) {
            len = width;
        }
        rt = tok + (size_t)row * (size_t)width;
        r1 = top1 + (size_t)row * (size_t)width;
        rc = chosen + (size_t)row * (size_t)width;
        for (k = 0; k < len; k++) {
            if (s->done[row]) {
                break;
            }
            apply_token(c, s, row, rt[k], r1[k], rc[k], hit_row, hit_mon, hit_strict, hit_fired, &n_hits, max_hits);
        }
    }
    return n_hits;
}

typedef struct {
    Py_buffer b;
    int held;
} Tmp;

static int grab(Tmp *t, PyObject *obj, Py_ssize_t nbytes) {
    if (PyObject_GetBuffer(obj, &t->b, PyBUF_SIMPLE) < 0) {
        return -1;
    }
    t->held = 1;
    if (!PyBuffer_IsContiguous(&t->b, 'C') || t->b.len < nbytes) {
        PyErr_SetString(PyExc_ValueError, "rare feed: state buffer is short or not contiguous");
        return -1;
    }
    return 0;
}

static void drop_all(Tmp *tmps, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (tmps[i].held) {
            PyBuffer_Release(&tmps[i].b);
            tmps[i].held = 0;
        }
    }
}

static PyObject *feed(PyObject *self, PyObject *args) {
    PyObject *cap_obj;
    PyObject *o_counts, *o_letters, *o_prev_last, *o_answer, *o_ring, *o_ring_len;
    PyObject *o_dec_n, *o_dec_b, *o_prev, *o_has, *o_hold, *o_lp1, *o_lpc, *o_t, *o_done, *o_mon;
    PyObject *o_seen, *o_rlen, *o_rtok, *o_rlp1, *o_rlpc;
    PyObject *o_rows, *o_tok, *o_top1, *o_chosen;
    PyObject *o_hrow, *o_hmon, *o_hstrict, *o_hfired;
    int cap, finish;
    Ctx *c;
    Tmp tmp[32];
    State st;
    int n_hits = 0;
    int n;
    Py_ssize_t nrows;
    int i;

    if (!PyArg_ParseTuple(
            args, "OiOOOOOOOOOOOOOOOOOOOOOOOOOiOOOO",
            &cap_obj, &cap,
            &o_counts, &o_letters, &o_prev_last, &o_answer, &o_ring, &o_ring_len,
            &o_dec_n, &o_dec_b, &o_prev, &o_has, &o_hold, &o_lp1, &o_lpc, &o_t, &o_done, &o_mon,
            &o_seen, &o_rlen, &o_rtok, &o_rlp1, &o_rlpc,
            &o_rows, &o_tok, &o_top1, &o_chosen, &finish,
            &o_hrow, &o_hmon, &o_hstrict, &o_hfired)) {
        return NULL;
    }
    c = (Ctx *)PyCapsule_GetPointer(cap_obj, "rare_feed_ctx");
    if (c == NULL) {
        return NULL;
    }
    if (cap < 1) {
        PyErr_SetString(PyExc_ValueError, "rare feed: empty state");
        return NULL;
    }
    memset(tmp, 0, sizeof(tmp));
#define GRAB(i, obj, nbytes)                                                                 \
    if (grab(&tmp[i], (obj), (nbytes)) < 0) {                                                \
        drop_all(tmp, 32);                                                                   \
        return NULL;                                                                         \
    }
    GRAB(0, o_counts, (Py_ssize_t)cap * c->ncls * (Py_ssize_t)sizeof(int64_t));
    GRAB(1, o_letters, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(2, o_prev_last, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(3, o_answer, (Py_ssize_t)cap);
    GRAB(4, o_ring, (Py_ssize_t)cap * WIN * (Py_ssize_t)sizeof(int16_t));
    GRAB(5, o_ring_len, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(6, o_dec_n, (Py_ssize_t)cap);
    GRAB(7, o_dec_b, (Py_ssize_t)cap * DEC_MAX);
    GRAB(8, o_prev, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(9, o_has, (Py_ssize_t)cap);
    GRAB(10, o_hold, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(11, o_lp1, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    GRAB(12, o_lpc, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    GRAB(13, o_t, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(14, o_done, (Py_ssize_t)cap);
    GRAB(15, o_mon, (Py_ssize_t)cap);
    GRAB(16, o_seen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(17, o_rlen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(18, o_rtok, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(int64_t));
    GRAB(19, o_rlp1, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
    GRAB(20, o_rlpc, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
    GRAB(21, o_rows, (Py_ssize_t)sizeof(int64_t));
    nrows = tmp[21].b.len / (Py_ssize_t)sizeof(int64_t);
    n = (int)nrows;
    GRAB(22, o_tok, (Py_ssize_t)n * (Py_ssize_t)sizeof(int64_t));
    GRAB(23, o_top1, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    GRAB(24, o_chosen, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    GRAB(25, o_hrow, (Py_ssize_t)n * (Py_ssize_t)sizeof(int32_t));
    GRAB(26, o_hmon, (Py_ssize_t)n * (Py_ssize_t)sizeof(int16_t));
    GRAB(27, o_hstrict, (Py_ssize_t)n * (Py_ssize_t)sizeof(int16_t));
    GRAB(28, o_hfired, (Py_ssize_t)n * (Py_ssize_t)sizeof(int64_t));
#undef GRAB

    st.counts = tmp[0].b.buf;
    st.letters = tmp[1].b.buf;
    st.prev_last = tmp[2].b.buf;
    st.answer = tmp[3].b.buf;
    st.ring = tmp[4].b.buf;
    st.ring_len = tmp[5].b.buf;
    st.dec_n = tmp[6].b.buf;
    st.dec_b = tmp[7].b.buf;
    st.prev = tmp[8].b.buf;
    st.has_hold = tmp[9].b.buf;
    st.hold_id = tmp[10].b.buf;
    st.hold_lp1 = tmp[11].b.buf;
    st.hold_lpc = tmp[12].b.buf;
    st.t = tmp[13].b.buf;
    st.done = tmp[14].b.buf;
    st.monitor_set = tmp[15].b.buf;
    st.tokens_seen = tmp[16].b.buf;
    st.recent_len = tmp[17].b.buf;
    st.recent_tok = tmp[18].b.buf;
    st.recent_lp1 = tmp[19].b.buf;
    st.recent_lpc = tmp[20].b.buf;
    st.cap = cap;
    st.ncls = c->ncls;

    Py_BEGIN_ALLOW_THREADS
    n_hits = feed_impl(c, &st, tmp[21].b.buf, tmp[22].b.buf, tmp[23].b.buf, tmp[24].b.buf, n, finish,
                       tmp[25].b.buf, tmp[26].b.buf, tmp[27].b.buf, tmp[28].b.buf);
    Py_END_ALLOW_THREADS
    drop_all(tmp, 32);
    (void)i;
    return PyLong_FromLong(n_hits);
}

static PyObject *feed_spans(PyObject *self, PyObject *args) {
    PyObject *cap_obj;
    PyObject *o_counts, *o_letters, *o_prev_last, *o_answer, *o_ring, *o_ring_len;
    PyObject *o_dec_n, *o_dec_b, *o_prev, *o_has, *o_hold, *o_lp1, *o_lpc, *o_t, *o_done, *o_mon;
    PyObject *o_seen, *o_rlen, *o_rtok, *o_rlp1, *o_rlpc;
    PyObject *o_rows, *o_lengths, *o_tok, *o_top1, *o_chosen;
    PyObject *o_hrow, *o_hmon, *o_hstrict, *o_hfired;
    int cap, width;
    Ctx *c;
    Tmp tmp[32];
    State st;
    int n_hits = 0;
    int nspan, buf_rows, max_hits, total;
    Py_ssize_t nrows;
    int i;

    if (!PyArg_ParseTuple(
            args, "OiOOOOOOOOOOOOOOOOOOOOOOOiOOOOOOO",
            &cap_obj, &cap,
            &o_counts, &o_letters, &o_prev_last, &o_answer, &o_ring, &o_ring_len,
            &o_dec_n, &o_dec_b, &o_prev, &o_has, &o_hold, &o_lp1, &o_lpc, &o_t, &o_done, &o_mon,
            &o_seen, &o_rlen, &o_rtok, &o_rlp1, &o_rlpc,
            &o_rows, &o_lengths, &width, &o_tok, &o_top1, &o_chosen,
            &o_hrow, &o_hmon, &o_hstrict, &o_hfired)) {
        return NULL;
    }
    c = (Ctx *)PyCapsule_GetPointer(cap_obj, "rare_feed_ctx");
    if (c == NULL) {
        return NULL;
    }
    if (cap < 1 || width < 1) {
        PyErr_SetString(PyExc_ValueError, "rare feed_spans: empty state");
        return NULL;
    }
    memset(tmp, 0, sizeof(tmp));
#define GRAB(i, obj, nbytes)                                                                 \
    if (grab(&tmp[i], (obj), (nbytes)) < 0) {                                                \
        drop_all(tmp, 32);                                                                   \
        return NULL;                                                                         \
    }
    GRAB(0, o_counts, (Py_ssize_t)cap * c->ncls * (Py_ssize_t)sizeof(int64_t));
    GRAB(1, o_letters, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(2, o_prev_last, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(3, o_answer, (Py_ssize_t)cap);
    GRAB(4, o_ring, (Py_ssize_t)cap * WIN * (Py_ssize_t)sizeof(int16_t));
    GRAB(5, o_ring_len, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(6, o_dec_n, (Py_ssize_t)cap);
    GRAB(7, o_dec_b, (Py_ssize_t)cap * DEC_MAX);
    GRAB(8, o_prev, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(9, o_has, (Py_ssize_t)cap);
    GRAB(10, o_hold, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(11, o_lp1, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    GRAB(12, o_lpc, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    GRAB(13, o_t, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(14, o_done, (Py_ssize_t)cap);
    GRAB(15, o_mon, (Py_ssize_t)cap);
    GRAB(16, o_seen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    GRAB(17, o_rlen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    GRAB(18, o_rtok, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(int64_t));
    GRAB(19, o_rlp1, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
    GRAB(20, o_rlpc, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
    GRAB(21, o_rows, (Py_ssize_t)sizeof(int64_t));
    nrows = tmp[21].b.len / (Py_ssize_t)sizeof(int64_t);
    nspan = (int)nrows;
    GRAB(22, o_lengths, (Py_ssize_t)nspan * (Py_ssize_t)sizeof(int32_t));
    GRAB(23, o_tok, (Py_ssize_t)width * (Py_ssize_t)sizeof(int64_t));
    buf_rows = (int)(tmp[23].b.len / ((Py_ssize_t)width * (Py_ssize_t)sizeof(int64_t)));
    GRAB(24, o_top1, (Py_ssize_t)buf_rows * width * (Py_ssize_t)sizeof(double));
    GRAB(25, o_chosen, (Py_ssize_t)buf_rows * width * (Py_ssize_t)sizeof(double));
    total = 0;
    for (i = 0; i < nspan; i++) {
        int len = ((const int32_t *)tmp[22].b.buf)[i];
        if (len > width) {
            len = width;
        }
        if (len > 0) {
            total += len;
        }
    }
    max_hits = total > 0 ? total : 1;
    GRAB(26, o_hrow, (Py_ssize_t)max_hits * (Py_ssize_t)sizeof(int32_t));
    GRAB(27, o_hmon, (Py_ssize_t)max_hits * (Py_ssize_t)sizeof(int16_t));
    GRAB(28, o_hstrict, (Py_ssize_t)max_hits * (Py_ssize_t)sizeof(int16_t));
    GRAB(29, o_hfired, (Py_ssize_t)max_hits * (Py_ssize_t)sizeof(int64_t));
#undef GRAB

    st.counts = tmp[0].b.buf;
    st.letters = tmp[1].b.buf;
    st.prev_last = tmp[2].b.buf;
    st.answer = tmp[3].b.buf;
    st.ring = tmp[4].b.buf;
    st.ring_len = tmp[5].b.buf;
    st.dec_n = tmp[6].b.buf;
    st.dec_b = tmp[7].b.buf;
    st.prev = tmp[8].b.buf;
    st.has_hold = tmp[9].b.buf;
    st.hold_id = tmp[10].b.buf;
    st.hold_lp1 = tmp[11].b.buf;
    st.hold_lpc = tmp[12].b.buf;
    st.t = tmp[13].b.buf;
    st.done = tmp[14].b.buf;
    st.monitor_set = tmp[15].b.buf;
    st.tokens_seen = tmp[16].b.buf;
    st.recent_len = tmp[17].b.buf;
    st.recent_tok = tmp[18].b.buf;
    st.recent_lp1 = tmp[19].b.buf;
    st.recent_lpc = tmp[20].b.buf;
    st.cap = cap;
    st.ncls = c->ncls;

    Py_BEGIN_ALLOW_THREADS
    n_hits = feed_spans_impl(c, &st, tmp[21].b.buf, tmp[22].b.buf, nspan, width, buf_rows, tmp[23].b.buf, tmp[24].b.buf,
                             tmp[25].b.buf, tmp[26].b.buf, tmp[27].b.buf, tmp[28].b.buf, tmp[29].b.buf, max_hits);
    Py_END_ALLOW_THREADS
    drop_all(tmp, 32);
    return PyLong_FromLong(n_hits);
}

typedef struct {
    uint64_t hash;
    int32_t row;
    uint32_t len;
    char *text;
} IdEnt;

typedef struct {
    IdEnt *slot;
    int cap;
    int used;
} IdMap;

static uint64_t id_hash(const char *text, Py_ssize_t n) {
    uint64_t h = 14695981039346656037ull;
    Py_ssize_t i;
    for (i = 0; i < n; i++) {
        h ^= (unsigned char)text[i];
        h *= 1099511628211ull;
    }
    return h == 0 ? 1 : h;
}

static void id_map_free(IdMap *m) {
    int i;
    if (m == NULL) {
        return;
    }
    if (m->slot != NULL) {
        for (i = 0; i < m->cap; i++) {
            PyMem_Free(m->slot[i].text);
        }
    }
    PyMem_Free(m->slot);
    PyMem_Free(m);
}

static void id_map_destroy(PyObject *capsule) { id_map_free((IdMap *)PyCapsule_GetPointer(capsule, "rare_id_map")); }

static int id_map_init(IdMap *m, int cap) {
    m->slot = (IdEnt *)PyMem_Calloc((size_t)cap, sizeof(IdEnt));
    if (m->slot == NULL) {
        return -1;
    }
    m->cap = cap;
    m->used = 0;
    return 0;
}

static int id_probe(const IdMap *m, uint64_t h, const char *text, uint32_t len, int *found) {
    int mask = m->cap - 1;
    int i = (int)(h & (uint64_t)mask);
    int tomb = -1;
    int n;
    for (n = 0; n < m->cap; n++) {
        IdEnt *e = &m->slot[i];
        if (e->text == NULL) {
            if (e->row == -2 && tomb < 0) {
                tomb = i;
            } else if (e->row != -2) {
                *found = 0;
                return tomb >= 0 ? tomb : i;
            }
        } else if (e->hash == h && e->len == len && memcmp(e->text, text, len) == 0) {
            *found = 1;
            return i;
        }
        i = (i + 1) & mask;
    }
    *found = 0;
    return tomb;
}

static int id_rehash(IdMap *m) {
    int cap = m->cap * 2;
    IdEnt *old = m->slot;
    int old_cap = m->cap;
    int i;
    if (id_map_init(m, cap) < 0) {
        m->slot = old;
        m->cap = old_cap;
        return -1;
    }
    m->used = 0;
    for (i = 0; i < old_cap; i++) {
        int found = 0;
        int at;
        if (old[i].text == NULL) {
            continue;
        }
        at = id_probe(m, old[i].hash, old[i].text, old[i].len, &found);
        m->slot[at] = old[i];
        m->used++;
    }
    PyMem_Free(old);
    return 0;
}

static PyObject *map_new(PyObject *self, PyObject *args) {
    IdMap *m = (IdMap *)PyMem_Calloc(1, sizeof(IdMap));
    PyObject *cap;
    (void)args;
    if (m == NULL) {
        return PyErr_NoMemory();
    }
    if (id_map_init(m, 1024) < 0) {
        PyMem_Free(m);
        return PyErr_NoMemory();
    }
    cap = PyCapsule_New(m, "rare_id_map", id_map_destroy);
    if (cap == NULL) {
        id_map_free(m);
        return NULL;
    }
    return cap;
}

static PyObject *map_put(PyObject *self, PyObject *args) {
    PyObject *cap_obj;
    const char *text;
    Py_ssize_t len;
    int row, found = 0, at;
    IdMap *m;
    IdEnt *e;
    (void)self;
    if (!PyArg_ParseTuple(args, "Os#i", &cap_obj, &text, &len, &row)) {
        return NULL;
    }
    m = (IdMap *)PyCapsule_GetPointer(cap_obj, "rare_id_map");
    if (m == NULL) {
        return NULL;
    }
    if (m->used * 2 >= m->cap && id_rehash(m) < 0) {
        return PyErr_NoMemory();
    }
    at = id_probe(m, id_hash(text, len), text, (uint32_t)len, &found);
    e = &m->slot[at];
    if (!found) {
        char *copy = (char *)PyMem_Malloc((size_t)len + 1);
        if (copy == NULL) {
            return PyErr_NoMemory();
        }
        memcpy(copy, text, (size_t)len);
        copy[len] = 0;
        e->text = copy;
        e->len = (uint32_t)len;
        e->hash = id_hash(text, len);
        m->used++;
    }
    e->row = row;
    Py_RETURN_NONE;
}

static PyObject *map_del(PyObject *self, PyObject *args) {
    PyObject *cap_obj;
    const char *text;
    Py_ssize_t len;
    int found = 0, at;
    IdMap *m;
    (void)self;
    if (!PyArg_ParseTuple(args, "Os#", &cap_obj, &text, &len)) {
        return NULL;
    }
    m = (IdMap *)PyCapsule_GetPointer(cap_obj, "rare_id_map");
    if (m == NULL) {
        return NULL;
    }
    at = id_probe(m, id_hash(text, len), text, (uint32_t)len, &found);
    if (found) {
        PyMem_Free(m->slot[at].text);
        m->slot[at].text = NULL;
        m->slot[at].row = -2;
        m->slot[at].len = 0;
        m->slot[at].hash = 0;
        m->used--;
    }
    Py_RETURN_NONE;
}

static PyObject *map_lookup(PyObject *self, PyObject *args) {
    PyObject *cap_obj, *ids, *fast, *out;
    IdMap *m;
    Py_ssize_t n, i;
    const char **ps;
    Py_ssize_t *ls;
    int32_t *rows;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &cap_obj, &ids)) {
        return NULL;
    }
    m = (IdMap *)PyCapsule_GetPointer(cap_obj, "rare_id_map");
    if (m == NULL) {
        return NULL;
    }
    fast = PySequence_Fast(ids, "req ids");
    if (fast == NULL) {
        return NULL;
    }
    n = PySequence_Fast_GET_SIZE(fast);
    ps = (const char **)PyMem_Malloc((size_t)n * sizeof(char *));
    ls = (Py_ssize_t *)PyMem_Malloc((size_t)n * sizeof(Py_ssize_t));
    rows = (int32_t *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (ps == NULL || ls == NULL || rows == NULL) {
        PyMem_Free(ps);
        PyMem_Free(ls);
        PyMem_Free(rows);
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyUnicode_Check(item) || PyUnicode_GET_LENGTH(item) == 0) {
            ps[i] = NULL;
            ls[i] = 0;
            continue;
        }
        ps[i] = PyUnicode_AsUTF8AndSize(item, &ls[i]);
        if (ps[i] == NULL) {
            PyMem_Free(ps);
            PyMem_Free(ls);
            PyMem_Free(rows);
            Py_DECREF(fast);
            return NULL;
        }
    }
    Py_BEGIN_ALLOW_THREADS
    for (i = 0; i < n; i++) {
        int found = 0;
        int at;
        if (ps[i] == NULL) {
            rows[i] = -2;
            continue;
        }
        at = id_probe(m, id_hash(ps[i], ls[i]), ps[i], (uint32_t)ls[i], &found);
        rows[i] = found ? m->slot[at].row : -1;
    }
    Py_END_ALLOW_THREADS
    out = PyBytes_FromStringAndSize((const char *)rows, n * (Py_ssize_t)sizeof(int32_t));
    PyMem_Free(ps);
    PyMem_Free(ls);
    PyMem_Free(rows);
    Py_DECREF(fast);
    return out;
}

typedef struct {
    Ctx *ctx;
    int cap;
    Tmp tmp[21];
    State st;
} Bound;

static void bind_state_ptrs(State *st, Tmp *tmp, const Ctx *c, int cap) {
    st->counts = tmp[0].b.buf;
    st->letters = tmp[1].b.buf;
    st->prev_last = tmp[2].b.buf;
    st->answer = tmp[3].b.buf;
    st->ring = tmp[4].b.buf;
    st->ring_len = tmp[5].b.buf;
    st->dec_n = tmp[6].b.buf;
    st->dec_b = tmp[7].b.buf;
    st->prev = tmp[8].b.buf;
    st->has_hold = tmp[9].b.buf;
    st->hold_id = tmp[10].b.buf;
    st->hold_lp1 = tmp[11].b.buf;
    st->hold_lpc = tmp[12].b.buf;
    st->t = tmp[13].b.buf;
    st->done = tmp[14].b.buf;
    st->monitor_set = tmp[15].b.buf;
    st->tokens_seen = tmp[16].b.buf;
    st->recent_len = tmp[17].b.buf;
    st->recent_tok = tmp[18].b.buf;
    st->recent_lp1 = tmp[19].b.buf;
    st->recent_lpc = tmp[20].b.buf;
    st->cap = cap;
    st->ncls = c->ncls;
}

static void bound_destroy(PyObject *capsule) {
    Bound *b = (Bound *)PyCapsule_GetPointer(capsule, "rare_feed_bound");
    if (b == NULL) {
        PyErr_Clear();
        return;
    }
    drop_all(b->tmp, 21);
    PyMem_Free(b);
}

/* Hold the 21 row-state buffers until the batch storage is replaced. */
static PyObject *bind_state(PyObject *self, PyObject *args) {
    PyObject *cap_obj;
    PyObject *o_counts, *o_letters, *o_prev_last, *o_answer, *o_ring, *o_ring_len;
    PyObject *o_dec_n, *o_dec_b, *o_prev, *o_has, *o_hold, *o_lp1, *o_lpc, *o_t, *o_done, *o_mon;
    PyObject *o_seen, *o_rlen, *o_rtok, *o_rlp1, *o_rlpc;
    int cap;
    Ctx *c;
    Bound *b;
    PyObject *capsule;
    (void)self;
    if (!PyArg_ParseTuple(
            args, "OiOOOOOOOOOOOOOOOOOOOOO",
            &cap_obj, &cap,
            &o_counts, &o_letters, &o_prev_last, &o_answer, &o_ring, &o_ring_len,
            &o_dec_n, &o_dec_b, &o_prev, &o_has, &o_hold, &o_lp1, &o_lpc, &o_t, &o_done, &o_mon,
            &o_seen, &o_rlen, &o_rtok, &o_rlp1, &o_rlpc)) {
        return NULL;
    }
    c = (Ctx *)PyCapsule_GetPointer(cap_obj, "rare_feed_ctx");
    if (c == NULL) {
        return NULL;
    }
    if (cap < 1) {
        PyErr_SetString(PyExc_ValueError, "rare bind_state: empty state");
        return NULL;
    }
    b = (Bound *)PyMem_Calloc(1, sizeof(Bound));
    if (b == NULL) {
        return PyErr_NoMemory();
    }
    b->ctx = c;
    b->cap = cap;
#define BGRAB(i, obj, nbytes)                                                                \
    if (grab(&b->tmp[i], (obj), (nbytes)) < 0) {                                             \
        drop_all(b->tmp, 21);                                                                \
        PyMem_Free(b);                                                                       \
        return NULL;                                                                         \
    }
    BGRAB(0, o_counts, (Py_ssize_t)cap * c->ncls * (Py_ssize_t)sizeof(int64_t));
    BGRAB(1, o_letters, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    BGRAB(2, o_prev_last, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    BGRAB(3, o_answer, (Py_ssize_t)cap);
    BGRAB(4, o_ring, (Py_ssize_t)cap * WIN * (Py_ssize_t)sizeof(int16_t));
    BGRAB(5, o_ring_len, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    BGRAB(6, o_dec_n, (Py_ssize_t)cap);
    BGRAB(7, o_dec_b, (Py_ssize_t)cap * DEC_MAX);
    BGRAB(8, o_prev, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    BGRAB(9, o_has, (Py_ssize_t)cap);
    BGRAB(10, o_hold, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    BGRAB(11, o_lp1, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    BGRAB(12, o_lpc, (Py_ssize_t)cap * (Py_ssize_t)sizeof(double));
    BGRAB(13, o_t, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    BGRAB(14, o_done, (Py_ssize_t)cap);
    BGRAB(15, o_mon, (Py_ssize_t)cap);
    BGRAB(16, o_seen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int64_t));
    BGRAB(17, o_rlen, (Py_ssize_t)cap * (Py_ssize_t)sizeof(int16_t));
    BGRAB(18, o_rtok, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(int64_t));
    BGRAB(19, o_rlp1, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
    BGRAB(20, o_rlpc, (Py_ssize_t)cap * RECENT_CAP * (Py_ssize_t)sizeof(double));
#undef BGRAB
    bind_state_ptrs(&b->st, b->tmp, c, cap);
    capsule = PyCapsule_New(b, "rare_feed_bound", bound_destroy);
    if (capsule == NULL) {
        drop_all(b->tmp, 21);
        PyMem_Free(b);
        return NULL;
    }
    return capsule;
}

/* Per-step inputs only. Row state stays bound from bind_state. */
static PyObject *feed_bound(PyObject *self, PyObject *args) {
    PyObject *bound_obj;
    PyObject *o_rows, *o_tok, *o_top1, *o_chosen, *o_hrow, *o_hmon, *o_hstrict, *o_hfired;
    int finish;
    Bound *b;
    Tmp tmp[8];
    int n, n_hits;
    Py_ssize_t nrows;
    (void)self;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOi", &bound_obj, &o_rows, &o_tok, &o_top1, &o_chosen, &o_hrow, &o_hmon, &o_hstrict,
            &o_hfired, &finish)) {
        return NULL;
    }
    b = (Bound *)PyCapsule_GetPointer(bound_obj, "rare_feed_bound");
    if (b == NULL) {
        return NULL;
    }
    memset(tmp, 0, sizeof(tmp));
#define SGRAB(i, obj, nbytes)                                                                \
    if (grab(&tmp[i], (obj), (nbytes)) < 0) {                                                \
        drop_all(tmp, 8);                                                                    \
        return NULL;                                                                         \
    }
    SGRAB(0, o_rows, (Py_ssize_t)sizeof(int64_t));
    nrows = tmp[0].b.len / (Py_ssize_t)sizeof(int64_t);
    n = (int)nrows;
    SGRAB(1, o_tok, (Py_ssize_t)n * (Py_ssize_t)sizeof(int64_t));
    SGRAB(2, o_top1, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    SGRAB(3, o_chosen, (Py_ssize_t)n * (Py_ssize_t)sizeof(double));
    SGRAB(4, o_hrow, (Py_ssize_t)n * (Py_ssize_t)sizeof(int32_t));
    SGRAB(5, o_hmon, (Py_ssize_t)n * (Py_ssize_t)sizeof(int16_t));
    SGRAB(6, o_hstrict, (Py_ssize_t)n * (Py_ssize_t)sizeof(int16_t));
    SGRAB(7, o_hfired, (Py_ssize_t)n * (Py_ssize_t)sizeof(int64_t));
#undef SGRAB
    Py_BEGIN_ALLOW_THREADS
    n_hits = feed_impl(
        b->ctx, &b->st, tmp[0].b.buf, tmp[1].b.buf, tmp[2].b.buf, tmp[3].b.buf, n, finish, tmp[4].b.buf, tmp[5].b.buf,
        tmp[6].b.buf, tmp[7].b.buf);
    Py_END_ALLOW_THREADS
    drop_all(tmp, 8);
    return PyLong_FromLong(n_hits);
}

static PyMethodDef methods[] = {
    {"make_ctx", make_ctx, METH_VARARGS, "Pack immutable rc4 tables."},
    {"feed", feed, METH_VARARGS, "Score one new token on each row. GIL is released."},
    {"bind_state", bind_state, METH_VARARGS, "Pin row-state buffers for later feed_bound calls."},
    {"feed_bound", feed_bound, METH_VARARGS, "Feed using pinned row state. GIL is released."},
    {"feed_spans", feed_spans, METH_VARARGS, "Score a variable-length token span on each row. GIL is released."},
    {"map_new", map_new, METH_NOARGS, "req_id to row map."},
    {"map_put", map_put, METH_VARARGS, "Insert req_id."},
    {"map_del", map_del, METH_VARARGS, "Remove req_id."},
    {"map_lookup", map_lookup, METH_VARARGS, "Resolve req ids. GIL released for the probe."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "rare_feed_native", "rc4 request-batch feed, GIL released", -1, methods,
};

PyMODINIT_FUNC PyInit_rare_feed_native(void) { return PyModule_Create(&moduledef); }
