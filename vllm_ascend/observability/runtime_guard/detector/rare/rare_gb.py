"""Learned token score of the rare_chars detector rc3 (runs/explore/rc3_report.md): gradient-boosted trees over per-token features
(detector/rare.py GB_INPUTS), evaluated only at gated tokens. sklearn's HistGradientBoostingClassifier is exported to complete
binary trees of one depth (as detector/trees.py, plus categorical splits) for storage; scoring uses the QuickScorer form built
from them at load (GB._quickscorer).
  feat  [T, 2^D - 1]  input index per internal node (heap order: children of k are 2k+1, 2k+2)
  thr   [T, 2^D - 1]  numeric node: go right iff x > thr (+inf on padding nodes: always left)
  cat   [T, 2^D - 1]  categorical node: go left iff bit x of mask is set (x: integer category code, 0..31)
  mask  [T, 2^D - 1]  uint32 left-category mask (unknown categories follow sklearn: its missing-value direction)
  leaf  [T, 2^D]      leaf values (an early leaf's value repeated over the leaves below it)
score = base + sum of the reached leaves (sklearn's decision_function); quant: the step lp1 / lpc / gap inputs are rounded to.
"""
import numpy as np


class GB:
    def __init__(self, spec):
        self.depth = int(spec['depth'])
        self.feat = np.asarray(spec['feat'], np.int64)
        self.thr = np.asarray(spec['thr'], np.float64)
        self.cat = np.asarray(spec['cat'], bool)
        self.mask = np.asarray(spec['mask'], np.int64)
        self.leaf = np.asarray(spec['leaf'], np.float64)
        self.base = float(spec['base'])
        self.inputs = [str(x) for x in spec['inputs']]
        self.quant = float(spec.get('quant', 0.0))     # lp1 / lpc / gap inputs rounded to this step (detector/rare.py gb_inputs)
        self._quickscorer()

    def _quickscorer(self):
        """scoring form (QuickScorer): the real nodes of every tree with, per node, the bitmask of the leaves it rules out when its
        test sends a row right (its left subtree; leaves numbered left to right). A row's exit leaf in a tree is the lowest leaf
        left after ANDing the masks of all of the tree's nodes whose test is true, so a whole forest is a few vectorised steps."""
        ni = 2 ** self.depth - 1
        feat, thr, cat, cmask, nmask, tree, leafv, leaf0 = [], [], [], [], [], [], [], []
        for t in range(len(self.feat)):
            leaves = []

            def walk(k, d):                              # -> bitmask of the leaves under heap slot k
                if d == self.depth or (not self.cat[t, k] and np.isinf(self.thr[t, k])):
                    span = 2 ** (self.depth - d)        # a leaf (or padding below an early leaf): its value
                    leaves.append(self.leaf[t, (k + 1) * span - 1 - ni])
                    return 1 << (len(leaves) - 1)
                j = len(feat)
                feat.append(self.feat[t, k])
                thr.append(self.thr[t, k])
                cat.append(self.cat[t, k])
                cmask.append(self.mask[t, k])
                nmask.append(0)
                tree.append(t)
                left = walk(2 * k + 1, d + 1)
                right = walk(2 * k + 2, d + 1)
                nmask[j] = left
                return left | right
            walk(0, 0)
            if len(leaves) > 63:
                raise ValueError('more than 63 leaves in a tree')
            leaf0.append(len(leafv))
            leafv += leaves
        self._qfeat = np.array(feat, np.int64)
        self._qthr = np.array(thr)
        self._qcat = np.flatnonzero(cat)
        self._qcfeat = self._qfeat[self._qcat]
        self._qcmask = np.array(cmask, np.int64)[self._qcat]
        full = np.int64(-1)                              # all leaves (two's complement: every bit set)
        self._qrule = np.array([~np.int64(m) for m in nmask], np.int64)   # AND-ed in when the test is true
        self._qfull = full
        self._qstart = np.r_[0, np.flatnonzero(np.diff(np.array(tree)))  + 1].astype(np.int64)
        self._qleaf = np.array(leafv)
        self._qleaf0 = np.array(leaf0, np.int64)
        codes = np.arange(32)                            # categorical node x code -> goes right (one-row path)
        self._qclut = ((self._qcmask[:, None] >> codes[None, :]) & 1) == 0
        self._qcrow = np.arange(len(self._qcat))

    def score(self, X):
        """X float64 [n, len(inputs)] (categorical inputs: integer codes 0..31) -> float64 [n]"""
        X = np.asarray(X, np.float64)
        if not len(X):
            return np.zeros(0)
        if len(X) == 1:                                  # the common case online: one row, 1-d arrays
            x = X[0]
            right = x[self._qfeat] > self._qthr
            if len(self._qcat):
                right[self._qcat] = self._qclut[self._qcrow, np.clip(x[self._qcfeat], 0, 31).astype(np.int64)]
            m = np.bitwise_and.reduceat(np.where(right, self._qrule, self._qfull), self._qstart)
            low = m & -m
            return np.array([self.base + self._qleaf[self._qleaf0 + np.frexp(low.astype(np.float64))[1] - 1].sum()])
        XT = np.ascontiguousarray(X.T)                   # inputs x rows: the gathers below are row takes
        right = XT.take(self._qfeat, 0) > self._qthr[:, None]   # numeric test: go right iff x > thr
        if len(self._qcat):
            c = np.clip(XT.take(self._qcfeat, 0), 0, 31).astype(np.int64)
            right[self._qcat] = ((self._qcmask[:, None] >> c) & 1) == 0
        m = np.bitwise_and.reduceat(np.where(right, self._qrule[:, None], self._qfull), self._qstart, axis=0)
        low = m & -m                                     # lowest set bit = the exit leaf
        idx = np.frexp(low.astype(np.float64))[1] - 1    # its position (an exact power of two)
        leaves = np.ascontiguousarray(self._qleaf[self._qleaf0[:, None] + idx].T)
        return self.base + leaves.sum(1)                 # same summation order as the one-row path (row sums)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls({k: z[k] for k in z.files})


def export(model, depth, inputs, quant=0.0):
    """sklearn HistGradientBoostingClassifier (binary, fitted on a numpy array) -> spec dict (complete trees of `depth`).
    sklearn re-orders the inputs (categorical first) and maps each categorical value to an ordinal code; the spec is in terms of
    the original inputs and raw values 0..31 (a value unseen in training goes where sklearn sends a missing value)."""
    n_int, n_leaf = 2 ** depth - 1, 2 ** depth
    pre = getattr(model, '_preprocessor', None)
    is_cat = np.zeros(len(inputs), bool) if model.is_categorical_ is None else np.asarray(model.is_categorical_, bool)
    order = np.r_[np.flatnonzero(is_cat), np.flatnonzero(~is_cat)] if pre is not None else np.arange(len(inputs))
    cats = pre.named_transformers_['encoder'].categories_ if pre is not None and is_cat.any() else []
    feat, thr, cat, mask, leaf = [], [], [], [], []
    for (pred,) in model._predictors:
        nodes = pred.nodes
        raw = pred.raw_left_cat_bitsets
        f, h, c, m, v = (np.zeros(n_int, np.int64), np.full(n_int, np.inf), np.zeros(n_int, bool), np.zeros(n_int, np.int64),
                         np.zeros(n_leaf))

        def walk(i, k, d):
            if nodes['is_leaf'][i]:
                span = 2 ** (depth - d)
                first = (k + 1) * span - 1 - n_int
                v[first:first + span] = nodes['value'][i]
                return
            if d >= depth:
                raise ValueError('tree deeper than depth')
            j = int(nodes['feature_idx'][i])
            f[k] = order[j]
            if nodes['is_categorical'][i]:
                words = raw[nodes['bitset_idx'][i]]
                mk = 0
                for val in range(32):
                    hit = np.flatnonzero(cats[j] == val)
                    if len(hit):
                        code = int(hit[0])
                        left = (int(words[code // 32]) >> (code % 32)) & 1
                    else:
                        left = int(bool(nodes['missing_go_to_left'][i]))
                    mk |= left << val
                c[k], m[k] = True, mk
            else:
                h[k] = nodes['num_threshold'][i]
            walk(nodes['left'][i], 2 * k + 1, d + 1)
            walk(nodes['right'][i], 2 * k + 2, d + 1)
        walk(0, 0, 0)
        feat.append(f)
        thr.append(h)
        cat.append(c)
        mask.append(m)
        leaf.append(v)
    return {'depth': depth, 'feat': np.array(feat), 'thr': np.array(thr), 'cat': np.array(cat), 'mask': np.array(mask),
            'leaf': np.array(leaf), 'base': float(np.ravel(model._baseline_prediction)[0]), 'inputs': np.array(inputs),
            'quant': float(quant)}
