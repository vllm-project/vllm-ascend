"""Independent CPU adapter for frozen M4 rules and model-specific tables."""

import operator

import numpy as np
from _m4_native import Detector
from m4_config import REASONS, Result


class RepDetectorCPP:
    def __init__(self, tables, think_end=None):
        if think_end is not None:
            # int(float) silently changes token identity. Accept integer-like
            # scalars only; None explicitly disables the answer boundary.
            if isinstance(think_end, (bool, np.bool_)):
                raise TypeError("think_end must be an integer token ID or None")
            think_end = operator.index(think_end)
            if not 0 <= think_end <= np.iinfo(np.int64).max:
                raise ValueError("think_end must be a non-negative int64 token ID")
        columns = ("norm", "chars", "ws", "run_char", "marker", "send")
        arrays = []
        size = None
        bounds = np.iinfo(np.int32)
        for key in columns:
            array = np.asarray(tables[key])
            if array.ndim != 1 or not array.size:
                raise ValueError(f"{key} must be a non-empty one-dimensional table")
            if size is not None and array.size != size:
                raise ValueError("vocabulary columns must have equal lengths")
            size = array.size
            if array.dtype.kind not in "biu":
                raise TypeError(f"{key} must contain integer or boolean values")
            if int(array.min()) < bounds.min or int(array.max()) > bounds.max:
                raise ValueError(f"{key} values must fit int32 without truncation")
            arrays.append(np.ascontiguousarray(array, dtype=np.int32))
        self.native = Detector(*arrays, -1 if think_end is None else think_end)

    def start(self):
        return self.native.start()

    def result(self, st):
        if st.failed:
            raise RuntimeError("failed detector state; discard it")
        return Result(True, st.fired_at, REASONS[st.reason]) if st.reason >= 0 else None

    def progress(self, st):
        """Exceptional-path receipt, not a retry/rollback promise.

        A batch can commit earlier slots before a later slot raises. Query each
        state to recover its committed prefix/first alert without consuming any
        token. A failed native block has no trustworthy partial counters.
        """
        if st.failed:
            return None
        return (st.received, st.t, st.fired_at, REASONS[st.reason] if st.reason >= 0 else "")

    def feed_events(self, states, token_lists):
        """Return only new (submission_slot, fired_at, reason) events.

        No Result is constructed for non-alert requests. The native layer
        preserves lazy token consumption and first-alert semantics.
        On exception the batch is NOT atomic: use progress() on every submitted
        state to recover committed prefixes/events; do not replay the whole batch.
        """
        return [
            (slot, fired_at, REASONS[reason]) for slot, fired_at, reason in self.native.feed_events(states, token_lists)
        ]

    def finish(self, st):
        self.native.finish(st)
        return self.result(st) or Result()
