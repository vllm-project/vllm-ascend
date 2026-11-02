# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU oracle: Python dictionaries and stable lists, no Triton."""

from copy import deepcopy


def resolve_reference(
    values, host_rows, block, attention_stride=0, source_block=0, resident_block=0, resident_null=0, union_bound=1
):
    out = deepcopy(values)
    rows = len(out["global_indices"])
    k = len(out["global_indices"][0]) if rows else 0
    valid_rows = min(rows, out.get("num_valid_rows", [rows])[0])
    request_ids = out.get("request_ids")
    states = out.get("request_state_indices")
    ownership = out["device_global_indices"]
    lru = out["lru_slots"]

    def request_of(row):
        return row if request_ids is None else request_ids[row] if row < valid_rows else -1

    def store(row, col, physical):
        out["hot_indices"][row][col] = physical
        if "attention_indices" in out:
            out["attention_indices"][row][col] = (
                physical // block * attention_stride + physical % block if physical >= 0 else -1
            )

    row = 0
    while row < rows:
        req = request_of(row)
        state = req if states is None else states[req] if 0 <= req < len(states) else -1
        if state < 0:
            for col in range(k):
                store(row, col, -1)
                if "resolved_global_indices" in out:
                    out["resolved_global_indices"][row][col] = -1
            for name in ("valid_counts", "swap_counts"):
                if name in out:
                    out[name][row] = 0
            row += 1
            continue
        end = row + 1
        if request_ids is not None:
            while end < rows and request_of(end) == req:
                end += 1
        union = {}  # insertion order is the first (row, column) reference
        refs = []
        for query in range(row, end):
            count = 0
            for col, token in enumerate(out["global_indices"][query]):
                g, resident = token, -1
                if "source_block_table" in out:
                    table = out["source_block_table"]
                    b = token // source_block if token >= 0 else -1
                    g = -1
                    if 0 <= req < len(table) and 0 <= b < len(table[req]) and table[req][b] > 0:
                        g = table[req][b] * source_block + token % source_block
                    table = out.get("resident_block_table")
                    b = token // resident_block if table is not None and token >= 0 else -1
                    if table is not None and state < len(table) and 0 <= b < len(table[state]):
                        physical = table[state][b]
                        if physical >= 0 and physical != resident_null:
                            resident = physical * resident_block + token % resident_block
                if g >= host_rows:
                    g = -1
                if "resolved_global_indices" in out:
                    out["resolved_global_indices"][query][col] = g
                if "miss_mask" in out:
                    out["miss_mask"][query][col] = 0
                count += resident >= 0 or g >= 0
                store(query, col, resident)
                if resident < 0 and g >= 0:
                    union.setdefault(g, (query, col))
                    refs.append((query, col, g))
            if "valid_counts" in out:
                out["valid_counts"][query] = count
            if "swap_counts" in out:
                out["swap_counts"][query] = 0
        if len(union) > union_bound:
            raise ValueError("union exceeds bound")
        if union:
            cached = {g: slot for slot, g in enumerate(ownership[state]) if g in union}
            protected = set(cached.values())
            hit_slots = [slot for slot in lru[state] if slot in protected]
            evict = [slot for slot in lru[state] if slot not in protected]
            misses = [g for g in union if g not in cached]
            for index, g in enumerate(misses):
                slot = evict[index]
                valid = 0 <= slot < len(ownership[state])
                cached[g] = slot if valid else -1
                physical = -1
                if valid:
                    ownership[state][slot] = g
                    physical = out["hot_block_table"][req][slot // block] * block + slot % block
                if "swap_counts" in out:
                    r, c = row + index // k, index % k
                    out["swap_host_physical_rows"][r][c] = g
                    out["swap_device_physical_rows"][r][c] = physical
            if "swap_counts" in out:
                for query in range(row, end):
                    out["swap_counts"][query] = min(k, max(0, len(misses) - (query - row) * k))
            for query, col, g in refs:
                slot = cached[g]
                physical = -1
                if slot >= 0:
                    physical = out["hot_block_table"][req][slot // block] * block + slot % block
                    first_miss = g in misses and union[g] == (query, col)
                    if "miss_mask" in out:
                        out["miss_mask"][query][col] = int(first_miss)
                    if "stats" in out:
                        index = int(first_miss)
                        out["stats"][index] = (out["stats"][index] + 1) % 2**64
                store(query, col, physical)
            lru[state] = evict[len(misses) :] + evict[: len(misses)] + hit_slots
        row = end
    return out


def gather_reference(
    storage,
    host_bytes,
    source,
    target,
    *,
    width,
    block,
    page_bytes,
    view_offset,
    hot_rows,
    mask=None,
    counts=None,
    states=None,
    attention_stride=0,
):
    """Predict the complete backing storage, including all guard bytes."""
    result = list(storage)
    attention = []
    for row, sources in enumerate(source):
        att = []
        active = states is None or states[row] >= 0
        for col, src in enumerate(sources):
            dst = target[row][col]
            att.append(dst // block * attention_stride + dst % block if active and dst >= 0 else -1)
            selected = col < min(len(sources), max(0, counts[row])) if counts is not None else mask[row][col] != 0
            if active and selected and src >= 0 and 0 <= dst < hot_rows:
                start = view_offset + dst // block * page_bytes + dst % block * width
                payload = host_bytes[src] if src < len(host_bytes) else [0] * width
                result[start : start + width] = payload
        attention.append(att)
    return result, attention
