# SPDX-License-Identifier: Apache-2.0
"""Static Engram owner membership and chunk-aligned global row layout."""

import hashlib
import json
from pathlib import Path

DEFAULT_CHUNK_ROWS = 1 << 22


def owner_set(path):
    return validate_owner_set(json.loads(Path(path).read_text()))


def validate_owner_set(spec):
    if not isinstance(spec, dict) or spec.get("schema") != 1:
        raise ValueError("Unsupported Engram owner-set schema")
    if not isinstance(spec.get("deployment_id"), str) or not spec["deployment_id"]:
        raise ValueError("Engram owner set requires a deployment_id")
    peers = spec.get("owners")
    if not isinstance(peers, list) or not 2 <= len(peers) <= 4:
        raise ValueError("Engram sharding requires two to four ordered owners")
    endpoints = []
    for peer in peers:
        if not isinstance(peer, dict) or set(peer) != {"host", "port"}:
            raise ValueError("Each Engram owner needs a host and port")
        if not isinstance(peer["host"], str) or not peer["host"] or type(peer["port"]) is not int:
            raise ValueError("Invalid Engram owner endpoint")
        if not 1 <= peer["port"] <= 65535:
            raise ValueError("Invalid Engram owner port")
        endpoints.append((peer["host"], peer["port"]))
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Duplicate Engram owner endpoint")
    chunk = spec.get("chunk_rows", DEFAULT_CHUNK_ROWS)
    if type(chunk) is not int or not 1 <= chunk <= DEFAULT_CHUNK_ROWS or chunk & (chunk - 1):
        raise ValueError("Engram chunk_rows must be a power of two up to 2^22")
    # Canonicalize before hashing so field order/omitted defaults cannot split a deployment.
    return dict(schema=1, deployment_id=spec["deployment_id"], owners=peers, chunk_rows=chunk)


def layout_identity(spec, checkpoint):
    encoded = json.dumps(dict(checkpoint=checkpoint, membership=spec, storage="int8-fp32-group32"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def row_ranges(rows, count, chunk):
    if type(rows) is not int or rows <= 0 or not 1 <= count <= 4 or chunk <= 0:
        raise ValueError("Invalid Engram row layout")
    chunks = (rows + chunk - 1) // chunk
    if chunks < count:
        raise ValueError("Engram table needs at least one chunk per owner")
    quotient, remainder = divmod(chunks, count)
    cursor, spans = 0, []
    for rank in range(count):
        end = cursor + quotient + (rank < remainder)
        spans.append((cursor * chunk, min(end * chunk, rows)))
        cursor = end
    return spans
