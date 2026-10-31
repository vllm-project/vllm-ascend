# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request identity across P/D processes, independent of local block hashing."""

import hashlib
import json
from typing import Any


def pool_request_digest(request: Any) -> str:
    """Compute before P's stateful final-token truncation.

    Block hashes may have process-local seeds and block granularities. The handoff
    therefore fingerprints the original logical input, including multimodal and
    adapter identities, rather than comparing P and D block hashes.
    """
    if request.prompt_token_ids is None or getattr(request, "prompt_embeds", None) is not None:
        raise ValueError("Pool PD request identity currently requires prompt token IDs")
    multimodal = []
    for feature in request.mm_features:
        identity = feature.mm_hash or feature.identifier
        if not isinstance(identity, str) or not identity:
            raise ValueError("Pool PD multimodal inputs require a stable content hash")
        multimodal.append((feature.modality, identity, feature.mm_position.offset, feature.mm_position.length))
    lora = request.lora_request
    adapter = None if lora is None else (lora.lora_name, lora.lora_int_id)
    payload = ("pool-request-v1", list(request.prompt_token_ids), request.cache_salt, adapter, multimodal)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
