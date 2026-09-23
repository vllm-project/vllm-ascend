# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Draft model module exclusions derived from checkpoint topology."""


def get_draft_exclusions(model) -> frozenset[str]:
    """Names to exclude from a draft model's transferable tensor set.

    Modules absent from the draft checkpoint will be shared with the target or
    rebuilt locally by align_draft_weights, so they must not participate in
    transfer. Both seed registration and receiver collection exclude them, which
    keeps the manifest and structural digest aligned. The checkpoint is read
    once during restore_load_derived_state; this hook returns the memoized
    result without repeated filesystem access.

    Returns a frozenset of exact module prefixes (e.g., "model.embed_tokens",
    "lm_head") whose parameters and buffers should be filtered out of the
    transferable tensor list. An empty set means every module participates.
    """
    return getattr(model, "_rfork_draft_exclusions", frozenset())
