"""Single source of truth for the ``@pytest.mark.e2e_coverage`` taxonomy.

This module defines the allowed dimensions and their legal values exactly
once. Three consumers import from here so they never drift apart:

- ``tests/e2e/generate_coverage_html.py`` — renders the coverage report and
  its taxonomy-driven HTML (filter options, bar charts, explorer).
- ``tests/e2e/conftest.py`` — enforces valid marker values at pytest
  collection time (``pytest_collection_modifyitems``).
- ``.github/workflows/scripts/coverage.py`` — CI gate that fails the LINT job
  when a marked test carries an out-of-taxonomy value.

Adding a new value: append it to the relevant set in :data:`ALLOWED_VALUES`.
That is the only edit needed — generators, runtime checks and CI all pick it
up automatically.

Multi-value dimensions (``feature``, ``parallel``) accept a comma-separated
string, e.g. ``feature="lora,mtp"``. Each comma-separated token must itself
be a member of the dimension's allowed set; see :func:`split_multi` and
:func:`validate_coverage_marker`.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Dimensions and their allowed values
# ---------------------------------------------------------------------------
ALLOWED_VALUES: dict[str, set[str]] = {
    "arch": {
        "dense",
        "moe",
        "embedding",
        "classification",
        "reranker",
        "mamba_ssm",
        "multimodal",
    },
    "feature": {
        "lora",
        "multi_lora",
        "runtime_lora",
        "fully_sharded_lora",
        "spec_decode",
        "mtp",
        "eagle3",
        "sfa_dsa",
        "dsa_cp",
        "prefix_caching",
        "chunked_prefill",
        "cpu_offloading",
        "cpu_weight_offload",
        "sleep_wake",
        "fa3",
        "fia_comparison",
        "batch_invariant",
        "guided_decoding",
        "pooling",
        "compile_fusion",
        "aclgraph",
        "xlite",
        "weight_transfer",
        "multi_instance",
        "profiling",
        "prompt_embeds",
        "score_api",
        "classification_api",
        "logprobs",
        "mixed_lengths",
        "long_sequence",
        "flashcomm1",
        "eplb",
        "dynamic_eplb",
        "multistream_moe",
        "mo_routing_replay",
    },
    "parallel": {
        "TP",
        "PP",
        "EP",
        "DP",
        "PCP",
        "DCP",
        "SP",
    },
    "deploy": {
        "pd_mix",
        "pd_disaggregation",
        "epd",
    },
    "hardware": {
        "310P",
        "A2",
        "A3",
        "A5",
    },
    "quantization": {
        "FP16",
        "BF16",
        "W8A8",
        "W4A8",
        "W8A8_dynamic",
    },
    "graph_mode": {
        "eager",
        "full_graph",
        "full_decode_only",
        "piecewise",
        "full_and_piecewise",
        "aclgraph",
        "xlite_decode_only",
        "xlite_full",
    },
}

# Human-readable label for each dimension (used in HTML/CLI output)
DIM_LABELS: dict[str, str] = {
    "arch": "Architecture",
    "feature": "Feature",
    "parallel": "Parallel",
    "deploy": "Deploy",
    "hardware": "Hardware",
    "quantization": "Quantization",
    "graph_mode": "Graph Mode",
}

# Dimension ordering — drives column/table ordering in the report
DIM_ORDER: list[str] = ["arch", "feature", "parallel", "deploy", "hardware", "quantization", "graph_mode"]

# Combinations that are semantically meaningless and should NOT be expected
# to be covered (e.g. a hardware chip that does not support some feature).
# Each rule is a partial dim->value mapping: a taxonomy combination is
# "invalid" iff it matches every constraint in at least one rule. Dims not
# named by a rule are wildcards. Rules whose dims are not all present in the
# combination being tested are skipped (see :func:`is_invalid_combo`), so a
# rule is only evaluated when every dim it names is in scope.
#
# This list is maintained by hand and refreshed periodically — it only
# affects reporting (the coverage HTML excludes these combinations from its
# "zero-coverage" enumeration), never marker validation.
INVALID_COMBOS: list[dict[str, str]] = [
    # {"hardware": "310P", "feature": "spec_decode"},
    # {"arch": "embedding", "parallel": "TP"},
]

# Every @pytest.mark.e2e_coverage must include all of these keys (empty
# string "" is allowed for a dimension that does not apply, but the key
# must be present).
REQUIRED_DIMS: set[str] = set(DIM_ORDER)

# Dimensions whose value may carry multiple comma-separated tokens. Each
# token must individually be a member of the dimension's allowed set.
MULTI_VALUE_DIMS: dict[str, bool] = {"feature": True, "parallel": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_multi(dim: str, raw: str) -> list[str]:
    """Split a marker value into its constituent tokens.

    For multi-value dimensions (``feature``, ``parallel``) this splits on
    comma, e.g. ``split_multi("feature", "lora,mtp") -> ["lora", "mtp"]``.
    For single-value dimensions the raw string is returned as a one-element
    list (after strip), so callers can treat all dimensions uniformly.

    Empty tokens (from ``"lora,,mtp"`` or a bare ``""``) are dropped.
    """
    vals = [v.strip() for v in raw.split(",")]
    return [v for v in vals if v]


def is_invalid_combo(combo: dict[str, str]) -> bool:
    """True iff *combo* matches every constraint of at least one rule in
    :data:`INVALID_COMBOS`.

    *combo* is a dim->value mapping (the caller may pass only the subset of
    dimensions it is enumerating). A rule is only evaluated when **every** dim
    it names is present in *combo*; otherwise the rule is skipped. This keeps
    the semantics correct when the coverage explorer enumerates a subset of
    dimensions: a rule referencing a dim that is not currently selected does
    not prune anything for that view.
    """
    return any(all(dim in combo and combo[dim] == val for dim, val in rule.items()) for rule in INVALID_COMBOS)


def _did_you_mean(value: str, allowed: set[str]) -> str:
    """Return a 'did you mean' hint for a value not in ``allowed``."""
    near = [a for a in allowed if value.lower() in a.lower() or a.lower() in value.lower()]
    if not near:
        return ""
    return f" (did you mean: {', '.join(sorted(near))}?)"


def validate_coverage_marker(coverage: dict[str, Any]) -> list[str]:
    """Validate one test's ``e2e_coverage`` marker kwargs.

    ``coverage`` is the mapping of dimension -> raw string value as written
    in the decorator (e.g. ``{"arch": "dense", "feature": "lora,mtp"}``).
    Multi-value dimensions are split by :func:`split_multi` and each token
    is checked individually against :data:`ALLOWED_VALUES`.

    Returns a list of human-readable warning strings — empty when the marker
    is fully valid. Callers that want hard enforcement should raise on a
    non-empty result (see ``conftest.py``).
    """
    warnings: list[str] = []

    provided = set(coverage.keys())
    missing = REQUIRED_DIMS - provided
    if missing:
        warnings.append(f"missing required key(s): {', '.join(sorted(missing))}")

    for dim, raw in coverage.items():
        allowed = ALLOWED_VALUES.get(dim)
        if allowed is None:
            warnings.append(f"unknown dimension '{dim}'")
            continue
        # Normalize to token list (single-value dims yield [raw])
        tokens = split_multi(dim, str(raw)) if MULTI_VALUE_DIMS.get(dim) else split_multi(dim, str(raw))
        for token in tokens:
            if token not in allowed:
                warnings.append(f"unknown value '{token}' in dimension '{dim}'{_did_you_mean(token, allowed)}")
    return warnings
