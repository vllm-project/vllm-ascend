# SPDX-License-Identifier: Apache-2.0
"""Declarative per-architecture reduction profiles.

A profile describes *how* a GLM family may be layer-cropped: config layout
(flat vs nested ``text_config``), the layer-count config keys (legacy ChatGLM
checkpoints carry both ``num_layers`` and ``num_hidden_layers``), exact
weight-name roots for decoder layers / embeddings / final norm / lm_head /
vision subtrees (verified against the pinned upstream index snapshots), MTP
layer handling, and the minimum prefix that still exercises the architecture's
distinctive blocks. Profiles carry workload shapes for the
precision/performance gates but deliberately no numeric baselines: baselines
are produced by running a reference build of the *same* checkpoint on the same
hardware with ``run_logits_dump.py`` / ``run_perf.py``.

Layer selection policy: reduction keeps a *prefix* of the source layers.
Suffix cropping would break rotary/cache assumptions and, for GLM-5.2/5.3,
would detach ``shared`` indexer layers from the ``full`` producer layers whose
top-k indices they reuse (see ``vllm_ascend/patch/worker/patch_deepseek_v2.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ProfileError

# Config keys whose values are per-layer lists of length <layer count> and
# must be truncated to the kept prefix. Applies at the layer-config level
# (top-level for flat configs, text_config for nested ones).
PER_LAYER_ARRAY_KEYS = ("indexer_types", "mlp_layer_types", "layer_types")

# Nested dicts inside the layer config that hold absolute layer-index lists.
# For a prefix crop, kept entries are unchanged; entries >= keep are removed.
INDEX_LIST_SUBKEYS = {"linear_attn_config": ("kda_layers", "full_attn_layers")}


@dataclass(frozen=True)
class NamingRules:
    """Exact checkpoint weight-name layout, verified against upstream indexes."""

    layer_roots: tuple[str, ...]  # decoder layer prefixes, e.g. ("model.layers.",)
    embed_names: tuple[str, ...]
    final_norm_names: tuple[str, ...]
    lm_head_names: tuple[str, ...]
    vision_roots: tuple[str, ...] = ()
    keep_names: tuple[str, ...] = ()  # extra globals kept verbatim (e.g. rotary inv_freq)


@dataclass(frozen=True)
class Workload:
    """Fixed workload shape for precision/performance runs (no baselines)."""

    prompt_lengths: tuple[int, ...]
    output_tokens: int
    logprob_top_k: int
    warmup_iterations: int
    measured_iterations: int


@dataclass(frozen=True)
class ReductionProfile:
    name: str
    family: str
    architectures: tuple[str, ...]
    # "flat": layer config at top level; "nested_text": under config["text_config"].
    config_layout: str
    # Config keys at the layer-config level that all hold the decoder layer
    # count. Every key present must agree (inconsistent input is rejected) and
    # every key present is updated on truncation.
    layer_count_keys: tuple[str, ...]
    naming: NamingRules
    # Smallest prefix that still covers the family's distinctive blocks.
    min_keep_layers: int
    default_keep_layers: int
    # Official released layer count, informational/auditing only.
    official_layers: int
    # Source checkpoints keep MTP layers at indices [num_layers, ...).
    # Remapped to follow the kept prefix when keep_mtp is set.
    keep_mtp: bool = True
    # Vision subtrees (naming.vision_roots) are kept in full; dropping them
    # would silently break a multimodal architecture, so no profile drops them.
    include_vision: bool = True
    precision_workload: Workload = field(
        default=Workload(
            prompt_lengths=(16, 128, 512),
            output_tokens=32,
            logprob_top_k=64,
            warmup_iterations=1,
            measured_iterations=1,
        )
    )
    performance_workload: Workload = field(
        default=Workload(
            prompt_lengths=(128, 1024),
            output_tokens=128,
            logprob_top_k=0,
            warmup_iterations=2,
            measured_iterations=5,
        )
    )

    def validate_keep_layers(self, keep_layers: int, source_layers: int) -> None:
        if keep_layers < self.min_keep_layers:
            raise ProfileError(
                f"profile {self.name!r} requires keep_layers >= {self.min_keep_layers} "
                f"(got {keep_layers}): a shorter prefix would miss blocks that define this "
                "architecture (see profile justification in tools/glm_reduced/README.md)"
            )
        if keep_layers >= source_layers:
            raise ProfileError(
                f"keep_layers={keep_layers} must be smaller than the source layer count "
                f"({source_layers}); a full copy is not a reduction"
            )


def _shared_indexer_closure_ok(indexer_types: list, keep_layers: int) -> list[str]:
    """Every kept `shared` layer needs its producing `full` layer in the prefix.

    Returns a list of human-readable problems (empty when valid).
    """
    problems = []
    last_full = None
    for idx, kind in enumerate(indexer_types[:keep_layers]):
        if not isinstance(kind, str):
            problems.append(f"indexer_types[{idx}]={kind!r} is not a string")
            continue
        if kind.lower() == "full":
            last_full = idx
        elif kind.lower() == "shared":
            if last_full is None:
                problems.append(f"layer {idx} is 'shared' but no 'full' producer precedes it in the kept prefix")
        else:
            problems.append(f"indexer_types[{idx}]={kind!r} is neither 'full' nor 'shared'")
    return problems


def read_layer_count(profile: ReductionProfile, layer_config: dict) -> int:
    """Read the decoder layer count, rejecting inconsistent duplicate keys."""
    values = {}
    for key in profile.layer_count_keys:
        value = layer_config.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ProfileError(f"config key {key!r}={value!r} is not a positive integer")
        values[key] = value
    if not values:
        raise ProfileError(
            f"profile {profile.name!r} found no layer-count key {profile.layer_count_keys} in the config"
        )
    if len(set(values.values())) != 1:
        raise ProfileError(
            f"inconsistent layer counts in config: {values}; refusing to guess which one is authoritative"
        )
    return next(iter(values.values()))


def write_layer_count(profile: ReductionProfile, layer_config: dict, keep_layers: int) -> None:
    for key in profile.layer_count_keys:
        if key in layer_config:
            layer_config[key] = keep_layers


def validate_layer_arrays(
    profile: ReductionProfile, layer_config: dict, keep_layers: int, source_layers: int
) -> list[str]:
    """Check config per-layer arrays against the requested prefix; returns problems."""
    problems: list[str] = []
    for key in PER_LAYER_ARRAY_KEYS:
        value = layer_config.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) != source_layers:
            problems.append(f"config key {key!r} must be a list of length {source_layers}, got {value!r}")
    indexer_types = layer_config.get("indexer_types")
    if isinstance(indexer_types, list) and len(indexer_types) == source_layers:
        problems.extend(_shared_indexer_closure_ok(indexer_types, keep_layers))
    layer_types = layer_config.get("layer_types")
    for subdict_key, list_keys in INDEX_LIST_SUBKEYS.items():
        subdict = layer_config.get(subdict_key)
        if subdict is None:
            continue
        if not isinstance(subdict, dict):
            problems.append(f"config key {subdict_key!r} must be a dict, got {subdict!r}")
            continue
        for list_key in list_keys:
            indices = subdict.get(list_key)
            if indices is None:
                continue
            if not isinstance(indices, list) or any(not isinstance(i, int) for i in indices):
                problems.append(f"{subdict_key}.{list_key} must be a list of ints, got {indices!r}")
                continue
            if any(i < 0 or i >= source_layers for i in indices):
                problems.append(f"{subdict_key}.{list_key} contains indices outside [0, {source_layers})")
        if isinstance(layer_types, list) and len(layer_types) == source_layers:
            kda = set(subdict.get("kda_layers") or [])
            full = set(subdict.get("full_attn_layers") or [])
            for idx in range(source_layers):
                expected = full if layer_types[idx] == "deepseek_sparse_attention" else kda
                if idx not in expected:
                    problems.append(
                        f"layer {idx} is {layer_types[idx]!r} but not listed in the matching "
                        f"linear_attn_config index list"
                    )
                    break
    return problems


def truncate_layer_config(profile: ReductionProfile, layer_config: dict, keep_layers: int, source_layers: int) -> dict:
    """Return a reduced copy of the layer config (input dict is not mutated)."""
    new_config = dict(layer_config)
    write_layer_count(profile, new_config, keep_layers)
    for key in PER_LAYER_ARRAY_KEYS:
        value = layer_config.get(key)
        if isinstance(value, list) and len(value) == source_layers:
            new_config[key] = value[:keep_layers]
    for subdict_key, list_keys in INDEX_LIST_SUBKEYS.items():
        subdict = layer_config.get(subdict_key)
        if isinstance(subdict, dict):
            new_subdict = dict(subdict)
            for list_key in list_keys:
                indices = subdict.get(list_key)
                if isinstance(indices, list):
                    new_subdict[list_key] = [i for i in indices if i < keep_layers]
            new_config[subdict_key] = new_subdict
    return new_config


_STD_NAMING = NamingRules(
    layer_roots=("model.layers.",),
    embed_names=("model.embed_tokens.weight",),
    final_norm_names=("model.norm.weight",),
    lm_head_names=("lm_head.weight",),
)

_PROFILES = {
    # GLM-4.5/4.6/4.7 (glm4_moe): flat config, 92 layers, first 3 dense MLP.
    # min prefix 8 keeps the dense->sparse MLP transition plus five routed-MoE
    # layers (160 experts, unchanged), so MoE routing/expert paths are covered.
    "glm4-moe": ReductionProfile(
        name="glm4-moe",
        family="glm4_moe",
        architectures=("Glm4MoeForCausalLM",),
        config_layout="flat",
        layer_count_keys=("num_hidden_layers",),
        naming=_STD_NAMING,
        min_keep_layers=8,
        default_keep_layers=8,
        official_layers=92,
    ),
    # GLM-5/5.1 (glm_moe_dsa, per-layer indexer weights, no indexer_types) and
    # GLM-5.2/5.3 (indexer_types full/shared; shared layers reuse the producer
    # layer's indexer weights and top-k indices). min prefix 8 covers 3 dense
    # layers, sparse MoE layers, and a complete full->shared indexer cycle
    # (producers at 0,1,2,6; consumers at 3,4,5,7) in the official 78-layer
    # checkpoints. GLM-5.3 ships FP8 blockwise quantization natively.
    "glm-moe-dsa": ReductionProfile(
        name="glm-moe-dsa",
        family="glm_moe_dsa",
        architectures=("GlmMoeDsaForCausalLM",),
        config_layout="flat",
        layer_count_keys=("num_hidden_layers",),
        naming=_STD_NAMING,
        min_keep_layers=8,
        default_keep_layers=8,
        official_layers=78,
    ),
    # GLM-4.7-Flash (glm4_moe_lite): flat config, 47 layers, MLA
    # (q_lora_rank/kv_lora_rank) + 64-expert MoE, first_k_dense_replace=1,
    # nextn=1. min prefix 4 covers the dense->sparse MLP transition and three
    # routed-MoE MLA layers.
    "glm4-moe-lite": ReductionProfile(
        name="glm4-moe-lite",
        family="glm4_moe_lite",
        architectures=("Glm4MoeLiteForCausalLM",),
        config_layout="flat",
        layer_count_keys=("num_hidden_layers",),
        naming=_STD_NAMING,
        min_keep_layers=4,
        default_keep_layers=8,
        official_layers=47,
    ),
    # GLM-5.3-Flash (glm5_next): nested text_config (45 layers), layer_types
    # alternating 3 linear_attention (KDA) + 1 deepseek_sparse_attention,
    # first 3 MLP dense then sparse, mHC hyper-connections, keypool indexer,
    # MTP layer at index 45, native FP8 quantization_config. Checkpoint text
    # root is "model.language_model." (verified against the upstream index);
    # vision subtree "model.visual." is always kept in full. min prefix 8
    # keeps two full hybrid cycles plus the dense->sparse MLP transition.
    "glm5-next": ReductionProfile(
        name="glm5-next",
        family="glm5_next",
        architectures=("Glm5NextForConditionalGeneration",),
        config_layout="nested_text",
        layer_count_keys=("num_hidden_layers",),
        naming=NamingRules(
            layer_roots=("model.language_model.layers.",),
            embed_names=("model.language_model.embed_tokens.weight",),
            final_norm_names=("model.language_model.norm.weight",),
            lm_head_names=("lm_head.weight",),
            vision_roots=("model.visual.",),
        ),
        min_keep_layers=8,
        default_keep_layers=8,
        official_layers=45,
    ),
    # GLM-4.1V-9B-Thinking (glm4v): dense 40-layer text tower under
    # "model.language_model." plus a "model.visual." vision tower (both kept;
    # vision is never cropped). Distinct from the old GLM-4V line, which the
    # support matrix marks unsupported; upstream vLLM carries a GLM-4.1V
    # processing test route (upstream_config.yaml -> test_glm4_1v.py). Dense
    # architecture: a prefix of 4 already covers attention+MLP+norms.
    "glm4v": ReductionProfile(
        name="glm4v",
        family="glm4v",
        architectures=("Glm4vForConditionalGeneration",),
        config_layout="nested_text",
        layer_count_keys=("num_hidden_layers",),
        naming=NamingRules(
            layer_roots=("model.language_model.layers.",),
            embed_names=("model.language_model.embed_tokens.weight",),
            final_norm_names=("model.language_model.norm.weight",),
            lm_head_names=("lm_head.weight",),
            vision_roots=("model.visual.",),
        ),
        min_keep_layers=4,
        default_keep_layers=8,
        official_layers=40,
        keep_mtp=False,
    ),
    # Legacy ChatGLM (chatglm3-6b, glm-4-9b-chat): remote-code architecture
    # with `transformer.encoder.layers.*` weights, word_embeddings /
    # final_layernorm / output_layer globals, and `num_layers` as the layer
    # count (glm-4-9b-chat carries both num_layers and num_hidden_layers;
    # both must agree and both are updated). Dense; prefix 4 covers
    # attention+MLP+norms. Loading route is upstream vLLM's chatglm support.
    "chatglm": ReductionProfile(
        name="chatglm",
        family="chatglm",
        architectures=("ChatGLMModel", "ChatGLMForCausalLM"),
        config_layout="flat",
        layer_count_keys=("num_layers", "num_hidden_layers"),
        naming=NamingRules(
            layer_roots=("transformer.encoder.layers.",),
            embed_names=("transformer.embedding.word_embeddings.weight",),
            final_norm_names=("transformer.encoder.final_layernorm.weight",),
            lm_head_names=("transformer.output_layer.weight",),
            keep_names=("transformer.rotary_pos_emb.inv_freq",),
        ),
        min_keep_layers=4,
        default_keep_layers=8,
        official_layers=0,  # 28 (chatglm3) / 40 (glm-4-9b): no single official count
        keep_mtp=False,
    ),
}


def get_profile(name: str) -> ReductionProfile:
    try:
        return _PROFILES[name]
    except KeyError:
        raise ProfileError(f"unknown profile {name!r}; available: {sorted(_PROFILES)}") from None


def list_profiles() -> list[ReductionProfile]:
    return list(_PROFILES.values())


def match_profile(architectures: list[str]) -> ReductionProfile | None:
    """Return the unique profile whose architectures cover the checkpoint's."""
    arch_set = set(architectures or [])
    matches = [p for p in _PROFILES.values() if arch_set and arch_set.issubset(set(p.architectures))]
    if not matches:
        return None
    return matches[0]
