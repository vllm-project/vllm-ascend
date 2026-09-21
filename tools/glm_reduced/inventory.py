# SPDX-License-Identifier: Apache-2.0
"""Auditable inventory of GLM models referenced by this repository.

Every entry maps an upstream checkpoint to a reduction recipe (a profile in
``profiles.py`` plus builder/test routes) or an explicit reason it cannot be
built safely. Sources: docs/source/user_guide/support_matrix/supported_models.md,
docs/source/tutorials/models/GLM*.md, vllm_ascend/models/__init__.py,
tests/e2e nightly/weekly YAML configs, .github/workflows/misc/model_dataset_list.json,
and the pinned public source descriptors shipped in ``tools/glm_reduced/sources.json``
(model IDs, revisions, config/index URLs, tensor counts, sizes).

Notes on scope honesty:

- "recipe-ready" means the builder + CPU transformation tests cover the
  layout; NPU runtime qualification of a reduced checkpoint is a separate,
  explicitly pending step (see README).
- Absence of a vllm_ascend-local model registration does NOT imply no loader:
  most models load through upstream vLLM's registry under the Ascend platform.
- Quantized checkpoint *variants* (e.g. Eco-Tech/GLM-5.2-w8a8c8) share the
  base architecture and are listed as variants, not separate architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ProfileError
from .profiles import get_profile, list_profiles

STATUS_RECIPE_READY = "recipe-ready"
STATUS_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class InventoryEntry:
    model_id: str
    architecture: str
    family: str
    config_layout: str
    layers: int | None
    variant_of: str | None = None  # quantized checkpoint variants share the base recipe
    quantization: str | None = None
    profile: str | None = None
    status: str = STATUS_RECIPE_READY
    reason: str = ""
    evidence: tuple[str, ...] = field(default=())


_ENTRIES = [
    # -- Legacy dense ChatGLM (remote-code `chatglm` architecture) ----------
    # Loader route: upstream vLLM's chatglm implementation (no Ascend-local
    # registration needed). Weight layout verified against the upstream index
    # snapshots: transformer.encoder.layers.*, word_embeddings/final_layernorm/
    # output_layer globals. glm-4-9b-chat carries both num_layers and
    # num_hidden_layers; the profile requires them to agree and updates both.
    InventoryEntry(
        model_id="THUDM/chatglm3-6b",
        architecture="ChatGLMModel",
        family="chatglm",
        config_layout="flat(num_layers)",
        layers=28,
        profile="chatglm",
        evidence=("tools/glm_reduced/sources.json", "upstream vLLM chatglm loader route"),
    ),
    InventoryEntry(
        model_id="THUDM/glm-4-9b-chat",
        architecture="ChatGLMModel",
        family="chatglm",
        config_layout="flat(num_layers+num_hidden_layers)",
        layers=40,
        profile="chatglm",
        evidence=("tools/glm_reduced/sources.json", "upstream vLLM chatglm loader route"),
    ),
    # -- GLM-4V lines --------------------------------------------------------
    InventoryEntry(
        model_id="zai-org/glm-4v-9b (GLM-4V line)",
        architecture="Glm4vForConditionalGeneration",
        family="glm4v-old",
        config_layout="nested_text",
        layers=None,
        status=STATUS_UNSUPPORTED,
        reason="the support matrix explicitly marks the old GLM-4V line unsupported on vllm-ascend "
        "(issue #2260); building an unvalidatable artifact would be dishonest coverage.",
        evidence=("docs/source/user_guide/support_matrix/supported_models.md",),
    ),
    InventoryEntry(
        # NOT the old GLM-4V line: GLM-4.1V has its own upstream processing
        # test route and model-dataset entry; treat its runtime status
        # independently.
        model_id="zai-org/GLM-4.1V-9B-Thinking",
        architecture="Glm4vForConditionalGeneration",
        family="glm4v",
        config_layout="nested_text",
        layers=40,
        profile="glm4v",
        evidence=(
            "tools/glm_reduced/sources.json",
            ".github/workflows/scripts/upstream_config.yaml -> tests/models/multimodal/processing/test_glm4_1v.py",
            ".github/workflows/misc/model_dataset_list.json",
        ),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-Image",
        architecture="unknown",
        family="glm-image",
        config_layout="unknown",
        layers=None,
        status=STATUS_UNSUPPORTED,
        reason="root config.json returns HTTP 404 at the pinned revision; the repository is listed in "
        "the model dataset manifest but is not a plain causal-LM checkpoint (likely component configs). "
        "No recipe is claimed without an inspected config/index.",
        evidence=(
            ".github/workflows/misc/model_dataset_list.json",
            "tools/glm_reduced/sources.json (resolution record)",
        ),
    ),
    # -- GLM-4.x MoE (dense prefix + routed MoE, upstream Glm4Moe) ----------
    InventoryEntry(
        model_id="zai-org/GLM-4.5",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        quantization="bf16 (native)",
        profile="glm4-moe",
        evidence=(
            "docs/source/tutorials/models/GLM4.x.md",
            "docs/source/user_guide/support_matrix/supported_models.md",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-4.6",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        quantization="bf16 (native)",
        profile="glm4-moe",
        evidence=("docs/source/tutorials/models/GLM4.x.md", "tools/glm_reduced/sources.json"),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-4.7",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        quantization="bf16 (native)",
        profile="glm4-moe",
        evidence=(
            "docs/source/tutorials/models/GLM4.x.md",
            "tests/e2e/nightly/single_node/models/configs/GLM-4.7.yaml",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-4.7-Flash",
        architecture="Glm4MoeLiteForCausalLM",
        family="glm4_moe_lite",
        config_layout="flat",
        layers=47,
        quantization="bf16 (native)",
        profile="glm4-moe-lite",
        evidence=("tools/glm_reduced/sources.json",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-4.5-w8a8",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        variant_of="zai-org/GLM-4.5",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm4-moe",
        evidence=("docs/source/tutorials/models/GLM4.x.md",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-4.6-w8a8",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        variant_of="zai-org/GLM-4.6",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm4-moe",
        evidence=("docs/source/tutorials/models/GLM4.x.md",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-4.7-W8A8-floatmtp",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        variant_of="zai-org/GLM-4.7",
        quantization="w8a8 + float MTP (quant_model_description.json)",
        profile="glm4-moe",
        evidence=(
            "tests/e2e/nightly/single_node/models/configs/GLM-4.7.yaml",
            ".github/workflows/misc/model_dataset_list.json",
        ),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-4.7-W8A8",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        variant_of="zai-org/GLM-4.7",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm4-moe",
        evidence=(".github/workflows/misc/model_dataset_list.json",),
    ),
    InventoryEntry(
        model_id="vllm-ascend/GLM-4.7-W8A8C8",
        architecture="Glm4MoeForCausalLM",
        family="glm4_moe",
        config_layout="flat",
        layers=92,
        variant_of="zai-org/GLM-4.7",
        quantization="w8a8c8 (quant_model_description.json)",
        profile="glm4-moe",
        evidence=("tests/e2e/weekly/multi_node/internal_dp/config/GLM-4.7-W8A8C8-Mooncake-Layerwise.yaml",),
    ),
    # -- GLM-5/5.1 DSA MoE (per-layer indexer weights, BF16) ----------------
    InventoryEntry(
        model_id="zai-org/GLM-5",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        quantization="bf16 (native)",
        profile="glm-moe-dsa",
        evidence=(
            "docs/source/tutorials/models/GLM5.md",
            "tests/e2e/weekly/single_node/configs/GLM-5.yaml",
            "vllm_ascend/models/__init__.py",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-5.1",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        quantization="bf16 (native)",
        profile="glm-moe-dsa",
        evidence=(
            "docs/source/tutorials/models/GLM5.md",
            "docs/source/user_guide/support_matrix/supported_models.md",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5-w4a8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5",
        quantization="w4a8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/weekly/single_node/configs/GLM-5.yaml",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5-w8a8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("docs/source/tutorials/models/GLM5.md", ".github/workflows/misc/model_dataset_list.json"),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5-W8A8-xLLM",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5",
        quantization="w8a8 xLLM layout (quant_model_description.json)",
        profile="glm-moe-dsa",
        reason="recipe applies; xLLM-specific sidecar files, if any, must classify or the build fails "
        "explicitly (unknown tensors are never silently dropped).",
        evidence=(".github/workflows/misc/model_dataset_list.json",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.1-w8a8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.1",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=(
            "tests/e2e/nightly/multi_node/internal_dp/config/GLM5_1-W8A8-EP.yaml",
            ".github/workflows/misc/model_dataset_list.json",
        ),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.1-w8a8c8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.1",
        quantization="w8a8c8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/nightly/multi_node/internal_dp/config/GLM-5.1-W8A8C8-A3_128k_90_50.yaml",),
    ),
    InventoryEntry(
        model_id="gdydems/GLM-5.1-w8a8c8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.1",
        quantization="w8a8c8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=(".github/workflows/misc/model_dataset_list.json",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.1-w4a4c8-mxfp4",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.1",
        quantization="w4a4c8 mxfp4 (950DT)",
        profile="glm-moe-dsa",
        reason="recipe applies only if the checkpoint stores per-tensor scales and a flat "
        "quant_model_description.json; packed MXFP4 layouts without that listing fail explicitly "
        "in the builder (UnsupportedQuantError) instead of emitting a corrupt model.",
        evidence=("tests/e2e/nightly/single_node/models/configs/GLM5_1_W4A4_A5.yaml",),
    ),
    # -- GLM-5.2 (BF16) / GLM-5.3 (native FP8) with shared indexer layers ---
    InventoryEntry(
        model_id="zai-org/GLM-5.2",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        quantization="bf16 (native)",
        profile="glm-moe-dsa",
        evidence=(
            "docs/source/tutorials/models/GLM5.2.md",
            "vllm_ascend/patch/worker/patch_deepseek_v2.py (shared-indexer semantics)",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="zai-org/GLM-5.3",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        quantization="fp8 blockwise (native quantization_config + weight_scale_inv tensors)",
        profile="glm-moe-dsa",
        evidence=("docs/source/tutorials/models/GLM5.3.md", "tools/glm_reduced/sources.json"),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.2-w4a8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.2",
        quantization="w4a8 (quant_model_description.json; real metadata validated in tests)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/nightly/single_node/models/configs/GLM-5.2-W4A8-MTP-A3.yaml",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.2-w8a8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.2",
        quantization="w8a8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/nightly/multi_node/internal_dp/config/GLM5_2-W8A8-A3-dual-nodes.yaml",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.2-w8a8c8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.2",
        quantization="w8a8c8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/weekly/multi_node/external_dp/config/GLM-5.2-W8A8C8-64k-1k-90-50-PD.yaml",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.2-w4a8c8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.2",
        quantization="w4a8c8 (quant_model_description.json)",
        profile="glm-moe-dsa",
        evidence=("tests/e2e/weekly/multi_node/external_dp/config/GLM-5.2-W4A8C8-1M-PD-DCP.yaml",),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.3-w8a8c8",
        architecture="GlmMoeDsaForCausalLM",
        family="glm_moe_dsa",
        config_layout="flat",
        layers=78,
        variant_of="zai-org/GLM-5.3",
        quantization="w8a8c8 (quant_model_description.json; real metadata validated in tests)",
        profile="glm-moe-dsa",
        evidence=("docs/source/tutorials/models/GLM5.3.md",),
    ),
    InventoryEntry(
        model_id="RedHatAI/GLM-5.2-speculator.dspark",
        architecture="(speculator draft)",
        family="glm_moe_dsa",
        config_layout="n/a",
        layers=None,
        variant_of="zai-org/GLM-5.2",
        status=STATUS_UNSUPPORTED,
        reason="draft/speculator-only artifact, not a standalone GLM target checkpoint; layer reduction "
        "of a target model does not apply to it.",
        evidence=(".github/workflows/misc/model_dataset_list.json",),
    ),
    # -- GLM-5.3-Flash hybrid multimodal (native FP8) ------------------------
    InventoryEntry(
        model_id="zai-org/GLM-5.3-Flash",
        architecture="Glm5NextForConditionalGeneration",
        family="glm5_next",
        config_layout="nested_text",
        layers=45,
        quantization="fp8 blockwise (native quantization_config + weight_scale_inv tensors)",
        profile="glm5-next",
        evidence=(
            "docs/source/tutorials/models/GLM5.3-Flash.md",
            "vllm_ascend/models/glm5next/model.py",
            "tools/glm_reduced/sources.json",
        ),
    ),
    InventoryEntry(
        model_id="Eco-Tech/GLM-5.3-Flash-w8a8",
        architecture="Glm5NextForConditionalGeneration",
        family="glm5_next",
        config_layout="nested_text",
        layers=45,
        variant_of="zai-org/GLM-5.3-Flash",
        quantization="w8a8 (quant_model_description.json; real metadata validated in tests)",
        profile="glm5-next",
        evidence=(".github/workflows/misc/model_dataset_list.json",),
    ),
]


def list_inventory() -> list[InventoryEntry]:
    return list(_ENTRIES)


def validate_inventory() -> list[str]:
    """Cross-check inventory entries against actual profiles; returns problems."""
    problems = []
    known = {p.name for p in list_profiles()}
    for entry in _ENTRIES:
        if entry.status == STATUS_UNSUPPORTED:
            if not entry.reason:
                problems.append(f"{entry.model_id}: unsupported entry without a reason")
            continue
        if not entry.profile:
            problems.append(f"{entry.model_id}: recipe-ready entry without a profile")
            continue
        try:
            profile = get_profile(entry.profile)
        except ProfileError:
            problems.append(f"{entry.model_id}: profile {entry.profile!r} does not exist")
            continue
        if entry.architecture not in profile.architectures:
            problems.append(
                f"{entry.model_id}: architecture {entry.architecture!r} not covered by profile "
                f"{entry.profile!r} {profile.architectures}"
            )
    unused = known - {e.profile for e in _ENTRIES if e.profile}
    if unused:
        problems.append(f"profiles without inventory entries: {sorted(unused)}")
    return problems


def render_markdown() -> str:
    lines = [
        "| Model | Architecture | Layout | Layers | Variant of | Quantization | Profile | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in _ENTRIES:
        lines.append(
            f"| {e.model_id} | {e.architecture} | {e.config_layout} | {e.layers or '-'} | "
            f"{e.variant_of or '-'} | {e.quantization or '-'} | {e.profile or '-'} | "
            f"{e.status}{(': ' + e.reason) if e.reason else ''} |"
        )
    return "\n".join(lines)
