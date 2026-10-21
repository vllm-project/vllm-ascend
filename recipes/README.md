# vLLM Ascend recipes

This directory holds Ascend overlays for recipes published by
[recipes.vllm.ai](https://recipes.vllm.ai). The initial import contains ten
recipes already merged upstream: nine owned by vLLM and MiniMax-H3 owned by
vLLM-Omni. Pending recipe PRs and Markdown-only tutorials are not imported.

The [recipes renderer](https://github.com/vllm-project/recipes) owns the site,
command builder, JSON API, schema, and shared vocabulary. Adding this
directory does not register a new website source or change published pages.

## Layout and ownership

```text
recipes/
├── README.md
├── manifest.yaml
├── validate.mjs
└── overlays/
    └── <hf_org>/<hf_repo>.yaml
```

- `manifest.yaml` declares the schema version, matching vLLM's manifest.
  It does not grant ownership or hardware scope.
- An overlay extends the base recipe with the same case-sensitive Hugging
  Face ID. It contains Ascend hardware settings, variants, dependencies,
  and guide sections rather than a second complete model recipe.
- Optional `models/<hf_org>/<hf_repo>.yaml` files may own complete
  Ascend-only recipes after source registration. The model must have no
  other owner; all `meta.hardware` entries and every variant's explicit
  `supported_hardware` list must be within the granted Huawei scope.

Overlays are additive: they cannot replace base values or change NVIDIA and
AMD settings. A new variant must explicitly restrict `supported_hardware`
to Ascend profiles. `guide_sections` fills
`<!-- overlay:vllm-ascend:<name> -->` markers in the base guide, or appends
sections when no marker exists.

## Imported content and provenance

The import uses the official renderer at
[`b08ad738`](https://github.com/vllm-project/recipes/commit/b08ad73889f9e33319d8ac9c887dec6a0975c55d)
and its committed `sources.lock.json`. Both source revisions are merged
official commits. No content is taken from an unmerged PR or fork branch.

| Base owner and revision | Imported overlays |
| --- | --- |
| [vLLM `52a65d7b`](https://github.com/vllm-project/vllm/tree/52a65d7b19e9e3ad86bf1e3b6954e4e11eeec7da/recipes/models) | Qwen3-30B-A3B, Qwen3.5-397B-A17B, Qwen3.6-27B, Qwen3.8-27B; DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp; Kimi-K3; GLM-5.3, GLM-5.3-Flash |
| [vLLM-Omni `24f0cf37`](https://github.com/vllm-project/vllm-omni/tree/24f0cf37b94076baece79a685dbf04f66b14aef4/recipes/models) | MiniMax-H3 |

The renderer's `split-overlay.mjs` extracts Huawei-scoped fields and guide
sections whose headings contain `Ascend`. YAML formatting and comments
change during extraction; configuration values are inherited unchanged.
Existing `verified` labels and measurements describe the source recipes,
not new NPU validation performed for this import.

Scattered Ascend prose, links, and shared metadata remain in the base
recipes. Only Qwen3.8-27B and MiniMax-H3 have standalone guide sections
extracted in this batch. They are not complete documentation migrations.

The current merge API places Huawei dependencies before base dependencies
(or at the end when requested). For MiniMax-H3, its Ascend pip prerequisite
moves before the shared weight-download step; it still precedes the
vLLM-Omni installation. The command contents are unchanged. Review this
ordering with the Omni maintainer before the published source cutover.

## Editing and validation

Follow the renderer's [contribution guide](https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md)
and [recipe authoring guide](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md).
Use [SCHEMA.md](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md)
and [taxonomy.yaml](https://github.com/vllm-project/recipes/blob/main/taxonomy.yaml)
for fields and hardware IDs. Schema, catalog, and command builder changes
belong in the renderer repository.

Keep shared model metadata in the base recipe. The
[recipe template](https://github.com/vllm-project/recipes/blob/main/templates/recipe.yaml)
and [JSON Schema](https://recipes.vllm.ai/recipe.schema.json) apply to
complete recipes; do not apply the full-recipe schema directly to partial
overlays. Link to [model tutorials](../docs/source/tutorials/models) for
deployment details and record actual hardware validation when adding
configurations.

Until the official channel registers Ascend, the `Recipes` PR check uses
`validate.mjs` with the merged renderer snapshot above. It invokes the
official sync, assemble, split, and validation tools to:

1. Resolve the pinned vLLM and Omni sources and check their ownership.
2. Split an isolated copy of the merged recipes into base and Huawei parts.
3. Check that this directory contains exactly the extracted overlays and
   values, rejecting missing content or additions from pending PRs.
4. Merge this directory into the temporary base with Huawei scope, check
   orphaned overlays and schema, and build the JSON API.

Run the same check from a separate renderer checkout:

```bash
git clone https://github.com/vllm-project/recipes.git recipes-renderer
cd recipes-renderer
git checkout --detach b08ad73889f9e33319d8ac9c887dec6a0975c55d
pnpm install --frozen-lockfile
node /path/to/vllm-ascend/recipes/validate.mjs
```

The adapter ignores local source overrides and creates temporary channels.
It does not change the upstream source recipes or the published website.
The check currently requires the imported merged snapshot to remain
unchanged; update the source snapshot deliberately when adding newly merged
content. Static validation does not establish NPU correctness or performance.

## Registering the source

Before publishing these overlays, register this directory in the renderer's
`sources.yaml` under the intended channel:

```yaml
vllm-ascend:
  repo: vllm-project/vllm-ascend
  ref: main
  path: recipes
  role: overlay
  scope: { brands: [Huawei] }
```

Coordinate the matching base changes in both vLLM and vLLM-Omni and update
the renderer's `sources.lock.json`. Its current merge rejects keys already
set by a base recipe even when their values match. Validate compatible
base and overlay revisions before changing the published pins; preserve
GPU behavior and reconcile pending Ascend recipe PRs separately.

Once the official channel reads this source with compatible base recipes,
remove `validate.mjs` and replace the bootstrap job with the reusable
`vllm-project/recipes/.github/workflows/validate-recipes.yml@main`
(`source: vllm-ascend`, `path: recipes`).

For preview after registration, add a gitignored `sources.local.yaml` in
the renderer checkout:

```yaml
overrides:
  vllm-ascend: /path/to/vllm-ascend/recipes
```

Run `pnpm dev` and inspect the commands and guides together. An override
alone does not register a new source. During cutover, override the base
sources with matching local vLLM and vLLM-Omni recipes as well. Do not
commit generated `public/`, `.content/`, or `.sources/` files.
