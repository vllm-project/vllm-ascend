# vLLM Ascend recipes

This directory holds Huawei-scoped overlays for
[recipes.vllm.ai](https://recipes.vllm.ai), following the migration in
[recipes PR #1160](https://github.com/vllm-project/recipes/pull/1160).
The initial import contains the Ascend portions of nine recipes already
merged in vLLM. MiniMax-H3 and other Omni content, pending recipe PRs, and
additional tutorials are deferred to later migrations.

The [recipes renderer](https://github.com/vllm-project/recipes) owns the site,
command builder, JSON API, schema, and shared vocabulary. Adding this
directory does not register a website source or change published pages.

## Layout and ownership

```text
recipes/
├── README.md
├── manifest.yaml
└── overlays/
    └── <hf_org>/<hf_repo>.yaml
```

- `manifest.yaml` declares the schema version, matching vLLM's manifest.
  The renderer's `sources.yaml` grants the source's role and hardware scope.
- An overlay extends the base recipe with the same case-sensitive Hugging
  Face ID. vLLM retains shared metadata and GPU settings; vLLM-Ascend owns
  the Ascend hardware settings, variants, dependencies, and guide sections.
- Overlays are additive and cannot change NVIDIA or AMD settings. New
  variants must explicitly restrict `supported_hardware` to Ascend profiles.
  PR #1160 allows entries identical to the base as no-ops during cutover;
  conflicting values remain errors.
- Qwen3.8-27B uses `guide_sections.ascend`, matching the base guide marker
  `<!-- overlay:vllm-ascend:ascend -->` in the companion vLLM split.
- Complete Ascend-only recipes may later live under `models/` if they have
  no other owner and all hardware and variant support stays in Huawei scope.
  No complete recipes are included in this initial migration.

## Imported content and provenance

All nine overlays come from the merged vLLM recipes at
[`52a65d7b`](https://github.com/vllm-project/vllm/tree/52a65d7b19e9e3ad86bf1e3b6954e4e11eeec7da/recipes/models):

- Qwen3-30B-A3B, Qwen3.5-397B-A17B, Qwen3.6-27B, Qwen3.8-27B
- DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp
- Kimi-K3
- GLM-5.3, GLM-5.3-Flash

Each overlay links to its fixed source file. The initial extraction used
the official renderer's `split-overlay.mjs` at
[`b08ad738`](https://github.com/vllm-project/recipes/commit/b08ad73889f9e33319d8ac9c887dec6a0975c55d),
with Qwen3.8's section name aligned to PR #1160's `ascend` marker.
Configuration values are preserved. Existing `verified` labels and
measurements describe source evidence, not new NPU validation for this import.

Only Qwen3.8-27B has standalone Ascend guide sections in this batch.
Scattered Ascend prose and reference links remain in the other base guides,
as in PR #1160. Moving those sections needs separate marker changes.

## Editing and validation

Follow the renderer's [contribution guide](https://github.com/vllm-project/recipes/blob/main/CONTRIBUTING.md),
[authoring guide](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md),
[schema](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md), and
[taxonomy](https://github.com/vllm-project/recipes/blob/main/taxonomy.yaml).
Keep shared metadata in the base recipe and schema/command-builder changes
in the renderer. Link to [model tutorials](../docs/source/tutorials/models)
for deployment details and record actual hardware validation for new settings.

The `Recipes` PR check calls the renderer's reusable
`validate-recipes.yml` workflow with `source: vllm-ascend` and `path: recipes`.
While source registration is pending, `renderer-ref` is pinned to PR #1160's
[`9ffe8f5d`](https://github.com/vllm-project/recipes/commit/9ffe8f5da08b1f2ae862464805ebc487529c34d4).
This is an **unmerged preview renderer**, whose lock uses the companion
vLLM split at
[`7846aee8`](https://github.com/esmeetu/vllm/commit/7846aee877aa763108476ed035d410fe8e175bca).
The validator replaces its Ascend source with the recipes under review,
checks ownership, Huawei scope, conflicts and orphaned overlays, then builds
the schema-validated JSON API. It does not enforce a fixed import count.

Run the same validation in a fresh, disposable renderer checkout:

```bash
git clone https://github.com/vllm-project/recipes.git recipes-renderer-check
cd recipes-renderer-check
git fetch origin pull/1160/head
git checkout --detach 9ffe8f5da08b1f2ae862464805ebc487529c34d4
pnpm install --frozen-lockfile
node scripts/validate-source.mjs --source vllm-ascend --dir /path/to/vllm-ascend/recipes
```

Use a disposable checkout because this renderer version writes and removes
`.content/validate`. Clear inherited `RECIPES_*` overrides when reproducing CI.
Static validation does not establish NPU correctness or performance.

## Staged source cutover

Follow PR #1160's order, validating compatible source pins at each step:

1. Merge these nine overlays into `vllm-project/vllm-ascend@main`.
2. Merge the renderer changes with official vLLM and vLLM-Ascend sources,
   keeping vLLM's Ascend content and leaving `exclude_brands` disabled.
   Identical overlay entries are no-ops during this overlap.
3. Merge the companion vLLM change that removes those Ascend fields and
   inserts the matching guide marker. The source-lock update then picks up
   the split, and the overlays supply the Ascend content.
4. Enable `scope.exclude_brands: [Huawei]` for the vLLM source so new
   structured Ascend settings must be contributed to vLLM-Ascend.

The renderer registration is:

```yaml
vllm-ascend:
  repo: vllm-project/vllm-ascend
  ref: main
  path: recipes
  role: overlay
  scope: { brands: [Huawei] }
```

Update the renderer's `sources.lock.json` to official revisions rather than
publishing its preview fork pins. Once renderer `main` includes registration
and identical-entry support, remove `renderer-ref` from this repository's
workflow so validation follows the published source configuration.

Compare the generated commands and guide placement before and after the
split. Overlay keys are appended after base keys, so variant display order
can change (for example Kimi-K3), as documented in PR #1160. Reconcile pending
Ascend PRs separately after this source is available. Do not commit generated
`public/`, `.content/`, or `.sources/` files.
