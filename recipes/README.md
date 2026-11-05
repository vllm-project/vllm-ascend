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
`validate-recipes.yml@main` workflow with `source: vllm-ascend` and
`path: recipes`. It uses the workflow's default `renderer-ref: main`, so
validation follows the renderer that publishes recipes.vllm.ai.

Once the Ascend source is registered, the validator replaces it with the
recipes under review, checks ownership, Huawei scope, conflicts and orphaned
overlays, then builds the schema-validated JSON API. Until registration lands
in renderer `main`, validation fails with
`has overlays/ but role is "core" (needs role: overlay)` because an unknown
source is treated as a standalone core source. Successful validation also
requires the staged cutover support described below.

Run the same validation in a fresh, disposable renderer checkout:

```bash
git clone --branch main --single-branch https://github.com/vllm-project/recipes.git recipes-renderer-check
cd recipes-renderer-check
pnpm install --frozen-lockfile
node scripts/validate-source.mjs --source vllm-ascend --dir /path/to/vllm-ascend/recipes
```

Use a disposable checkout because the validator writes and removes
`.content/validate`. Clear inherited `RECIPES_*` overrides when reproducing CI.
Static validation does not establish NPU correctness or performance.

## Pull request previews

The existing PR labeler adds the `recipe` label to PRs changing `recipes/**`.
Once enabled, the `Recipes Preview` workflow sends events to
`vllm-project/recipes`, which owns preview builds, deployment, and cleanup:

- PRs targeting `main`: opening, pushing, or reopening sends `preview-pr`
  with `source: vllm-ascend` and the PR number. Closing sends the same event
  so the renderer can delete the preview branch.
- Pushes to `main` changing `recipes/**`: `recipes-updated` requests a
  production source-lock update with `source: vllm-ascend`.

The workflow's `recipes-preview` environment provides a **View deployment**
link to
`https://vllm-recipes-git-preview-vllm-ascend-pr-<PR number>-inferact-inc.vercel.app`.
The renderer uses `preview/vllm-ascend-pr-<PR number>` as the preview branch.
The link can appear before the deployment is ready; allow a few minutes for
the build and check the preview banner for the rendered commit. A successful
dispatch only confirms that the event was accepted, not that a build passed.
Push runs link to [recipes.vllm.ai](https://recipes.vllm.ai); close events do
not advertise a preview URL.

### Administrator setup and activation

This workflow follows [vLLM PR #63259](https://github.com/vllm-project/vllm/pull/63259).
It is disabled until the repository Actions variable `RECIPES_PREVIEW_ENABLED`
is set to `true`. Leave it unset or `false` during setup: the original recipes
receiver ignores `source` and would interpret an Ascend PR number as a vLLM PR.

1. Before merging the labeler rule, create the `recipe` label in
   `vllm-project/vllm-ascend`. The existing PR labeler can apply labels but
   does not have the `issues: write` permission needed to create them. It
   runs independently of the preview activation variable.
2. In `vllm-project/recipes`, finish the Ascend source registration and
   compatible source-lock cutover described below. Extend the dispatch
   receiver to validate and route `source: vllm-ascend`, include the source
   in concurrency groups, and scan Ascend PRs with the `recipe` label in
   the scheduled fallback. Confirm production source updates include Ascend.
3. Reuse or create a dispatch GitHub App installed only on
   `vllm-project/recipes`, with **Contents: read and write**. Keep it separate
   from the publisher App. Confirm the receiver's rulesets prevent this App
   from creating, updating, or deleting non-preview branches and tags; it
   must not be a bypass actor. See the
   [existing rollout](https://github.com/vllm-project/recipes/pull/1169).
4. In `vllm-project/vllm-ascend`, create the `recipes-preview` environment.
   Restrict its deployment branches to `main` only, then add the environment
   variable `RECIPES_APP_ID` and environment secret `RECIPES_APP_PRIVATE_KEY`.
   The workflow mints a short-lived installation token scoped to `recipes`.
   It never checks out or executes PR code, including for fork PRs.
5. Set `RECIPES_PREVIEW_ENABLED=true` as a **repository Actions variable**
   (not an environment variable, since the job condition is evaluated before
   entering the environment). Only enable it after the receiver is ready.
6. Use a real overlay PR to verify the `recipe` label, initial preview,
   updated commit after a push, close cleanup, and rebuild after reopening.
   Verify that merging the PR updates the production source lock. A workflow
   rerun or a new push can trigger a preview for an existing PR after activation.

GitHub App credentials and environment settings are administrator actions;
adding these files does not configure them. Disabling the repository variable
stops new dispatches but does not remove existing preview branches. Preview
activation does not change the validation renderer, which follows `main`.

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
publishing its preview fork pins. This repository's workflow already follows
renderer `main`; source registration and identical-entry support must land
there for validation and the staged source cutover to succeed.

Compare the generated commands and guide placement before and after the
split. Overlay keys are appended after base keys, so variant display order
can change (for example Kimi-K3), as documented in PR #1160. Reconcile pending
Ascend PRs separately after this source is available. Do not commit generated
`public/`, `.content/`, or `.sources/` files.
