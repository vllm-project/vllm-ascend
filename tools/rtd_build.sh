#!/usr/bin/env bash
# tools/rtd_build.sh
#
# Single build entry point for Read the Docs. Works in two contexts:
#
#   1. RTD CI: $READTHEDOCS_PROJECT is set to the project slug, and
#      $READTHEDOCS_OUTPUT points to the staging directory RTD will
#      publish. The script auto-detects Chinese mode when
#      $READTHEDOCS_PROJECT == "vllm-ascend-cn" and otherwise builds the
#      English site.
#
#   2. Local dev: invoke with DOCS_LANG=zh (or en) to override the
#      auto-detect. Output goes to ./site by default; set SITE_DIR to
#      redirect.
#
# All DOCS_* env vars consumed by mkdocs.yml's !ENV tags and
# docs/hooks/nav_titles.py are set here, in one place. The previous
# design had each language's env vars duplicated across two
# .readthedocs.*.yaml files; this script collapses that to a single
# dispatch.

set -euo pipefail

# --- Language detection ----------------------------------------------------
# Explicit override (DOCS_LANG) wins, so local dev can bypass the
# RTD-only project-slug heuristic.
if [ -z "${DOCS_LANG:-}" ]; then
    case "${READTHEDOCS_PROJECT:-}" in
        vllm-ascend-cn) DOCS_LANG=zh ;;
        *)              DOCS_LANG=en ;;
    esac
fi
export DOCS_LANG

# --- DOCS_IS_RELEASE for tagged builds -------------------------------------
# RTD sets READTHEDOCS_VERSION_TYPE=tag for tagged releases. The HTML
# override in docs/overrides/main.html uses config.extra.is_release to
# decide whether to render the "developer preview" banner.
export DOCS_IS_RELEASE=$( [ "${READTHEDOCS_VERSION_TYPE:-}" = tag ] && echo true || echo false )

# --- Chinese-mode settings -------------------------------------------------
if [ "$DOCS_LANG" = "zh" ]; then
    export DOCS_DIR=docs/source/zh
    export DOCS_MACROS_DIR=docs/source/zh
    export DOCS_SITE_NAME="vLLM Ascend (中文)"
    export DOCS_SITE_URL="https://docs.vllm.ai/projects/ascend/zh/latest/"
    export DOCS_THEME_LANG=zh
    export DOCS_PALETTE_LIGHT_TOGGLE_NAME="切换到深色模式"
    export DOCS_PALETTE_DARK_TOGGLE_NAME="切换到浅色模式"

    # Generate Chinese markdown from .po files. Must run before
    # mkdocs so the generated sources exist when mkdocs scans
    # DOCS_DIR.
    echo "[rtd-build] Generating Chinese sources from .po files..."
    python tools/generate_zh_docs.py

    # Mirror shared static assets into the Chinese DOCS_DIR. mkdocs resolves
    # `extra_css` and other static paths relative to DOCS_DIR, so anything
    # we want served on both language builds must physically live under
    # docs/source/zh/. The English DOCS_DIR is docs/source/, so the source
    # of truth stays there and we copy at build time.
    echo "[rtd-build] Mirroring shared static assets into docs/source/zh/..."
    mkdir -p docs/source/zh/stylesheets
    cp docs/source/stylesheets/extra.css docs/source/zh/stylesheets/extra.css
fi

# --- Output directory ------------------------------------------------------
# Default to ./site for local dev. RTD sets $READTHEDOCS_OUTPUT to its
# staging dir; we write the html/ subdir there so RTD's post-build
# index.html check finds the file at $READTHEDOCS_OUTPUT/html/index.html.
SITE_DIR="${SITE_DIR:-${READTHEDOCS_OUTPUT:+${READTHEDOCS_OUTPUT}/html}}"
SITE_DIR="${SITE_DIR:-$(pwd)/site}"

# --- Run mkdocs ------------------------------------------------------------
echo "[rtd-build] Building docs (DOCS_LANG=$DOCS_LANG, DOCS_IS_RELEASE=$DOCS_IS_RELEASE, site_dir=$SITE_DIR)..."
exec mkdocs build \
    --site-dir "$SITE_DIR" \
    --clean \
    --strict
