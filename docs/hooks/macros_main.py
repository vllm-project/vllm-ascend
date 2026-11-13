"""MkDocs macros plugin main module.

Exposes version variables from mkdocs.yml extra configuration as Jinja2 macros.
Variables are defined in the 'extra' section of mkdocs.yml.
"""

from pathlib import Path

import regex as re

# UI string translations keyed by DOCS_LANG. Injected into config.extra.i18n
# in define_env() so override templates (e.g. docs/overrides/main.html) can
# read them via config.extra.i18n[config.extra.docs_lang]. The macros
# plugin's env.variables are only available in markdown content, not in
# override templates, hence the detour through config.extra.
UI_STRINGS = {
    "en": {
        # Banner shown if this build were treated as a developer preview
        # (defensive; release branches always render the release banner).
        "banner_preview": "You are viewing the latest developer preview documentation.",
        "banner_link": "Click here",
        "banner_suffix": "to access the official release version ({version}) documentation.",
        # Banner shown on this official release build, linking back to the
        # latest developer-preview (main) docs.
        "banner_release_preview": (
            "You are currently viewing the documentation for the official release version ({version})."
        ),
        "banner_release_suffix": "to access the latest developer preview documentation.",
        "btn_report": "report issue",
        "aria_report": "Report a documentation issue",
    },
    "zh": {
        # 若本构建被当作开发者预览版时展示（防御性；release 分支始终渲染发布版提示框）。
        "banner_preview": "您正在查看最新的开发者预览版文档。",
        "banner_link": "点击此处",
        "banner_suffix": "可访问正式发布版本（{version}）文档。",
        # 正式发布版本构建上展示的提示框，链接回最新的开发者预览版（main）文档。
        "banner_release_preview": "您正在查看正式发布版本({version})的文档。",
        "banner_release_suffix": "可访问最新的开发者预览版文档。",
        "btn_report": "反馈问题",
        "aria_report": "报告文档问题",
    },
}


def _is_release_version(version):
    """Return True when the Read the Docs version slug denotes a release.

    RTD publishes release branches as plain branches (READTHEDOCS_VERSION_TYPE
    is "branch", not "tag"), so READTHEDOCS_VERSION_TYPE cannot be used to spot
    an official release. The version slug itself can. Depending on how the RTD
    version was created the slug is one of:

      - "v0.30.0"            (short slug, matches the public /en/v0.30.0/ URL)
      - "releases-v0.30.0"   (auto-created from the branch, "/" replaced by "-")
      - "releases/v0.30.0"   (raw branch name)

    Release candidates ("releases/v0.30.0rc2") are not official releases and
    must not match: the trailing "rc..." suffix fails the fullmatch.
    "latest" (main) and PR-number slugs never match.
    """
    return bool(re.fullmatch(r"(?:releases[/-])?v\d+\.\d+\.\d+", str(version or "").strip()))


def define_env(env):
    """Define environment variables and macros for mkdocs-macros plugin."""

    # Load commit info from files
    repo_root = Path(__file__).resolve().parent.parent.parent
    commit_path = repo_root / ".github" / "vllm-main-verified.commit"
    tag_path = repo_root / ".github" / "vllm-release-tag.commit"

    main_vllm_commit = "unknown"
    main_vllm_tag = "unknown"

    if commit_path.exists():
        main_vllm_commit = commit_path.read_text(encoding="utf-8").strip()
    if tag_path.exists():
        main_vllm_tag = tag_path.read_text(encoding="utf-8").strip()

    # Static release/main stack variables come exclusively from mkdocs.yml so
    # maintainers have one place to update dependency versions.
    env.variables["main_vllm_commit"] = main_vllm_commit
    env.variables["main_vllm_tag"] = main_vllm_tag

    # Expose UI translations to override templates by attaching the dict
    # to config.extra. See module docstring for the rationale.
    config = env.variables.get("config")
    if config is not None:
        # is_release must be a real bool: RTD publishes release branches as
        # plain branches, and DOCS_IS_RELEASE would arrive as the string
        # "false"/"true" (always truthy), so it cannot be trusted here.
        config.extra["is_release"] = _is_release_version(config.extra.get("rtd_version"))
        stable_version = env.variables["stable_vllm_ascend_version"]
        # The current build's own version (v0.30.0 on this branch). Used by
        # the release banner to identify the docs being viewed.
        release_version = env.variables["vllm_ascend_version"]
        config.extra["i18n"] = {
            language: {
                **strings,
                "banner_suffix": strings["banner_suffix"].format(version=stable_version),
                "banner_release_preview": strings["banner_release_preview"].format(version=release_version),
            }
            for language, strings in UI_STRINGS.items()
        }

    @env.macro
    def include_code(file_path, start_after=None, end_before=None, language="python"):
        """Include a portion of a source file as a code block.

        Args:
            file_path: Path relative to the repository root.
            start_after: Include content after this marker line.
            end_before: Stop before this marker line.
            language: Code block language.
        """
        full_path = repo_root / file_path
        if not full_path.exists():
            return f"```\n# File not found: {file_path}\n```"

        lines = full_path.read_text(encoding="utf-8").splitlines()

        if start_after:
            try:
                start_idx = next(i for i, line in enumerate(lines) if start_after in line) + 1
            except StopIteration:
                start_idx = 0
        else:
            start_idx = 0

        if end_before:
            try:
                end_idx = next(i for i in range(start_idx, len(lines)) if end_before in lines[i])
            except StopIteration:
                end_idx = len(lines)
        else:
            end_idx = len(lines)

        content = "\n".join(lines[start_idx:end_idx])
        return f"```{language}\n{content}\n```"
