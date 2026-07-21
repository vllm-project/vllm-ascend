"""MkDocs macros plugin main module.

Exposes version variables from mkdocs.yml extra configuration as Jinja2 macros.
Variables are defined in the 'extra' section of mkdocs.yml.
"""

from pathlib import Path

# UI string translations keyed by DOCS_LANG. Injected into config.extra.i18n
# in define_env() so override templates (e.g. docs/overrides/main.html) can
# read them via config.extra.i18n[config.extra.docs_lang]. The macros
# plugin's env.variables are only available in markdown content, not in
# override templates, hence the detour through config.extra.
UI_STRINGS = {
    "en": {
        "banner_preview": "You are viewing the latest developer preview docs.",
        "banner_link": "Click here",
        "banner_suffix": "to view docs for the latest stable release (v0.18.0).",
        "btn_report": "report issue",
        "aria_report": "Report a documentation issue",
    },
    "zh": {
        "banner_preview": "您正在查看最新的开发者预览版文档。",
        "banner_link": "点击此处",
        "banner_suffix": "查看最新稳定版（v0.18.0）的文档。",
        "btn_report": "反馈问题",
        "aria_report": "报告文档问题",
    },
}


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

    # Declare variables (these are also in mkdocs.yml 'extra' section)
    env.variables["vllm_version"] = env.variables.get("vllm_version", "v0.22.1")
    env.variables["vllm_ascend_version"] = env.variables.get("vllm_ascend_version", "v0.22.1rc1")
    env.variables["pip_vllm_ascend_version"] = env.variables.get("pip_vllm_ascend_version", "0.22.1rc1")
    env.variables["pip_vllm_version"] = env.variables.get("pip_vllm_version", "0.22.1")
    env.variables["cann_image_tag"] = env.variables.get("cann_image_tag", "9.0.1-910b-ubuntu22.04-py3.12")
    env.variables["main_python_version"] = env.variables.get("main_python_version", ">= 3.10, < 3.13")
    env.variables["main_cann_version"] = env.variables.get("main_cann_version", "9.0.1")
    env.variables["main_pytorch_torch_npu_version"] = env.variables.get(
        "main_pytorch_torch_npu_version", "2.10.0 / 2.10.0.post2"
    )
    env.variables["main_triton_ascend_version"] = env.variables.get("main_triton_ascend_version", "3.2.1")
    env.variables["main_vllm_commit"] = main_vllm_commit
    env.variables["main_vllm_tag"] = main_vllm_tag

    # Expose UI translations to override templates by attaching the dict
    # to config.extra. See module docstring for the rationale.
    config = env.variables.get("config")
    if config is not None:
        config.extra["i18n"] = UI_STRINGS

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
