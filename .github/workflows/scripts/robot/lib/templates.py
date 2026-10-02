"""Issue and PR template loading for robot workflows."""

from pathlib import Path

import yaml

from .prefix_map import TITLE_TO_TEMPLATE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
TEMPLATE_DIR = REPO_ROOT / "ISSUE_TEMPLATE"
PR_TEMPLATE_PATH = REPO_ROOT / "PULL_REQUEST_TEMPLATE.md"


def load_issue_template(issue_type: str) -> str:
    """Load the YAML issue form template for *issue_type* and format it as
    a human-readable Markdown checklist.

    Args:
        issue_type: A prefix from ``TITLE_TO_TEMPLATE`` (e.g. ``"[Bug]"``).

    Returns:
        Formatted template text, or a placeholder message if the template
        file does not exist.
    """
    template_file = TITLE_TO_TEMPLATE.get(issue_type)
    if not template_file:
        return "(No template found for this issue type)"

    path = TEMPLATE_DIR / template_file
    if not path.exists():
        return f"(Template file {template_file} not found)"

    with open(path) as f:
        data = yaml.safe_load(f)

    lines = [f"## Template: {data.get('name', template_file)}"]
    lines.append(f"Description: {data.get('description', 'N/A')}")
    lines.append("")

    for field in data.get("body", []):
        if field.get("type") == "markdown":
            continue
        label = field.get("attributes", {}).get("label", "(unnamed)")
        required = field.get("validations", {}).get("required", False)
        req_mark = " (required)" if required else ""
        desc = field.get("attributes", {}).get("description", "")
        lines.append(f"- **{label}**{req_mark}")
        if desc:
            desc_short = desc[:200] + "..." if len(desc) > 200 else desc
            lines.append(f"  {desc_short}")
        lines.append("")

    return "\n".join(lines)


def load_pr_template() -> str:
    """Load the Pull Request template Markdown file.

    Returns:
        The template contents prefixed with a heading, or a placeholder
        message if the file does not exist.
    """
    if not PR_TEMPLATE_PATH.exists():
        return "(PR template file not found)"

    content = PR_TEMPLATE_PATH.read_text()
    return f"## PR Template\n{content}"
