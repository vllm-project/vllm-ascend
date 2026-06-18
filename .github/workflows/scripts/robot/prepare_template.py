#!/usr/bin/env python3
"""Step 2: Load the issue/PR template.

Reads the type from a file (written by step 1), loads the matching
template, and writes the human-readable template to a file.

Supports two input formats:
  - issue_type.txt: contains prefix like "[Bug]" → loads YAML issue template
  - pr_info.json: contains PR metadata → loads PR_TEMPLATE.md
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

TITLE_TO_TEMPLATE = {
    "[Bug]": "400-bug-report.yml",
    "[Installation]": "200-installation.yml",
    "[Usage]": "300-usage.yml",
    "[Doc]": "100-documentation.yml",
    "[Misc]": "800-others.yml",
}

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ISSUE_TEMPLATE"
PR_TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "PULL_REQUEST_TEMPLATE.md"


def load_issue_template(issue_type: str) -> str:
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
    if not PR_TEMPLATE_PATH.exists():
        return "(PR template file not found)"

    content = PR_TEMPLATE_PATH.read_text()
    return f"## PR Template\n{content}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare issue/PR template")
    parser.add_argument("--input", default="issue_type.txt", help="File containing the issue type or PR info")
    parser.add_argument("--output", default="template.txt", help="File to write the template to")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix == ".json":
        print("Loading PR template...")
        template_text = load_pr_template()
    else:
        issue_type = input_path.read_text().strip()
        print(f"Loading template for: {issue_type}")
        template_text = load_issue_template(issue_type)

    Path(args.output).write_text(template_text)
    print(f"Template prepared ({len(template_text)} chars)")


if __name__ == "__main__":
    main()
