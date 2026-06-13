#!/usr/bin/env python3
"""Step 2: Load the issue template for the extracted issue type.

Reads issue_type from a file (written by step 1), loads the matching
YAML template, and writes the human-readable template to a file.
"""

import argparse
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


def load_template(issue_type: str) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare issue template")
    parser.add_argument("--input", default="issue_type.txt", help="File containing the issue type")
    parser.add_argument("--output", default="template.txt", help="File to write the template to")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    issue_type = input_path.read_text().strip()
    print(f"Loading template for: {issue_type}")

    template_text = load_template(issue_type)
    Path(args.output).write_text(template_text)
    print(f"Template prepared ({len(template_text)} chars)")


if __name__ == "__main__":
    main()
