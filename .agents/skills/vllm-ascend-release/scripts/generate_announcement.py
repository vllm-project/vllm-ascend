#!/usr/bin/env python3
"""
Generate release announcement for broadcasting.

This script creates a formatted announcement message
suitable for various channels (GitHub, blog, social media).

Usage:
    python generate_announcement.py \
        --version v0.15.0rc1 \
        --release-notes release-notes.md \
        --output announcement.md
"""

import argparse
import re
from pathlib import Path


def extract_highlights(release_notes: str) -> list[str]:
    """Extract key highlights from release notes."""
    highlights = []

    # Look for ### Highlights section
    match = re.search(r"###\s+Highlights?\s*\n(.*?)(?=\n###|\Z)", release_notes, re.DOTALL)
    if match:
        section = match.group(1)
        # Extract bullet points
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                highlights.append(line[2:])

    return highlights[:5]  # Limit to 5 highlights


def generate_github_announcement(version: str, highlights: list[str]) -> str:
    """Generate GitHub release announcement."""
    lines = [
        f"# vLLM Ascend {version} Released!",
        "",
        f"We are excited to announce the release of vLLM Ascend {version}!",
        "",
        "## Highlights",
        "",
    ]

    for h in highlights:
        lines.append(f"- {h}")

    lines.extend(
        [
            "",
            "## Getting Started",
            "",
            "Install via pip:",
            "```bash",
            f"pip install vllm-ascend=={version.lstrip('v')}",
            "```",
            "",
            "Or use Docker:",
            "```bash",
            f"docker pull quay.io/ascend/vllm-ascend:{version}",
            "```",
            "",
            "## Documentation",
            "",
            "- [Official Documentation](https://docs.vllm.ai/projects/ascend/en/latest)",
            "- [Release Notes](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html)",
            "",
            "## Feedback",
            "",
            "Please report any issues or provide feedback on our [GitHub Issues](https://github.com/vllm-project/vllm-ascend/issues).",
            "",
            "Thank you to all contributors who made this release possible!",
        ]
    )

    return "\n".join(lines)


def generate_short_announcement(version: str, highlights: list[str]) -> str:
    """Generate short announcement for social media."""
    lines = [
        f"🚀 vLLM Ascend {version} is now available!",
        "",
        "Key highlights:",
    ]

    for h in highlights[:3]:
        lines.append(f"• {h}")

    lines.extend(
        [
            "",
            f"📦 pip install vllm-ascend=={version.lstrip('v')}",
            "📖 https://docs.vllm.ai/projects/ascend",
            "",
            "#vLLM #Ascend #NPU #LLM #AI",
        ]
    )

    return "\n".join(lines)


def generate_chinese_announcement(version: str, highlights: list[str]) -> str:
    """Generate Chinese announcement."""
    lines = [
        f"# vLLM Ascend {version} 发布！",
        "",
        f"我们很高兴地宣布 vLLM Ascend {version} 正式发布！",
        "",
        "## 主要更新",
        "",
    ]

    for h in highlights:
        lines.append(f"- {h}")

    lines.extend(
        [
            "",
            "## 快速开始",
            "",
            "通过 pip 安装：",
            "```bash",
            f"pip install vllm-ascend=={version.lstrip('v')}",
            "```",
            "",
            "或使用 Docker：",
            "```bash",
            f"docker pull quay.io/ascend/vllm-ascend:{version}",
            "```",
            "",
            "## 文档",
            "",
            "- [官方文档](https://docs.vllm.ai/projects/ascend/en/latest)",
            "- [发布说明](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html)",
            "",
            "## 反馈",
            "",
            "如有问题或建议，请在 [GitHub Issues](https://github.com/vllm-project/vllm-ascend/issues) 反馈。",
            "",
            "感谢所有贡献者的付出！",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate release announcement")
    parser.add_argument("--version", required=True, help="Release version")
    parser.add_argument("--release-notes", required=True, help="Release notes file")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    release_notes_path = Path(args.release_notes)
    if release_notes_path.exists():
        with open(release_notes_path) as f:
            release_notes = f.read()
        highlights = extract_highlights(release_notes)
    else:
        print(f"Warning: Release notes not found at {release_notes_path}")
        highlights = ["Various improvements and bug fixes"]

    if not highlights:
        highlights = ["Various improvements and bug fixes"]

    print(f"Generating announcement for {args.version}...")
    print(f"Found {len(highlights)} highlights")

    output_lines = [
        "=" * 60,
        "GITHUB RELEASE ANNOUNCEMENT",
        "=" * 60,
        "",
        generate_github_announcement(args.version, highlights),
        "",
        "=" * 60,
        "SHORT ANNOUNCEMENT (Social Media)",
        "=" * 60,
        "",
        generate_short_announcement(args.version, highlights),
        "",
        "=" * 60,
        "CHINESE ANNOUNCEMENT",
        "=" * 60,
        "",
        generate_chinese_announcement(args.version, highlights),
    ]

    output = "\n".join(output_lines)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output)

    print(f"Announcement saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
