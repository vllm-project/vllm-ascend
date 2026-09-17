#!/usr/bin/env python3
"""Set is_release flag in mkdocs config for tagged releases."""

import os
import pathlib
import sys

mkdocs_file = sys.argv[1] if len(sys.argv) > 1 else "mkdocs.yml"

if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag":
    p = pathlib.Path(mkdocs_file)
    text = p.read_text(encoding="utf-8")
    text = text.replace("is_release: false", "is_release: true")
    p.write_text(text, encoding="utf-8")
    print(f"[release] Set is_release: true in {mkdocs_file}")
else:
    print(f"[release] Not a tag build, skipping {mkdocs_file}")
