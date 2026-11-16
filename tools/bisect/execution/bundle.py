# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Freeze tool code outside the target checkout before switching commits."""

import base64
import hashlib
import io
import zipfile
from pathlib import Path


def tool_bundle() -> tuple[str, str]:
    tools = Path(__file__).resolve().parents[2]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(tools.rglob("*.py")):
            if "__pycache__" not in path.parts:
                relative = path.relative_to(tools)
                if relative.parts[0] == "bisect" or relative == Path("__init__.py"):
                    bundle.writestr("tools/" + relative.as_posix(), path.read_bytes())
        requirements = tools / "bisect/build_analyzer/requirements.txt"
        bundle.writestr("tools/bisect/build_analyzer/requirements.txt", requirements.read_bytes())
    data = archive.getvalue()
    return hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode("ascii")


# This bootstrap only unpacks our own Python snapshot; it does not install packages.
# Data and test environment values travel on stdin rather than command arguments.
BOOTSTRAP = r"""
import base64, hashlib, io, json, os, pathlib, runpy, sys, tempfile, zipfile
request = json.load(sys.stdin)
data = base64.b64decode(request.pop("bundle"))
digest = request.pop("bundle_digest")
if hashlib.sha256(data).hexdigest() != digest:
    raise RuntimeError("Tool snapshot checksum mismatch")
owner = str(os.getuid()) if hasattr(os, "getuid") else "user"
root = pathlib.Path(tempfile.gettempdir()) / ("bisect-tools-" + digest + "-" + owner)
if root.exists() and root.is_symlink():
    raise RuntimeError("Tool snapshot path is a symlink")
root.mkdir(mode=0o700, exist_ok=True)
if os.name == "posix" and (root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077):
    raise RuntimeError("Tool snapshot directory permissions are not private")
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if root.resolve() not in target.parents:
            raise RuntimeError("Invalid tool snapshot path")
    archive.extractall(root)
sys.path.insert(0, str(root))
from tools.bisect.orchestration.node_agent import handle
response = handle(request)
print("\nBISECT-RESULT-" + request["nonce"] + " " + json.dumps(response), flush=True)
"""
