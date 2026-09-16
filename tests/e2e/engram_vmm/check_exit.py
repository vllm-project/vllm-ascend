# SPDX-License-Identifier: Apache-2.0
"""Normal process exit releases owner descriptors; same-path fresh restart."""

import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="engram-exit-", dir="/dev/shm"))
code = """
import sys,torch,torch_npu
from types import SimpleNamespace
from vllm_ascend.models.deepseek_v41.engram_vmm.mapping import VmmMapping
torch.npu.set_device(0)
m=VmmMapping(sys.argv[1],17,256,torch.int8,SimpleNamespace(rank=0))
m.publish(0,torch.ones(17,256,dtype=torch.int8))
print("OWNER_READY",flush=True)
# Intentionally leave the live tensor/mapping to the process-exit fallback.
"""
for cycle in range(2):
    p = subprocess.run(
        [sys.executable, "-u", "-c", code, str(root / "table")], text=True, capture_output=True, timeout=90
    )
    print(p.stdout, flush=True)
    assert p.returncode == 0, p.stderr
    assert "ENGRAM_VMM_EXIT_CLEAN" in p.stdout, p.stdout
    assert not list(root.iterdir()), list(root.iterdir())
root.rmdir()
print("EXIT_RESTART_PASS", flush=True)
