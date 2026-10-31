import subprocess
import sys
from types import SimpleNamespace

import torch

from tests.e2e.pull_request.one_card.rlhf import conftest as rlhf


def test_server_does_not_block_when_stderr_exceeds_pipe_capacity(tmp_path, monkeypatch, capfd):
    ready = tmp_path / "ready"
    child = tmp_path / "noisy_server.py"
    child.write_text(
        "import pathlib, sys, time\n"
        "sys.stderr.write('x' * (2 * 1024 * 1024))\n"
        "sys.stderr.flush()\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "time.sleep(30)\n"
    )
    real_popen = subprocess.Popen

    def launch(_cmd, **kwargs):
        return real_popen([sys.executable, str(child)], **kwargs)

    def health(*args, **kwargs):
        return SimpleNamespace(status_code=200 if ready.exists() else 503)

    monkeypatch.setattr(
        rlhf,
        "subprocess",
        SimpleNamespace(
            Popen=launch, TimeoutExpired=subprocess.TimeoutExpired, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE
        ),
    )
    monkeypatch.setattr(rlhf, "requests", SimpleNamespace(get=health))
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda *args: (1, 1), raising=False)
    with rlhf.server(timeout=3):
        assert ready.exists()
    assert len(capfd.readouterr().err) >= 2 * 1024 * 1024
