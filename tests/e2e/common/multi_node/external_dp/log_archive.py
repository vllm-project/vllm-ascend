# SPDX-License-Identifier: Apache-2.0
import logging
import os
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)
ARCHIVE_INTERVAL_SECONDS = 30
ARCHIVE_JOIN_TIMEOUT_SECONDS = 1


def collect_logs(src_dir: Path, output_tar: Path) -> None:
    if not src_dir.exists():
        return
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    # A killed writer must leave the previous complete archive available to CI.
    with tempfile.NamedTemporaryFile(dir=output_tar.parent, suffix=".tar.gz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            archive.add(src_dir, arcname=src_dir.name)
        temporary_path.replace(output_tar)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def periodic_rank_log_archive(log_root: Path, node_index: int, interval: float = ARCHIVE_INTERVAL_SECONDS):
    log_prefix = os.environ.get("LOG_PREFIX")
    if not log_prefix:
        yield
        return
    src_dir = log_root / f"node-{node_index}"
    output_tar = Path(log_prefix) / f"node_{node_index}_external_dp_logs.tar.gz"
    stop_event = threading.Event()

    def snapshot():
        try:
            collect_logs(src_dir, output_tar)
        except Exception:
            logger.exception("Failed to snapshot external DP logs for node %d", node_index)

    def archive_periodically():
        while not stop_event.is_set():
            snapshot()
            if stop_event.wait(interval):
                break

    thread = threading.Thread(target=archive_periodically, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=ARCHIVE_JOIN_TIMEOUT_SECONDS)
