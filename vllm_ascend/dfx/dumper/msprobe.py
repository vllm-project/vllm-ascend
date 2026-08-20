#
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

"""Msprobe debugger bridge: JSON dump_enable, AclGraph switch, start/finalize."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager, suppress
from pathlib import Path

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


class MsprobeBridgeMixin:
    """Mixin: PrecisionDebugger / AclGraphDumper lifecycle and dump_enable I/O."""

    def _init_debugger(self, cudagraph_mode: CUDAGraphMode):
        """Best-effort msprobe debugger init. Never raises — dump stays optional.

        Skip construction when DFX ``dump.enabled=false`` even if
        ``dump_config_path`` is set — PrecisionDebugger may wrap torch APIs
        when msprobe ``dump_enable`` is omitted (treated as on). Hot-reload
        that flips dump on retries via ``Dumper._enforce_dump_requires_debugger``.

        Missing ``dump_config_path`` or msprobe import/construct failure leaves
        ``_debugger=None``. Callers that want dump must then force
        ``dump.enabled=false`` (see ``Dumper._enforce_dump_requires_debugger``).
        """
        if not bool(self.dfx_config.dump_enabled()):
            self._debugger = None
            self._uses_aclgraph_dumper = False
            return None
        dump_cfg = self.runner.ascend_config.dump_config_path
        if dump_cfg is None:
            self._debugger = None
            self._uses_aclgraph_dumper = False
            return None
        try:
            if cudagraph_mode == CUDAGraphMode.NONE:
                from msprobe.pytorch import PrecisionDebugger

                self._debugger = PrecisionDebugger(dump_cfg)
                self._uses_aclgraph_dumper = False
                return self._debugger

            from msprobe.pytorch import AclGraphDumper

            self._debugger = AclGraphDumper(dump_cfg)
            self._uses_aclgraph_dumper = True
            return self._debugger
        except Exception as exc:
            logger.error(
                "[Anomaly msprobe] debugger init failed (dump will stay off until "
                "msprobe is available and dump.enabled is re-enabled): %s",
                exc,
            )
            self._debugger = None
            self._uses_aclgraph_dumper = False
            return None

    def _is_aclgraph_dumper(self) -> bool:
        return bool(getattr(self, "_uses_aclgraph_dumper", False))

    @staticmethod
    def _clear_aclgraph_stats() -> None:
        """Drop buffered acl stats so the next step() only covers the dump window."""
        try:
            from msprobe.pytorch.aclgraph_dump import get_acl_stat_dict

            get_acl_stat_dict(clear=True)
        except Exception:
            pass

    def _ensure_aclgraph_hooks(self) -> None:
        """Install AclGraphDumper patches once **before** graph capture.

        Hooks must be active (``_running=True``) during capture so ACLGraph
        replay includes dump instrumentation. Tensor write is gated by device
        ``switch`` (``dump_enable``); ``step()`` archives staging into ``step*``.
        """
        if self._debugger is None or not self._is_aclgraph_dumper():
            return
        if self._aclgraph_hooks_installed:
            return
        model = getattr(self.runner, "model", None)
        if model is None:
            return
        self._debugger.start(model)
        self._aclgraph_hooks_installed = True
        self._clear_aclgraph_stats()
        logger.info(
            "[Anomaly msprobe] AclGraphDumper hooks installed (step gated by dump window) %s",
            self.dump_rank_tag(),
        )

    def start_dump_data(self) -> None:
        if self._debugger is None:
            return

        # ACLGraph: install hooks early (load_model) with collection enabled so
        # capture/replay include dump ops. Eager PrecisionDebugger must NOT
        # start unconditionally — that breaks Dynamo/AOT in profile_run.
        if self._is_aclgraph_dumper():
            self._ensure_aclgraph_hooks()
            if self._msprobe_dump_active:
                # Discard stats from non-dump steps; finalize.step writes this window.
                self._clear_aclgraph_stats()
                self._debugger_started = True
        elif self._msprobe_dump_active and not self._debugger_started:
            self._debugger.start(self.runner.model)
            self._debugger_started = True

        will_mark_forward_seen = bool(self._msprobe_dump_active and self._dump_needs_forward)
        if will_mark_forward_seen:
            self._dump_forward_seen = True
            # Never log full token lists here — that stalls TP0 before forward
            # collectives and hangs the request. I/O details go to DFX report.
            logger.info(
                "[Anomaly msprobe] start dump-forward %s %s",
                self.dump_rank_tag(),
                self._dump_state_tag(),
            )

    def finalize_dump_data(self, *, dump: bool = True) -> None:
        if self._debugger is None or not self._debugger_started:
            return
        dumping = bool(self._msprobe_dump_active)
        is_acl = self._is_aclgraph_dumper()
        # PrecisionDebugger: stop then step. AclGraphDumper: never stop here —
        # stop would drop hooks needed for graph replay / later dump windows.
        if not is_acl and hasattr(self._debugger, "stop"):
            self._debugger.stop()
            self._debugger_started = False

        try:
            if dump:
                self._debugger.step()
            else:
                self._debugger.step(dump=False)
        except Exception as exc:
            # Multi-rank races on shared msprobe JSON must not kill the worker.
            logger.error(
                "[Anomaly msprobe] debugger.step failed %s error=%s",
                self.dump_rank_tag(),
                exc,
            )

        if is_acl:
            # Keep hooks + _running for future replay; only clear the step flag.
            self._debugger_started = False

        # capture/dummy (dump=False): must not consume the pending dump-forward window.
        if not dump:
            if self._dump_needs_forward:
                self._dump_forward_seen = False
            return
        # Barrier before flipping shared dump_enable=false: peer step() may
        # re-read the JSON (AclGraphDumper._refresh_dump_enable) and skip write.
        if dumping and self._use_pending_dump_sync():
            try:
                tp_group = get_tp_group()
                if tp_group.world_size > 1:
                    torch.distributed.barrier(group=tp_group.cpu_group)
            except Exception as exc:
                logger.warning(
                    "[Anomaly msprobe] dump finalize barrier failed %s error=%s",
                    self.dump_rank_tag(),
                    exc,
                )
        dump_path = getattr(self._debugger, "dump_path", None)
        if dumping and dump_path:
            logger.info(
                "[Anomaly msprobe] step done %s dump_path=%s (check step*/rank*/ under this dir)",
                self.dump_rank_tag(),
                dump_path,
            )
        self.disable_msprobe_dump_if_needed()
        if dumping:
            logger.debug(
                "[Anomaly msprobe] finalize after dump-forward %s %s",
                self.dump_rank_tag(),
                self._dump_state_tag(),
            )

    @contextmanager
    def lock_msprobe_config(self, config_path: Path):
        lock_path = Path(f"{config_path}.lock")
        os.makedirs(lock_path.parent, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def disable_msprobe_dump_if_needed(self) -> None:
        if not self._msprobe_dump_active:
            return
        if self._debugger is None:
            return
        # Async check may enable dump after this step's start (or even after
        # forward). Keep dump_enable until a later start→finalize pair runs.
        if self._dump_needs_forward and not self._dump_forward_seen:
            logger.debug(
                "[Anomaly msprobe] disable deferred (needs forward) %s %s",
                self.dump_rank_tag(),
                self._dump_state_tag(),
            )
            return
        if not self.set_msprobe_dump_state(False):
            return
        self._msprobe_dump_active = False
        self._dump_needs_forward = False
        self._dump_forward_seen = False
        logger.info(
            "[Anomaly msprobe] disable succeeded %s",
            self.dump_rank_tag(),
        )

    def set_msprobe_dump_state(self, dump_state: bool) -> bool:
        """Write dump_enable and reload debugger config under the same lock.

        Reload must stay next to the write: any work between them (logging,
        sample-param dump, another thread's start/finalize, or another TP
        flipping the shared JSON) can make start() see a stale in-memory flag.
        """
        dump_cfg = self.runner.ascend_config.dump_config_path
        if not dump_cfg:
            logger.error("[Anomaly msprobe] set msprobe dump state failed, because dump_config_path is empty")
            return False

        config_path = Path(dump_cfg)
        if not config_path.exists():
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, because config file not found. path=%s",
                str(config_path),
            )
            return False

        try:
            with self.lock_msprobe_config(config_path):
                with config_path.open("r", encoding="utf-8") as f:
                    config_obj = json.load(f)

                if not isinstance(config_obj, dict):
                    logger.error(
                        "[Anomaly msprobe] set msprobe dump state failed, because json root is not object. type=%s",
                        type(config_obj).__name__,
                    )
                    return False

                ori_value = config_obj.get("dump_enable")
                if ori_value != dump_state:
                    config_obj["dump_enable"] = dump_state
                    # Atomic replace: open("w") truncates first and other ranks'
                    # AclGraphDumper.step()/_refresh_dump_enable may read empty JSON.
                    self._atomic_write_msprobe_json(config_path, config_obj)
                # Reload while still holding the lock so this process picks up
                # the value we just wrote before another rank can change it.
                # PrecisionDebugger: ``_maybe_reload_config``.
                # AclGraphDumper: no such helper — sync device ``switch`` below.
                if self._debugger is not None:
                    maybe_reload = getattr(self._debugger, "_maybe_reload_config", None)
                    if callable(maybe_reload):
                        maybe_reload(force=True)
                    else:
                        self._sync_aclgraph_dump_enable(dump_state)
            return True
        except Exception as e:
            logger.error(
                "[Anomaly msprobe] set msprobe dump state failed, path=%s error=%s",
                str(config_path),
                e,
            )
            return False

    def _sync_aclgraph_dump_enable(self, dump_state: bool) -> None:
        """Align AclGraphDumper in-memory ``dump_enable`` / ``switch`` with JSON.

        Graph replay gates tensor saves via the device ``switch`` tensor. If we
        only flip the shared JSON, ``step()`` may refresh ``switch`` to 1 during
        the dump window, then DFX writes ``dump_enable=false`` but never calls
        ``step()`` again — ``switch`` sticks at 1 and every forward keeps writing
        staging ``dump_tensor_data`` with no ``step*`` archive.
        """
        if self._debugger is None or not self._is_aclgraph_dumper():
            return
        dbg = self._debugger
        enabled = bool(dump_state)
        try:
            if hasattr(dbg, "dump_enable"):
                dbg.dump_enable = enabled
            switch = getattr(dbg, "switch", None)
            if switch is not None and hasattr(switch, "fill_"):
                switch.fill_(int(enabled))
            # Keep config signature current so a later ``step()`` refresh does
            # not treat disk as unchanged while switch was set from DFX.
            get_sig = getattr(dbg, "_get_config_signature", None)
            cfg_path = getattr(dbg, "config_path", None)
            if callable(get_sig) and cfg_path:
                dbg._config_signature = get_sig(cfg_path)
        except Exception as exc:
            logger.warning(
                "[Anomaly msprobe] sync AclGraphDumper switch failed dump_enable=%s %s error=%s",
                enabled,
                self.dump_rank_tag(),
                exc,
            )

    @staticmethod
    def _atomic_write_msprobe_json(config_path: Path, config_obj: dict) -> None:
        """Write JSON via temp file + ``os.replace`` so concurrent readers never see a truncated file."""
        tmp_path = config_path.with_name(f"{config_path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(config_obj, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, config_path)
        finally:
            if tmp_path.exists():
                with suppress(OSError):
                    tmp_path.unlink()
