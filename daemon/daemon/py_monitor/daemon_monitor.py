#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import math
import queue
import time
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

from rich.table import Table
from rich.text import Text

from py_monitor.control_api import ControlApiServer, ControlCommand, parse_int_field
from py_monitor.env_config import env_float, env_scaled_size
from py_monitor.models import PoolSpec, PoolTuneOverride
from py_monitor.parsing import parse_gb_to_bytes, parse_mb_to_bytes, parse_pool_key, parse_pool_spec
from py_monitor.units import GB, MB, fmt_bytes


def _safe_enum_name(value) -> str:
    # pybind enums stringify inconsistently across versions; be defensive.
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    # textual prints like 'ModelState.REGISTERED'
    if "." in text:
        return text.split(".")[-1]
    return text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mdaemon monitor (nvitop-like TUI for multimodel daemon)")
    parser.add_argument(
        "--pool",
        action="append",
        type=parse_pool_spec,
        default=[],
        help=(
            "Initial pool spec device:granularity_mb:total_gb[:cap_gb] (repeatable). "
            "Examples: --pool 0:16:4 --pool 1:16:4:12. "
            "If cap_gb is omitted, only MDAEMON_DEVICE_TOTAL_CAP_GB is applied."
        ),
    )
    parser.add_argument("--refresh-ms", type=int, default=500, help="UI refresh interval in milliseconds. Default: 500ms")
    parser.add_argument(
        "--sync-model-shm",
        action="store_true",
        help="If set, calls snapshot_models(sync=True) to sync from shared memory each refresh",
    )
    parser.add_argument(
        "--control-enable",
        action="store_true",
        help="Enable localhost HTTP control API for pool operations",
    )
    parser.add_argument(
        "--control-host",
        type=str,
        default="127.0.0.1",
        help="Control API bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=18080,
        help="Control API bind port (default: 18080)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.pool:
        raise SystemExit("At least one --pool is required. Example: --pool 0:16:4 or --pool 0:16:4:10")

    try:
        import mdaemon_py  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Failed to import mdaemon_py. Build it first in multimodel/daemon via ./build_pybind.sh\n"
            f"Import error: {exc}"
        )

    from textual.app import App, ComposeResult, SystemCommand
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import DataTable, Header, Input, Label, RichLog, Static

    class PromptInputScreen(ModalScreen[Optional[str]]):
        def __init__(self, title: str, help_text: str,
                     placeholder: str) -> None:
            super().__init__()
            self._title = title
            self._help_text = help_text
            self._placeholder = placeholder

        CSS = """
        PromptInputScreen {
            align: center middle;
            background: rgba(0, 0, 0, 0.55);
        }
        #prompt_box {
            width: 84;
            height: auto;
            border: round #4ea1ff;
            padding: 1 2;
            background: #111a2e;
        }
        #prompt_title {
            color: #7be3ff;
            text-style: bold;
        }
        #prompt_help {
            color: #b6c7ff;
            margin: 1 0;
        }
        #prompt_input {
            margin-top: 1;
        }
        """

        def compose(self) -> ComposeResult:
            with Vertical(id="prompt_box"):
                yield Static(self._title, id="prompt_title")
                yield Static(self._help_text, id="prompt_help")
                yield Input(placeholder=self._placeholder, id="prompt_input")

        def on_mount(self) -> None:
            self.query_one("#prompt_input", Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.dismiss(event.value)

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    class PoolRow(Static):
        def __init__(self, key: Tuple[int, int]):
            super().__init__(classes="pool-row")
            self.key = key
            self.label = Label("")
            self.bar = Static("", classes="pool-bar")
            self.right = Label("")

        @staticmethod
        def _render_bar(used_h: int, total_h: int, width: int, height: int = 2) -> Text:
            total = max(1, total_h)
            ratio = max(0.0, min(1.0, used_h / total))
            filled = int(round(ratio * width))
            empty = max(0, width - filled)

            bar_text = Text()
            for row in range(height):
                row_text = Text()
                if filled > 0:
                    row_text.append("█" * filled, style="bold #3ad47a")
                if empty > 0:
                    row_text.append("░" * empty, style="#3b4f78")
                bar_text.append_text(row_text)
                if row < height - 1:
                    bar_text.append("\n")
            return bar_text

        def compose(self) -> ComposeResult:
            yield Horizontal(self.label, self.bar, self.right)

        def update_from_snapshot(self, snap) -> None:
            gran = int(snap.granularity)
            dev = int(snap.device_id)
            total_h = int(snap.total_handles)
            used_h = int(snap.used_handles)
            avail_b = int(snap.available_bytes)
            total_b = int(snap.total_bytes)

            self.label.update(f"dev{dev}  {gran // MB:>4}MB")

            # Responsive width: use current rendered bar widget width.
            # Fallback keeps first paint stable before layout finalizes.
            bar_width = max(10, int(getattr(self.bar.size, "width", 0) or 34) - 1)
            self.bar.update(
                self._render_bar(used_h=used_h, total_h=total_h, width=bar_width)
            )

            used_b = int(snap.used_bytes)
            ratio = (used_h / max(1, total_h)) * 100.0
            self.right.update(
                f"{ratio:5.1f}%  |  used {fmt_bytes(used_b):>8} / total {fmt_bytes(total_b):>8}  |  avail {fmt_bytes(avail_b):>8}"
            )

    class DaemonMonitor(App):
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+p", "open_palette", "Palette"),
            ("ctrl+shift+1", "pool_set_cap", "Set Pool Cap (local)"),
            ("ctrl+shift+2", "pool_set_tuning", "Set Pool Auto-Tuning (local)"),
            ("ctrl+shift+3", "pool_create", "Create Pool (local)"),
            ("ctrl+shift+4", "pool_show_info", "Show Pool Runtime Info (local)"),
        ]
        CSS = """
        Screen {
            background: #0b1020;
            color: #d7e3ff;
        }

        #pools {
            width: 58%;
            border: round #4ea1ff;
            padding: 0 1;
            margin: 0 1 0 0;
            background: #121a33;
        }

        #right_col {
            width: 42%;
            height: 1fr;
        }

        #models {
            border: round #67d17a;
            background: #111a2e;
            height: 58%;
            margin: 0 0 1 0;
        }

        #messages {
            border: round #ffb454;
            background: #111a2e;
            height: 42%;
            padding: 0 1;
        }

        #status_bar {
            dock: bottom;
            height: 1;
            background: #1b2440;
            padding: 0 1;
        }

        #status {
            width: 1fr;
            color: #7be3ff;
            text-style: bold;
        }

        #help {
            width: auto;
            color: #b6c7ff;
            content-align: right middle;
        }

        .pool-row {
            height: 3;
            border: round #3f5d8f;
            padding: 0 1;
            margin: 0 0 1 0;
            background: #192544;
        }

        .pool-bar {
            width: 1fr;
            margin: 0 1;
        }
        """

        def __init__(
            self,
            daemon,
            pool_specs: List[PoolSpec],
            refresh_ms: int,
            sync_model_shm: bool,
            control_enable: bool,
            control_host: str,
            control_port: int,
        ):
            super().__init__()
            self.daemon = daemon
            self.pool_specs = pool_specs
            self.refresh_ms = refresh_ms
            self.sync_model_shm = sync_model_shm
            self.control_enable = control_enable
            self.control_host = control_host
            self.control_port = control_port
            self.pool_caps: Dict[Tuple[int, int], int] = {
                (int(spec.device_id), int(spec.granularity_bytes)): int(spec.cap_bytes)
                for spec in pool_specs
            }
            self.pool_tune_overrides: Dict[Tuple[int, int], PoolTuneOverride] = {}
            self._known_pools: set[Tuple[int, int]] = set(self.pool_caps.keys())
            self._pending_cap_pool: Optional[Tuple[int, int]] = None
            self._pending_tune_pool: Optional[Tuple[int, int]] = None
            self._pending_create_pool: Optional[Tuple[int, int]] = None

            self.pool_rows: Dict[Tuple[int, int], PoolRow] = {}
            self.prev_model_msg_states: Dict[int, str] = {}
            self.message_lines: Deque[Tuple[str, str]] = collections.deque(maxlen=400)
            self.last_action = ""
            self.control_cmd_q: "queue.Queue[ControlCommand]" = queue.Queue()
            self.control_api: Optional[ControlApiServer] = None

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                self.pools_view = VerticalScroll(id="pools")
                yield self.pools_view
                with Vertical(id="right_col"):
                    self.models_table = DataTable(id="models")
                    self.messages_log = RichLog(id="messages", wrap=True, auto_scroll=True)
                    yield self.models_table
                    yield self.messages_log
            with Horizontal(id="status_bar"):
                self.status = Label("", id="status")
                self.help = Label("Palette: Ctrl+P | Quit: Q", id="help")
                yield self.status
                yield self.help

        def on_mount(self) -> None:
            for spec in self.pool_specs:
                self.daemon.initialize_handle_pool_device(
                    int(spec.granularity_bytes), int(spec.device_id), int(spec.total_bytes)
                )
            self.daemon.start()

            self.pools_view.border_title = " Handle Pools "
            self.models_table.border_title = " Models "
            self.messages_log.border_title = " Messages "
            self.models_table.zebra_stripes = True

            self.models_table.add_columns(
                "model_id",
                "state",
                "msg",
                "npuid",
                "osid",
                "alloc",
                "handles",
            )
            self._append_message("info", "monitor started")
            if self.control_enable:
                self.control_api = ControlApiServer(
                    host=self.control_host,
                    port=self.control_port,
                    command_queue=self.control_cmd_q,
                    snapshot_provider=self.daemon.snapshot_handle_pools,
                )
                self.control_api.start()
                self._append_message(
                    "info",
                    f"control API enabled at http://{self.control_host}:{self.control_port}",
                )
            self._set_status("initializing monitor...")
            self.set_interval(self.refresh_ms / 1000.0, self._refresh)

        def on_unmount(self) -> None:
            if self.control_api is not None:
                try:
                    self.control_api.stop()
                except Exception:
                    pass
            try:
                self.daemon.stop()
            except Exception:
                pass

        def _drain_control_commands(self) -> None:
            while True:
                try:
                    cmd = self.control_cmd_q.get_nowait()
                except queue.Empty:
                    break

                try:
                    def _find_pool_snapshot(dev: int, gran: int):
                        for snap in self.daemon.snapshot_handle_pools():
                            if int(snap.device_id) == dev and int(snap.granularity) == gran:
                                return snap
                        return None

                    if cmd.action == "create":
                        dev = parse_int_field(cmd.payload, "device_id", 0)
                        gran = parse_int_field(cmd.payload, "granularity", 1)
                        total_bytes = parse_int_field(cmd.payload, "total_bytes", 1)
                        cap_bytes = int(cmd.payload.get("cap_bytes", 0))
                        if cap_bytes < 0:
                            raise ValueError("cap_bytes must be >= 0")
                        if cap_bytes > 0 and cap_bytes < total_bytes:
                            raise ValueError("cap_bytes must be >= total_bytes")
                        pool_key = (dev, gran)
                        if self._pool_exists(pool_key):
                            cmd.response_queue.put({"ok": False, "status": 409, "error": "pool already exists"})
                            continue
                        self.daemon.initialize_handle_pool_device(int(gran), int(dev), int(total_bytes))
                        self._known_pools.add(pool_key)
                        if cap_bytes > 0:
                            self.pool_caps[pool_key] = cap_bytes
                        self._append_message("info", f"api create pool {self._pool_text(pool_key)} total={fmt_bytes(total_bytes)}")
                        cmd.response_queue.put({
                            "ok": True,
                            "status": 200,
                            "device_id": dev,
                            "granularity": gran,
                            "total_bytes": total_bytes,
                            "cap_bytes": cap_bytes,
                        })
                        continue

                    if cmd.action in ("extend", "remove"):
                        dev = parse_int_field(cmd.payload, "device_id", 0)
                        gran = parse_int_field(cmd.payload, "granularity", 1)
                        target_bytes = parse_int_field(cmd.payload, "target_bytes", 1)
                        pool_key = (dev, gran)
                        if not self._pool_exists(pool_key):
                            cmd.response_queue.put({"ok": False, "status": 404, "error": "pool not found"})
                            continue
                        snap = _find_pool_snapshot(dev, gran)
                        if snap is None:
                            cmd.response_queue.put({"ok": False, "status": 404, "error": "pool snapshot not found"})
                            continue
                        current_total_bytes = int(snap.total_bytes)
                        if cmd.action == "extend" and target_bytes <= current_total_bytes:
                            cmd.response_queue.put(
                                {
                                    "ok": False,
                                    "status": 409,
                                    "error": (
                                        f"extend target_bytes {target_bytes} is not above current total "
                                        f"{current_total_bytes}"
                                    ),
                                }
                            )
                            continue
                        if cmd.action == "remove" and target_bytes >= current_total_bytes:
                            cmd.response_queue.put(
                                {
                                    "ok": False,
                                    "status": 409,
                                    "error": (
                                        f"remove target_bytes {target_bytes} is not below current total "
                                        f"{current_total_bytes}"
                                    ),
                                }
                            )
                            continue

                        delta_bytes = abs(target_bytes - current_total_bytes)
                        count = int(math.ceil(delta_bytes / max(1, gran)))
                        if count <= 0:
                            cmd.response_queue.put(
                                {
                                    "ok": False,
                                    "status": 200,
                                    "message": "no-op: target does not require handle change",
                                    "current_total_bytes": current_total_bytes,
                                    "target_bytes": target_bytes,
                                }
                            )
                            continue

                        if cmd.action == "extend":
                            ok = bool(self.daemon.extend_handles(gran, dev, count))
                        else:
                            ok = bool(self.daemon.remove_handles(gran, dev, count))
                        self._append_message(
                            "info",
                            f"api {cmd.action} {self._pool_text(pool_key)} target={fmt_bytes(target_bytes)} count={count} -> {ok}",
                        )
                        cmd.response_queue.put(
                            {
                                "ok": bool(ok),
                                "status": 200 if ok else 409,
                                "device_id": dev,
                                "granularity": gran,
                                "count": count,
                                "current_total_bytes": current_total_bytes,
                                "target_bytes": target_bytes,
                                "action": cmd.action,
                            }
                        )
                        continue

                    cmd.response_queue.put({"ok": False, "status": 400, "error": f"unknown action: {cmd.action}"})
                except Exception as exc:
                    cmd.response_queue.put({"ok": False, "status": 400, "error": str(exc)})

        def action_open_palette(self) -> None:
            if hasattr(self, "action_command_palette"):
                self.action_command_palette()
                return
            self._set_status(
                "error: command palette is not available in this Textual version; update textual and retry"
                ,
                is_error=True,
            )

        def action_pool_set_cap(self) -> None:
            """Set Pool Cap (local): dev:gran:new_cap or guided dev:gran then cap."""
            self._pending_cap_pool = None
            self._prompt_for_pool_cap_first()

        def action_pool_set_tuning(self) -> None:
            """Set Pool Auto-Tuning (local): dev:gran:extend:ex_step:remove:rm_step or guided."""
            self._pending_tune_pool = None
            self._prompt_for_pool_tuning_first()

        def action_pool_create(self) -> None:
            """Create Pool (local): dev:gran:init:cap or guided dev:gran then init:cap."""
            self._pending_create_pool = None
            self._prompt_for_pool_create_first()

        def action_pool_show_info(self) -> None:
            """Show Pool Runtime Info (local): cap and tuning overrides/effective values."""
            extend_threshold = env_scaled_size("MDAEMON_EXTEND_THRESHOLD_inGB", 1, GB)
            remove_threshold = env_scaled_size("MDAEMON_REMOVE_THRESHOLD_inGB", 2, GB)
            extend_bytes = env_scaled_size("MDAEMON_EXTEND_BYTES_inMB", 256, MB)
            remove_bytes = env_scaled_size("MDAEMON_REMOVE_BYTES_inMB", 512, MB)

            pools = sorted(self._known_pools, key=lambda x: (x[0], x[1]))
            if not pools:
                self._set_status("no known pools yet", is_error=False)
                self._append_message("info", "pool info: no known pools yet")
                return

            table = Table(title="Pool Runtime Info", header_style="bold #7be3ff")
            table.add_column("Pool", style="#d7e3ff")
            table.add_column("Cap(local)", style="#d7e3ff")
            table.add_column("Override(local)", style="#d7e3ff")
            table.add_column("Effective Tuning", style="#d7e3ff")

            for dev, gran in pools:
                pool_key = (dev, gran)
                cap_bytes = int(self.pool_caps.get(pool_key, 0))
                cap_text = "none" if cap_bytes <= 0 else fmt_bytes(cap_bytes)

                override = self.pool_tune_overrides.get(pool_key)
                if override is None:
                    ov_text = "none"
                    eff_extend = extend_threshold
                    eff_ex_step = extend_bytes
                    eff_remove = remove_threshold
                    eff_rm_step = remove_bytes
                else:
                    ov_text = (
                        f"extend={fmt_bytes(override.extend_threshold_bytes)} "
                        f"step={fmt_bytes(override.extend_step_bytes)} "
                        f"remove={fmt_bytes(override.remove_threshold_bytes)} "
                        f"step={fmt_bytes(override.remove_step_bytes)}"
                    )
                    eff_extend = override.extend_threshold_bytes
                    eff_ex_step = override.extend_step_bytes
                    eff_remove = override.remove_threshold_bytes
                    eff_rm_step = override.remove_step_bytes

                eff_text = (
                    f"extend={fmt_bytes(eff_extend)} "
                    f"step={fmt_bytes(eff_ex_step)} "
                    f"remove={fmt_bytes(eff_remove)} "
                    f"step={fmt_bytes(eff_rm_step)}"
                )

                table.add_row(f"dev{dev}:{gran // MB}MB", cap_text, ov_text, eff_text)

            self.messages_log.write(table)

            self.last_action = f"listed {len(pools)} pool runtime configs"

        def get_system_commands(self, screen) -> Iterable[SystemCommand]:
            # Register pool management actions so they are discoverable in Ctrl+P.
            yield from super().get_system_commands(screen)
            yield SystemCommand(
                "Set Pool Cap (local)",
                "Set per-pool cap override used only by this monitor process",
                self.action_pool_set_cap,
            )
            yield SystemCommand(
                "Set Pool Auto-Tuning (local)",
                "Set per-pool extend/remove thresholds and steps for this monitor",
                self.action_pool_set_tuning,
            )
            yield SystemCommand(
                "Create Pool (local)",
                "Create a pool via monitor and attach local cap metadata",
                self.action_pool_create,
            )
            yield SystemCommand(
                "Show Pool Runtime Info (local)",
                "Show local cap/tuning overrides and effective runtime values",
                self.action_pool_show_info,
            )

        def _prompt(self, title: str, help_text: str, placeholder: str,
                    callback: Callable[[Optional[str]], None]) -> None:
            self.push_screen(PromptInputScreen(title, help_text, placeholder), callback)

        def _pool_exists(self, pool_key: Tuple[int, int]) -> bool:
            return pool_key in self._known_pools

        def _pool_text(self, pool_key: Tuple[int, int]) -> str:
            dev, gran = pool_key
            return f"dev{dev}:{gran // MB}MB"

        def _parse_tuning_values(self, text: str) -> PoolTuneOverride:
            parts = [p.strip() for p in text.split(":")]
            if len(parts) != 4:
                raise ValueError("expected extend:extend_step:remove:remove_step")
            extend_threshold = parse_gb_to_bytes(parts[0])
            extend_step = parse_mb_to_bytes(parts[1])
            remove_threshold = parse_gb_to_bytes(parts[2])
            remove_step = parse_mb_to_bytes(parts[3])
            return PoolTuneOverride(
                extend_threshold_bytes=extend_threshold,
                extend_step_bytes=extend_step,
                remove_threshold_bytes=remove_threshold,
                remove_step_bytes=remove_step,
            )

        def _prompt_for_pool_cap_first(self) -> None:
            self._prompt(
                "Set Pool Cap (local override)",
                "Input full: dev:gran_mb:new_cap_gb\n"
                "or input pool: dev:gran_mb then next step input cap.\n"
                "Examples: 0:4:5 , 0:4:5GB",
                "dev:gran[:cap]",
                self._on_pool_cap_first,
            )

        def _on_pool_cap_first(self, value: Optional[str]) -> None:
            if value is None:
                return
            text = value.strip()
            if not text:
                self._set_status("empty input for pool cap", is_error=True)
                return
            try:
                parts = [p.strip() for p in text.split(":")]
                if len(parts) == 3:
                    pool_key = parse_pool_key(":".join(parts[:2]))
                    if not self._pool_exists(pool_key):
                        self._set_status(f"unknown pool {parts[0]}:{parts[1]}", is_error=True)
                        return
                    cap_bytes = parse_gb_to_bytes(parts[2])
                    self.pool_caps[pool_key] = cap_bytes
                    self.last_action = (
                        f"set cap {self._pool_text(pool_key)} -> {fmt_bytes(cap_bytes)} (local)"
                    )
                    self._append_message("info", self.last_action)
                    return

                if len(parts) == 2:
                    pool_key = parse_pool_key(text)
                    if not self._pool_exists(pool_key):
                        self._set_status(f"unknown pool {parts[0]}:{parts[1]}", is_error=True)
                        return
                    self._pending_cap_pool = pool_key
                    self._prompt(
                        f"Set Cap for {self._pool_text(pool_key)}",
                        "Input cap in GB. If no unit, GB is assumed.\n"
                        "Examples: 5, 5GB, 4.5GB",
                        "new_cap",
                        self._on_pool_cap_second,
                    )
                    return
                raise ValueError("expected dev:gran or dev:gran:cap")
            except Exception as exc:
                self._set_status(f"invalid pool cap input: {exc}", is_error=True)

        def _on_pool_cap_second(self, value: Optional[str]) -> None:
            if value is None:
                self._pending_cap_pool = None
                return
            pool_key = self._pending_cap_pool
            self._pending_cap_pool = None
            if pool_key is None:
                self._set_status("missing selected pool for cap update", is_error=True)
                return
            try:
                cap_bytes = parse_gb_to_bytes(value.strip())
                self.pool_caps[pool_key] = cap_bytes
                self.last_action = (
                    f"set cap {self._pool_text(pool_key)} -> {fmt_bytes(cap_bytes)} (local)"
                )
                self._append_message("info", self.last_action)
            except Exception as exc:
                self._set_status(f"invalid cap value: {exc}", is_error=True)

        def _prompt_for_pool_tuning_first(self) -> None:
            self._prompt(
                "Set Pool Auto-Tuning (local override)",
                "Input full: dev:gran_mb:extend:extend_step:remove:remove_step\n"
                "or input pool: dev:gran_mb then next step tuning values.\n"
                "Defaults when no unit: extend/remove in GB, steps in MB.\n"
                "Examples: 0:4:1GB:200MB:2GB:200MB or 0:4:1:200:2:200",
                "dev:gran[:extend:step:remove:step]",
                self._on_pool_tuning_first,
            )

        def _on_pool_tuning_first(self, value: Optional[str]) -> None:
            if value is None:
                return
            text = value.strip()
            if not text:
                self._set_status("empty input for pool tuning", is_error=True)
                return
            try:
                parts = [p.strip() for p in text.split(":")]
                if len(parts) == 6:
                    pool_key = parse_pool_key(":".join(parts[:2]))
                    if not self._pool_exists(pool_key):
                        self._set_status(f"unknown pool {parts[0]}:{parts[1]}", is_error=True)
                        return
                    override = self._parse_tuning_values(":".join(parts[2:]))
                    self.pool_tune_overrides[pool_key] = override
                    self.last_action = (
                        "set tuning "
                        f"{self._pool_text(pool_key)} -> "
                        f"extend={fmt_bytes(override.extend_threshold_bytes)} "
                        f"step={fmt_bytes(override.extend_step_bytes)} "
                        f"remove={fmt_bytes(override.remove_threshold_bytes)} "
                        f"step={fmt_bytes(override.remove_step_bytes)} (local)"
                    )
                    self._append_message("info", self.last_action)
                    return

                if len(parts) == 2:
                    pool_key = parse_pool_key(text)
                    if not self._pool_exists(pool_key):
                        self._set_status(f"unknown pool {parts[0]}:{parts[1]}", is_error=True)
                        return
                    self._pending_tune_pool = pool_key
                    self._prompt(
                        f"Set Tuning for {self._pool_text(pool_key)}",
                        "Input extend:extend_step:remove:remove_step\n"
                        "Defaults when no unit: GB:MB:GB:MB\n"
                        "Examples: 1GB:200MB:2GB:200MB or 1:200:2:200",
                        "extend:step:remove:step",
                        self._on_pool_tuning_second,
                    )
                    return
                raise ValueError("expected dev:gran or dev:gran:extend:step:remove:step")
            except Exception as exc:
                self._set_status(f"invalid pool tuning input: {exc}", is_error=True)

        def _on_pool_tuning_second(self, value: Optional[str]) -> None:
            if value is None:
                self._pending_tune_pool = None
                return
            pool_key = self._pending_tune_pool
            self._pending_tune_pool = None
            if pool_key is None:
                self._set_status("missing selected pool for tuning update", is_error=True)
                return
            try:
                override = self._parse_tuning_values(value.strip())
                self.pool_tune_overrides[pool_key] = override
                self.last_action = (
                    "set tuning "
                    f"{self._pool_text(pool_key)} -> "
                    f"extend={fmt_bytes(override.extend_threshold_bytes)} "
                    f"step={fmt_bytes(override.extend_step_bytes)} "
                    f"remove={fmt_bytes(override.remove_threshold_bytes)} "
                    f"step={fmt_bytes(override.remove_step_bytes)} (local)"
                )
                self._append_message("info", self.last_action)
            except Exception as exc:
                self._set_status(f"invalid tuning value: {exc}", is_error=True)

        def _prompt_for_pool_create_first(self) -> None:
            self._prompt(
                "Create Pool (local monitor setup)",
                "Input full: dev:gran_mb:init_gb:cap_gb\n"
                "or input pool: dev:gran_mb then next step init:cap.\n"
                "Examples: 0:4:2:10 or 0:4:2GB:10GB",
                "dev:gran:init:cap",
                self._on_pool_create_first,
            )

        def _apply_create_pool(self, pool_key: Tuple[int, int], init_bytes: int,
                               cap_bytes: int) -> bool:
            if self._pool_exists(pool_key):
                self._set_status(f"pool already exists: {self._pool_text(pool_key)}", is_error=True)
                return False
            dev, gran = pool_key
            self.daemon.initialize_handle_pool_device(int(gran), int(dev), int(init_bytes))
            self.pool_caps[pool_key] = cap_bytes
            self._known_pools.add(pool_key)
            self.last_action = (
                f"create pool {self._pool_text(pool_key)} init={fmt_bytes(init_bytes)} "
                f"cap={fmt_bytes(cap_bytes)} (local)"
            )
            self._append_message("info", self.last_action)
            return True

        def _on_pool_create_first(self, value: Optional[str]) -> None:
            if value is None:
                return
            text = value.strip()
            if not text:
                self._set_status("empty input for create pool", is_error=True)
                return
            try:
                parts = [p.strip() for p in text.split(":")]
                if len(parts) == 4:
                    pool_key = parse_pool_key(":".join(parts[:2]))
                    init_bytes = parse_gb_to_bytes(parts[2])
                    cap_bytes = parse_gb_to_bytes(parts[3])
                    if cap_bytes < init_bytes:
                        raise ValueError("cap must be >= init")
                    self._apply_create_pool(pool_key, init_bytes, cap_bytes)
                    return

                if len(parts) == 2:
                    pool_key = parse_pool_key(text)
                    if self._pool_exists(pool_key):
                        self._set_status(f"pool already exists: {self._pool_text(pool_key)}", is_error=True)
                        return
                    self._pending_create_pool = pool_key
                    self._prompt(
                        f"Create {self._pool_text(pool_key)}",
                        "Input init:cap in GB. If no unit, GB is assumed.\n"
                        "Examples: 2:10 or 2GB:10GB",
                        "init:cap",
                        self._on_pool_create_second,
                    )
                    return
                raise ValueError("expected dev:gran or dev:gran:init:cap")
            except Exception as exc:
                self._set_status(f"invalid create input: {exc}", is_error=True)

        def _on_pool_create_second(self, value: Optional[str]) -> None:
            if value is None:
                self._pending_create_pool = None
                return
            pool_key = self._pending_create_pool
            self._pending_create_pool = None
            if pool_key is None:
                self._set_status("missing selected pool for create", is_error=True)
                return
            try:
                parts = [p.strip() for p in value.strip().split(":")]
                if len(parts) != 2:
                    raise ValueError("expected init:cap")
                init_bytes = parse_gb_to_bytes(parts[0])
                cap_bytes = parse_gb_to_bytes(parts[1])
                if cap_bytes < init_bytes:
                    raise ValueError("cap must be >= init")
                self._apply_create_pool(pool_key, init_bytes, cap_bytes)
            except Exception as exc:
                self._set_status(f"invalid init/cap value: {exc}", is_error=True)

        def _append_message(self, level: str, message: str) -> None:
            lvl = level.lower().strip() or "info"
            ts = time.strftime("%H:%M:%S")
            self.message_lines.append((lvl, f"{ts} [{lvl.upper():5}] {message}"))

            if not self.is_mounted:
                return

            if lvl == "error":
                style = "bold #ff5f7a"
            elif lvl == "debug":
                style = "#8ba0d6"
            else:
                style = "#d7e3ff"
            self.messages_log.write(Text(self.message_lines[-1][1], style=style))

        def _append_daemon_line(self, line: str) -> None:
            if "[ERROR]" in line:
                lvl = "error"
                style = "bold #ff5f7a"
            elif "[DEBUG]" in line:
                lvl = "debug"
                style = "#8ba0d6"
            else:
                lvl = "info"
                style = "#d7e3ff"

            self.message_lines.append((lvl, line))
            if self.is_mounted:
                self.messages_log.write(Text(line, style=style))

        def _set_status(self, message: str, is_error: bool = False) -> None:
            status = self.query_one("#status", Label)
            status.styles.color = "#ff5f7a" if is_error else "#7be3ff"
            status.update(message)
            if is_error:
                self._append_message("error", message)

        def _auto_tune(self, pool_snaps: Iterable) -> None:
            extend_threshold = env_scaled_size("MDAEMON_EXTEND_THRESHOLD_inGB", 1, GB)
            remove_threshold = env_scaled_size("MDAEMON_REMOVE_THRESHOLD_inGB", 2, GB)
            extend_bytes = env_scaled_size("MDAEMON_EXTEND_BYTES_inMB", 256, MB)
            remove_bytes = env_scaled_size("MDAEMON_REMOVE_BYTES_inMB", 512, MB)
            # Optional hard cap in GB from env. 0 means unlimited.
            per_device_total_cap_gb = env_float("MDAEMON_DEVICE_TOTAL_CAP_GB", 0.0)
            per_device_total_cap = int(per_device_total_cap_gb * GB)

            snaps = list(pool_snaps)
            per_device_total: Dict[int, int] = collections.defaultdict(int)
            for snap in snaps:
                per_device_total[int(snap.device_id)] += int(snap.total_bytes)

            for snap in snaps:
                gran = int(snap.granularity)
                dev = int(snap.device_id)
                avail_b = int(snap.available_bytes)
                avail_h = int(snap.available_handles)
                self._known_pools.add((dev, gran))

                tuning_override = self.pool_tune_overrides.get((dev, gran))
                pool_extend_threshold = (
                    tuning_override.extend_threshold_bytes
                    if tuning_override is not None
                    else extend_threshold
                )
                pool_extend_step = (
                    tuning_override.extend_step_bytes
                    if tuning_override is not None
                    else extend_bytes
                )
                pool_remove_threshold = (
                    tuning_override.remove_threshold_bytes
                    if tuning_override is not None
                    else remove_threshold
                )
                pool_remove_step = (
                    tuning_override.remove_step_bytes
                    if tuning_override is not None
                    else remove_bytes
                )

                if avail_b < pool_extend_threshold:
                    count = int(math.ceil(pool_extend_step / max(1, gran)))

                    pool_cap = int(self.pool_caps.get((dev, gran), 0))
                    effective_pool_cap = pool_cap
                    if per_device_total_cap > 0:
                        effective_pool_cap = (
                            min(pool_cap, per_device_total_cap)
                            if pool_cap > 0
                            else per_device_total_cap
                        )

                    pool_total = int(snap.total_bytes)
                    if effective_pool_cap > 0:
                        if pool_total >= effective_pool_cap:
                            self.last_action = (
                                f"extend skipped dev{dev} {gran // MB}MB: "
                                f"pool total {fmt_bytes(pool_total)} >= cap {fmt_bytes(effective_pool_cap)}"
                            )
                            self._append_message("debug", self.last_action)
                            continue

                        pool_remaining = effective_pool_cap - pool_total
                        max_count_by_pool_cap = int(pool_remaining // max(1, gran))
                        if max_count_by_pool_cap <= 0:
                            self.last_action = (
                                f"extend skipped dev{dev} {gran // MB}MB: "
                                f"remaining pool cap {fmt_bytes(pool_remaining)} < one handle {fmt_bytes(gran)}"
                            )
                            self._append_message("debug", self.last_action)
                            continue
                        if count > max_count_by_pool_cap:
                            count = max_count_by_pool_cap

                    if per_device_total_cap > 0:
                        used_total = int(per_device_total.get(dev, 0))
                        if used_total >= per_device_total_cap:
                            self.last_action = (
                                f"extend skipped dev{dev} {gran // MB}MB: "
                                f"device total {fmt_bytes(used_total)} >= cap {fmt_bytes(per_device_total_cap)}"
                            )
                            self._append_message("debug", self.last_action)
                            continue

                        remaining = per_device_total_cap - used_total
                        max_count_by_cap = int(remaining // max(1, gran))
                        if max_count_by_cap <= 0:
                            self.last_action = (
                                f"extend skipped dev{dev} {gran // MB}MB: "
                                f"remaining cap {fmt_bytes(remaining)} < one handle {fmt_bytes(gran)}"
                            )
                            self._append_message("debug", self.last_action)
                            continue
                        if count > max_count_by_cap:
                            count = max_count_by_cap

                    if count > 0:
                        ok = bool(self.daemon.extend_handles(gran, dev, count))
                        if ok:
                            per_device_total[dev] = int(per_device_total.get(dev, 0)) + count * gran
                        self.last_action = (
                            f"extend dev{dev} {gran // MB}MB +{count} ({pool_extend_step // MB}MB) -> {ok}"
                        )
                        self._append_message("info", self.last_action)

                elif avail_b > pool_remove_threshold:
                    count = int(pool_remove_step // max(1, gran))
                    count = min(count, avail_h)
                    if count > 0:
                        ok = bool(self.daemon.remove_handles(gran, dev, count))
                        self.last_action = (
                            f"remove dev{dev} {gran // MB}MB -{count} ({pool_remove_step // MB}MB) -> {ok}"
                        )
                        self._append_message("info", self.last_action)

        def _refresh(self) -> None:
            try:
                self._drain_control_commands()

                if hasattr(self.daemon, "drain_logs"):
                    for line in self.daemon.drain_logs():
                        self._append_daemon_line(str(line))

                pool_snaps = self.daemon.snapshot_handle_pools()
                self._auto_tune(pool_snaps)

                # Pools (progress bars)
                seen_keys = set()
                for snap in sorted(pool_snaps, key=lambda s: (int(s.device_id), int(s.granularity))):
                    key = (int(snap.device_id), int(snap.granularity))
                    seen_keys.add(key)
                    row = self.pool_rows.get(key)
                    if row is None:
                        row = PoolRow(key)
                        self.pool_rows[key] = row
                        self.pools_view.mount(row)
                    row.update_from_snapshot(snap)

                # Models (table)
                model_snaps = self.daemon.snapshot_models(self.sync_model_shm)
                self.models_table.clear()
                for snap in sorted(model_snaps, key=lambda s: int(s.model_id)):
                    model_id = int(snap.model_id)
                    msg_state = _safe_enum_name(snap.message_state)
                    prev = self.prev_model_msg_states.get(model_id)
                    if prev is not None and prev != msg_state:
                        self._append_message("debug", f"model {model_id} message_state: {prev} -> {msg_state}")
                    self.prev_model_msg_states[model_id] = msg_state

                    self.models_table.add_row(
                        str(model_id),
                        _safe_enum_name(snap.state),
                        msg_state,
                        str(int(snap.model_npuid)),
                        str(int(snap.model_osid)),
                        fmt_bytes(int(snap.allocated_bytes)),
                        str(int(snap.allocated_handles)),
                    )

                now = time.strftime("%H:%M:%S")
                self._set_status(
                    "daemon running="
                    f"{bool(self.daemon.is_running())}"
                    f" | refresh={self.refresh_ms}ms"
                    f" | tune(thr<={fmt_bytes(env_scaled_size('MDAEMON_EXTEND_THRESHOLD_inGB', 1, GB))}/>={fmt_bytes(env_scaled_size('MDAEMON_REMOVE_THRESHOLD_inGB', 2, GB))}"
                    f", add={fmt_bytes(env_scaled_size('MDAEMON_EXTEND_BYTES_inMB', 256, MB))}"
                    f", drop={fmt_bytes(env_scaled_size('MDAEMON_REMOVE_BYTES_inMB', 512, MB))}"
                    f" | {now} | {self.last_action}"
                )
            except Exception as exc:
                self._set_status(f"error: {exc}", is_error=True)

    daemon = mdaemon_py.Daemon()
    app = DaemonMonitor(
        daemon,
        args.pool,
        args.refresh_ms,
        args.sync_model_shm,
        args.control_enable,
        args.control_host,
        args.control_port,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
