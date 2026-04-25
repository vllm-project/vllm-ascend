import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from vllm_ascend import ai_qos

VISIBLE_DEVICE_ENV = "ASCEND_RT_VISIBLE_DEVICES"

MASTER_ID_AIV_DATA = 11
MASTER_ID_AIV_INS = 12
MASTER_ID_SDMA = 13
MASTER_ID_PCIEDMA = 7
FUSE_SELECT_MAX = 1
SDMA_MATA_BW_LOW = 0
SDMA_MATA_BW_HIGH = 1
SDMA_MATA_HARDLIMIT = 0
D2D_VL_INIT = 0
H2D_VL_INIT = 1
STATE_SDMA_MATA_LEN = 4
STATE_QOS_TUPLE_LEN = 5
STATE_FUSE_GBL_LEN = 3
FUSE_APPLY_ENABLE = 1
FUSE_APPLY_AUTOQOS_FUSE_EN = 1

DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "ai_qos_state.json"

# Shown when unset cannot parse the state file; state file is removed after printing.
UNSET_STATE_PARSE_FAILED_MSG = (
    "Failed to parse the state file. Please reboot the server to restore endpoint-side QoS settings. "
    "On the switch side, log in and run `sys-view` to enter system view, then run "
    "`display current-configuration` to show the current configuration. Find the previously applied "
    "switch-side QoS commands, re-enter each command with the `undo` prefix, and finally run `commit` "
    "to complete configuration rollback."
)


def _remove_state_file_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError as e:
        print(
            f"Warning: could not remove state file {path}: {e}. Please remove this file manually.",
            file=sys.stderr,
        )


def _unset_state_parse_failed(state_path: Path) -> None:
    print(UNSET_STATE_PARSE_FAILED_MSG, file=sys.stderr)
    _remove_state_file_quiet(state_path)
    sys.exit(1)


def _parse_and_validate_unset_state(data: object) -> tuple[dict, list[str], dict[str, list[int]], dict[str, list[int]]]:
    """Validate unset JSON shape and types; raise ValueError on any failure."""
    if not isinstance(data, dict):
        raise ValueError("root must be object")
    oq = data.get("original_qos")
    if not isinstance(oq, dict):
        raise ValueError("original_qos")
    pc = data.get("printed_commands")
    if not isinstance(pc, list) or not all(isinstance(x, str) for x in pc):
        raise ValueError("printed_commands")
    osm_raw = data.get("original_sdma_mata", {})
    if not isinstance(osm_raw, dict):
        raise ValueError("original_sdma_mata")
    ofu_raw = data.get("original_fuse", {})
    if not isinstance(ofu_raw, dict):
        raise ValueError("original_fuse")

    validated_oq: dict[str, dict[str, list[int]]] = {}
    for dev_s, masters in oq.items():
        try:
            _dev = int(dev_s)
        except (TypeError, ValueError) as e:
            raise ValueError("original_qos device key") from e
        if not isinstance(masters, dict):
            raise ValueError("original_qos masters")
        vm: dict[str, list[int]] = {}
        for m_s, tup in masters.items():
            try:
                _ = int(m_s)
            except (TypeError, ValueError) as e:
                raise ValueError("original_qos master key") from e
            if not isinstance(tup, list) or len(tup) != STATE_QOS_TUPLE_LEN:
                raise ValueError("original_qos tuple")
            try:
                vm[str(m_s)] = [int(x) for x in tup]
            except (TypeError, ValueError) as e:
                raise ValueError("original_qos tuple values") from e
        validated_oq[str(_dev)] = vm

    validated_osm: dict[str, list[int]] = {}
    for dev_s, mata in osm_raw.items():
        try:
            _dev = int(dev_s)
        except (TypeError, ValueError) as e:
            raise ValueError("original_sdma_mata device key") from e
        if not isinstance(mata, list) or len(mata) != STATE_SDMA_MATA_LEN:
            raise ValueError("original_sdma_mata tuple")
        try:
            validated_osm[str(_dev)] = [int(x) for x in mata]
        except (TypeError, ValueError) as e:
            raise ValueError("original_sdma_mata values") from e

    validated_ofu: dict[str, list[int]] = {}
    for dev_s, gbl in ofu_raw.items():
        try:
            _dev = int(dev_s)
        except (TypeError, ValueError) as e:
            raise ValueError("original_fuse device key") from e
        if not isinstance(gbl, list) or len(gbl) != STATE_FUSE_GBL_LEN:
            raise ValueError("original_fuse tuple")
        try:
            validated_ofu[str(_dev)] = [int(x) for x in gbl]
        except (TypeError, ValueError) as e:
            raise ValueError("original_fuse values") from e

    return validated_oq, pc, validated_osm, validated_ofu


def _print_config_block(lines: list[str]) -> None:
    print("system-view")
    for line in lines:
        print(line)
    print("commit")


def _device_list() -> list[int]:
    device_str = os.getenv(VISIBLE_DEVICE_ENV, "").strip()
    if not device_str:
        try:
            torch = importlib.import_module("torch")
            count = int(torch.npu.device_count())
        except Exception as e:
            print(
                f"Error: {VISIBLE_DEVICE_ENV} is unset and failed to run torch.npu.device_count().",
                file=sys.stderr,
            )
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)
        if count <= 0:
            print("Error: no visible NPU devices found.", file=sys.stderr)
            sys.exit(1)
        return list(range(count))
    out: list[int] = []
    for dev in device_str.split(","):
        part = dev.strip()
        if not part:
            print(
                f"Error: invalid {VISIBLE_DEVICE_ENV} value (empty segment): {device_str!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            d = int(part, 10)
        except ValueError:
            print(
                f"Error: {VISIBLE_DEVICE_ENV} must be comma-separated integers; got {device_str!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        if d < 0:
            print(
                f"Error: {VISIBLE_DEVICE_ENV} device id must be non-negative; got {d}",
                file=sys.stderr,
            )
            sys.exit(1)
        out.append(d)
    if not out:
        print(f"Error: {VISIBLE_DEVICE_ENV} must list at least one device.", file=sys.stderr)
        sys.exit(1)
    return out


def _capture_original_qos(device_list: list[int], masterid_table: dict[str, int]) -> dict[str, dict[str, list]]:
    original: dict[str, dict[str, list]] = {}
    for device_id in device_list:
        key_d = str(device_id)
        original[key_d] = {}
        for _accu, master_id in masterid_table.items():
            ret, m, mpamid, q, pmg, mode = ai_qos.get_qos(device_id, master_id)
            if ret != 0:
                print(
                    f"Warning: get_qos failed (dev={device_id} master={master_id} ret={ret}); not saved for restore.",
                    file=sys.stderr,
                )
                continue
            original[key_d][str(master_id)] = [m, mpamid, q, pmg, mode]
    return original


def _capture_original_sdma_mata(
    device_list: list[int],
) -> dict[str, list]:
    out: dict[str, list] = {}
    for device_id in device_list:
        ret_q, _m, mpamid, _q, _pmg, _mode = ai_qos.get_qos(device_id, MASTER_ID_SDMA)
        if ret_q != 0:
            print(
                f"Warning: get_qos(SDMA) failed (dev={device_id} ret={ret_q}); SDMA mata not saved for restore.",
                file=sys.stderr,
            )
            continue
        ret, bw_lo, bw_hi, hard = ai_qos.get_bw(device_id, mpamid)
        if ret == 0:
            out[str(device_id)] = [int(mpamid), int(bw_lo), int(bw_hi), int(hard)]
    return out


def _capture_original_fuse(
    device_list: list[int],
) -> dict[str, list]:
    out: dict[str, list] = {}
    for device_id in device_list:
        ret, en, aut, fuse = ai_qos.get_fuse_mode(device_id)
        if ret == 0:
            out[str(device_id)] = [int(en), int(aut), int(fuse)]
    return out


def _merge_baseline_for_new_devices(
    device_list: list[int],
    original_qos: dict[str, dict[str, list]],
    original_sdma_mata: dict[str, list],
    original_fuse: dict[str, list],
    masterid_table: dict[str, int],
) -> None:
    missing: list[int] = []
    for d in device_list:
        ds = str(d)
        if ds not in original_qos or ds not in original_sdma_mata or ds not in original_fuse:
            missing.append(d)
    if not missing:
        return
    oq = _capture_original_qos(missing, masterid_table)
    for k, v in oq.items():
        original_qos[k] = v
    osm = _capture_original_sdma_mata(missing)
    for k, v in osm.items():
        original_sdma_mata[k] = v
    ofu = _capture_original_fuse(missing)
    for k, v in ofu.items():
        original_fuse[k] = v


def _load_first_apply_baseline(
    state_path: Path,
) -> tuple[dict, dict, dict] | None:
    if not state_path.is_file():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    oq = data.get("original_qos")
    if not isinstance(oq, dict):
        return None
    osm = data.get("original_sdma_mata", {})
    ofu = data.get("original_fuse", {})
    if not isinstance(osm, dict):
        osm = {}
    if not isinstance(ofu, dict):
        ofu = {}
    return (oq, osm, ofu)


def run_unset(state_path: Path) -> None:
    if not state_path.is_file():
        print(f"No state file at {state_path}; nothing to undo.", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _unset_state_parse_failed(state_path)

    try:
        original_qos, printed, original_sdma_mata, original_fuse = _parse_and_validate_unset_state(data)
    except ValueError:
        _unset_state_parse_failed(state_path)

    for dev_s, mata in original_sdma_mata.items():
        device_id = int(dev_s)
        mid, bw_lo, bw_hi, hard = mata
        r = ai_qos.set_bw(device_id, mid, bw_lo, bw_hi, hard)
        if r != 0:
            print(
                f"Warning: restore SDMA mata (dev={device_id} mpamid={mid}) failed, ret = {r}",
                file=sys.stderr,
            )

    for dev_s, masters in original_qos.items():
        device_id = int(dev_s)
        for m_s, tup in masters.items():
            master_id = int(m_s)
            _m, mpamid, qos, pmg, mode = tup
            ai_qos.set_qos(device_id, master_id, mpamid, qos, pmg, mode)

    for dev_s, gbl in original_fuse.items():
        device_id = int(dev_s)
        en, aut, fmode = gbl
        r = ai_qos.set_fuse_gbl_config(device_id, en, aut, fmode)
        if r != 0:
            print(
                f"Warning: restore fuse gbl (dev={device_id}) failed, ret = {r}",
                file=sys.stderr,
            )

    _print_config_block([f"undo {line}" for line in printed])

    try:
        state_path.unlink()
    except OSError as e:
        print(f"Warning: could not remove state file {state_path}: {e}", file=sys.stderr)


class AiqosConfig:
    def __init__(self, aiqos_config: dict):
        self.mode = aiqos_config.get("mode")
        self.aiqos_priority = aiqos_config.get("aiqos_priority")
        self.aiqos_table = {
            "AIV_D2D": {"low": (1, 0, 0, 1), "middle": (3, 4, 1, 2), "high": (5, 5, 2, 3)},
            "AIV_H2D": {"low": (1, 0, 3, 1), "middle": (3, 4, 4, 2), "high": (5, 5, 5, 3)},
            "SDMA_D2D": {"low": (2, 0, 0, 1), "middle": (4, 4, 1, 2), "high": (6, 5, 2, 3)},
            "SDMA_H2D": {"low": (2, 0, 3, 1), "middle": (4, 4, 4, 2), "high": (6, 5, 5, 3)},
            "PCIEDMA_H2D": {"low": (0, 0, 3, 1), "high": (7, 5, 5, 3)},
        }
        self.masterid_table = {
            "AIV_DATA": MASTER_ID_AIV_DATA,
            "AIV_INS": MASTER_ID_AIV_INS,
            "SDMA": MASTER_ID_SDMA,
            "PCIEDMA": MASTER_ID_PCIEDMA,
        }

    def set_qos(self, state_path: Path) -> None:
        device_list = _device_list()
        baseline = _load_first_apply_baseline(state_path)
        if baseline is not None:
            original_qos, original_sdma_mata, original_fuse = baseline
            _merge_baseline_for_new_devices(
                device_list,
                original_qos,
                original_sdma_mata,
                original_fuse,
                self.masterid_table,
            )
        else:
            original_qos = _capture_original_qos(device_list, self.masterid_table)
            original_sdma_mata = _capture_original_sdma_mata(
                device_list,
            )
            original_fuse = _capture_original_fuse(device_list)

        attributes = ["sqos", "dqos", "vl", "pri"]
        for op_type in self.aiqos_table:
            level = self.aiqos_priority.get(op_type)
            config_tuple = self.aiqos_table.get(op_type).get(level)
            for idx, attr in enumerate(attributes):
                var_name = f"{op_type.lower()}_{attr}"
                setattr(self, var_name, config_tuple[idx])
        aiv_qos = min(self.aiv_d2d_sqos, self.aiv_h2d_sqos)
        sdma_qos = min(self.sdma_d2d_sqos, self.sdma_h2d_sqos)
        pcie_qos = self.pciedma_h2d_sqos
        fuse_mode = FUSE_SELECT_MAX

        command_types = {
            "aiv_d2d": aiv_qos,
            "aiv_h2d": aiv_qos,
            "sdma_d2d": sdma_qos,
            "sdma_h2d": sdma_qos,
            "pciedma_h2d": pcie_qos,
        }

        def generate_command(qos_value: int, dqos: int, vl: int, pri: int, vl_init: int) -> str:
            return (
                f"hccs qos remap {qos_value} {vl_init} {dqos}\n"
                f"hccs vl remap peer-type cpu {dqos} {vl_init} {vl}\n"
                f"hccs vl remap peer-type npu {dqos} {vl_init} {vl}\n"
                f"hccs vl remap peer-type sw {dqos} {vl_init} {vl}\n"
                f"hccs sp peer-type cpu {vl} {pri}\n"
                f"hccs sp peer-type npu {vl} {pri}\n"
                f"hccs sp peer-type sw {vl} {pri}\n"
            )

        cmd_set: set[str] = set()
        for cmd_type, qos_value in command_types.items():
            dqos = getattr(self, f"{cmd_type}_dqos")
            vl = getattr(self, f"{cmd_type}_vl")
            pri = getattr(self, f"{cmd_type}_pri")
            vl_init = D2D_VL_INIT
            if "h2d" in cmd_type:
                vl_init = H2D_VL_INIT
            cmd_str = generate_command(qos_value, dqos, vl, pri, vl_init)
            for sub_str in cmd_str.split("\n"):
                if sub_str.strip():
                    cmd_set.add(sub_str)
        printed_commands = sorted(cmd_set)

        for device_id in device_list:
            ai_qos.set_fuse_gbl_config(device_id, FUSE_APPLY_ENABLE, FUSE_APPLY_AUTOQOS_FUSE_EN, fuse_mode)
            for accu, master_id in self.masterid_table.items():
                ret, _master, mpamid, qos, pmg, mode = ai_qos.get_qos(device_id, master_id)
                if ret != 0:
                    print(
                        f"get_qos failed (dev={device_id} master={master_id} ret={ret}).",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                if accu.startswith("AIV"):
                    ai_qos.set_qos(device_id, master_id, mpamid, aiv_qos, pmg, mode)
                elif accu.startswith("SDMA"):
                    ai_qos.set_bw(device_id, mpamid, SDMA_MATA_BW_LOW, SDMA_MATA_BW_HIGH, SDMA_MATA_HARDLIMIT)
                    ai_qos.set_qos(device_id, master_id, mpamid, sdma_qos, pmg, mode)
                else:
                    ai_qos.set_qos(device_id, master_id, mpamid, pcie_qos, pmg, mode)

        _print_config_block(printed_commands)

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "original_qos": original_qos,
                    "original_sdma_mata": original_sdma_mata,
                    "original_fuse": original_fuse,
                    "printed_commands": printed_commands,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI QoS tuning for Ascend NPU.  "
        " Multiple apply reuses the "
        "first snapshot in the state file; unset restores that baseline and removes the file."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="apply",
        choices=["apply", "unset"],
        help='Run "unset" to restore the first-apply snapshot and delete the state file.',
    )
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "manual"])
    parser.add_argument("--AIV_D2D", type=str, default="high", choices=["low", "middle", "high"])
    parser.add_argument("--AIV_H2D", type=str, default="high", choices=["low", "middle", "high"])
    parser.add_argument("--SDMA_D2D", type=str, default="high", choices=["low", "middle", "high"])
    parser.add_argument("--SDMA_H2D", type=str, default="low", choices=["low", "middle", "high"])
    parser.add_argument("--PCIEDMA_H2D", type=str, default="high", choices=["low", "high"])
    args = parser.parse_args()
    state_path = DEFAULT_STATE_PATH

    if args.command == "unset":
        run_unset(state_path)
    else:
        aiqos_config = {
            "mode": args.mode,
            "aiqos_priority": {
                "AIV_D2D": args.AIV_D2D,
                "AIV_H2D": args.AIV_H2D,
                "SDMA_D2D": args.SDMA_D2D,
                "SDMA_H2D": args.SDMA_H2D,
                "PCIEDMA_H2D": args.PCIEDMA_H2D,
            },
        }
        AiqosConfig(aiqos_config).set_qos(state_path)
