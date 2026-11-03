# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OSCAR INT2 configuration surface and path resolution.

Covers the three invariants the extraction must keep:

1. Default DISABLED — with a clean environment the stock BF16 path is
   never opted into.
2. The legacy-env shim maps the frozen RC53 posture env vars onto the
   semantic config (so historical evidence legs reproduce).
3. Path resolution is env-driven and fail-closed (no home-directory
   fallbacks, no silent stale-vendor loading).
"""

import pytest


def _clean_env() -> dict:
    keys = [
        "OSCAR_INT2_ENABLED",
        "R218_INT2_SERVE",
        "R109_TT_PREFIX",
        "R109_TT_RECENT",
        "SGLANG_OSCAR_K_ROTATION_PATH",
        "SGLANG_OSCAR_V_ROTATION_PATH",
        "R152_EAGER",
        "R161_BATCHED_MID",
        "R113_QUANT_CPP",
        "R174_INT2_MSD",
        "R144_OPP_PATH",
        "OSCAR_OPP_PATH",
        "ASCEND_CUSTOM_OPP_PATH",
        "R144_LAUNCHER_SRC",
        "OSCAR_LAUNCHER_SRC",
        "OSCAR_LSE_BRIDGE_SRC",
        "OSCAR_DIAG",
        "OSCAR_DIAG_DIR",
    ]
    return {k: "" for k in keys}


def _fresh_module(monkeypatch, env: dict):
    import importlib

    import vllm_ascend.oscar.config as cfg
    import vllm_ascend.oscar.paths as paths

    for k, v in env.items():
        if v == "":
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    importlib.reload(cfg)
    importlib.reload(paths)
    return cfg, paths


class TestDefaultDisabled:
    def test_clean_env_is_disabled(self, monkeypatch, tmp_path):
        cfg, _ = _fresh_module(monkeypatch, _clean_env())
        c = cfg.get_oscar_config()
        assert c.enabled is False

    def test_legacy_serve_flag_enables(self, monkeypatch):
        env = _clean_env()
        env["R218_INT2_SERVE"] = "1"
        cfg, _ = _fresh_module(monkeypatch, env)
        assert cfg.get_oscar_config().enabled is True

    def test_config_is_cached_per_process(self, monkeypatch):
        env = _clean_env()
        cfg, _ = _fresh_module(monkeypatch, env)
        c1 = cfg.get_oscar_config()
        monkeypatch.setenv("R218_INT2_SERVE", "1")
        assert cfg.get_oscar_config() is c1  # env not re-read mid-run


class TestLegacyEnvMapping:
    def test_canonical_posture(self, monkeypatch):
        env = _clean_env()
        env.update(
            {
                "R218_INT2_SERVE": "1",
                "R109_TT_PREFIX": "128",
                "R109_TT_RECENT": "896",
                "SGLANG_OSCAR_K_ROTATION_PATH": "/rot/k.pt",
                "SGLANG_OSCAR_V_ROTATION_PATH": "/rot/v.pt",
            }
        )
        cfg, _ = _fresh_module(monkeypatch, env)
        c = cfg.get_oscar_config()
        assert c.enabled and c.prefix_window == 128 and c.recent_window == 896
        assert c.k_rotation_path == "/rot/k.pt"
        assert c.v_rotation_path == "/rot/v.pt"
        # Frozen canonical posture: eager OFF/BATCHED OFF/CPP OFF/MSD OFF
        # unless explicitly set (legacy monolith per-site defaults).
        assert c.eager is False and c.batched_middle is False
        assert c.cpp_quantizer is False and c.int2_msd is False

    def test_semantic_defaults_documented(self, monkeypatch):
        env = _clean_env()
        cfg, _ = _fresh_module(monkeypatch, env)
        c = cfg.OscarInt2Config()
        assert c.prefix_window == 128 and c.recent_window == 896
        assert c.page_block == 128 and c.eager is True


class TestPathResolution:
    def test_opp_path_fail_closed(self, monkeypatch):
        env = _clean_env()
        _, paths = _fresh_module(monkeypatch, env)
        with pytest.raises(FileNotFoundError):
            paths.opp_path()

    def test_opp_path_first_chain_entry(self, monkeypatch):
        env = _clean_env()
        env["R144_OPP_PATH"] = "/v/a:/v/b"
        _, paths = _fresh_module(monkeypatch, env)
        assert paths.opp_path() == "/v/a"

    def test_launcher_src_env_override(self, monkeypatch, tmp_path):
        env = _clean_env()
        env["R144_LAUNCHER_SRC"] = str(tmp_path / "x.cpp")
        _, paths = _fresh_module(monkeypatch, env)
        assert paths.launcher_src() == str(tmp_path / "x.cpp")

    def test_no_home_directory_references(self, monkeypatch):
        env = _clean_env()
        _, paths = _fresh_module(monkeypatch, env)
        with open(paths.__file__) as fh:
            src = fh.read()
        assert "/home/" not in src

    def test_lse_bridge_missing_raises(self, monkeypatch, tmp_path):
        env = _clean_env()
        env["OSCAR_LSE_BRIDGE_SRC"] = str(tmp_path / "nope.cpp")
        _, paths = _fresh_module(monkeypatch, env)
        with pytest.raises(FileNotFoundError):
            paths.lse_bridge_src()


class TestGatedDiagnostics:
    def test_diag_write_off_by_default(self, monkeypatch, tmp_path):
        import vllm_ascend.oscar.diag as diag

        monkeypatch.delenv("OSCAR_DIAG", raising=False)
        monkeypatch.setenv("OSCAR_DIAG_DIR", str(tmp_path))
        diag.write("probe.txt", "should-not-land")
        assert not (tmp_path / "probe.txt").exists()

    def test_diag_write_gated(self, monkeypatch, tmp_path):
        import vllm_ascend.oscar.diag as diag

        monkeypatch.setenv("OSCAR_DIAG", "1")
        monkeypatch.setenv("OSCAR_DIAG_DIR", str(tmp_path))
        diag.write("probe.txt", "lands")
        assert (tmp_path / "probe.txt").read_text() == "lands\n"


class TestRepoRootDepth:
    def test_repo_root_resolves_to_checkout_root(self, monkeypatch):
        # no env override: the in-repo default must resolve to the actual
        # checkout root (three dirname calls from vllm_ascend/oscar/), not
        # its parent
        monkeypatch.delenv("OSCAR_OPP_PATH", raising=False)
        monkeypatch.delenv("R144_OPP_PATH", raising=False)
        import os

        import vllm_ascend.oscar.paths as P

        assert os.path.isfile(os.path.join(P._REPO_ROOT, "pyproject.toml"))
        assert os.path.isdir(os.path.join(P._REPO_ROOT, "vllm_ascend", "oscar"))

    def test_runner_tmp_default_is_uid_isolated(self, monkeypatch):
        # /tmp is shared on multi-user hosts: the default scratch dir must
        # carry the uid so extension builds cannot collide across users
        import os

        import vllm_ascend.oscar.paths as P

        monkeypatch.delenv("OSCAR_RUNNER_TMP", raising=False)
        d = P.runner_tmp("ut_probe")
        assert d.startswith(f"/tmp/oscar_runner-{os.getuid()}/")
