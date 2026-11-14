# SPDX-License-Identifier: Apache-2.0
"""OSCAR INT2 configuration surface (extraction keystone).

The RC53-frozen behavior lives behind 312 historical environment gates
(docs/env_inventory.tsv in the delivery workspace). This module is the
first step of the cleanup: ONE place that defines the configuration for
the OSCAR INT2 three-tier path, with

  - a small, documented set of semantic options (the ones the frozen
    canonical posture actually exercises),
  - a legacy-env compatibility shim (read once, mapped onto the semantic
    config) so the frozen evidence legs keep working unchanged while the
    experiment switches are removed from the code paths.

Default: DISABLED — the stock BF16 path is never touched unless OSCAR is
explicitly enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _int(name: str, default: int) -> int:
    """Legacy-env integer knob.

    Unset OR EMPTY falls back to the default: the RC53 docker whitelist
    exports empty strings to defeat launcher setdefaults, so "" is an
    established "not set" spelling here. A non-empty value that does not
    parse fails closed with the variable name instead of a bare
    ``ValueError: invalid literal``.
    """
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name}={v!r} is not a valid integer") from None


@dataclass(frozen=True)
class OscarInt2Config:
    """Semantic configuration for the OSCAR INT2 three-tier KV path.

    Tier layout:
      prefix_window  BF16 high-precision prefix tokens (default 128)
      recent_window  BF16 recent-ring tokens (default 896)
      middle is INT2-packed paged KV between them (block 128, 2-bit,
      per-token scale+offset lanes).
    """

    # -- lifecycle / opt-in -------------------------------------------------
    enabled: bool = False  # master switch (default OFF)

    # -- tier geometry ------------------------------------------------------
    prefix_window: int = 128
    recent_window: int = 896
    page_block: int = 128

    # -- rotation (REQUIRED for INT2 quality) --------------------------------
    # Calibrated per-layer orthogonal rotations for K (qqt) and V (sst);
    # quantizing unrotated K/V fails retrieval outright.
    k_rotation_path: str = ""
    v_rotation_path: str = ""

    # -- executor ------------------------------------------------------------
    eager: bool = True  # validated executor
    batched_middle: bool = True  # B>=2 askv-masked carrier staging
    cpp_quantizer: bool = True  # single-call C++ K/V quantize (byte-exact)

    # -- numerics ------------------------------------------------------------
    quant_clip_k: float = 0.96
    quant_clip_v: float = 0.92
    int2_msd: bool = True  # MSD anti-overflow mode (kernel arg 1/1)


def _from_legacy_env() -> OscarInt2Config:
    """Map the frozen canonical posture env onto the semantic config.

    Source of truth for the frozen posture: the RC53 serve chain
    (r216_serve_full4.sh docker defaults + RC_ENV.sh). Only the gates the
    canonical posture distinguishes are honored here; the remaining ~280
    historical experiment switches default to their canonical values inside
    the extracted modules (they are being deleted, not mapped).
    """
    # Defaults here replicate the FROZEN monolith's per-site env defaults
    # exactly (e.g. R161_BATCHED_MID unset = OFF), so an environment the
    # frozen overlay accepts behaves identically on this branch. The
    # dataclass field defaults above are the documented semantic defaults
    # for direct construction, NOT the legacy-env fallbacks.
    #
    # LAYERING (runtime-proven, regression/probe_out): the serve entry
    # (r168_serve_can.py) additionally setdefaults R112_LSE_FAST /
    # R113_MIDDLE_CPP / R113_QUANT_CPP / R113_QROT_DEDUPE to "1" in MODE=B
    # BEFORE the engine import -- those keys are NOT in the docker -e
    # whitelist, so "absent from the whitelist" does not mean "unset". The
    # frozen chain therefore ran the C++ single-call quantize (launcher-ext
    # oscar_quant_kv, ATen-only) as its production path. R113_SCATTER_CPP
    # WAS whitelisted with an EMPTY default, which defeats the setdefault
    # (present-but-empty) -- paged scatter writes stayed on the Python
    # path. This shim reads the env exactly as the monolith did; no
    # behavior change on either side.
    return OscarInt2Config(
        enabled=_flag("OSCAR_INT2_ENABLED") or _flag("R218_INT2_SERVE"),
        prefix_window=_int("R109_TT_PREFIX", 128),
        recent_window=_int("R109_TT_RECENT", 896),
        k_rotation_path=os.environ.get("SGLANG_OSCAR_K_ROTATION_PATH", ""),
        v_rotation_path=os.environ.get("SGLANG_OSCAR_V_ROTATION_PATH", ""),
        eager=_flag("R152_EAGER", "0"),
        batched_middle=_flag("R161_BATCHED_MID", "0"),
        cpp_quantizer=_flag("R113_QUANT_CPP", "0"),
        int2_msd=_flag("R174_INT2_MSD", "0"),
    )


_cached: OscarInt2Config | None = None


def get_oscar_config() -> OscarInt2Config:
    """Read the config once per process (env is not re-read mid-run)."""
    global _cached
    if _cached is None:
        _cached = _from_legacy_env()
    return _cached
