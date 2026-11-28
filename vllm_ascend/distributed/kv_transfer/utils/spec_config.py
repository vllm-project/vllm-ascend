# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Prefill/Decode speculative-decoding config parity for KV transfer.

Prefill and Decode must register the same KV cache tensors: the transfer pairs
the two sides' base addresses by position. Enabling speculative decoding on one
side only, or with different layout-relevant knobs, changes that tensor set, and
the transfer then pairs addresses that do not describe the same layers. The
handshake metadata therefore carries a small fingerprint of the layout-relevant
speculative config, and the receiving engine rejects a mismatch instead of
serving output decoded from KV cache it never received.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

# Bump when the encoding changes. Peers advertising another version are skipped
# so a rolling upgrade does not fail on an encoding change alone.
SPEC_FINGERPRINT_VERSION = 2

# Field every connector's handshake metadata must expose.
SPEC_FINGERPRINT_FIELD = "spec_config_fingerprint"


class KVTransferSpecConfigMismatch(RuntimeError):
    """Prefill and Decode run different speculative decoding configs."""


@dataclass(frozen=True)
class SpecConfigFingerprint:
    """Layout-relevant part of ``vllm_config.speculative_config``."""

    enabled: bool
    method: str
    num_speculative_tokens: int
    draft_layers: int
    # Digest of the draft model's KV layout (architecture, head geometry,
    # dtype). Two different drafts with the same method/token/layer counts can
    # still register different tensors, so the digest is part of the config.
    draft_layout: str = "none"

    def encode(self) -> str:
        return (
            f"v{SPEC_FINGERPRINT_VERSION}"
            f"|enabled={int(self.enabled)}"
            f"|method={self.method}"
            f"|n={self.num_speculative_tokens}"
            f"|draft_layers={self.draft_layers}"
            f"|draft={self.draft_layout}"
        )

    @classmethod
    def decode(cls, raw: str) -> "SpecConfigFingerprint | None":
        """Parse ``raw``; ``None`` for an unknown/older encoding."""
        if not raw or "|" not in raw:
            return None
        version, _, rest = raw.partition("|")
        if version != f"v{SPEC_FINGERPRINT_VERSION}":
            return None
        fields = dict(item.split("=", 1) for item in rest.split("|") if "=" in item)
        try:
            return cls(
                enabled=fields["enabled"] == "1",
                method=fields["method"],
                num_speculative_tokens=int(fields["n"]),
                draft_layers=int(fields["draft_layers"]),
                draft_layout=fields.get("draft", "none"),
            )
        except (KeyError, ValueError):
            return None

    def cli_config(self) -> str:
        """``--speculative-config`` value that reproduces this side's config.

        Methods that need a draft checkpoint get a placeholder for its path:
        the fingerprint intentionally does not carry paths, so the operator
        still has to point at the same draft model.
        """
        config: dict[str, object] = {
            "method": self.method,
            "num_speculative_tokens": self.num_speculative_tokens,
        }
        if self.method not in ("mtp", "none"):
            config["model"] = "<draft-model-path>"
        return json.dumps(config)


def _draft_layout_digest(draft_model_config: Any) -> str:
    """Digest the draft model's KV-relevant geometry, ignoring its path."""
    if draft_model_config is None:
        return "none"
    hf_config = getattr(draft_model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    parts = [
        ",".join(str(item) for item in architectures),
        str(getattr(hf_config, "hidden_size", "")),
        str(getattr(hf_config, "num_attention_heads", "")),
        str(getattr(hf_config, "num_key_value_heads", "")),
        str(getattr(hf_config, "head_dim", "")),
        str(getattr(draft_model_config, "dtype", "")),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def fingerprint_from_vllm_config(vllm_config: Any) -> str:
    """Return the fingerprint of ``vllm_config.speculative_config``."""
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return SpecConfigFingerprint(enabled=False, method="none", num_speculative_tokens=0, draft_layers=0).encode()

    # ``draft_model_config`` is None for MTP, where the draft layers come from
    # the target model; zero then means "not reported", not "no draft layers".
    draft_model_config = getattr(speculative, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    draft_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
    return SpecConfigFingerprint(
        enabled=True,
        method=str(getattr(speculative, "method", "") or "unknown"),
        num_speculative_tokens=int(getattr(speculative, "num_speculative_tokens", 0) or 0),
        draft_layers=draft_layers,
        draft_layout=_draft_layout_digest(draft_model_config),
    ).encode()


def peer_fingerprint(metadata: Any) -> str:
    """Read the fingerprint from a peer handshake metadata object."""
    return getattr(metadata, SPEC_FINGERPRINT_FIELD, "") or ""


class SpecConfigParityGuard:
    """Compare the local speculative config with each peer's fingerprint."""

    def __init__(self, vllm_config: Any, role: str):
        self._role = role or "local"
        self._local_raw = fingerprint_from_vllm_config(vllm_config)
        self._reported: set[str] = set()

    @property
    def local_fingerprint(self) -> str:
        return self._local_raw

    def check(self, peer_raw: str | None, peer_label: str) -> None:
        """Raise when the peer cannot describe the same KV cache layout.

        An empty fingerprint means an older peer that does not report its
        config; the check is skipped so rolling upgrades keep working.
        """
        if not peer_raw or peer_raw == self._local_raw:
            return

        local = SpecConfigFingerprint.decode(self._local_raw)
        peer = SpecConfigFingerprint.decode(peer_raw)
        if local is None or peer is None:
            if peer_label not in self._reported:
                self._reported.add(peer_label)
                logger.warning(
                    "Cannot verify the speculative decoding config of KV transfer peer %s "
                    "(local=%r, peer=%r); continuing without the check.",
                    peer_label,
                    self._local_raw,
                    peer_raw,
                )
            return

        message = self._mismatch_message(local, peer, peer_label)
        if peer_label not in self._reported:
            self._reported.add(peer_label)
            logger.error(message)
        raise KVTransferSpecConfigMismatch(message)

    def _mismatch_message(self, local: SpecConfigFingerprint, peer: SpecConfigFingerprint, peer_label: str) -> str:
        peer_role = "prefill" if self._role == "decode" else "decode"
        enabled_side = local if local.enabled else peer
        parts = [
            f"Speculative decoding config mismatch with {peer_role} engine {peer_label}: "
            f"{peer_role}={peer.encode()!r}, {self._role}={local.encode()!r}.",
            "Prefill and Decode must run the same speculative decoding config, because the draft "
            "layers change the KV cache tensor set and the transferred KV cache cannot be paired "
            "otherwise.",
        ]
        if enabled_side.enabled:
            owner = self._role if enabled_side is local else peer_role
            parts.append(f"The {owner} engine runs --speculative-config '{enabled_side.cli_config()}'.")
        parts.append("Start both engines with the same speculative config, or disable speculative decoding on both.")
        return " ".join(parts)
