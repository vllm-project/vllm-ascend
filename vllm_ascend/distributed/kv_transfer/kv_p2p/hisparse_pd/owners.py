# SPDX-License-Identifier: Apache-2.0
"""Loader-proven transfer roles without changing shared-group planning flags."""

from .execution import draft_transfer_enabled


def bind_loaded_owners(config, speculator, cache_config):
    transfer = config.kv_transfer_config
    if transfer is None:
        return
    children = transfer.kv_connector_extra_config.get("connectors", ())
    names = [x.get("kv_connector") for x in children]
    if transfer.kv_connector != "HiSparsePDConnector" and "HiSparsePDConnector" not in names:
        return
    enabled = draft_transfer_enabled(config)
    layers = {name for g in cache_config.kv_cache_groups if g.enable_kv_transfer for name in g.layer_names}
    draft = set(getattr(speculator, "draft_attn_layer_names", ()))
    if enabled and (not draft or not draft <= layers):
        raise ValueError("MTP P/D needs every loaded draft cache in the transfer groups")
    if draft and not enabled:
        raise ValueError("Loaded draft caches require the explicit MTP prefix policy")
    context = config.compilation_config.static_forward_context
    for name in layers:
        context[name].hisparse_pd_model_role = "draft" if name in draft else "target"
