"""Target-only handoff outside graph capture; caches stay owned upstream."""

TARGET_CONTRACT = "target_completed_source_indexer_v1"
MTP_CONTRACT = "mtp3_completed_prefix_rebuild_proposals_v1"


def draft_transfer_enabled(config):
    spec = config.speculative_config
    if spec is None:
        return False
    if spec.method != "mtp" or spec.num_speculative_tokens != 3:
        raise ValueError("HiSparse P/D currently supports only the explicit MTP3 prefix policy")
    if spec.draft_model_config.model != config.model_config.model:
        raise ValueError("MTP P/D requires one checkpoint identity for target and draft")
    if not config.model_config.enforce_eager and spec.enforce_eager:
        raise ValueError("MTP P/D graph mode requires draft graphs too")
    return True


def execution_contract(config):
    return MTP_CONTRACT if draft_transfer_enabled(config) else TARGET_CONTRACT


def draft_state(count):
    return {"policy": MTP_CONTRACT, "prefix_tokens": count, "lookahead_tokens": 0}


def validate_draft_state(value, count, enabled):
    if enabled and value != draft_state(count):
        raise ValueError("MTP P/D requires an exact prefix and fresh D proposals")
    if not enabled and value is not None:
        raise ValueError("Unexpected draft state in target-only P/D")


def export_mirror_ready(common_worker, enabled):
    if not enabled:
        return True
    if common_worker is None:
        raise RuntimeError("MTP export has no bound common HiSparse worker")
    return not common_worker._draft_mirror_pending


def validate_execution(config):
    draft_transfer_enabled(config)
    if config.model_config.enforce_eager:
        return "eager"
    mode = config.compilation_config.cudagraph_mode
    if getattr(mode, "name", None) != "FULL_DECODE_ONLY":
        raise ValueError("HiSparse P/D target graphs require FULL_DECODE_ONLY")
    return "decode_graph"


def assert_transfer_boundary(*, capturing, closing, closed):
    if capturing or closing or closed:
        raise RuntimeError("HiSparse P/D transfer cannot enter capture or teardown")


def validate_completion(value, contract=TARGET_CONTRACT):
    if (
        not isinstance(value, dict)
        or value.get("contract") != contract
        or type(value.get("generation")) is not int
        or (value["generation"] <= 0)
    ):
        raise ValueError("Producer did not publish a completed target handoff")


def _binding(layouts, regions, owners):
    geometry = tuple(
        sorted(
            (
                (
                    x.key(),
                    x.role,
                    x.group_id,
                    x.block_size,
                    x.row_bytes,
                    x.base,
                    x.block_stride,
                    x.num_blocks,
                    x.payload_format,
                )
                for x in layouts
            )
        )
    )
    return (geometry, tuple(sorted(regions.items())), tuple((id(x) for x in owners)))


class CacheBindingGuard:
    def __init__(self, layouts, regions, owners):
        self.expected = _binding(layouts, regions, owners)

    def validate(self, layouts, regions, owners):
        if _binding(layouts, regions, owners) != self.expected:
            raise RuntimeError("HiSparse P/D registered cache binding changed")
