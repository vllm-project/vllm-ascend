# SPDX-License-Identifier: Apache-2.0
"""HiSparse P/D transport composed with the upstream HiSparseConnector.

Dependency: the common HiSparse backend/platform-hook patch used by the Ascend
HiSparse backend. No scheduler/coordinator/host-import algorithm is replaced.
"""

from __future__ import annotations

import ipaddress
import uuid
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)

from .configuration import validate_transport_options
from .execution import draft_state, draft_transfer_enabled, execution_contract, validate_draft_state, validate_execution
from .protocol import VERSION, digest


@dataclass
class HiSparsePDMetadata(KVConnectorMetadata):
    exports: list[dict] = field(default_factory=list)
    imports: list[dict] = field(default_factory=list)


class HiSparsePDConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def supports_sleep_mode(cls, kv_transfer_config):
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import optimized_profile_enabled

        if not optimized_profile_enabled():
            raise RuntimeError("HiSparse P/D requires the explicit paired A5 optimization profile")
        super().__init__(vllm_config, role, kv_cache_config)
        self.config = vllm_config
        self._composed = False
        self.cache_config = kv_cache_config
        transfer = vllm_config.kv_transfer_config
        extra = transfer.kv_connector_extra_config
        validate_transport_options(extra)
        parallel = vllm_config.parallel_config
        self.execution_mode = validate_execution(vllm_config)
        self.transfer_draft = draft_transfer_enabled(vllm_config)
        if getattr(vllm_config, "lora_config", None) is not None:
            raise ValueError("HiSparse P/D does not yet support per-request LoRA identity")
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "pipeline_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
                "data_parallel_size",
            )
        ):
            raise ValueError("HiSparse P/D supports only TP, with PP/DP/CP size one")
        if vllm_config.attention_config.hisparse_config is None or not kv_cache_config.hisparse_host_num_blocks:
            raise ValueError("Both P/D endpoints require HiSparse and a sibling HiSparseConnector")
        if transfer.is_kv_producer == transfer.is_kv_consumer:
            raise ValueError("Configure HiSparsePDConnector as kv_producer or kv_consumer, not kv_both")
        self.producer = transfer.is_kv_producer
        self.group_ids = tuple(kv_cache_config.transfer_group_ids)
        roles = set()
        for index in self.group_ids:
            group = kv_cache_config.kv_cache_groups[index]
            group_role = str(getattr(group.role, "value", group.role))
            roles.add(group_role)
            if group.is_eagle_group or group_role not in ("hisparse_source", "hisparse_indexer"):
                raise ValueError("HiSparse P/D cannot transfer draft or non-HiSparse groups")
        if roles != {"hisparse_source", "hisparse_indexer"}:
            raise ValueError("HiSparse P/D needs both SOURCE and INDEXER groups")
        for key in ("host",) if self.producer else ("host", "producer_host"):
            ipaddress.IPv4Address(extra[key])
        for key in ("control_port",) if self.producer else ("control_port", "producer_control_port"):
            if not 1024 <= int(extra[key]) <= 65535 - parallel.tensor_parallel_size:
                raise ValueError("Invalid HiSparse P/D control port range")
        # Explicit deployment identity binds weights/revision/tokenizer; model
        # config alone cannot identify two different fine-tuned checkpoints.
        if not isinstance(extra.get("model_identity"), str) or not extra["model_identity"].strip():
            raise ValueError("Set the same immutable model_identity on both P/D endpoints")
        hf = vllm_config.model_config.hf_config.to_dict()
        hf.pop("_name_or_path", None)
        self.identity = digest(
            {
                "checkpoint": extra["model_identity"],
                "hf_config": hf,
                "tp": parallel.tensor_parallel_size,
                "version": VERSION,
                "transfer_contract": execution_contract(vllm_config),
            }
        )
        self.exports = []
        self.imports = []
        self.loads = {}
        self.pending_exports = set()
        self.connector_worker = None
        if role == KVConnectorRole.WORKER:
            from .worker import HiSparsePDWorker

            self.connector_worker = HiSparsePDWorker(
                vllm_config,
                kv_cache_config,
                identity=self.identity,
                producer=self.producer,
            )

    def get_loaded_kv_cache_group_ids(self, request):
        return self.group_ids

    def bind_hisparse_sibling(self, sibling):
        if type(sibling).__name__ != "HiSparseConnector":
            raise ValueError("HiSparse P/D requires the upstream HiSparseConnector sibling")
        self._composed = True
        if self.connector_worker is not None:
            self.connector_worker.common_worker = sibling.connector_worker

    def _require_composition(self):
        if not self._composed:
            raise ValueError("HiSparsePDConnector must run inside the supported MultiConnector composition")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._require_composition()
        params = request.kv_transfer_params or {}
        if self.producer or not params.get("do_remote_prefill"):
            return 0, False
        ticket = params.get("hisparse_pd")
        if not isinstance(ticket, dict) or ticket.get("version") != VERSION or ticket.get("identity") != self.identity:
            raise ValueError("Missing or incompatible HiSparse P/D ticket")
        tokens = request.prompt_token_ids or []
        count = ticket["num_tokens"]
        validate_draft_state(ticket.get("draft_state"), count, self.transfer_draft)
        if (
            not isinstance(count, int)
            or not 0 < count <= len(tokens)
            or digest(tokens[:count]) != ticket["tokens_sha256"]
        ):
            raise ValueError("HiSparse P/D ticket does not match the requested token prefix")
        matched = max(0, count - num_computed_tokens)
        if matched:
            self.loads[request.request_id] = {
                "request_id": request.request_id,
                "ticket": dict(ticket),
                "start_token": num_computed_tokens,
            }
        return matched, bool(matched)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            job = self.loads.pop(request.request_id)
            # Async HMA import already called record_pending_host_import via
            # the SOURCE manager. Copy only transferable manager block lists.
            all_ids = blocks.get_block_ids()
            job["blocks"] = {index: list(all_ids[index]) for index in self.group_ids}
            self.imports.append(job)

    def request_finished_all_groups(self, request, block_ids):
        self._require_composition()
        self.loads.pop(request.request_id, None)
        params = request.kv_transfer_params or {}
        if not self.producer or not params.get("do_remote_decode"):
            return False, None
        status = getattr(request.status, "name", str(request.status))
        if status not in ("FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"):
            return False, None
        count = min(request.num_computed_tokens, request.num_prompt_tokens)
        if count <= 0:
            return False, None
        ticket = {
            "version": VERSION,
            "identity": self.identity,
            "lease_id": uuid.uuid4().hex,
            "num_tokens": count,
            "tokens_sha256": digest(list(request.prompt_token_ids[:count])),
        }
        if self.transfer_draft:
            ticket["draft_state"] = draft_state(count)
        self.exports.append(
            {
                "request_id": request.request_id,
                "ticket": ticket,
                "blocks": {index: list(block_ids[index]) for index in self.group_ids},
            }
        )
        self.pending_exports.add(request.request_id)
        return True, {"do_remote_prefill": True, "do_remote_decode": False, "hisparse_pd": ticket}

    def build_connector_meta(self, scheduler_output):
        result = HiSparsePDMetadata(self.exports, self.imports)
        self.exports, self.imports = [], []
        return result

    def update_connector_output(self, connector_output):
        self.pending_exports.difference_update(connector_output.finished_sending or ())
        # HiSparseConnector receives the unchanged aggregated finished_recving
        # and failed_recving sets through MultiConnector, and calls the common
        # coordinator.finish_host_import itself. Do not publish twice here.

    def has_pending_push_work(self):
        return bool(self.pending_exports or self.exports or self.imports)

    def has_pending_block_frees(self):
        return bool(self.pending_exports)

    def register_kv_caches(self, kv_caches):
        self._require_composition()
        self.connector_worker.register(kv_caches)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HiSparsePDMetadata):
            raise TypeError("HiSparse P/D received the wrong child metadata")
        self.connector_worker.queue(metadata)
        if self.producer:
            self.connector_worker.publish_ready_exports()

    def get_transfer_results(self, finished_req_ids):
        sending, receiving, failed = self.connector_worker.poll(finished_req_ids)
        return KVConnectorTransferResults(finished_sending=sending, finished_recving=receiving, failed_recving=failed)

    def get_finished(self, finished_req_ids):
        results = self.get_transfer_results(finished_req_ids)
        return results.finished_sending or None, results.finished_recving or None

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        return

    def prepare_cache_release(self):
        """Drain remote readers before a sibling releases the mapped host pool."""
        if self.connector_worker is not None:
            self.connector_worker.close()

    def shutdown(self):
        self.prepare_cache_release()
