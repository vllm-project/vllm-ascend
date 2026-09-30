# SPDX-License-Identifier: Apache-2.0
"""Request QoS for native P -> D layer writes; no transport selection changes.

Reuses the installed QosPDPool's engine initialization, registration rollback,
lane locks and lifetime. The pool and KV storage remain worker-owned. Forward
finalization drains the send queue before the scheduler may recycle KV blocks.
"""

import logging
import time
from dataclasses import dataclass, field

# vLLM configures the "vllm" logger, not Python's root logger. Ascend's
# module namespace is a sibling, so getLogger(__name__) can discard INFO.
# Inherit vLLM's handlers/level; do not change global logging configuration.
logger = logging.getLogger("vllm.kv_transfer.layerwise_qos")


def QosPDWritePool(*args, **kwargs):
    """Construct the optional lane pool only for an enabled connector."""
    from mooncake.qos_pd_lane import QosPDPool  # type: ignore[import-not-found, import-untyped]

    class WritePool(QosPDPool):
        def write(self, qos, session, local, remote, sizes):
            if type(qos) is not int or qos not in self._locks:
                raise ValueError("requested QoS lane is unavailable")
            with self._locks[qos]:
                if self._closed:
                    raise RuntimeError("QoS pool is closed")
                if getattr(self, "_broken", False):
                    raise RuntimeError("QoS pool is broken after registration rollback")
                if not sizes or not (len(local) == len(remote) == len(sizes)):
                    raise ValueError("empty or inconsistent transfer batch")
                for src, dst, size in zip(local, remote, sizes):
                    if any(type(v) is not int or v <= 0 for v in (src, dst, size)):
                        raise ValueError("invalid transfer range")
                    if not any(base <= src and src + size <= base + n for base, n in self._buffers.items()):
                        raise ValueError("local transfer range is not registered")
                rc = self._engines[qos].batch_transfer_sync_write(session, local, remote, sizes)
                if rc != 0:
                    raise RuntimeError(f"QoS {qos}: layerwise WRITE failed: {rc}")
                return rc

    return WritePool(*args, **kwargs)


def validate_policy(policy):
    # This extension uses the unified label contract on both peers. Reject a
    # legacy reversed mapping before creating engines or publishing metadata.
    if policy is not None and policy.priority_to_qos != {0: 0, 3: 3, 7: 7}:
        raise ValueError("layerwise QoS requires canonical low=0 medium=3 high=7 mapping")


def bind_request(policy, request):
    """D is the authority; P consumes D's already resolved canonical priority."""
    params = request.kv_transfer_params
    if not params or not (params.get("do_remote_prefill") or params.get("do_remote_decode")):
        return
    if policy is None:
        if params.get("layerwise_qos_version") is not None:
            raise ValueError("layerwise QoS must be enabled on both P and D")
        return
    if params.get("do_remote_prefill"):
        priority = policy.request_priority(request)
    else:
        if (
            type(params.get("layerwise_qos_version")) is not int
            or params["layerwise_qos_version"] != 1
            or type(params.get("kv_priority")) is not int
        ):
            raise ValueError("missing canonical layerwise QoS metadata from D")
        priority = params["kv_priority"]
        policy.select(priority)
    # Keep a private copy; don't mutate another consumer's input dictionary.
    request.kv_transfer_params = dict(params, kv_priority=priority, layerwise_qos_version=1)


def peer_ports(policy, agent):
    ports = agent.qos_te_rpc_ports
    if policy is None:
        if ports or agent.layerwise_qos_version:
            raise ValueError("layerwise QoS must be enabled on both P and D")
        return {}
    if type(agent.layerwise_qos_version) is not int or agent.layerwise_qos_version != 1 or not isinstance(ports, dict):
        raise ValueError("D does not advertise layerwise QoS lanes")
    expected = set(policy.priority_to_qos.values())
    if set(ports) != expected or any(type(q) is not int for q in ports):
        raise ValueError("P/D QoS lane sets differ")
    if any(type(p) is not int or not 1 <= p <= 65535 for p in ports.values()):
        raise ValueError("invalid peer QoS port")
    if len(set(ports.values())) != len(ports):
        raise ValueError("peer QoS lanes share a port")
    return dict(ports)


@dataclass
class LayerwiseBatch:
    src: list = field(default_factory=list)
    dst: list = field(default_factory=list)
    length: list = field(default_factory=list)
    members: list = field(default_factory=list)


def group_batches(sender, task, group_idx):
    batches: dict[tuple[int, str], LayerwiseBatch] = {}
    for rid, meta in task.send_request.items():
        qos = sender.qos_policy.select(meta.kv_priority)
        if qos not in meta.remote_qos_te_rpc_ports:
            raise ValueError("matching D lane is missing; refusing default-lane fallback")
        session = f"{meta.remote_host}:{meta.remote_qos_te_rpc_ports[qos]}"
        batch = batches.setdefault((qos, session), LayerwiseBatch())
        for layer_name in task.layer_names or [task.layer_name]:
            assert sender.layer_metadata[layer_name].tensor_group_idx[0] == group_idx
            src, dst, sizes = sender.get_transfer_meta(task, rid, meta, layer_name, group_idx)
            if not (len(src) == len(dst) == len(sizes)):
                raise ValueError("layer transfer descriptor shape mismatch")
            batch.src.extend(src)
            batch.dst.extend(dst)
            batch.length.extend(sizes)
            batch.members.append(
                dict(
                    request_id=rid,
                    priority=meta.kv_priority,
                    layer_name=layer_name,
                    src=list(src),
                    dst=list(dst),
                    sizes=list(sizes),
                )
            )
    return batches


def transfer_batch(sender, task, qos, session, batch):
    """Single physical batch; test observers bracket this exact native WRITE."""
    if sender.qos_policy.log_enabled:
        for member in batch.members:
            logger.info(
                "KV_QOS_LAYER request=%s priority=%d lane_qos=%d op=write layer=%s layer_idx=%d session=%s bytes=%d",
                member["request_id"],
                member["priority"],
                qos,
                member["layer_name"],
                task.layer_idx,
                session,
                sum(member["sizes"]),
            )
    return sender.qos_pool.write(qos, session, batch.src, batch.dst, batch.length)


def send_batches(sender, task, batches, group_idx):
    # Completion accounting belongs to handle_request, including empty batches.
    for (qos, session), batch in batches.items():
        if batch.src:
            transfer_batch(sender, task, qos, session, batch)


def handle_request(sender, task):
    """Own terminal notification, buffer reuse completion and queue accounting."""
    error = sender.qos_error
    completion = {**task.send_request, **task.failed_requests}
    last_layer = task.layer_idx == sender.total_layers - 1
    try:
        if error is None:
            error = sender._transfer_kv_cache(task)
        if task.failed_requests and error is None:
            error = "layerwise request metadata failed"
    except Exception as exc:
        error = str(exc)
    finally:
        if error is not None:
            sender.qos_error = error
            sender.failed_reqs.update(completion)
        sender.failed_reqs.update(task.failed_requests)
        try:
            group_idx = sender.layer_metadata[task.layer_name].tensor_group_idx[0]
            for rid, meta in completion.items():
                key = (rid, group_idx)
                failed = rid in sender.failed_reqs
                if key in sender.qos_terminal_reqs:
                    continue
                if not failed and not (last_layer and meta.chunk_finish):
                    continue
                # Mark before invoking: a callback may send and then fail on ACK.
                sender.qos_terminal_reqs.add(key)
                try:
                    sender.callback_func(rid, meta, group_idx, trans_flag=not failed)
                except Exception as exc:
                    error = f"{error or ''}; terminal notification failed: {exc}"
                    sender.qos_error = error
                    sender.failed_reqs.add(rid)
        except Exception as exc:
            error = f"{error or ''}; finalization failed: {exc}"
            sender.qos_error = error
            sender.failed_reqs.update(completion)
        finally:
            try:
                if sender.reuse_completion_callback is not None:
                    sender.reuse_completion_callback(task.layer_idx, error)
            except Exception as exc:
                sender.qos_error = f"{error or ''}; reuse callback failed: {exc}"
            finally:
                sender.send_queue.task_done()
        if last_layer:
            for rid, meta in completion.items():
                if meta.chunk_finish:
                    sender.failed_reqs.discard(rid)
                    sender.qos_terminal_reqs = {key for key in sender.qos_terminal_reqs if key[0] != rid}


def drain(sender, timeout=60):
    deadline = time.monotonic() + timeout
    with sender.send_queue.all_tasks_done:
        while sender.send_queue.unfinished_tasks:
            if not sender.is_alive():
                raise RuntimeError("layerwise sender exited with pending KV writes")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("layerwise writes did not drain; buffers must remain registered")
            sender.send_queue.all_tasks_done.wait(timeout=min(1, remaining))
    if sender.qos_error is not None:
        raise RuntimeError("layerwise QoS write failed: " + sender.qos_error)
