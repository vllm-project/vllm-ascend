# SPDX-License-Identifier: Apache-2.0
"""Worker-local observation for checkpoint integration tests."""


class ExpertOffloadAuditWorker:
    def expert_offload_test_audit(self, action):
        """Inspect a real worker, and observe dispatch without changing its decisions."""
        runner = self.model_runner
        manager = runner.expert_offload_manager
        if action == "install":
            assert manager.state.name == "READY"
            audit = {"steps": [], "replays": 0}
            runner._expert_offload_test_audit = audit
            prepare = runner._prepare_inputs_impl

            def observed_prepare(scheduler_output, batch_req_state, batch_desc):
                audit["steps"].append(
                    {
                        "prefill": bool(batch_req_state.has_prefill),
                        "tokens": int(batch_desc.num_tokens),
                        "mode": batch_desc.cg_mode.name,
                    }
                )
                return prepare(scheduler_output, batch_req_state, batch_desc)

            runner._prepare_inputs_impl = observed_prepare
            replay = runner.cudagraph_manager.run_fullgraph

            def observed_replay(*args, **kwargs):
                audit["replays"] += 1
                return replay(*args, **kwargs)

            runner.cudagraph_manager.run_fullgraph = observed_replay
        audit = runner._expert_offload_test_audit
        bindings = tuple(manager.bindings.values())
        draft = getattr(runner.speculator, "model", None)
        target_bindings = [
            binding
            for layer in runner.model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        ]
        draft_bindings = (
            [
                binding
                for layer in draft.modules()
                if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
            ]
            if draft is not None
            else []
        )
        snapshot = {
            "state": manager.state.name,
            "steps": list(audit["steps"]),
            "replays": audit["replays"],
            "capture_sizes": list(runner.cudagraph_manager.capture_sizes),
            "layer_names": [binding.layer_name for binding in bindings],
            "loaded": all(binding.host_store.finalized and binding.host_store.prepared for binding in bindings),
            "target_layers": len(target_bindings),
            "draft_layers": len(draft_bindings),
            "shared_manager": all(binding.manager is manager for binding in (*target_bindings, *draft_bindings))
            and (draft is None or runner.speculator.expert_offload_manager is manager),
            "addresses": {
                binding.layer_name: {name: tensor.data_ptr() for name, tensor in binding.resident.items()}
                for binding in bindings
            },
        }
        if action == "close":
            runner.shutdown()
            snapshot["closed_state"] = manager.state.name
            snapshot["released"] = (
                not manager.bindings
                and not manager.prefill_pool.tensors
                and all(not binding.resident and not binding.graph_staging for binding in bindings)
            )
        return snapshot
