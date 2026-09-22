# SPDX-License-Identifier: Apache-2.0
"""Worker-isolated probe: preserve the A1 observation order exactly."""


class LogitsGateWorkerExtension:
    def baseline_install(self, expected_layers):
        import torch
        from vllm.distributed import get_tp_group

        runner = self.model_runner
        model = runner.model
        layers = model.model.layers
        if len(layers) != expected_layers:
            raise AssertionError(f"Expected {expected_layers} layers, got {len(layers)}")
        rank = get_tp_group().rank_in_group
        state = {"armed": None, "captures": 0, "last_layer_calls": 0, "pending": None, "rank": rank}

        def capture(_module, _inputs, result):
            sample = state["armed"]
            if sample is None:
                raise AssertionError("Unarmed layer-10 forward during collection")
            if state["pending"] is not None:
                raise AssertionError("More than one forward for a non-chunked request")
            hidden, residual = result
            n = len(sample["input_token_ids"])
            # SP is allowed only with explicit restoration of the global row layout.
            if hidden.shape[0] < n:
                hidden = get_tp_group().all_gather(hidden, dim=0)[:n]
                residual = get_tp_group().all_gather(residual, dim=0)[:n]
            if hidden.shape[0] < n or residual.shape[0] < n:
                raise AssertionError("Incomplete intermediate hidden-state rows")
            positions = torch.tensor(sample["prediction_positions"], device=hidden.device, dtype=torch.long)
            h = hidden.index_select(0, positions).clone()
            r = residual.index_select(0, positions).clone()
            pre_norm = h.float() + r.float()
            normalized, _ = model.model.norm(h, r)
            logits = model.compute_logits(normalized)
            if logits is None or logits.shape != (len(sample["answer_token_ids"]), 154880):
                raise AssertionError("Full-vocabulary intermediate logits unavailable")
            if (
                not torch.isfinite(pre_norm).all()
                or not torch.isfinite(normalized).all()
                or not torch.isfinite(logits).all()
            ):
                raise AssertionError("Non-finite intermediate baseline values")
            info = {
                "id": sample["id"],
                "rows": logits.shape[0],
                "vocab_size": logits.shape[1],
                "input_sha256": sample["input_sha256"],
                "rank": rank,
            }
            if rank == 0:
                # Use CPU NumPy argmax, exactly as the archived float32 A1 arrays.
                info["argmax"] = logits.detach().float().cpu().numpy().argmax(axis=-1).tolist()
            state["pending"] = info
            state["captures"] += 1

        def last_layer(_module, _inputs, _result):
            state["last_layer_calls"] += 1

        state["handles"] = [layers[10].register_forward_hook(capture), layers[-1].register_forward_hook(last_layer)]
        self._baseline = state
        return {
            "rank": rank,
            "layers": len(layers),
            "layer_class": type(layers[10]).__name__,
            "norm_class": type(model.model.norm).__name__,
            "model_class": type(model).__name__,
        }

    def baseline_arm(self, sample):
        state = self._baseline
        if state["armed"] is not None:
            raise AssertionError("Previous sample not finished")
        state["armed"] = sample
        state["pending"] = None
        state["before_last"] = state["last_layer_calls"]
        return True

    def baseline_finish(self):
        state = self._baseline
        if state["pending"] is None or state["last_layer_calls"] != state["before_last"] + 1:
            raise AssertionError("Missing capture or incomplete full-model forward")
        result = state["pending"]
        state["armed"] = None
        state["pending"] = None
        return result

    def baseline_remove(self):
        state = self._baseline
        for handle in state["handles"]:
            handle.remove()
        result = {key: state[key] for key in ("rank", "captures", "last_layer_calls")}
        del self._baseline
        return result
