# SPDX-License-Identifier: Apache-2.0
from importlib.util import find_spec

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


class DMTPBackendCheck:
    def check_dmtp_attention_backends(self):
        from importlib.util import find_spec

        from vllm_ascend.device.hardware_profile import FlashAttentionNpu3Model, get_current_hardware_profile

        runner = self.model_runner
        fa3_expected = (
            find_spec("flash_attn_npu_3") is not None
            and FlashAttentionNpu3Model.QWEN in get_current_hardware_profile().flash_attn_npu_3_models
        )
        fa3 = ("vllm_ascend.attention.flash_attention_v3", "AscendFlashAttentionBackend")
        draft_groups = [group for groups in runner.speculator.attn_groups for group in groups]
        assert all((group.backend.full_cls_name() == fa3) == fa3_expected for group in draft_groups)
        assert runner._can_skip_spec_cpu_length_sync() == fa3_expected
        if fa3_expected:
            draft_layers = {name for group in draft_groups for name in group.layer_names}
            target_fa3_layers = sum(
                len(set(group.layer_names) - draft_layers)
                for groups in runner.attn_groups
                for group in groups
                if group.backend.full_cls_name() == fa3
            )
            assert target_fa3_layers == 16
        return True

    def count_dmtp_graph_replays(self):
        self.dmtp_graph_replays = dict(target=0, native=0, draft=0)
        runner = self.model_runner
        managers = dict(
            target=runner.cudagraph_manager,
            native=runner.speculator.native_cudagraph_manager,
            draft=runner.speculator.query_cudagraph_manager,
        )
        for name, manager in managers.items():
            assert manager.graphs, f"{name} has no captured graphs"
            original = manager.run_fullgraph

            def replay(desc, name=name, original=original):
                self.dmtp_graph_replays[name] += 1
                return original(desc)

            manager.run_fullgraph = replay

    def get_dmtp_graph_replays(self):
        return self.dmtp_graph_replays


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "aclgraph"])
def test_qwen36_dmtp_matches_target_greedy(enforce_eager):
    if not enforce_eager:
        from vllm_ascend.device.hardware_profile import FlashAttentionNpu3Model, get_current_hardware_profile

        if (
            find_spec("flash_attn_npu_3") is None
            or FlashAttentionNpu3Model.QWEN not in get_current_hardware_profile().flash_attn_npu_3_models
        ):
            pytest.skip("DMTP ACLGraph requires the Qwen FA3 backend")
    target = "Qwen/Qwen3.6-27B"
    draft = "Piggy12/Qwen3.6-27B-DMTP"
    prompts = ["The capital of France is", "Continue the sequence of even integers: 2, 4, 6, 8,"]
    sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    common = dict(
        revision="1cda14dd71a3f47115850eb790f416f165da1cb5",
        worker_extension_cls="tests.e2e.pull_request.two_card.spec_decode.test_dmtp.DMTPBackendCheck",
        tensor_parallel_size=2,
        block_size=128,
        dtype="bfloat16",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        language_model_only=True,
        max_model_len=3072,
        max_num_seqs=2,
        disable_log_stats=False,
    )
    with VllmRunner(target, **common) as llm:
        reference = [r.outputs[0].token_ids for r in llm.model.generate(prompts, sampling)]
    spec = {
        "method": "dmtp",
        "model": draft,
        "revision": "d3d8d02162c64b88a9000735d260ef479825863e",
        "num_speculative_tokens": 7,
    }
    common["enforce_eager"] = enforce_eager
    if not enforce_eager:
        common["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8, 16]}
    with VllmRunner(target, speculative_config=spec, **common) as llm:
        assert all(llm.model.collective_rpc("check_dmtp_attention_backends"))
        if not enforce_eager:
            llm.model.collective_rpc("count_dmtp_graph_replays")
        stop_after_first = SamplingParams(temperature=0, max_tokens=32, stop_token_ids=[reference[0][0]])
        stopped = llm.model.generate(prompts[:1], stop_after_first)[0].outputs[0]
        assert stopped.finish_reason == "stop" and len(stopped.token_ids) <= 1
        for batch in ([prompts[0]], prompts):
            actual = [r.outputs[0].token_ids for r in llm.model.generate(batch, sampling)]
            assert actual == reference[: len(batch)]
        metrics = {m.name: m for m in llm.model.get_metrics()}
        assert metrics["vllm:spec_decode_num_drafts"].value > 0
        # A verifier that silently rejects every draft can still pass output parity.
        accepted = metrics["vllm:spec_decode_num_accepted_tokens_per_pos"].values
        assert len(accepted) == 7 and sum(accepted[1:]) > 0
        if not enforce_eager:
            for counts in llm.model.collective_rpc("get_dmtp_graph_replays"):
                assert all(count > 0 for count in counts.values()), counts
