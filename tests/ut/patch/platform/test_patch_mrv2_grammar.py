# SPDX-License-Identifier: Apache-2.0

import pickle
from types import SimpleNamespace

import numpy as np
from vllm.v1.core.sched.output import GrammarOutput as VllmGrammarOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.grammar_output import GrammarOutput
from vllm_ascend.patch.platform import patch_mrv2_grammar


def test_scheduler_uses_only_the_backfilled_prefix():
    scheduler = object.__new__(Scheduler)
    scheduler.requests = {
        req_id: SimpleNamespace(use_structured_output=True, is_prefill_chunk=False) for req_id in ["a", "b", "c"]
    }
    bitmask = np.zeros((9, 1), dtype=np.int32)
    scheduler.structured_output_manager = SimpleNamespace(grammar_bitmask=lambda *_: bitmask)
    output = SchedulerOutput.make_empty()
    output.has_structured_output_requests = True
    output.num_scheduled_tokens = {"a": 3, "b": 3, "c": 3}
    output.scheduled_spec_decode_tokens = {"a": [7, -1, -1], "b": [8, 9]}
    grammar = patch_mrv2_grammar.get_grammar_bitmask(scheduler, output)
    assert isinstance(grammar, VllmGrammarOutput)
    assert grammar.structured_output_request_ids == ["a", "b", "c"]
    assert grammar.num_acceptable_drafts == [1, 2, 0]
    assert grammar.grammar_bitmask is bitmask


def test_metadata_survives_worker_transport():
    grammar = GrammarOutput(["json"], np.array([[1], [2]], dtype=np.int32), [1])
    buffers = []
    payload = pickle.dumps(grammar, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append)
    received = pickle.loads(payload, buffers=buffers)
    assert isinstance(received, VllmGrammarOutput)
    assert received.structured_output_request_ids == ["json"]
    assert received.num_acceptable_drafts == [1]
    np.testing.assert_array_equal(received.grammar_bitmask, grammar.grammar_bitmask)


def test_plain_batch_keeps_no_grammar_output():
    scheduler = object.__new__(Scheduler)
    output = SchedulerOutput.make_empty()
    output.has_structured_output_requests = False
    assert patch_mrv2_grammar.get_grammar_bitmask(scheduler, output) is None
