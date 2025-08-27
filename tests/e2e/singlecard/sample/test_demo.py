# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import sys
from typing import Union

import pytest

from tests.e2e.utils import fork_new_process_for_each_test
# yapf: disable
from tests.e2e.singlecard.sample.utils import (DUMMY_LOGITPROC_ARG,
                                              DUMMY_LOGITPROC_FQCN,
                                              DUMMY_LOGITPROC_MODULE,
                                              MAX_TOKENS, MODEL_NAME,
                                              TEMP_GREEDY,
                                              CustomLogitprocSource,
                                              DummyLogitsProcessor,
                                              dummy_module)
from tests.e2e.singlecard.sample.utils import entry_points as fake_entry_points
from tests.e2e.singlecard.sample.utils import prompts
# yapf: enable
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import (LogitsProcessor)

# Create a mixture of requests which do and don't utilize the dummy logitproc
sampling_params_list = [
    SamplingParams(temperature=TEMP_GREEDY,
                   max_tokens=MAX_TOKENS,
                   extra_args={DUMMY_LOGITPROC_ARG: 128}),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
    SamplingParams(temperature=TEMP_GREEDY,
                   max_tokens=MAX_TOKENS,
                   extra_args={DUMMY_LOGITPROC_ARG: 67}),
    SamplingParams(temperature=TEMP_GREEDY, max_tokens=MAX_TOKENS),
]


def _run_test(kwargs: dict, logitproc_loaded: bool) -> None:
    """Compare `LLM` instance initialized with specified `kwargs` against
    reference `LLM` instance.

    Two scenarios:
    1. Server has loaded dummy logitproc; test that requests which specify
       dummy logitproc arg value behave as if logitproc is operating (output
       token value should repeat), while requests that don't specify dummy
       logitproc arg value should match reference `LLM` output.
    2. Server has *not* loaded dummy logitproc; test that all requests
       behave as if logitproc is *not* operating (output matches reference
       `LLM` output.)
    
    Args:
      kwargs: `LLM` constructor kwargs
      logitproc_loaded: server has loaded dummy logitproc if True
    """

    # Create a vLLM instance and load custom logitproc
    llm_logitproc = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.1,
        **kwargs,
    )

    # Create a reference vLLM instance without custom logitproc
    llm_ref = LLM(model=MODEL_NAME, gpu_memory_utilization=0.1)

    # Run inference with logitproc loaded
    outputs_logitproc = llm_logitproc.generate(prompts, sampling_params_list)

    # Reference run
    outputs_ref = llm_ref.generate(prompts, sampling_params_list)

    # Validate outputs
    for bdx, (out_lp, out_ref, params) in enumerate(
            zip(outputs_logitproc, outputs_ref, sampling_params_list)):
        lp_toks = out_lp.outputs[0].token_ids
        if logitproc_loaded and params.extra_args:
            # This request exercises custom logitproc; validate that logitproc
            # forces `target_token` to be decoded in each step
            target_token = params.extra_args[DUMMY_LOGITPROC_ARG]
            if not all(x == target_token for x in lp_toks):
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, shoud all be "
                    f"{target_token}")
        else:
            # This request does not exercise custom logitproc (or custom
            # logitproc is not enabled on this server); validate against
            # reference result
            ref_toks = out_ref.outputs[0].token_ids
            if lp_toks != ref_toks:
                raise AssertionError(
                    f"Request {bdx} generated {lp_toks}, should match "
                    f"{ref_toks}")


@fork_new_process_for_each_test
@pytest.mark.parametrize("logitproc_source", list(CustomLogitprocSource))
def test_custom_logitsprocs(logitproc_source: CustomLogitprocSource):
    """Test offline Python interface for passing custom logitsprocs

    Construct an `LLM` instance which loads a custom logitproc that has a
    well-defined behavior (mask out all tokens except one `target_token`)

    Construct a reference `LLM` instance with no custom logitproc

    Pass in a batch of requests, 50% of which pass a `target_token` value
    in through `SamplingParams.extra_args`, 50% of which do not.

    Validate that
    * Requests which do not activate the custom logitproc, yield the same
      results for both `LLM` instances
    * Requests which activate the custom logitproc, only output `target_token`

    Test four scenarios, corresponding to `logitproc_source` value
    * No logitsprocs loaded - test that generated tokens match reference `LLM`
      instance output
    * Logitproc passed in via {entrypoint, class object, fully-qualified class
      name (FQCN)} - test that dummy logitproc is utilized correctly when
      provided via any of these three possible sources 

    Args:
      logitproc_source: what source (entrypoint, fully-qualified class name
                        (FQCN), class object, or None) the user pulls the
                        logitproc from
    """

    # Test that logitproc info is passed to workers
    random.seed(40)

    # Choose LLM args based on logitproc source
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_NONE:
        # Scenario: the server does not load any custom logitproc
        # Every other scenario is a different way of loading a custom logitproc
        _run_test({}, logitproc_loaded=False)
        return

    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT:
        # Scenario: vLLM loads a logitproc from a preconfigured entrypoint
        # To that end, mock a dummy logitproc entrypoint
        import importlib.metadata
        importlib.metadata.entry_points = fake_entry_points  # type: ignore

        _run_test({}, logitproc_loaded=True)
        return

    kwargs: dict[str, list[Union[str, type[LogitsProcessor]]]] = {}
    if logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_FQCN:
        # Scenario: load logitproc based on fully-qualified class name (FQCN)
        # Inject dummy module which defines logitproc
        sys.modules[DUMMY_LOGITPROC_MODULE] = dummy_module
        kwargs["logits_processors"] = [DUMMY_LOGITPROC_FQCN]
    elif logitproc_source == CustomLogitprocSource.LOGITPROC_SOURCE_CLASS:
        # Scenario: load logitproc from provided class object
        kwargs["logits_processors"] = [DummyLogitsProcessor]

    _run_test(kwargs, logitproc_loaded=True)
