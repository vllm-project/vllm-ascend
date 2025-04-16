import dataclasses
import functools
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.sampler import (PromptLogprobs, SampleLogprobs,
                                                SamplerOutput,
                                                SamplingMetadata, get_logprobs,
                                                get_pythonized_sample_results)
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.utils import current_stream
from vllm.worker.model_runner_base import (
    _init_attn_metadata_from_tensor_dict,
    _init_frozen_model_input_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)
from vllm.worker.multi_step_model_runner import (ModelOutput,
                                                 PythonizationCache,
                                                 StatefulModelInput)

from vllm_ascend.worker.model_runner import (
    ModelInputForNPUWithSamplingMetadata, NPUModelRunnerBase)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend


@dataclass(frozen=False)
class StatefulModelInputForNPU(StatefulModelInput):

    # actual frozen model input dataclass passed to _base_model_runner
    frozen_model_input: Optional[ModelInputForNPUWithSamplingMetadata] = None

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "StatefulModelInputForNPU":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        tensor_dict = _init_frozen_model_input_from_tensor_dict(
            ModelInputForNPUWithSamplingMetadata, tensor_dict)
        return cls(**tensor_dict)

    def maybe_advance_frozen_model_input(self, device: str, pin_memory: bool):
        """
        Advancing the datastructures of StatefulModelInput::frozen_model_input
        is only required when prefills are scheduled with decodes to run in
        multi-step. This advancement/correction is required to account for
        the conversion of Prefills to Decodes after the first multi-step.
        """
        if self.current_step != 1 or self.num_single_step_prefills == 0:
            return

        assert self.frozen_model_input is not None
        fmi = self.frozen_model_input

        # Truncate input_tokens
        assert fmi.input_tokens is not None
        assert fmi.input_tokens.shape[0] >= self.num_seqs
        fmi_new_input_tokens: torch.Tensor = fmi.input_tokens[:self.num_seqs]

        # Update frozen_model_input::input_positons.
        assert fmi.input_positions is not None
        assert fmi.input_positions.shape[0] >= self.num_seqs
        fmi_new_input_positions: torch.Tensor = fmi.input_positions[:self.
                                                                    num_seqs]

        # Assert unsupported
        # TODO Uncomment the following codes when NPU supported
        # assert fmi.lora_mapping is None
        # assert fmi.lora_requests is not None
        # assert len(fmi.lora_requests) == 0
        # assert fmi.prompt_adapter_mapping is None
        # assert fmi.prompt_adapter_requests is not None
        # assert len(fmi.prompt_adapter_requests) == 0
        assert fmi.attn_metadata is not None
        assert fmi.multi_modal_kwargs is not None
        assert len(fmi.multi_modal_kwargs) == 0

        self.frozen_model_input = dataclasses.replace(
            self.frozen_model_input,
            input_tokens=fmi_new_input_tokens,
            input_positions=fmi_new_input_positions)

        self.maybe_advance_sampling_metadata(device, pin_memory)


class MultiStepModelNPURunner(NPUModelRunnerBase[StatefulModelInputForNPU]):
    # mypy: enable-error-code=type-var

    def __init__(self, base_model_runner: NPUModelRunnerBase, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # uses the base model runner to execute the model and wraps it with
        # multi-step logic
        self._base_model_runner: NPUModelRunnerBase = base_model_runner

        self.is_multi_step = self.scheduler_config.is_multi_step
        self.pinned_sampled_token_ids: Optional[torch.Tensor] = None

        # Using the PythonizationCache in Pipeline-Parallel clobbers the
        # SequenceOutput and CompletionSequenceGroupOutput object.
        # When cache-reset happens at the last step of a multi-step
        # execution, there may be other on-going single-step/multi-step
        # executions. The current caching implementation does not check
        # for this.
        self.pythonization_cache = PythonizationCache() \
            if self.parallel_config.pipeline_parallel_size == 1 else None

    def get_model(self) -> nn.Module:
        return self._base_model_runner.get_model()

    @functools.cached_property
    def _copy_stream(self):
        # used to copy tensors from NPU to CPU asynchronously
        return torch.npu.Stream()

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> StatefulModelInputForNPU:
        model_input = (StatefulModelInputForNPU.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        ))
        return model_input

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> StatefulModelInputForNPU:
        frozen_model_input: ModelInputForNPUWithSamplingMetadata = \
              self._base_model_runner.prepare_model_input(
                    seq_group_metadata_list,
                    virtual_engine,
                    finished_requests_ids)

        assert frozen_model_input.query_lens is not None
        assert frozen_model_input.seq_lens is not None
        assert frozen_model_input.attn_metadata is not None
        num_queries = len(frozen_model_input.query_lens)
        num_seqs = len(frozen_model_input.seq_lens)
        num_single_step_prefills = frozen_model_input.attn_metadata.num_prefills

        model_input = StatefulModelInputForNPU(
            frozen_model_input=frozen_model_input,
            num_seqs=num_seqs,
            num_queries=num_queries,
            num_single_step_prefills=num_single_step_prefills,
            step_cuda_events=[torch.npu.Event(blocking=True)] * 2,
        )

        return model_input

    def _async_process_outputs(self, model_input: StatefulModelInputForNPU,
                               output_proc_callback: Callable):
        # Proceed with pythonization and output_proc in order.
        # Stop on the first one that fails to pythonize
        output_proc_callback()

        cont = True
        for step_num, model_output in enumerate(model_input.cached_outputs):
            if not model_output.pythonized:
                model_output.maybe_pythonize(model_input, self._copy_stream,
                                             self.pinned_sampled_token_ids)
                if model_output.pythonized:
                    ctx = output_proc_callback.keywords["ctx"]  # type: ignore
                    ctx.append_output(
                        outputs=[model_output.sampler_output],
                        seq_group_metadata_list=ctx.seq_group_metadata_list,
                        scheduler_outputs=ctx.scheduler_outputs,
                        is_async=False,
                        is_last_step=False,
                        is_first_step_output=step_num == 0)

                    output_proc_callback()
                else:
                    cont = False

            if not cont:
                break

    def _final_process_outputs(
            self, model_input: StatefulModelInputForNPU,
            output_proc_callback: Optional[Callable]) -> List[SamplerOutput]:
        assert model_input.frozen_model_input is not None

        has_async_callback = output_proc_callback is not None

        outputs = []
        for step_num, output in enumerate(model_input.cached_outputs):
            is_last_step = step_num == len(model_input.cached_outputs) - 1

            # For non-async case:
            #   -- We simply add the outputs
            # For async case:
            #   -- Invoke callback, pythonize, add to callback queue and repeat
            #   -- For last output, just add to callback queue
            if has_async_callback:
                assert output_proc_callback is not None

                # Invoke callback before pythonize (to overlap with NPU)
                output_proc_callback()

                # Pythonize
                if not output.pythonized:
                    output.pythonize(model_input, self._copy_stream,
                                     self.pinned_sampled_token_ids)

                    # For non last step, add to callback queue to chain
                    # callbacks=>pythonize pairs (for NPU overlap)
                    if not is_last_step:
                        ctx = output_proc_callback.keywords[  # type: ignore
                            "ctx"]  # type: ignore
                        ctx.append_output(
                            outputs=[output.sampler_output],
                            seq_group_metadata_list=ctx.
                            seq_group_metadata_list,
                            scheduler_outputs=ctx.scheduler_outputs,
                            is_async=False,
                            is_last_step=False,
                            is_first_step_output=step_num == 0)
                    else:
                        outputs.append(output.sampler_output)
            else:
                output.pythonize(model_input, self._copy_stream,
                                 self.pinned_sampled_token_ids)
                outputs.append(output.sampler_output)

        return outputs

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: StatefulModelInputForNPU,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        """ 
        Execute the model for a single step and update multi-step
        metadata
        """
        assert num_steps == 1, "MultiStepModelRunner only supports num_steps=1"
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None

        # path for warm up runs
        if not model_input.is_multi_step:
            return self._base_model_runner.execute_model(
                frozen_model_input, kv_caches, intermediate_tensors, num_steps)

        # make sure we skip the sampler on the lask rank and only pythonize
        # if CPU is ahead.
        if self.is_driver_worker and get_pp_group().is_last_rank:
            if self.pinned_sampled_token_ids is None:
                self.pinned_sampled_token_ids = torch.zeros(
                    (self.scheduler_config.max_num_seqs, 1),
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=True)

            self._base_model_runner.model.sampler.include_gpu_probs_tensor = (
                True)
            if frozen_model_input.sampling_metadata:
                frozen_model_input.sampling_metadata.skip_sampler_cpu_output = (
                    True)

        # some pre-execute model logic for multi-step:
        #   - if it's the first step, we need to reset the sampling tensors
        #   - if it's not the first step, we need to advance the step using the
        #   appended sampler output from last iteration
        #   - also maybe pythonize if CPU is ahead of NPU

        stream = current_stream()
        if not model_input.is_first_multi_step:
            # Explicitly block on the previous step's forward to make sure we
            # don't clobber any NPU tensors still in use.
            # This is not needed for flashattn backend, but for other attn
            # backends such as flashinfer that performs extra CPU operations on
            # input metadata we may need to synchronize any CPU operations that
            # might clobber enqueued forwards. (prevents CPU from running too
            # far ahead if needed)
            model_input.wait_previous_step()
            model_input = self._advance_step(
                model_input, model_input.cached_outputs[-1].sampler_output)

            # frozen_model_input may have been updated
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None

        if model_input.base_output_proc_callback is None:
            assert frozen_model_input is not None
            model_input.base_output_proc_callback = \
                        frozen_model_input.async_callback

        if frozen_model_input.async_callback is not None:
            assert model_input.base_output_proc_callback is not None
            async_callback = functools.partial(
                self._async_process_outputs,
                model_input=model_input,
                output_proc_callback=model_input.base_output_proc_callback)

            model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                model_input.frozen_model_input,
                async_callback=async_callback)
            # Update the local instance
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None

        # Execute the model
        output = self._base_model_runner.execute_model(frozen_model_input,
                                                       kv_caches,
                                                       intermediate_tensors,
                                                       num_steps=1)

        # record the event for the current step so that the next step can sync
        model_input.record_step_event(stream)

        if get_pp_group().is_last_rank and self.is_driver_worker:
            assert isinstance(output, list)
            assert len(
                output
            ) == 1, "MultiStepModelRunner requires single-step base_models"

            # event for the pythonization so that we only pythonize if the
            # tensors are ready. May be able to be combined with the step event
            output_ready_event = torch.npu.Event()
            output_ready_event.record(stream)
            if self.parallel_config.pipeline_parallel_size > 1:
                output[0].sampled_token_ids_cpu = output[
                    0].sampled_token_ids.cpu()
            model_input.cached_outputs.append(
                ModelOutput(output[0], output_ready_event,
                            output[0].sampled_token_ids, False,
                            output[0].logprobs, self.pythonization_cache))

            # These NPU tensors are not required by multi-step;
            # erase them to ensure they are not pythonized or
            # transferred to CPU
            output[0].sampled_token_ids = None
            output[0].sampled_token_probs = None
            output[0].logprobs = None

            # Pythonize the output if CPU is ahead and the previous step is
            # ready.
            if frozen_model_input.async_callback is None:
                for model_output in model_input.cached_outputs:
                    model_output.maybe_pythonize(model_input,
                                                 self._copy_stream,
                                                 self.pinned_sampled_token_ids)

        model_input.current_step += 1

        if not get_pp_group().is_last_rank:
            # Should be IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            return output
        if not self.is_driver_worker:
            return []

        # Pythonize the output and block if needed since it is the last step
        if model_input.is_last_step:
            outputs = self._final_process_outputs(
                model_input, model_input.base_output_proc_callback)
            if self.pythonization_cache:
                self.pythonization_cache.reset()
            return outputs

        # should be [SamplerOutput]
        return output

    def _update_sampling_metadata(self, sampling_metadata: SamplingMetadata,
                                  num_seqs: Optional[int], num_queries: int):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _advance_step(self, model_input: StatefulModelInputForNPU,
                      out: SamplerOutput) -> StatefulModelInputForNPU:

        model_input.maybe_advance_frozen_model_input(self.device,
                                                     self.pin_memory)
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        assert frozen_model_input.input_tokens is not None
        assert frozen_model_input.input_tokens.shape[0] == model_input.num_seqs
        assert frozen_model_input.attn_metadata is not None

        sampled_token_ids = model_input.cached_outputs[-1].sampled_token_ids
        num_seqs = model_input.num_seqs
        num_queries = model_input.num_queries
        frozen_model_input = model_input.frozen_model_input
        assert frozen_model_input is not None
        attn_metadata = frozen_model_input.attn_metadata
        assert attn_metadata is not None

        turn_prefills_into_decodes: bool = model_input.current_step == 1 and \
                                    model_input.num_single_step_prefills != 0
        attn_metadata.advance_step(
            frozen_model_input,
            sampled_token_ids,
            self.block_size,
            num_seqs,
            num_queries,
            turn_prefills_into_decodes=turn_prefills_into_decodes)

        return model_input

    def load_model(self) -> None:
        self._base_model_runner.load_model()
        self.model_memory_usage = self._base_model_runner.model_memory_usage

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        return self._base_model_runner.save_sharded_state(
            path, pattern, max_size)

    def save_tensorized_model(self,
                              tensorizer_config: TensorizerConfig) -> None:
        return self._base_model_runner.save_tensorized_model(tensorizer_config)

    def profile_run(self) -> None:
        return self._base_model_runner.profile_run()

    def remove_all_loras(self):
        return self._base_model_runner.remove_all_loras()

    def capture_model(self, kv_caches: List[List]) -> None:
        return self._base_model_runner.capture_model(kv_caches)

    @property
    def vocab_size(self) -> int:
        return self._base_model_runner.vocab_size


DeferredLogprobsReturnType = Tuple[Optional[List[Optional[PromptLogprobs]]],
                                   Optional[List[SampleLogprobs]]]


def deferred_pythonize_logprobs(
    output: SamplerOutput,
    sampling_metadata: SamplingMetadata,
    logprobs_tensor: Optional[torch.Tensor],
) -> DeferredLogprobsReturnType:
    """Perform deferred logprob Pythonization.

    1. Pythonize NPU-side sampler result tensors into CPU-side sampler result.
    2. Pythonize NPU-side logprobs tensor into CPU-side logprobs lists,
       utilizing  the Pythonized sampler result computed in step 1.
    
    These deferred computations are not required for single-step scheduling
    or the `profile_run()` phase of multi-step scheduling.

    Args:
        output: sampler output (under deferred Pythonization)
        sampling_metadata
        
    Returns:
        prompt_logprobs (CPU), sample_logprobs (CPU)
    """

    # - Deferred pythonization of sample result
    sampler_result = get_pythonized_sample_results(
        output.deferred_sample_results_args)

    # - Erase the NPU-side deferred sample_result
    #   computation args to ensure it is never
    #   pythonized or transferred to CPU
    output.deferred_sample_results_args = None

    # - Deferred pythonization of logprobs
    (
        prompt_logprobs,
        sample_logprobs,
    ) = get_logprobs(logprobs_tensor, sampling_metadata, sampler_result)
    assert len(prompt_logprobs) == len(sampling_metadata.seq_groups)
    assert len(sample_logprobs) == len(sampling_metadata.seq_groups)

    return prompt_logprobs, sample_logprobs


def _pythonize_sampler_output(
    model_input: StatefulModelInputForNPU,
    output: SamplerOutput,
    pinned_sampled_token_buffer: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    logprobs_tensor: Optional[torch.Tensor],
    cache: Optional[PythonizationCache],
) -> None:
    """ This function is only called when the output tensors are ready. 
    See :class:`ModelOutput`. 
    
    Modifies `output.outputs` and `pinned_sampled_token_buffer` in-place, 
    adding a Pythonized output data structure
    (:class:`CompletionSequenceGroupOutput`) for each :class:`SequenceGroup`.

    Args:
      model_input
      output: sampler output
      pinned_sampled_token_token_buffer: CPU-side pinned memory
                                         (receives copy of
                                         NPU-side token buffer.)
      sampled_token_ids: NPU-side token buffer
      logprobs_tensor: NPU-side tensor containing 
                       logprobs computed during sampling
    """

    assert model_input.frozen_model_input is not None

    frozen_model_input = model_input.frozen_model_input
    assert frozen_model_input.sampling_metadata is not None
    sampling_metadata = frozen_model_input.sampling_metadata
    # samples generation should have been skipped
    assert not output.outputs

    pinned_buffer = pinned_sampled_token_buffer[:model_input.num_queries]

    # We guarantee output tensors are ready, so it is safe to
    # pythonize the sampler output & obtain CPU-side logprobs.
    #
    # However we should check whether logprobs pythonization may
    # be skipped entirely, i.e. because no logprobs were requested
    # or pythonization was not deferred. To that end,
    #
    # * `prompt_logprobs_are_requested_for_prefill` signals that
    #   there are *any* prefill-phase requests which specify that
    #   prompt logprobs should be returned.
    #
    # * `any_logprobs_are_requested` signals that there are any
    #   requests which (1) specify that sample logprobs should be
    #   returned, or (2) are in the prefill phase AND specify that
    #   prompt logprobs should be returned.
    #
    # Later on, these flags cause adjustments to the pythonization
    # process to accommodate logprobs.

    seq_groups = sampling_metadata.seq_groups
    prompt_logprobs_are_requested_for_prefill = any([
        sg.sampling_params.prompt_logprobs is not None and sg.is_prompt
        for sg in seq_groups
    ])
    any_logprobs_are_requested = (
        prompt_logprobs_are_requested_for_prefill
        or any([sg.sampling_params.logprobs is not None for sg in seq_groups]))

    if prompt_logprobs_are_requested_for_prefill:
        # CPU NPU sync, after gathering *only* sampled tokens (since
        # requesting prompt logprobs leads `sampled_token_ids` to
        # include prompt token ids in addition to sampled token ids.)
        sample_idx_tensor = torch.tensor(
            [sdx for sg in seq_groups for sdx in sg.sample_indices])
        pinned_buffer = pinned_buffer.copy_(
            sampled_token_ids[sample_idx_tensor, :], non_blocking=False)
    else:
        # CPU NPU sync
        pinned_buffer = pinned_buffer.copy_(sampled_token_ids,
                                            non_blocking=False)

    # this will not block as the tensors are already on CPU
    samples_list = pinned_buffer.tolist()

    skip_sampler_cpu_output = (
        frozen_model_input.sampling_metadata.skip_sampler_cpu_output)

    # *Don't* skip logprobs pythonization *if*:
    # * Any requests require logprobs to be returned in this
    # iteration AND
    # * These requests are being scheduled in a fashion which
    # defers pythonization (i.e. multi-step scheduling.)
    do_pythonize_logprobs = (skip_sampler_cpu_output
                             and any_logprobs_are_requested)
    (
        prompt_logprobs,
        sample_logprobs,
    ) = (deferred_pythonize_logprobs(output, sampling_metadata,
                                     logprobs_tensor)
         if do_pythonize_logprobs else (None, None))

    for sgdx, (seq_group,
               sample_result) in enumerate(zip(seq_groups, samples_list)):
        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        # (Check for Guided Decoding)
        if seq_group.sampling_params.logits_processors:
            assert len(seq_group.sampling_params.logits_processors) == 0, (
                "Logits Processors are not supported in multi-step decoding")

        if do_pythonize_logprobs:
            assert prompt_logprobs is not None
            assert sample_logprobs is not None

            (
                group_prompt_logprobs,
                group_sample_logprobs,
            ) = (  # Utilize deferred pythonization results
                prompt_logprobs[sgdx],
                sample_logprobs[sgdx],
            )
        elif any_logprobs_are_requested:
            (
                group_prompt_logprobs,
                group_sample_logprobs,
            ) = (
                # profile_run: use already-computed logprobs
                output.outputs[sgdx].prompt_logprobs,
                [sample.logprobs for sample in output.outputs[sgdx].samples])

        seq_ids = seq_group.seq_ids
        next_token_ids = sample_result
        parent_ids = [0]
        seq_outputs: List[SequenceOutput]

        if cache is not None:
            completion_seq_group_output: CompletionSequenceGroupOutput = \
                cache.cached_completion_seq_group_output.get_object()
            completion_seq_group_output.samples.clear()
            seq_outputs = completion_seq_group_output.samples
        else:
            seq_outputs = []

        for tdx, (parent_id,
                  next_token_id) in enumerate(zip(parent_ids, next_token_ids)):
            if cache is not None:
                seq_output: SequenceOutput = cache.cached_seq_output.get_object(
                )
                seq_output.parent_seq_id = seq_ids[parent_id]
                seq_output.output_token = next_token_id

                if any_logprobs_are_requested:
                    seq_output.logprobs = group_sample_logprobs[tdx]
                else:
                    logprobs = next(iter(seq_output.logprobs.values()))
                    seq_output.logprobs.clear()

                    logprobs.logprob = float('inf')
                    logprobs.rank = None
                    logprobs.decoded_token = None

                    seq_output.logprobs[next_token_id] = logprobs

                seq_outputs.append(seq_output)

            else:
                seq_outputs.append(
                    SequenceOutput(seq_ids[parent_id], next_token_id,
                                   (group_sample_logprobs[tdx]
                                    if any_logprobs_are_requested else {
                                        next_token_id:
                                        Logprob(logprob=float('inf'),
                                                rank=None,
                                                decoded_token=None)
                                    })))
        if cache is not None:
            completion_seq_group_output.prompt_logprobs = \
                group_prompt_logprobs if any_logprobs_are_requested else None
            output.outputs.append(completion_seq_group_output)
        else:
            output.outputs.append(
                CompletionSequenceGroupOutput(
                    seq_outputs, (group_prompt_logprobs
                                  if any_logprobs_are_requested else None)))

    assert len(output.outputs) > 0
