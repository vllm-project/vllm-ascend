import copy
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import (ExecuteModelRequest, HiddenStates, SequenceData,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.worker.model_runner_base import BroadcastableModelInput
from vllm.worker.multi_step_model_runner import StatefulModelInput
from vllm.worker.worker_base import DelegateWorkerBase

from vllm_ascend.worker.draft_model_runner import TP1DraftModelRunner
from vllm_ascend.worker.multi_step_runner import MultiStepModelNPURunner
from vllm_ascend.worker.worker import WorkerInput


@dataclass
class MultiStepState:
    worker_input: WorkerInput
    model_input: StatefulModelInput


class MultiStepWorker(ProposerWorkerBase, DelegateWorkerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base_model_runner = self.model_runner
        # for multi-step model, wrap the model runner with MultiStepModelRunner
        self.model_runner = MultiStepModelNPURunner(
            base_model_runner,
            vllm_config=base_model_runner.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=base_model_runner.is_driver_worker,
        )

        pipeline_parallel_size = self.parallel_config.pipeline_parallel_size
        self.multi_step_states: List[
            Optional[MultiStepState]] = [None] * pipeline_parallel_size
        self.temp_output = None
        self._proposer: SpeculativeProposer

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """
        Get the driver input and broadcast it to other workers.
        """
        assert self.is_driver_worker
        virtual_engine = execute_model_req.virtual_engine
        is_first_multi_step = execute_model_req.is_first_multi_step
        if is_first_multi_step:
            # on first step we prepare the worker input and model input normally
            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: StatefulModelInput = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))

            if execute_model_req.async_callback:
                model_input.frozen_model_input = dataclasses.replace(  # type: ignore
                    model_input.frozen_model_input,
                    async_callback=execute_model_req.async_callback)
        else:
            # on subsequent steps we reuse the worker input and model input
            multi_step_state = self.multi_step_states[virtual_engine]
            worker_input = multi_step_state.worker_input
            model_input = multi_step_state.model_input
            frozen_model_input = model_input.frozen_model_input
            assert frozen_model_input is not None
            assert frozen_model_input.attn_metadata is not None
            # clear the cached metadata so that it can be recomputed on
            # the workers.
            frozen_model_input.attn_metadata._cached_prefill_metadata = None
            frozen_model_input.attn_metadata._cached_decode_metadata = None

        model_input.is_first_multi_step = is_first_multi_step
        model_input.is_last_step = execute_model_req.is_last_step

        if not is_first_multi_step:
            # we broadcast the last sampled token ids to all TP workers so they
            # can update their model input metadata in-place.
            self._prepare_last_sampled_token_ids_for_tp_workers(
                execute_model_req=execute_model_req, model_input=model_input)

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_tensor_dict(broadcast_data, src=0)

        # Retuning empty dict here to keep this compatible with
        # `LocalOrDistributedWorkerBase._get_driver_input_and_broadcast`
        return model_input, worker_input, {}

    def _prepare_last_sampled_token_ids_for_tp_workers(
        self,
        execute_model_req: ExecuteModelRequest,
        model_input: StatefulModelInput,
    ) -> None:
        """ 
        Prepare the last sampled token ids for TP workers. If it's the last 
        PP rank, then the last sampled token ids are already in the model_input.
        If it is NOT the last PP rank, then we need to get the last sampled
        token that is cached in the execute_model_req.
        """
        if get_pp_group().is_last_rank:
            assert model_input.cached_outputs[
                -1].sampler_output.sampled_token_ids is None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None
            model_input.last_sampled_token_ids = model_input.cached_outputs[
                -1].sampled_token_ids
            # free sampled token ids from the previous step if it has been
            # pythonized. Cannot free the last sampled token ids because
            # we need it for GPU advance_step.
            for output in model_input.cached_outputs[:-1]:
                if output.pythonized:
                    output.sampled_token_ids = None
        else:
            # otherwise we need to get the cached sampled token ids from the
            # execute_model_req
            assert execute_model_req.last_sampled_token_ids is not None
            model_input.last_sampled_token_ids = (
                execute_model_req.last_sampled_token_ids.cuda())
            model_input.add_sampler_output(
                SamplerOutput(outputs=[], sampled_token_ids=None),
                model_input.last_sampled_token_ids)

            # free sampled token ids from the previous step.
            # TODO(will) we could reuse the sampled token ids tensor from
            # the previous step instead.
            for output in model_input.cached_outputs[:-1]:
                output.sampled_token_ids = None
            assert model_input.cached_outputs[-1].sampled_token_ids is not None

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[Tuple[StatefulModelInput, WorkerInput, Dict[str,
                                                              torch.Tensor]]]:
        """
        Depending on the current state of the request and multi step worker,
        this method may skip the normal _prepare_model_input and
        _prepare_worker_input methods and instead used cached values.
        """
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            virtual_engine = execute_model_req.virtual_engine
            (model_input, worker_input,
             kwargs) = self._get_driver_input_and_broadcast(execute_model_req)
            assert isinstance(model_input, StatefulModelInput)
            if execute_model_req.is_first_multi_step:
                # cache the worker input and model input for the next steps
                self.multi_step_states[virtual_engine] = MultiStepState(
                    worker_input=worker_input, model_input=model_input)
        # if TP workers
        else:
            broadcast_data = self._get_worker_input_from_broadcast()
            # if the driver has sent an empty input, we should stop the worker
            # loop
            if broadcast_data is None:
                return None
            model_input, worker_input, kwargs = broadcast_data
            assert isinstance(model_input, StatefulModelInput)
            virtual_engine = worker_input.virtual_engine
            if model_input.is_first_multi_step:
                pass
                # TODO(will) Can cache the worker input and model input for the
                # next steps. See below for details
            else:
                # TODO(will) possible to also cache and reuse the cached worker
                # input and model input. The idea is essentially the delta
                # optimization for model_inputs. Where the TP workers can cache
                # the model input states and we only broadcast the delta need
                # for the next step (sampled_token_ids from the previous step)

                assert isinstance(model_input, StatefulModelInput)
                # we need to update the last sampled token ids in the model
                # input for the workers so that they can run inplace
                # advance_step
                model_input.add_sampler_output(
                    SamplerOutput(outputs=[], sampled_token_ids=None),
                    model_input.last_sampled_token_ids)

        assert model_input is not None
        assert worker_input is not None
        return model_input, worker_input, kwargs

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)
        # Expand the batch for sequences with a bonus token.
        # Perform a forward pass on the expanded batch and filter the
        # response to retain only the original sequences' responses.
        expanded_request, indices_of_seq_with_bonus_tokens =\
            self._expand_execute_model_request(
                execute_model_req, seq_ids_with_bonus_token_in_last_step)

        # Run model sample_len times.
        model_outputs: List[SamplerOutput] = []
        if isinstance(self.model_runner, TP1DraftModelRunner) \
            and self.model_runner.supports_gpu_multi_step(expanded_request):
            # Here we run the draft_model_runner with multi-step prepare
            # on the NPU directly
            expanded_request.num_steps = sample_len
            self.model_runner.set_indices_of_seq_with_bonus_tokens(
                indices_of_seq_with_bonus_tokens)
            model_outputs = self.execute_model(
                execute_model_req=expanded_request)
        else:
            # Here we run multi-step directly, with every step prepared
            # on the CPU.
            # TODO: Remove this branch once DraftModelRunner supports TP>1
            # and other restrictions that are part of DraftModelRunner's
            # supports_gpu_multi_step(..)
            for _ in range(sample_len):
                model_output: List[SamplerOutput] = self.worker.execute_model(
                    execute_model_req=expanded_request)
                assert (len(model_output) == 1
                        ), "composing multistep workers not supported"
                model_output = model_output[0]

                self._append_new_tokens(
                    model_output, expanded_request.seq_group_metadata_list,
                    indices_of_seq_with_bonus_tokens)
                model_outputs.append(model_output)

        # move indices to device to avoid stream sync
        indices_of_seq_with_bonus_tokens = torch.tensor(
            indices_of_seq_with_bonus_tokens, device=self.device)
        filtered_model_outputs = self._filter_model_output(
            model_outputs, indices_of_seq_with_bonus_tokens)
        return filtered_model_outputs, True

    @staticmethod
    def _expand_execute_model_request(
        execute_model_req: ExecuteModelRequest,
        seq_with_bonus_token_in_last_step: set,
    ) -> Tuple[ExecuteModelRequest, List[int]]:
        """
        Expands the execute model request based on sequences with bonus
        tokens.

        For each sequence with a bonus token, this method creates a new
        sequence without the bonus token and adds it to the execute model
        request. The original sequence groups are also retained. The indices
        of the original sequence groups are returned for further processing.

        Args:
            execute_model_req (ExecuteModelRequest): The original execute
            model request.
            seq_with_bonus_token_in_last_step (set): Set of sequence IDs that 
            contain bonus tokens.

        Returns:
            Tuple[ExecuteModelRequest, List[int]]: The updated execute model
            request with expanded sequences and a list of indices corresponding
            to the original sequence groups.
        """
        updated_seq_group_metadata_list: List[SequenceGroupMetadata] = []
        updated_execute_model_req = execute_model_req.clone(
            updated_seq_group_metadata_list)
        indices_of_original_sequence_groups = []
        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_group_has_bonus_tokens = False
            for seq_id, _ in seq_group.seq_data.items():
                # Identify sequences with bonus tokens in the sequence group.
                if seq_id in seq_with_bonus_token_in_last_step:
                    seq_group_has_bonus_tokens = True
                    break
            if seq_group_has_bonus_tokens:
                #Create new sequences without the last bonus token. These new
                # sequence have the same sequence id as the original sequence.
                # We create a new sequence group and add them there.
                updated_seq_group_without_bonus_token  = \
                    MultiStepWorker._copy_seq_metadata_excluding_last_token(
                        seq_group, seq_with_bonus_token_in_last_step)
                updated_seq_group_metadata_list.append(
                    updated_seq_group_without_bonus_token)
            # Add the original sequence group.
            updated_seq_group_metadata_list.append(
                MultiStepWorker._shallow_copy_seq_group_metadata(seq_group))
            # Record the index of the original sequence group.
            indices_of_original_sequence_groups.append(
                len(updated_seq_group_metadata_list) - 1)

        updated_execute_model_req.seq_group_metadata_list =\
            updated_seq_group_metadata_list

        if isinstance(updated_execute_model_req.previous_hidden_states,
                      HiddenStates):
            updated_execute_model_req.previous_hidden_states\
                .expand_with_bonus_tokens(seq_with_bonus_token_in_last_step)

        return updated_execute_model_req, indices_of_original_sequence_groups

    @staticmethod
    def _shallow_copy_seq_group_metadata(
        seq_group_metadata: SequenceGroupMetadata, ) -> SequenceGroupMetadata:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
        """
        # Shallow-copy the SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        # We must shallow-copy seq_group_metadata as is_prompt could change.
        new_seq_group_metadata = copy.copy(seq_group_metadata)

        # We must shallow-copy seq_data as we will append token ids
        new_seq_data: Dict[int, SequenceData] = {}
        for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
            new_seq_data[seq_id] = copy.copy(old_seq_data)
            new_seq_data[seq_id].output_token_ids =\
                old_seq_data.output_token_ids[:]

        new_seq_group_metadata.seq_data = new_seq_data
        return new_seq_group_metadata

    @staticmethod
    def _copy_seq_metadata_excluding_last_token(
        seq_group_metadata: SequenceGroupMetadata,
        seq_ids_to_copy: Set[int],
    ) -> SequenceGroupMetadata:
        """
        Creates a shallow copy of the given SequenceGroupMetadata, retaining
        only the sequence IDs specified in seq_ids_to_copy. For each of these
        sequence IDs, all output_token_ids except the last one are copied.
        Sequence IDs not in seq_ids_to_copy are excluded from the copy.
        
        Parameters:
        seq_group_metadata (SequenceGroupMetadata): The original sequence
            group metadata.
        seq_ids_to_copy (Set[int]): The set of sequence IDs to include in the
            copy.
        
        Returns:
        SequenceGroupMetadata: A shallow copy of the sequence group metadata
            with the specified modifications.
        """
        # Shallow-copy the SequenceGroupMetadata.
        new_seq_group_metadata = copy.copy(seq_group_metadata)
        # Shallow-copy seq_data and modify the output_token_ids.
        new_seq_data: Dict[int, SequenceData] = {}
        for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
            if (seq_id in seq_ids_to_copy):
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                # Copy all the output token ids except the last.
                # Also reduce num_computed_tokens by 1 since we are not
                # including the last output token.
                # NOTE: num_computed_tokens is not directly used by the
                # speculative decoding workers, as it is only relevant for
                # chunked prefill, which is disabled for speculative decoding.
                # However, to maintain consistency in num_computed_tokens,
                # we update it here.
                new_seq_data[seq_id].output_token_ids =\
                    old_seq_data.output_token_ids[:-1]
                new_seq_data[seq_id].update_num_computed_tokens(-1)
        new_seq_group_metadata.seq_data = new_seq_data
        return new_seq_group_metadata

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        """MultiStepWorker does not yet implement support for cache swap
        operations or beam search.
        """
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "MultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "MultiStepWorker does not support beam search.")

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: set,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    def maybe_load_lm_head_weight(
        self,
        lm_head_weight: torch.Tensor,
    ) -> None:
        weight_loader = getattr(
            self.worker.model_runner.model_runner.model.lm_head.weight,
            "weight_loader", default_weight_loader)
        weight_loader(
            self.worker.model_runner.model_runner.model.lm_head.weight,
            lm_head_weight)
