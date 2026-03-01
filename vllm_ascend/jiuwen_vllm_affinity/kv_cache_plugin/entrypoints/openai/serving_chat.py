import time
from collections.abc import AsyncGenerator, Sequence

import jinja2
from fastapi import Request
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import GenerationError
from vllm.entrypoints.utils import get_max_tokens
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import (
    BeamSearchParams,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters
from vllm.v1.serial_utils import bytestr

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.entrypoints.openai.protocol import (
    ReleaseKvCacheRequest,
    ReleaseKvCacheResponse,
)
from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.engine.core import (
    encode_engine_core_request,
    pack_request_sharing_cache_salt,
)

logger = init_logger(__name__)


class OpenAIServingChatEx(OpenAIServingChat):
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )

            if request.tools is None or (request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Common case.
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret
                conversation, engine_prompts = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                # For GPT-OSS.
                conversation, engine_prompts = self._make_request_with_harmony(request)
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text, _, _ = self._get_prompt_components(engine_prompt)
                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=len(engine_prompt["prompt_token_ids"]),
                    default_sampling_params=self.default_sampling_params,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )
                    validate_logits_processors_parameters(
                        self.logits_processors,
                        sampling_params,
                    )

                self._log_inputs(
                    sub_request_id,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=sub_request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                else:
                    sharing_cache_salt = (
                        request.cache_salt
                        if getattr(request, "cache_sharing", False) and hasattr(request, "cache_salt")
                        else None
                    )
                    if sharing_cache_salt is not None:
                        sub_request_id = pack_request_sharing_cache_salt(sub_request_id, sharing_cache_salt)
                    engine_request, tokenization_kwargs = await self._process_inputs(
                        sub_request_id,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

                    if sharing_cache_salt is not None and request.cache_sharing:
                        engine_request.cache_salt = None

                    generator = self.engine_client.generate(
                        engine_request,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        prompt_text=prompt_text,
                        tokenization_kwargs=tokenization_kwargs,
                        data_parallel_rank=data_parallel_rank,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
            )
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def release_kv_cache(
        self, request: ReleaseKvCacheRequest, raw_request: Request
    ) -> ReleaseKvCacheResponse | ErrorResponse:
        """
        Release kv cache API similar to OpenAI's API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )

            if request.tools is None or (request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            message_length = len(request.messages)
            message_begin = request.messages_released_index if request.messages_released_index > 0 else 0
            message_begin = min(message_begin, message_length)

            after_tool_dicts = tool_dicts
            if after_tool_dicts is not None and request.tools_released_index is not None:
                after_tool_dicts = tool_dicts[: request.tools_released_index]
            tool_changed = after_tool_dicts is not None and len(tool_dicts) != len(after_tool_dicts)
            if message_begin >= message_length and not tool_changed:
                # nothing to release
                return ReleaseKvCacheResponse(cache_salt=request.cache_salt, block_released=0)
            if not self.use_harmony:
                # Common case.
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret
                before_request_prompts, before_engine_prompts = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    add_special_tokens=request.add_special_tokens,
                )
                (after_request_prompts, after_engine_prompts) = (
                    ([], [])
                    if (message_begin == 0)
                    else await self._preprocess_chat(
                        request,
                        tokenizer,
                        request.messages[:message_begin],
                        chat_template=request.chat_template or self.chat_template,
                        chat_template_content_format=self.chat_template_content_format,
                        add_generation_prompt=request.add_generation_prompt,
                        continue_final_message=request.continue_final_message,
                        tool_dicts=after_tool_dicts,
                        documents=request.documents,
                        chat_template_kwargs=request.chat_template_kwargs,
                        tool_parser=tool_parser,
                        add_special_tokens=request.add_special_tokens,
                    )
                )
            else:
                logger.error("Error in preprocessing prompt inputs")
                return self.create_error_response("harmony not supported")
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_id = f"chatcmpl-{self._base_request_id(raw_request, str(time.time()))}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if len(before_engine_prompts) != 1 or (message_begin > 0 and len(after_engine_prompts) != 1):
            error_msg = (
                f"engine prompts should be 1, while older is {len(before_engine_prompts)} "
                f"and newer is {len(after_engine_prompts)}"
            )
            logger.error(error_msg)
            return self.create_error_response(error_msg)

        before_token_ids: list[int] = []
        prompt_token_ids_key: str = "prompt_token_ids"
        for i, engine_prompt in enumerate(before_engine_prompts):
            if isinstance(engine_prompt, dict) and prompt_token_ids_key in engine_prompt:
                before_token_ids.extend(engine_prompt[prompt_token_ids_key])

        after_token_ids: list[int] = []
        for i, engine_prompt in enumerate(after_engine_prompts):
            if isinstance(engine_prompt, dict) and prompt_token_ids_key in engine_prompt:
                after_token_ids.extend(engine_prompt[prompt_token_ids_key])

        assert len(before_token_ids) > len(after_token_ids)
        released_token_index = len(after_token_ids)
        if tool_changed:
            for i in range(len(after_token_ids)):
                if before_token_ids[i] != after_token_ids[i]:
                    released_token_index = i
                    break
        elif len(after_token_ids) > 0:
            last_match_index_after = 0
            last_match_index_before = 0
            match_count = len(after_token_ids)
            eos_token_count = 5
            if match_count > eos_token_count:
                match_count -= eos_token_count
            for idx, val in enumerate(before_token_ids):
                if val == after_token_ids[last_match_index_after]:
                    last_match_index_before = idx
                    last_match_index_after += 1
                    if last_match_index_after >= match_count:
                        break
            released_token_index = last_match_index_before + 1
        request_params: list[tuple[Sequence[bytestr], int]] = []
        try:
            elapsed_tokens = 0
            for i, engine_prompt in enumerate(before_engine_prompts):
                if not isinstance(engine_prompt, dict) or prompt_token_ids_key not in engine_prompt:
                    continue
                if elapsed_tokens + len(engine_prompt[prompt_token_ids_key]) <= released_token_index:
                    elapsed_tokens += len(engine_prompt[prompt_token_ids_key])
                    continue
                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(before_engine_prompts) == 1 else f"{request_id}_{i}"

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                sampling_params = SamplingParams()
                sampling_params.structured_outputs = StructuredOutputsParams()
                validate_logits_processors_parameters(
                    self.logits_processors,
                    sampling_params,
                )

                self._log_inputs(
                    sub_request_id,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)
                sharing_cache_salt = (
                    request.cache_salt if hasattr(request, "cache_sharing") and request.cache_sharing else None
                )
                if sharing_cache_salt is not None:
                    sub_request_id = pack_request_sharing_cache_salt(sub_request_id, sharing_cache_salt)
                engine_request, _ = await self._process_inputs(
                    sub_request_id,
                    engine_prompt,
                    sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=0,
                )

                if sharing_cache_salt is not None and request.cache_sharing:
                    engine_request.cache_salt = None

                request_params.append(
                    (
                        encode_engine_core_request(engine_request),
                        released_token_index - elapsed_tokens,
                    )
                )
                elapsed_tokens += len(engine_prompt[prompt_token_ids_key])
                released_token_index = elapsed_tokens
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
        released_num = await self.engine_client.release_kv_cache(request.cache_salt, request_params)
        return ReleaseKvCacheResponse(cache_salt=request.cache_salt, block_released=released_num)


def register_openai_serving():
    OpenAIServingChat.release_kv_cache = OpenAIServingChatEx.release_kv_cache
    OpenAIServingChat.create_chat_completion = OpenAIServingChatEx.create_chat_completion
