# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-only, non-streaming adapter; no reasoning-number fallback."""

from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat

from tools.kimi_k3.final_answer import final_message


class GSM8KFinalAPIChat(VLLMCustomAPIChat):
    async def parse_text_response(self, json_content, output):
        await self._parse_usage(json_content, output)
        choices = json_content.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        output.content = final_message(message, finish)
        output.reasoning_content = ""
        output.update_extra_details_data_from_text_response(json_content)
        output.extra_details_data["gsm8k_finish_reason"] = finish
        output.extra_details_data["gsm8k_final_present"] = bool(output.content)
