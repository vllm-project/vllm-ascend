import unittest

from starlette.responses import StreamingResponse

from vllm_ascend.openai_proxy_utils import (
    append_text_to_chat_content,
    append_text_to_prompt,
    streaming_response_kwargs,
)


class TestAppendTextToPrompt(unittest.TestCase):

    def test_append_to_str(self):
        self.assertEqual(append_text_to_prompt("hello", " world"), "hello world")

    def test_append_to_list_of_str(self):
        self.assertEqual(append_text_to_prompt(["hello"], " world"), ["hello world"])

    def test_append_to_empty_list(self):
        self.assertEqual(append_text_to_prompt([], "x"), ["x"])


class TestAppendTextToChatContent(unittest.TestCase):

    def test_append_to_str(self):
        self.assertEqual(append_text_to_chat_content("hello", " world"), "hello world")

    def test_append_to_list_of_text_parts(self):
        content = [{"type": "text", "text": "hello"}]
        updated = append_text_to_chat_content(content, " world")
        self.assertEqual(updated, [{"type": "text", "text": "hello world"}])
        self.assertEqual(content, [{"type": "text", "text": "hello"}])

    def test_append_to_list_ending_with_non_text_part(self):
        content = [{"type": "image_url", "image_url": "https://example.com/a.png"}]
        updated = append_text_to_chat_content(content, "hello")
        self.assertEqual(content, [{"type": "image_url", "image_url": "https://example.com/a.png"}])
        self.assertEqual(updated[-1], {"type": "text", "text": "hello"})

    def test_append_to_list_ending_with_str(self):
        self.assertEqual(append_text_to_chat_content(["hello"], " world"), ["hello world"])


class TestStreamingResponseKwargs(unittest.TestCase):

    def test_streaming_response_sets_exact_sse_content_type(self):
        response = StreamingResponse(iter([b"data: ok\n\n"]), **streaming_response_kwargs(True))
        self.assertEqual(response.headers["content-type"], "text/event-stream")

    def test_non_streaming_response_uses_json_media_type(self):
        response = StreamingResponse(iter([b"{}"]), **streaming_response_kwargs(False))
        self.assertEqual(response.headers["content-type"], "application/json")

