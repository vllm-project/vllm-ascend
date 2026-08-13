from examples.disaggregated_prefill_v1.proxy_utils import append_generated_text_to_chat_content


def test_append_generated_text_preserves_multimodal_content():
    original_content = [
        {"type": "text", "text": "Read the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    updated_content = append_generated_text_to_chat_content(original_content, "partial answer")

    assert updated_content == [
        *original_content,
        {"type": "text", "text": "partial answer"},
    ]
    assert original_content == [
        {"type": "text", "text": "Read the image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_append_generated_text_keeps_text_content_compatible():
    assert append_generated_text_to_chat_content("prompt", " continuation") == "prompt continuation"
