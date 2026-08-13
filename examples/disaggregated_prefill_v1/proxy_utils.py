from typing import Any


def append_generated_text_to_chat_content(content: str | list[Any], generated_text: str) -> str | list[Any]:
    """Append generated text without flattening multimodal chat content."""
    if isinstance(content, str):
        return content + generated_text
    if isinstance(content, list):
        updated_content = list(content)
        if generated_text:
            updated_content.append({"type": "text", "text": generated_text})
        return updated_content
    raise TypeError(f"Unsupported chat content type: {type(content).__name__}")
