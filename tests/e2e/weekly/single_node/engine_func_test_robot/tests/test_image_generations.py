import pytest

from tests.e2e.weekly.single_node.engine_func_test_robot.utility import assertion, image_helper


def _generation_request(**overrides):
    data = {
        "prompt": "a beautiful sunset over the ocean",
        "size": "512x512",
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"n": 2},
        {"size": "1024x1024"},
        {"negative_prompt": "blurry"},
        {"seed": 42},
        {"num_inference_steps": 20},
        {"guidance_scale": 7.5},
        {"true_cfg_scale": 5.0},
        {"response_format": "b64_json"},
        {"output_format": "jpeg", "output_compression": 80},
    ],
    ids=[
        "basic",
        "multi_image",
        "size",
        "negative_prompt",
        "seed",
        "steps",
        "guidance",
        "true_cfg",
        "response_format",
        "output_format",
    ],
)
def test_image_generations_accepts_representative_options(api_client, overrides):
    # Each entry represents one option family; exhaustive boundary matrices are intentionally collapsed.
    response = image_helper.send_image_generation_request(api_client, _generation_request(**overrides))
    response_json = image_helper.assert_image_generation_response_fields(response)
    if "n" in overrides:
        assert len(response_json["data"]) == overrides["n"]


@pytest.mark.parametrize(
    "prompt",
    [
        "a cat",
        "cyberpunk city at night with neon lights",
        "A detailed landscape with mountains and rivers. " * 20,
    ],
    ids=["short", "descriptive", "long"],
)
def test_image_generations_accepts_representative_prompt_shapes(api_client, prompt):
    # Prompt coverage is kept to short, descriptive, and long text shapes.
    response = image_helper.send_image_generation_request(api_client, _generation_request(prompt=prompt))
    image_helper.assert_image_generation_response_fields(response)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": ""},
        {"n": 0},
        {"n": 11},
        {"n": "2"},
        {"size": "invalid"},
        {"size": "0x0"},
        {"response_format": "url"},
        {"num_inference_steps": 0},
        {"num_inference_steps": 201},
        {"guidance_scale": -0.1},
        {"guidance_scale": 20.1},
        {"true_cfg_scale": -0.1},
        {"seed": "not-an-int"},
        {"output_compression": 101, "output_format": "jpeg"},
    ],
    ids=[
        "empty_prompt",
        "n_low",
        "n_high",
        "n_type",
        "bad_size",
        "zero_size",
        "bad_response_format",
        "steps_low",
        "steps_high",
        "guidance_low",
        "guidance_high",
        "true_cfg_low",
        "bad_seed",
        "bad_compression",
    ],
)
def test_image_generations_rejects_invalid_values(api_client, overrides):
    # Invalid inputs are grouped by validation class instead of repeating every equivalent literal.
    response = image_helper.send_image_generation_request(api_client, _generation_request(**overrides))
    assertion.assert_validation_error_response(response)


def test_image_generations_rejects_missing_prompt(api_client):
    # Missing prompt is the only required-field case that cannot be represented as an override.
    response = image_helper.send_image_generation_request(api_client, {"size": "512x512"})
    assertion.assert_validation_error_response(response)
