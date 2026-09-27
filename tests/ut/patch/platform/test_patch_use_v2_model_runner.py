from vllm_ascend.patch.platform import patch_use_v2_model_runner


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_v1_model_runner_unsupported_features",
        lambda _: [
            "prefill context parallel",
            "dspark speculative decoding",
            "dflash2 drafts",
            "diffusion models",
        ],
    )

    unsupported = patch_use_v2_model_runner._patched_get_v1_model_runner_unsupported_features(object())

    assert unsupported == ["prefill context parallel", "diffusion models"]


def test_release_pcp_is_not_rejected_as_v2_unsupported_feature(monkeypatch):
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_unsupported_features",
        lambda _: ["prefill context parallelism", "diffusion models"],
    )
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "resolve_spec_pp_support",
        lambda _: None,
    )

    unsupported = patch_use_v2_model_runner._patched_get_unsupported_features(object())

    # Both supported pins delegate PCP checks to the manager (#53853).
    # The Ascend wrapper must preserve any remaining upstream restriction.
    assert unsupported == ["prefill context parallelism", "diffusion models"]
