import pytest

from translation_audit.runtime.run_workspace import assert_config_is_secret_free


def test_public_configuration_is_accepted() -> None:
    assert_config_is_secret_free({"model": {"revision": "abc"}, "runtime": {"device": "cuda"}})


@pytest.mark.parametrize("key", ["api_key", "hf_token", "client_secret", "password"])
def test_secret_shaped_hydra_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_config_is_secret_free({"runtime": {key: "value"}})
