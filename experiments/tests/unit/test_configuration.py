import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from translation_audit.config import EXPERIMENT_ROOT
from translation_audit.resolvers import register_resolvers


def test_default_configuration_is_nonexecuting_and_secret_free() -> None:
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(EXPERIMENT_ROOT / "conf")):
        cfg = compose(config_name="config", return_hydra_config=True)
    assert cfg.runtime.execute is False
    assert cfg.runtime.name == "local"
    assert cfg.stage.name == "prepare_cohort"
    rendered = OmegaConf.to_yaml(cfg, resolve=False).lower()
    assert "runpod_api_key" not in rendered
    assert "hf_token" not in rendered


def test_active_source_has_no_archived_service_clients() -> None:
    source_root = EXPERIMENT_ROOT / "src"
    active_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in source_root.rglob("*.py")
    ).lower()
    assert "receptiviti" not in active_text
    assert "openrouter" not in active_text


@pytest.mark.parametrize(
    "stage_name",
    [
        "prepare_cohort",
        "validate_languages",
        "compute_embeddings",
        "estimate_translation_quality",
        "compute_emotion_features",
        "analyze",
    ],
)
@pytest.mark.parametrize("runtime_name", ["local", "runpod_secure"])
def test_every_stage_composes_with_the_common_run_contract(
    stage_name: str,
    runtime_name: str,
) -> None:
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(EXPERIMENT_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=[f"stage={stage_name}", f"runtime={runtime_name}"])
    assert int(cfg.runtime.checkpoint_every_rows) > 0
    assert cfg.runtime.resume is True
    assert cfg.runtime.store_response_text_in_logs is False
    assert str(cfg.logging.json_filename).endswith(".jsonl")


def test_runpod_runtime_requires_deletion_and_a_bounded_lease() -> None:
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(EXPERIMENT_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=["runtime=runpod_secure"])
    assert cfg.runtime.delete_pod_on_completion is True
    assert cfg.runtime.delete_pod_on_failure is True
    assert cfg.runtime.verify_pod_deletion is True
    assert 0 < int(cfg.runtime.automatic_terminate_hours) <= 6


def test_embedding_stage_uses_the_active_implementation_and_pinned_qwen() -> None:
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(EXPERIMENT_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=["stage=compute_embeddings"])
    assert cfg.stage.implementation == "compute_embeddings"
    assert cfg.stage.backend == "sentence_transformers"
    assert cfg.encoder.repository == "Qwen/Qwen3-Embedding-4B"
    assert len(str(cfg.encoder.revision)) == 40
    assert cfg.encoder.frozen is True
