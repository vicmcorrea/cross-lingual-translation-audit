import numpy as np
import polars as pl

from translation_audit.analysis.metrics import compute_retrieval_metrics
from translation_audit.analysis.sensitivity import (
    affective_sensitivities,
    paired_encoder_differences,
    summarize_language_matched,
    summarize_retrieval_sensitivity,
)
from translation_audit.emotion.features import EMOTIONS, VALENCES


def _analysis_frame() -> pl.DataFrame:
    rows = 6
    data: dict[str, object] = {
        "pair_id": [f"p{index}" for index in range(rows)],
        "participant_id": ["a", "a", "b", "b", "c", "c"],
        "cohort_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
        "prompt_family": ["positive", "improvement", "mental"] * 2,
        "pt_word_count": [1, 2, 3, 5, 10, 30],
        "exact_normalized_copy": [False] * rows,
        "pt_duplicate_group_id": [f"pt{index}" for index in range(rows)],
        "en_duplicate_group_id": [f"en{index}" for index in range(rows)],
        "cometkiwi_score": np.linspace(0.5, 0.8, rows),
        "emotion_jensen_shannon_distance": np.linspace(0.1, 0.3, rows),
        "valence_jensen_shannon_distance": np.linspace(0.05, 0.2, rows),
        "emotion_mean_absolute_share_delta": np.linspace(0.02, 0.08, rows),
        "valence_mean_absolute_share_delta": np.linspace(0.01, 0.04, rows),
        "semantic_density_absolute_delta": np.linspace(0.01, 0.06, rows),
    }
    for direction in ("en_to_pt", "pt_to_en"):
        data[f"paired_cosine_{direction}"] = np.linspace(0.6, 0.9, rows)
        data[f"retrieval_rank_{direction}"] = [1, 2, 1, 3, 1, 1]
        data[f"reciprocal_rank_{direction}"] = [1.0, 0.5, 1.0, 1 / 3, 1.0, 1.0]
        data[f"hit_at_1_{direction}"] = [True, False, True, False, True, True]
        data[f"hit_at_5_{direction}"] = [True] * rows
    return pl.DataFrame(data)


def _emotion_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(6):
        record: dict[str, object] = {
            "pair_id": f"p{index}",
            "emotion_jensen_shannon_distance": 0.1 * index,
            "valence_jensen_shannon_distance": 0.05 * index,
            "emotion_mean_absolute_share_delta": 0.01 * index,
            "valence_mean_absolute_share_delta": 0.005 * index,
            "pt_emotion_lexicon_coverage": 0.0 if index == 0 else 0.5,
            "en_emotion_lexicon_coverage": 0.0 if index == 1 else 0.5,
            "pt_valence_lexicon_coverage": 0.0 if index == 0 else 0.4,
            "en_valence_lexicon_coverage": 0.0 if index == 1 else 0.4,
            "pt_semantic_node_count": 0.0 if index == 0 else 4.0,
            "en_semantic_node_count": 0.0 if index == 1 else 4.0,
        }
        for prefix in ("pt", "en"):
            for emotion in EMOTIONS:
                record[f"{prefix}_emotion_{emotion}_share"] = 0.0
            for valence in VALENCES:
                record[f"{prefix}_valence_{valence}_share"] = 0.0
        if index != 0:
            record["pt_emotion_joy_share"] = 1.0
            record["pt_valence_positive_share"] = 1.0
        if index != 1:
            record["en_emotion_joy_share"] = 1.0
            record["en_valence_positive_share"] = 1.0
        rows.append(record)
    return pl.from_dicts(rows)


def test_language_model_and_affective_sensitivities_are_clustered_and_paired() -> None:
    primary = _analysis_frame()
    language = pl.DataFrame(
        {
            "pair_id": primary["pair_id"],
            "pt_matches_expected": [True, True, True, False, True, True],
            "en_matches_expected": [True, False, True, True, True, True],
            "pt_ambiguous": [False, False, False, False, False, False],
            "en_ambiguous": [False, True, False, False, False, False],
        }
    )
    matched, coverage = summarize_language_matched(
        primary,
        language,
        repetitions=50,
        confidence_level=0.95,
        seed=7,
        bootstrap_batch_size=11,
    )
    assert matched["n_pairs"].unique().to_list() == [4]
    both = coverage.filter(
        (pl.col("stratum_type") == "overall")
        & (pl.col("metric") == "both_languages_match")
    )
    assert both["estimate"].item() == 4 / 6
    expected_language_rates = {
        row["metric"]: row["estimate"]
        for row in coverage.filter(pl.col("stratum_type") == "overall").iter_rows(
            named=True
        )
    }
    assert expected_language_rates["pt_ambiguous"] == 0.0
    assert expected_language_rates["en_ambiguous"] == 1 / 6
    assert expected_language_rates["pt_explicit_mismatch"] == 1 / 6
    assert expected_language_rates["en_explicit_mismatch"] == 0.0

    comparison = primary.with_columns(
        pl.Series("hit_at_1_en_to_pt", [True, True, True, False, True, True])
    )
    differences = paired_encoder_differences(
        primary,
        comparison,
        primary_model="4B",
        comparison_model="8B",
        repetitions=50,
        confidence_level=0.95,
        seed=7,
        bootstrap_batch_size=11,
    )
    delta = differences.filter(
        (pl.col("direction") == "en_to_pt") & (pl.col("metric") == "hit_at_1")
    )["estimate_delta_comparison_minus_primary"].item()
    assert delta == 1 / 6

    affective = affective_sensitivities(
        primary,
        _emotion_frame(),
        repetitions=50,
        confidence_level=0.95,
        seed=7,
        bootstrap_batch_size=11,
    )
    assert affective.nonzero_metrics["n_pairs"].unique().to_list() == [4]
    assert set(affective.coverage_by_length["stratum_value"].unique()) == {
        "all",
        "1-2 words",
        "3-5 words",
        "6-20 words",
        "21+ words",
    }


def test_direction_specific_target_deduplication_reduces_candidate_count() -> None:
    metadata = pl.DataFrame(
        {
            "source_row": [0, 1, 2],
            "pair_id": ["p0", "p1", "p2"],
            "participant_id": ["a", "b", "c"],
            "pt_duplicate_group_id": ["pt-a", "pt-a", "pt-b"],
            "en_duplicate_group_id": ["en-a", "en-b", "en-b"],
        }
    )
    identity = np.eye(3, dtype=np.float32)
    rows: list[dict[str, object]] = []
    for direction in ("en_to_pt", "pt_to_en"):
        rows.extend(
            {
                "pair_id": f"p{index}",
                "source_row": index,
                "direction": direction,
                "query_embedding": identity[index].tolist(),
                "document_embedding": identity[index].tolist(),
            }
            for index in range(3)
        )
    retrieval = compute_retrieval_metrics(
        pl.DataFrame(rows),
        metadata,
        batch_size=2,
        requested_device="cpu",
        deduplicate_targets=True,
    )
    summary = summarize_retrieval_sensitivity(
        retrieval.frame,
        metadata,
        sensitivity_name="target_deduplication",
        candidate_counts={"en_to_pt": 2, "pt_to_en": 2},
        repetitions=50,
        confidence_level=0.95,
        seed=7,
        bootstrap_batch_size=11,
    )
    assert summary["n_target_candidates"].unique().to_list() == [2]
    assert summary.height == 6
