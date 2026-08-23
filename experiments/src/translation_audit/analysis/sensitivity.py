"""Prespecified, text-free sensitivity summaries for the translation audit."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl

from translation_audit.analysis.metrics import (
    clustered_bootstrap_conditional_means,
)
from translation_audit.emotion.features import EMOTIONS, VALENCES

_DIRECTIONS = ("en_to_pt", "pt_to_en")
_RETRIEVAL_METRICS = ("hit_at_1", "hit_at_5", "reciprocal_rank")
_ANALYSIS_INDEPENDENT_METRICS = (
    "cometkiwi_score",
    "emotion_jensen_shannon_distance",
    "valence_jensen_shannon_distance",
    "emotion_mean_absolute_share_delta",
    "valence_mean_absolute_share_delta",
    "semantic_density_absolute_delta",
)
_COMET_LABEL = "COMETKiwi automated reference-free criterion; not human gold"


@dataclass(frozen=True, slots=True)
class AffectiveSensitivityOutput:
    """Nonzero-profile estimates and length-stratified coverage diagnostics."""

    nonzero_metrics: pl.DataFrame
    coverage_by_length: pl.DataFrame


def _seed(seed: int, *labels: str) -> int:
    digest = hashlib.sha256("\u0000".join(labels).encode()).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _length_band_expression() -> pl.Expr:
    return (
        pl.when(pl.col("pt_word_count") <= 2)
        .then(pl.lit("1-2 words"))
        .when(pl.col("pt_word_count") <= 5)
        .then(pl.lit("3-5 words"))
        .when(pl.col("pt_word_count") <= 20)
        .then(pl.lit("6-20 words"))
        .otherwise(pl.lit("21+ words"))
        .alias("length_band")
    )


def _bootstrap_matrix(
    values: np.ndarray,
    participants: Sequence[str],
    cluster_universe: Sequence[str],
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return clustered_bootstrap_conditional_means(
        values,
        participants,
        cluster_universe,
        repetitions=repetitions,
        confidence_level=confidence_level,
        seed=seed,
        batch_size=bootstrap_batch_size,
    )


def summarize_language_matched(
    analysis_frame: pl.DataFrame,
    language_frame: pl.DataFrame,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize primary metrics after both expected language labels are returned."""
    required_language = {
        "pair_id",
        "pt_matches_expected",
        "en_matches_expected",
        "pt_ambiguous",
        "en_ambiguous",
    }
    if not required_language.issubset(language_frame.columns):
        raise ValueError("Language sensitivity input lacks expected-label flags")
    joined = analysis_frame.join(
        language_frame.select(required_language), on="pair_id", how="inner", validate="1:1"
    ).with_columns(
        (pl.col("pt_matches_expected") & pl.col("en_matches_expected")).alias(
            "both_languages_match"
        ),
        _length_band_expression(),
    ).with_columns(
        (
            (~pl.col("pt_matches_expected") & ~pl.col("pt_ambiguous"))
            | (~pl.col("en_matches_expected") & ~pl.col("en_ambiguous"))
        ).alias("diagnostic_explicit_mismatch"),
    ).with_columns(
        (
            ~pl.col("both_languages_match") & ~pl.col("diagnostic_explicit_mismatch")
        ).alias("diagnostic_ambiguous_only"),
    )
    if joined.height != analysis_frame.height:
        raise ValueError("Language sensitivity input does not cover every analysis pair")
    matched = joined.filter(pl.col("both_languages_match"))
    cluster_universe = cast(list[str], joined["participant_id"].to_list())
    metric_specifications: list[tuple[str, str, str, str]] = [
        (
            "not_applicable",
            metric,
            metric,
            _COMET_LABEL if metric == "cometkiwi_score" else "automated feature",
        )
        for metric in _ANALYSIS_INDEPENDENT_METRICS
    ]
    for direction in _DIRECTIONS:
        metric_specifications.extend(
            (
                direction,
                metric,
                f"{metric}_{direction}",
                "embedding-based automated measure",
            )
            for metric in ("paired_cosine", *_RETRIEVAL_METRICS)
        )
    estimates, lowers, uppers = _bootstrap_matrix(
        np.column_stack(
            [
                np.asarray(matched[column].to_list(), dtype=np.float64)
                for _, _, column, _ in metric_specifications
            ]
        ),
        cast(list[str], matched["participant_id"].to_list()),
        cluster_universe,
        repetitions=repetitions,
        confidence_level=confidence_level,
        seed=_seed(seed, "both_languages_match"),
        bootstrap_batch_size=bootstrap_batch_size,
    )
    summary_rows = [
        {
            "stratum_type": "sensitivity",
            "stratum_value": "both_languages_match",
            "direction": direction,
            "metric": metric,
            "n_pairs": matched.height,
            "n_participants": matched["participant_id"].n_unique(),
            "estimate": float(estimates[index]),
            "ci_lower": float(lowers[index]),
            "ci_upper": float(uppers[index]),
            "confidence_level": confidence_level,
            "bootstrap_repetitions": repetitions,
            "evidence_role": evidence_role,
            "subset_definition": "Lingua expected-label subset; not verified language ground truth",
        }
        for index, (direction, metric, _, evidence_role) in enumerate(metric_specifications)
    ]
    summary = pl.DataFrame(summary_rows).sort(["direction", "metric"])

    coverage_rows: list[dict[str, object]] = []
    strata = [("overall", "all", joined)]
    for value in ("1-2 words", "3-5 words", "6-20 words", "21+ words"):
        strata.append(("pt_length_band", value, joined.filter(pl.col("length_band") == value)))
    for stratum_type, stratum_value, subset in strata:
        participants = cast(list[str], subset["participant_id"].to_list())
        metrics: Mapping[str, pl.Series] = {
            "pt_matches_expected": subset["pt_matches_expected"],
            "en_matches_expected": subset["en_matches_expected"],
            "pt_ambiguous": subset["pt_ambiguous"],
            "en_ambiguous": subset["en_ambiguous"],
            "pt_explicit_mismatch": ~subset["pt_matches_expected"]
            & ~subset["pt_ambiguous"],
            "en_explicit_mismatch": ~subset["en_matches_expected"]
            & ~subset["en_ambiguous"],
            "both_languages_match": subset["both_languages_match"],
            "diagnostic_ambiguous_only": subset["diagnostic_ambiguous_only"],
            "diagnostic_explicit_mismatch": subset["diagnostic_explicit_mismatch"],
        }
        metric_items = list(metrics.items())
        estimates, lowers, uppers = _bootstrap_matrix(
            np.column_stack(
                [np.asarray(series.to_list(), dtype=np.float64) for _, series in metric_items]
            ),
            participants,
            cluster_universe,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_seed(seed, "language_coverage", stratum_type, stratum_value),
            bootstrap_batch_size=bootstrap_batch_size,
        )
        for index, (metric, _) in enumerate(metric_items):
            coverage_rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": stratum_value,
                    "metric": metric,
                    "n_pairs": subset.height,
                    "n_participants": subset["participant_id"].n_unique(),
                    "estimate": float(estimates[index]),
                    "ci_lower": float(lowers[index]),
                    "ci_upper": float(uppers[index]),
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": repetitions,
                    "evidence_role": "automated language-identification coverage",
                }
            )
    return summary, pl.DataFrame(coverage_rows).sort(
        ["stratum_type", "stratum_value", "metric"]
    )


def length_stratified_r1(
    analysis_frame: pl.DataFrame,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> pl.DataFrame:
    """Estimate length-stratified bidirectional R@1 with the full cluster universe."""
    enriched = analysis_frame.with_columns(_length_band_expression())
    cluster_universe = cast(list[str], enriched["participant_id"].to_list())
    rows: list[dict[str, object]] = []
    for length_band in ("1-2 words", "3-5 words", "6-20 words", "21+ words"):
        subset = enriched.filter(pl.col("length_band") == length_band)
        participants = cast(list[str], subset["participant_id"].to_list())
        estimates, lowers, uppers = _bootstrap_matrix(
            np.column_stack(
                [
                    np.asarray(subset[f"hit_at_1_{direction}"].to_list(), dtype=np.float64)
                    for direction in _DIRECTIONS
                ]
            ),
            participants,
            cluster_universe,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_seed(seed, "figure1_length_r1", length_band),
            bootstrap_batch_size=bootstrap_batch_size,
        )
        for index, direction in enumerate(_DIRECTIONS):
            estimate = float(estimates[index])
            lower = float(lowers[index])
            upper = float(uppers[index])
            rows.append(
                {
                    "length_band": length_band,
                    "direction": direction,
                    "n_pairs": subset.height,
                    "n_participants": subset["participant_id"].n_unique(),
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_error_minus": estimate - lower,
                    "ci_error_plus": upper - estimate,
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pl.DataFrame(rows).sort(["direction", "length_band"])


def summarize_retrieval_sensitivity(
    retrieval_frame: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    sensitivity_name: str,
    candidate_counts: Mapping[str, int],
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> pl.DataFrame:
    """Summarize one alternate retrieval construction with clustered intervals."""
    required_retrieval = {"pair_id", "direction", *_RETRIEVAL_METRICS}
    if not required_retrieval.issubset(retrieval_frame.columns):
        raise ValueError("Retrieval sensitivity input lacks required metrics")
    required_metadata = {"pair_id", "participant_id"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError("Retrieval sensitivity metadata lacks participant clusters")
    rows: list[dict[str, object]] = []
    cluster_universe = cast(list[str], metadata["participant_id"].to_list())
    for direction in _DIRECTIONS:
        subset = retrieval_frame.filter(pl.col("direction") == direction).join(
            metadata.select(required_metadata), on="pair_id", how="inner", validate="1:1"
        )
        if subset.height != metadata.height:
            raise ValueError("Retrieval sensitivity does not cover every analysis pair")
        participants = cast(list[str], subset["participant_id"].to_list())
        estimates, lowers, uppers = _bootstrap_matrix(
            np.column_stack(
                [np.asarray(subset[metric].to_list(), dtype=np.float64) for metric in _RETRIEVAL_METRICS]
            ),
            participants,
            cluster_universe,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_seed(seed, sensitivity_name, direction),
            bootstrap_batch_size=bootstrap_batch_size,
        )
        for index, metric in enumerate(_RETRIEVAL_METRICS):
            rows.append(
                {
                    "sensitivity": sensitivity_name,
                    "direction": direction,
                    "metric": metric,
                    "n_pairs": subset.height,
                    "n_participants": subset["participant_id"].n_unique(),
                    "n_target_candidates": int(candidate_counts[direction]),
                    "estimate": float(estimates[index]),
                    "ci_lower": float(lowers[index]),
                    "ci_upper": float(uppers[index]),
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": repetitions,
                    "evidence_role": "embedding-based automated measure",
                }
            )
    return pl.DataFrame(rows).sort(["direction", "metric"])


def paired_encoder_differences(
    primary_frame: pl.DataFrame,
    comparison_frame: pl.DataFrame,
    *,
    primary_model: str,
    comparison_model: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> pl.DataFrame:
    """Bootstrap paired per-response retrieval differences between two encoders."""
    metric_columns = [
        f"{metric}_{direction}" for direction in _DIRECTIONS for metric in _RETRIEVAL_METRICS
    ]
    required = {"pair_id", "participant_id", *metric_columns}
    if not required.issubset(primary_frame.columns) or not required.issubset(
        comparison_frame.columns
    ):
        raise ValueError("Encoder sensitivity input lacks paired retrieval metrics")
    comparison_selected = comparison_frame.select(
        "pair_id", *(pl.col(column).alias(f"{column}_comparison") for column in metric_columns)
    )
    joined = primary_frame.select(required).join(
        comparison_selected, on="pair_id", how="inner", validate="1:1"
    )
    if joined.height != primary_frame.height or joined.height != comparison_frame.height:
        raise ValueError("Encoder sensitivity inputs do not contain the same response pairs")
    participants = cast(list[str], joined["participant_id"].to_list())
    metric_specifications: list[tuple[str, str, np.ndarray]] = []
    for direction in _DIRECTIONS:
        for metric in _RETRIEVAL_METRICS:
            column = f"{metric}_{direction}"
            differences = (
                joined[f"{column}_comparison"].cast(pl.Float64)
                - joined[column].cast(pl.Float64)
            )
            metric_specifications.append(
                (direction, metric, np.asarray(differences.to_list(), dtype=np.float64))
            )
    estimates, lowers, uppers = _bootstrap_matrix(
        np.column_stack([values for _, _, values in metric_specifications]),
        participants,
        participants,
        repetitions=repetitions,
        confidence_level=confidence_level,
        seed=_seed(seed, "paired_encoder_difference"),
        bootstrap_batch_size=bootstrap_batch_size,
    )
    rows: list[dict[str, object]] = []
    for index, (direction, metric, _) in enumerate(metric_specifications):
        rows.append(
            {
                "primary_model": primary_model,
                "comparison_model": comparison_model,
                "direction": direction,
                "metric": metric,
                "n_pairs": joined.height,
                "n_participants": joined["participant_id"].n_unique(),
                "estimate_delta_comparison_minus_primary": float(estimates[index]),
                "ci_lower": float(lowers[index]),
                "ci_upper": float(uppers[index]),
                "confidence_level": confidence_level,
                "bootstrap_repetitions": repetitions,
                "paired_by": "pair_id with survey-specific respondent-record cluster draws",
            }
        )
    return pl.DataFrame(rows).sort(["direction", "metric"])


def affective_sensitivities(
    metadata: pl.DataFrame,
    emotion_frame: pl.DataFrame,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
) -> AffectiveSensitivityOutput:
    """Estimate profile distances when both sides are nonzero and coverage by length."""
    required_metadata = {"pair_id", "participant_id", "pt_word_count"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError("Affective sensitivity metadata lacks required columns")
    profile_specifications = {
        "emotion": {
            "labels": EMOTIONS,
            "metrics": (
                "emotion_jensen_shannon_distance",
                "emotion_mean_absolute_share_delta",
            ),
        },
        "valence": {
            "labels": VALENCES,
            "metrics": (
                "valence_jensen_shannon_distance",
                "valence_mean_absolute_share_delta",
            ),
        },
    }
    required_emotion = {"pair_id", "pt_semantic_node_count", "en_semantic_node_count"}
    for family, specification in profile_specifications.items():
        labels = cast(Sequence[str], specification["labels"])
        required_emotion.update(f"pt_{family}_{label}_share" for label in labels)
        required_emotion.update(f"en_{family}_{label}_share" for label in labels)
        required_emotion.update(cast(Sequence[str], specification["metrics"]))
        required_emotion.update(
            {f"pt_{family}_lexicon_coverage", f"en_{family}_lexicon_coverage"}
        )
    if not required_emotion.issubset(emotion_frame.columns):
        raise ValueError("Affective sensitivity input lacks profile or coverage features")
    joined = metadata.select(required_metadata).join(
        emotion_frame.select(required_emotion), on="pair_id", how="inner", validate="1:1"
    )
    if joined.height != metadata.height:
        raise ValueError("Affective sensitivity input does not cover every analysis pair")
    joined = joined.with_columns(
        _length_band_expression(),
        (pl.col("pt_semantic_node_count") == 0).alias("pt_semantic_node_count_zero"),
        (pl.col("en_semantic_node_count") == 0).alias("en_semantic_node_count_zero"),
    )
    cluster_universe = cast(list[str], joined["participant_id"].to_list())
    for family, specification in profile_specifications.items():
        labels = cast(Sequence[str], specification["labels"])
        joined = joined.with_columns(
            (pl.sum_horizontal([pl.col(f"pt_{family}_{label}_share") for label in labels]) > 0)
            .alias(f"pt_{family}_profile_nonzero"),
            (pl.sum_horizontal([pl.col(f"en_{family}_{label}_share") for label in labels]) > 0)
            .alias(f"en_{family}_profile_nonzero"),
        ).with_columns(
            (
                pl.col(f"pt_{family}_profile_nonzero")
                & pl.col(f"en_{family}_profile_nonzero")
            ).alias(f"both_{family}_profiles_nonzero"),
            (
                ~pl.col(f"pt_{family}_profile_nonzero")
                & ~pl.col(f"en_{family}_profile_nonzero")
            ).alias(f"both_{family}_profiles_zero"),
            (
                pl.col(f"pt_{family}_profile_nonzero")
                ^ pl.col(f"en_{family}_profile_nonzero")
            ).alias(f"exactly_one_{family}_profile_zero"),
        )

    nonzero_rows: list[dict[str, object]] = []
    for family, specification in profile_specifications.items():
        subset = joined.filter(pl.col(f"both_{family}_profiles_nonzero"))
        participants = cast(list[str], subset["participant_id"].to_list())
        profile_metrics = cast(Sequence[str], specification["metrics"])
        estimates, lowers, uppers = _bootstrap_matrix(
            np.column_stack(
                [
                    np.asarray(subset[metric].to_list(), dtype=np.float64)
                    for metric in profile_metrics
                ]
            ),
            participants,
            cluster_universe,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_seed(seed, "affective_nonzero", family),
            bootstrap_batch_size=bootstrap_batch_size,
        )
        for index, metric in enumerate(profile_metrics):
            nonzero_rows.append(
                {
                    "profile_family": family,
                    "subset": f"both_{family}_profiles_nonzero",
                    "metric": metric,
                    "n_pairs": subset.height,
                    "n_participants": subset["participant_id"].n_unique(),
                    "estimate": float(estimates[index]),
                    "ci_lower": float(lowers[index]),
                    "ci_upper": float(uppers[index]),
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": repetitions,
                    "evidence_role": "exploratory bilingual lexical-profile comparison",
                }
            )

    coverage_rows: list[dict[str, object]] = []
    strata = [("overall", "all", joined)]
    for value in ("1-2 words", "3-5 words", "6-20 words", "21+ words"):
        strata.append(("pt_length_band", value, joined.filter(pl.col("length_band") == value)))
    for stratum_type, stratum_value, subset in strata:
        participants = cast(list[str], subset["participant_id"].to_list())
        metric_specifications: list[tuple[str, str, pl.Series, str]] = []
        for family in profile_specifications:
            metrics: dict[str, pl.Series] = {
                f"pt_{family}_profile_nonzero": subset[f"pt_{family}_profile_nonzero"],
                f"en_{family}_profile_nonzero": subset[f"en_{family}_profile_nonzero"],
                f"both_{family}_profiles_nonzero": subset[f"both_{family}_profiles_nonzero"],
                f"both_{family}_profiles_zero": subset[f"both_{family}_profiles_zero"],
                f"exactly_one_{family}_profile_zero": subset[
                    f"exactly_one_{family}_profile_zero"
                ],
                f"pt_{family}_lexicon_coverage": subset[f"pt_{family}_lexicon_coverage"],
                f"en_{family}_lexicon_coverage": subset[f"en_{family}_lexicon_coverage"],
                f"pt_{family}_lexicon_coverage_nonzero": (
                    subset[f"pt_{family}_lexicon_coverage"] > 0
                ),
                f"en_{family}_lexicon_coverage_nonzero": (
                    subset[f"en_{family}_lexicon_coverage"] > 0
                ),
                f"pt_{family}_coverage_profile_disagreement": (
                    (subset[f"pt_{family}_lexicon_coverage"] > 0)
                    != subset[f"pt_{family}_profile_nonzero"]
                ),
                f"en_{family}_coverage_profile_disagreement": (
                    (subset[f"en_{family}_lexicon_coverage"] > 0)
                    != subset[f"en_{family}_profile_nonzero"]
                ),
            }
            metric_specifications.extend(
                (family, metric, series, "lexicon and profile coverage diagnostic")
                for metric, series in metrics.items()
            )
        for metric in ("pt_semantic_node_count_zero", "en_semantic_node_count_zero"):
            metric_specifications.append(
                (
                    "semantic_network",
                    metric,
                    subset[metric],
                    "semantic-network coverage diagnostic",
                )
            )
        estimates, lowers, uppers = _bootstrap_matrix(
            np.column_stack(
                [
                    np.asarray(series.to_list(), dtype=np.float64)
                    for _, _, series, _ in metric_specifications
                ]
            ),
            participants,
            cluster_universe,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_seed(seed, "affective_coverage", stratum_type, stratum_value),
            bootstrap_batch_size=bootstrap_batch_size,
        )
        for index, (family, metric, _, evidence_role) in enumerate(metric_specifications):
            coverage_rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": stratum_value,
                    "profile_family": family,
                    "metric": metric,
                    "n_pairs": subset.height,
                    "n_participants": subset["participant_id"].n_unique(),
                    "estimate": float(estimates[index]),
                    "ci_lower": float(lowers[index]),
                    "ci_upper": float(uppers[index]),
                    "confidence_level": confidence_level,
                    "bootstrap_repetitions": repetitions,
                    "evidence_role": evidence_role,
                }
            )
    return AffectiveSensitivityOutput(
        nonzero_metrics=pl.DataFrame(nonzero_rows).sort(["profile_family", "metric"]),
        coverage_by_length=pl.DataFrame(coverage_rows).sort(
            ["stratum_type", "stratum_value", "profile_family", "metric"]
        ),
    )
