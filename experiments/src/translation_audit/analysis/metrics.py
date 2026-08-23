"""Deterministic retrieval, bootstrap, sensitivity, and transfer metrics."""

import hashlib
import importlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import polars as pl

_DIRECTIONS = ("en_to_pt", "pt_to_en")
_EMOTION_METRICS = (
    "emotion_jensen_shannon_distance",
    "valence_jensen_shannon_distance",
    "emotion_mean_absolute_share_delta",
    "valence_mean_absolute_share_delta",
    "semantic_density_absolute_delta",
)
_RIDGE_FEATURES = (
    "paired_cosine_en_to_pt",
    "paired_cosine_pt_to_en",
    "emotion_jensen_shannon_distance",
    "valence_jensen_shannon_distance",
    "emotion_mean_absolute_share_delta",
    "valence_mean_absolute_share_delta",
    "semantic_density_absolute_delta",
    "log1p_pt_word_count",
)
_COMET_LABEL = "COMETKiwi automated reference-free criterion; not human gold"


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    """Per-pair retrieval metrics and the matrix backend used."""

    frame: pl.DataFrame
    device: str


@dataclass(frozen=True, slots=True)
class ValidationOutput:
    """Leave-one-cohort-out metrics and predictions."""

    metrics: pl.DataFrame
    predictions: pl.DataFrame


def _normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Embedding matrix must be non-empty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Embedding matrix contains a zero vector")
    return values / norms


def iter_similarity_batches(
    queries: np.ndarray,
    documents: np.ndarray,
    batch_size: int,
    requested_device: str,
) -> tuple[Iterator[np.ndarray], str]:
    """Compute bounded query batches on CUDA or MPS, otherwise NumPy CPU."""
    if batch_size <= 0:
        raise ValueError("Retrieval batch size must be positive")
    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Analysis compute device must be auto, cpu, cuda, or mps")
    torch_module: Any | None = None
    torch_device: str | None = None
    if requested_device != "cpu":
        try:
            candidate: Any = importlib.import_module("torch")
            if requested_device in {"auto", "cuda"} and bool(candidate.cuda.is_available()):
                torch_module = candidate
                torch_device = "cuda"
            elif requested_device in {"auto", "mps"} and bool(
                candidate.backends.mps.is_available()
            ):
                torch_module = candidate
                torch_device = "mps"
        except (ImportError, AttributeError):
            torch_module = None
            torch_device = None
    if requested_device in {"cuda", "mps"} and torch_module is None:
        raise RuntimeError(f"{requested_device.upper()} retrieval was requested but is unavailable")

    if torch_module is not None and torch_device is not None:
        document_tensor: Any = torch_module.from_numpy(documents).to(torch_device)

        def accelerator_batches() -> Iterator[np.ndarray]:
            for start in range(0, queries.shape[0], batch_size):
                query_tensor: Any = torch_module.from_numpy(
                    queries[start : start + batch_size]
                ).to(torch_device)
                similarities: Any = query_tensor @ document_tensor.T
                yield cast(np.ndarray, similarities.float().cpu().numpy())

        return accelerator_batches(), torch_device

    document_transpose = documents.T

    def cpu_batches() -> Iterator[np.ndarray]:
        for start in range(0, queries.shape[0], batch_size):
            yield queries[start : start + batch_size] @ document_transpose

    return cpu_batches(), "cpu"


def _positive_indices(group_ids: Sequence[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    return [np.asarray(groups[group_id], dtype=np.int64) for group_id in group_ids]


def compute_retrieval_metrics(
    embeddings: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    batch_size: int,
    requested_device: str,
    deduplicate_targets: bool = False,
) -> RetrievalOutput:
    """Compute paired cosine and duplicate-aware full-corpus ranks in bounded batches."""
    required_embeddings = {
        "pair_id",
        "source_row",
        "direction",
        "query_embedding",
        "document_embedding",
    }
    if set(embeddings.columns) != required_embeddings:
        raise ValueError("Embedding artifacts have an invalid retrieval schema")
    required_metadata = {"pair_id", "pt_duplicate_group_id", "en_duplicate_group_id"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError("Curated metadata lacks duplicate groups")
    metadata_sorted = metadata.sort("source_row") if "source_row" in metadata.columns else metadata
    expected_ids = cast(list[str], metadata_sorted["pair_id"].to_list())
    outputs: list[pl.DataFrame] = []
    devices: set[str] = set()

    for direction in _DIRECTIONS:
        direction_frame = embeddings.filter(pl.col("direction") == direction).sort("source_row")
        if direction_frame.height != metadata_sorted.height:
            raise ValueError("Embedding direction does not cover every response pair")
        pair_ids = cast(list[str], direction_frame["pair_id"].to_list())
        if pair_ids != expected_ids:
            raise ValueError("Embedding pair order differs from curated metadata")
        queries = _normalized(np.asarray(direction_frame["query_embedding"].to_list(), dtype=np.float32))
        documents = _normalized(
            np.asarray(direction_frame["document_embedding"].to_list(), dtype=np.float32)
        )
        if queries.shape != documents.shape:
            raise ValueError("Query and document embedding shapes differ")
        group_column = "pt_duplicate_group_id" if direction == "en_to_pt" else "en_duplicate_group_id"
        group_ids = cast(list[str], metadata_sorted[group_column].to_list())
        if deduplicate_targets:
            group_values = np.asarray(group_ids, dtype=object)
            group_reduction_order = np.argsort(group_values, kind="stable")
            ordered_group_values = group_values[group_reduction_order]
            group_starts = np.flatnonzero(
                np.r_[True, ordered_group_values[1:] != ordered_group_values[:-1]]
            )
            ordered_groups = ordered_group_values[group_starts]
            rank_index_by_group = {
                str(group_id): index for index, group_id in enumerate(ordered_groups)
            }
            positives = [
                np.asarray([rank_index_by_group[group_id]], dtype=np.int64)
                for group_id in group_ids
            ]
            candidate_count = len(rank_index_by_group)
        else:
            group_reduction_order = None
            group_starts = None
            positives = _positive_indices(group_ids)
            candidate_count = documents.shape[0]
        paired_cosine = np.sum(queries * documents, axis=1, dtype=np.float64)
        ranks = np.empty(queries.shape[0], dtype=np.int64)
        similarity_batches, device = iter_similarity_batches(
            queries, documents, batch_size, requested_device
        )
        devices.add(device)
        query_offset = 0
        document_indices = np.arange(candidate_count, dtype=np.int64)
        for similarity_batch in similarity_batches:
            ranking_batch = (
                np.maximum.reduceat(
                    similarity_batch[:, group_reduction_order], group_starts, axis=1
                )
                if group_reduction_order is not None and group_starts is not None
                else similarity_batch
            )
            for local_index, row_scores in enumerate(ranking_batch):
                query_index = query_offset + local_index
                positive = positives[query_index]
                positive_scores = row_scores[positive]
                best_score = float(np.max(positive_scores))
                best_positive = int(np.min(positive[positive_scores == best_score]))
                strictly_better = int(np.count_nonzero(row_scores > best_score))
                tied_before = int(
                    np.count_nonzero((row_scores == best_score) & (document_indices < best_positive))
                )
                ranks[query_index] = strictly_better + tied_before + 1
            query_offset += similarity_batch.shape[0]
        outputs.append(
            pl.DataFrame(
                {
                    "pair_id": pair_ids,
                    "direction": [direction] * len(pair_ids),
                    "paired_cosine": paired_cosine,
                    "retrieval_rank": ranks,
                    "reciprocal_rank": 1.0 / ranks,
                    "hit_at_1": ranks <= 1,
                    "hit_at_5": ranks <= 5,
                }
            )
        )
    if len(devices) != 1:
        raise RuntimeError("Retrieval directions used inconsistent compute devices")
    return RetrievalOutput(frame=pl.concat(outputs, how="vertical"), device=next(iter(devices)))


def _derived_seed(seed: int, *labels: str) -> int:
    digest = hashlib.sha256("\u0000".join(labels).encode()).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def clustered_bootstrap_mean(
    values: np.ndarray,
    participants: Sequence[str],
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    batch_size: int,
) -> tuple[float, float, float]:
    """Estimate a row-weighted mean CI by resampling complete participant clusters."""
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) != len(participants) or numeric.size == 0:
        raise ValueError("Cluster bootstrap inputs are empty or misaligned")
    if not np.isfinite(numeric).all():
        raise ValueError("Cluster bootstrap values must be finite")
    if repetitions <= 0 or batch_size <= 0:
        raise ValueError("Bootstrap repetitions and batch size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence level must be between zero and one")
    participant_values = np.asarray(list(participants), dtype=object)
    unique_participants, inverse = np.unique(participant_values, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=numeric)
    cluster_counts = np.bincount(inverse)
    estimate = float(numeric.mean())
    if unique_participants.size == 1:
        return estimate, estimate, estimate

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        sampled = rng.integers(
            0,
            unique_participants.size,
            size=(stop - start, unique_participants.size),
        )
        bootstrap[start:stop] = cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    return estimate, float(lower), float(upper)


def clustered_bootstrap_conditional_mean(
    values: np.ndarray,
    participants: Sequence[str],
    cluster_universe: Sequence[str],
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    batch_size: int,
) -> tuple[float, float, float]:
    """Bootstrap a filtered row mean over the full respondent-record cluster universe."""
    estimates, lower, upper = clustered_bootstrap_conditional_means(
        np.asarray(values, dtype=np.float64)[:, np.newaxis],
        participants,
        cluster_universe,
        repetitions=repetitions,
        confidence_level=confidence_level,
        seed=seed,
        batch_size=batch_size,
    )
    return float(estimates[0]), float(lower[0]), float(upper[0])


def clustered_bootstrap_conditional_means(
    values: np.ndarray,
    participants: Sequence[str],
    cluster_universe: Sequence[str],
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap several filtered row means with shared full-universe cluster draws."""
    numeric = np.asarray(values, dtype=np.float64)
    if (
        numeric.ndim != 2
        or numeric.shape[0] != len(participants)
        or numeric.shape[0] == 0
        or numeric.shape[1] == 0
    ):
        raise ValueError("Conditional cluster bootstrap inputs are empty or misaligned")
    if not np.isfinite(numeric).all():
        raise ValueError("Conditional cluster bootstrap values must be finite")
    if repetitions <= 0 or batch_size <= 0:
        raise ValueError("Bootstrap repetitions and batch size must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence level must be between zero and one")

    universe = np.unique(np.asarray(list(cluster_universe), dtype=object))
    eligible_participants = np.asarray(list(participants), dtype=object)
    if universe.size == 0:
        raise ValueError("Eligible clusters are not contained in the bootstrap universe")
    eligible_indices = np.searchsorted(universe, eligible_participants)
    if np.any(eligible_indices >= universe.size) or not np.array_equal(
        universe[eligible_indices], eligible_participants
    ):
        raise ValueError("Eligible clusters are not contained in the bootstrap universe")
    cluster_sums = np.zeros((universe.size, numeric.shape[1]), dtype=np.float64)
    np.add.at(cluster_sums, eligible_indices, numeric)
    cluster_counts = np.bincount(eligible_indices, minlength=universe.size)
    estimates = numeric.mean(axis=0)
    if universe.size == 1:
        return estimates, estimates.copy(), estimates.copy()

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((repetitions, numeric.shape[1]), dtype=np.float64)
    completed = 0
    attempts = 0
    while completed < repetitions:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Conditional bootstrap could not draw an eligible response")
        draw_count = min(batch_size, repetitions - completed)
        sampled = rng.integers(0, universe.size, size=(draw_count, universe.size))
        offsets = np.arange(draw_count, dtype=np.int64)[:, np.newaxis] * universe.size
        weights = np.bincount(
            (sampled + offsets).ravel(), minlength=draw_count * universe.size
        ).reshape(draw_count, universe.size)
        denominators = weights @ cluster_counts
        valid = denominators > 0
        if not valid.any():
            continue
        numerators = weights @ cluster_sums
        batch_estimates = numerators[valid] / denominators[valid, np.newaxis]
        accepted = min(batch_estimates.shape[0], repetitions - completed)
        bootstrap[completed : completed + accepted] = batch_estimates[:accepted]
        completed += accepted
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail], axis=0)
    return estimates, lower, upper


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


def _strata(frame: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    enriched = frame.with_columns(_length_band_expression())
    strata: list[tuple[str, str, pl.DataFrame]] = [("overall", "all", enriched)]
    for column, stratum_type in (
        ("cohort_id", "cohort"),
        ("prompt_family", "prompt_family"),
        ("length_band", "pt_length_band"),
    ):
        for value in sorted(cast(list[str], enriched[column].unique().to_list())):
            strata.append((stratum_type, value, enriched.filter(pl.col(column) == value)))
    strata.extend(
        [
            ("sensitivity", "exclude_2_words_or_fewer", enriched.filter(pl.col("pt_word_count") > 2)),
            ("sensitivity", "exclude_5_words_or_fewer", enriched.filter(pl.col("pt_word_count") > 5)),
            (
                "sensitivity",
                "exclude_exact_normalized_copy",
                enriched.filter(~pl.col("exact_normalized_copy")),
            ),
            (
                "sensitivity",
                "one_per_repeated_pt_response",
                enriched.sort("pair_id").unique("pt_duplicate_group_id", keep="first"),
            ),
        ]
    )
    return strata


def summarize_strata(
    frame: pl.DataFrame,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
    bootstrap_batch_size: int,
    cluster_universe: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Create descriptive primary and sensitivity estimates with clustered CIs."""
    required = {
        "pair_id",
        "participant_id",
        "cohort_id",
        "prompt_family",
        "pt_word_count",
        "exact_normalized_copy",
        "pt_duplicate_group_id",
        "cometkiwi_score",
        *_EMOTION_METRICS,
    }
    if not required.issubset(frame.columns):
        raise ValueError("Analysis frame lacks required summary columns")
    rows: list[dict[str, object]] = []
    all_clusters = (
        cast(list[str], frame["participant_id"].to_list())
        if cluster_universe is None
        else list(cluster_universe)
    )
    for stratum_type, stratum_value, subset in _strata(frame):
        if subset.is_empty():
            continue
        participants = cast(list[str], subset["participant_id"].to_list())
        metric_specifications: list[tuple[str, str, str, str]] = [
            (
                "not_applicable",
                metric,
                metric,
                _COMET_LABEL if metric == "cometkiwi_score" else "automated feature",
            )
            for metric in ("cometkiwi_score", *_EMOTION_METRICS)
        ]
        for direction in _DIRECTIONS:
            for metric in ("paired_cosine", "hit_at_1", "hit_at_5", "reciprocal_rank"):
                metric_specifications.append(
                    (
                        direction,
                        metric,
                        f"{metric}_{direction}",
                        "embedding-based automated measure",
                    )
                )
        matrix = np.column_stack(
            [
                np.asarray(subset[column].to_list(), dtype=np.float64)
                for _, _, column, _ in metric_specifications
            ]
        )
        estimates, lowers, uppers = clustered_bootstrap_conditional_means(
            matrix,
            participants,
            all_clusters,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=_derived_seed(seed, stratum_type, stratum_value),
            batch_size=bootstrap_batch_size,
        )
        for index, (direction, metric, _, evidence_role) in enumerate(metric_specifications):
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_value": stratum_value,
                    "direction": direction,
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
    return pl.DataFrame(rows).sort(["stratum_type", "stratum_value", "direction", "metric"])


def _ridge_features(frame: pl.DataFrame) -> np.ndarray:
    enriched = frame.with_columns(pl.col("pt_word_count").log1p().alias("log1p_pt_word_count"))
    if not set(_RIDGE_FEATURES).issubset(enriched.columns):
        raise ValueError("Analysis frame lacks ridge predictor columns")
    matrix = np.asarray(enriched.select(_RIDGE_FEATURES).to_numpy(), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Ridge predictor matrix contains non-finite values")
    return matrix


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Any:
    linear_model: Any = importlib.import_module("sklearn.linear_model")
    pipeline: Any = importlib.import_module("sklearn.pipeline")
    preprocessing: Any = importlib.import_module("sklearn.preprocessing")
    model: Any = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model


def _select_alpha(
    frame: pl.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    alphas: Sequence[float],
    default_alpha: float,
) -> float:
    cohort_values = np.asarray(frame["cohort_id"].to_list(), dtype=object)
    training_cohorts = sorted(set(cast(list[str], cohort_values[train_mask].tolist())))
    if len(training_cohorts) < 2:
        return default_alpha
    losses: list[tuple[float, float]] = []
    for alpha in alphas:
        fold_losses: list[float] = []
        for validation_cohort in training_cohorts:
            inner_validation = train_mask & (cohort_values == validation_cohort)
            inner_training = train_mask & (cohort_values != validation_cohort)
            if not inner_validation.any() or not inner_training.any():
                continue
            prediction_value: Any = _fit_ridge(
                x[inner_training], y[inner_training], alpha
            ).predict(x[inner_validation])
            prediction = np.asarray(prediction_value, dtype=np.float64)
            fold_losses.append(float(np.mean((y[inner_validation] - prediction) ** 2)))
        if fold_losses:
            losses.append((float(np.mean(fold_losses)), alpha))
    return min(losses, key=lambda item: (item[0], item[1]))[1] if losses else default_alpha


def automated_convergent_validation(
    frame: pl.DataFrame,
    splits: pl.DataFrame,
    *,
    alphas: Sequence[float],
    default_alpha: float,
) -> ValidationOutput:
    """Predict the automated COMET criterion with leave-one-cohort-out ridge models."""
    if any(alpha <= 0.0 for alpha in alphas) or default_alpha <= 0.0:
        raise ValueError("Ridge alphas must be positive")
    if set(splits.columns) != {"participant_id", "fold_id", "role"}:
        raise ValueError("Curated split table has an invalid schema")
    x = _ridge_features(frame)
    y = np.asarray(frame["cometkiwi_score"].to_list(), dtype=np.float64)
    participants = np.asarray(frame["participant_id"].to_list(), dtype=object)
    pair_ids = cast(list[str], frame["pair_id"].to_list())
    cohorts = np.asarray(frame["cohort_id"].to_list(), dtype=object)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold_id in sorted(cast(list[str], splits["fold_id"].unique().to_list())):
        fold = splits.filter(pl.col("fold_id") == fold_id)
        train_participants = set(
            cast(list[str], fold.filter(pl.col("role") == "train")["participant_id"].to_list())
        )
        test_participants = set(
            cast(list[str], fold.filter(pl.col("role") == "test")["participant_id"].to_list())
        )
        if (
            not train_participants
            or not test_participants
            or train_participants.intersection(test_participants)
        ):
            raise ValueError("Curated split fold has invalid train/test membership")
        train_mask = np.fromiter((value in train_participants for value in participants), dtype=bool)
        test_mask = np.fromiter((value in test_participants for value in participants), dtype=bool)
        held_out = sorted(set(cast(list[str], cohorts[test_mask].tolist())))
        if len(held_out) != 1 or np.any(cohorts[train_mask] == held_out[0]):
            raise ValueError("Curated split fold is not leave-one-cohort-out")
        alpha = _select_alpha(frame, x, y, train_mask, alphas, default_alpha)
        model = _fit_ridge(x[train_mask], y[train_mask], alpha)
        prediction_value: Any = model.predict(x[test_mask])
        predicted = np.asarray(prediction_value, dtype=np.float64)
        observed = y[test_mask]
        residual = observed - predicted
        rmse = float(math.sqrt(float(np.mean(residual**2))))
        mae = float(np.mean(np.abs(residual)))
        target_variance = float(np.sum((observed - observed.mean()) ** 2))
        r2 = None if target_variance == 0.0 else float(1.0 - np.sum(residual**2) / target_variance)
        pearson = (
            None
            if observed.size < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0
            else float(np.corrcoef(observed, predicted)[0, 1])
        )
        metric_rows.append(
            {
                "fold_id": fold_id,
                "held_out_cohort": held_out[0],
                "n_train_pairs": int(train_mask.sum()),
                "n_test_pairs": int(test_mask.sum()),
                "selected_alpha": alpha,
                "mae": mae,
                "rmse": rmse,
                "r_squared": r2,
                "pearson_r": pearson,
                "criterion": _COMET_LABEL,
                "interpretation": "automated convergent validation; descriptive, not human validation",
            }
        )
        test_indices = np.flatnonzero(test_mask)
        prediction_rows.extend(
            {
                "pair_id": pair_ids[index],
                "fold_id": fold_id,
                "held_out_cohort": held_out[0],
                "cometkiwi_score": float(observed[position]),
                "predicted_cometkiwi_score": float(predicted[position]),
                "residual": float(residual[position]),
            }
            for position, index in enumerate(test_indices)
        )
    predictions = pl.DataFrame(prediction_rows)
    if predictions.height != frame.height or predictions["pair_id"].n_unique() != frame.height:
        raise ValueError("Leave-one-cohort-out predictions do not cover each pair exactly once")
    return ValidationOutput(
        metrics=pl.DataFrame(metric_rows).sort("held_out_cohort"),
        predictions=predictions.sort("pair_id"),
    )
