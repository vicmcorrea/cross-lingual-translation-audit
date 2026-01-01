"""Build the immutable paired-response cohort without exposing source identifiers."""

import argparse
import json
import os
import shutil
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import polars as pl

from translation_audit.config import DATA_ROOT
from translation_audit.runtime.files import sha256_file

SNAPSHOT_ID = "source_snapshot_v1"
CURATED_VERSION = "paired_triads_v1"
NORMALIZATION_VERSION = "nfkc_whitespace_v1"
EXPECTED_RAW_ROWS = 61_596
EXPECTED_PAIRS = 44_118
EXPECTED_PARTICIPANTS = 14_706
EXPECTED_PROMPTS = 12

COHORT_SPECS: dict[str, tuple[int, int]] = {
    "cohort_2023_a": (2023, 4_858),
    "cohort_2024_a": (2024, 4_022),
    "cohort_2025_a": (2025, 5_049),
    "cohort_2025_b": (2025, 777),
}
PROMPT_FAMILIES = {1: "positive_workplace", 2: "improvement", 3: "mental_health_wellbeing"}


def normalize_for_qc(value: str) -> str:
    """Normalize only for deduplication and quality-control comparisons."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def stable_id(namespace: str, *parts: object) -> str:
    """Create a reproducible opaque identifier from local lineage values."""
    material = "\x1f".join((CURATED_VERSION, namespace, *(str(part) for part in parts)))
    return sha256(material.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any], mode: int = 0o640) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(descriptor, _json_bytes(payload))
    finally:
        os.close(descriptor)


def _validated_existing(destination: Path) -> dict[str, Any] | None:
    success = destination / "_SUCCESS"
    manifest_path = destination / "manifest.json"
    if not destination.exists():
        return None
    if not success.is_file() or not manifest_path.is_file():
        raise FileExistsError(f"Incomplete curated directory already exists at {destination}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("curated_version") != CURATED_VERSION:
        raise FileExistsError(f"Conflicting curated version already exists at {destination}")
    for filename, expected in manifest["output_sha256"].items():
        actual = sha256_file(destination / filename)
        if actual != expected:
            raise ValueError(f"Checksum mismatch in existing curated file {filename}")
    return manifest


def _load_private_selection(path: Path) -> dict[str, tuple[str, int, int]]:
    """Load source hashes from a local-only manifest and validate its public metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Private cohort-selection manifest is missing at {path}")
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Private cohort-selection manifest must be a JSON object")
    selection: dict[str, tuple[str, int, int]] = {}
    for source_key, metadata_value in cast(dict[object, object], loaded).items():
        if not isinstance(source_key, str) or len(source_key) != 64 or not isinstance(metadata_value, dict):
            raise ValueError("Malformed private cohort-selection manifest")
        metadata = cast(dict[object, object], metadata_value)
        cohort_value = metadata.get("id")
        year_value = metadata.get("year")
        participant_value = metadata.get("participants")
        if (
            not isinstance(cohort_value, str)
            or not isinstance(year_value, int)
            or not isinstance(participant_value, int)
        ):
            raise ValueError("Malformed private cohort metadata")
        cohort_id = cohort_value
        year = year_value
        participants = participant_value
        if COHORT_SPECS.get(cohort_id) != (year, participants):
            raise ValueError(f"Private metadata conflicts with the public contract for {cohort_id}")
        selection[source_key] = (cohort_id, year, participants)
    if {value[0] for value in selection.values()} != set(COHORT_SPECS):
        raise ValueError("Private cohort-selection manifest does not cover the four public cohorts")
    return selection


def _load_selected(
    raw_snapshot: Path,
    selection: dict[str, tuple[str, int, int]],
) -> tuple[pl.DataFrame, dict[str, int]]:
    files = sorted(raw_snapshot.glob("paired_comments_*.parquet"))
    if len(files) != 3 or not (raw_snapshot / "_SUCCESS").is_file():
        raise FileNotFoundError(f"Expected a complete three-file raw snapshot at {raw_snapshot}")
    cohort_ids = {survey_key: cohort[0] for survey_key, cohort in selection.items()}
    candidates = (
        pl.scan_parquet(files)
        .filter(
            pl.col("survey_key_sha256").is_in(list(selection))
            & pl.col("question_index").is_in([1, 2, 3])
            & (pl.col("reported_source_language") == "pt")
            & (pl.col("target_language") == "en")
            & (pl.col("source_text").str.strip_chars().str.len_chars() > 0)
            & (pl.col("translated_text").str.strip_chars().str.len_chars() > 0)
        )
        .with_columns(
            pl.col("survey_key_sha256").replace_strict(cohort_ids).alias("cohort_id"),
            pl.col("question_index").replace_strict(PROMPT_FAMILIES).alias("prompt_family"),
            pl.col("source_text")
            .map_elements(normalize_for_qc, return_dtype=pl.String)
            .alias("normalized_pt"),
            pl.col("translated_text")
            .map_elements(normalize_for_qc, return_dtype=pl.String)
            .alias("normalized_en"),
        )
        .collect()
    )
    logical_keys = [
        "survey_key_sha256",
        "respondent_key_sha256",
        "question_index",
        "normalized_pt",
        "normalized_en",
    ]
    complete_participants = (
        candidates.unique(subset=logical_keys)
        .group_by(["survey_key_sha256", "respondent_key_sha256"])
        .agg(pl.col("question_index").n_unique().alias("question_count"))
        .filter(pl.col("question_count") == 3)
        .select("survey_key_sha256", "respondent_key_sha256")
    )
    selected = candidates.join(
        complete_participants,
        on=["survey_key_sha256", "respondent_key_sha256"],
        how="inner",
    )
    if selected.height != EXPECTED_RAW_ROWS:
        raise ValueError(f"Expected {EXPECTED_RAW_ROWS} selected raw rows, found {selected.height}")
    selection_qc = {
        "candidate_raw_rows": candidates.height,
        "incomplete_participant_raw_rows_excluded": candidates.height - selected.height,
        "candidate_participants": candidates.select("survey_key_sha256", "respondent_key_sha256")
        .unique()
        .height,
        "incomplete_participants_excluded": candidates.select("survey_key_sha256", "respondent_key_sha256")
        .unique()
        .height
        - complete_participants.height,
    }
    return selected, selection_qc


def _prepare_tables(
    selected: pl.DataFrame,
    selection: dict[str, tuple[str, int, int]],
) -> dict[str, pl.DataFrame]:
    dedup_keys = [
        "cohort_id",
        "respondent_key_sha256",
        "question_index",
        "normalized_pt",
        "normalized_en",
    ]
    duplicate_counts = selected.group_by(dedup_keys).len(name="ingestion_record_count")
    translation_variants = selected.group_by(
        ["cohort_id", "respondent_key_sha256", "question_index", "normalized_pt"]
    ).agg(pl.col("normalized_en").n_unique().alias("translation_variant_count"))
    selected = (
        selected.sort("source_record_id")
        .join(duplicate_counts, on=dedup_keys)
        .join(
            translation_variants,
            on=["cohort_id", "respondent_key_sha256", "question_index", "normalized_pt"],
        )
    )
    logical = selected.unique(subset=dedup_keys, keep="first", maintain_order=True)
    if logical.height != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} logical pairs, found {logical.height}")
    if logical["translation_variant_count"].max() != 1:
        raise ValueError("A source response has competing English translations")

    participant_question_counts = logical.group_by(["cohort_id", "respondent_key_sha256"]).agg(
        pl.col("question_index").n_unique().alias("question_count")
    )
    if participant_question_counts.filter(pl.col("question_count") != 3).height:
        raise ValueError("The selected cohort contains an incomplete response triad")
    if participant_question_counts.height != EXPECTED_PARTICIPANTS:
        raise ValueError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, found {participant_question_counts.height}"
        )

    cohort_counts = {
        row["cohort_id"]: row["len"]
        for row in participant_question_counts.group_by("cohort_id").len().to_dicts()
    }
    for cohort_id, (_, expected) in COHORT_SPECS.items():
        if cohort_counts.get(cohort_id) != expected:
            raise ValueError(f"Unexpected participant count for {cohort_id}")

    cohort_years = {cohort[0]: cohort[1] for cohort in selection.values()}
    logical = logical.with_columns(
        pl.struct(["cohort_id", "respondent_key_sha256"])
        .map_elements(
            lambda row: stable_id("participant", row["cohort_id"], row["respondent_key_sha256"]),
            return_dtype=pl.String,
        )
        .alias("participant_id"),
        pl.struct(dedup_keys)
        .map_elements(
            lambda row: stable_id("pair", *(row[key] for key in dedup_keys)),
            return_dtype=pl.String,
        )
        .alias("pair_id"),
        pl.concat_str([pl.col("cohort_id"), pl.lit("_q"), pl.col("question_index")]).alias("prompt_id"),
        pl.col("cohort_id").replace_strict(cohort_years).cast(pl.Int32).alias("year"),
        pl.col("normalized_pt").str.split(" ").list.len().cast(pl.Int32).alias("pt_word_count"),
        pl.col("normalized_en").str.split(" ").list.len().cast(pl.Int32).alias("en_word_count"),
        (pl.col("normalized_pt") == pl.col("normalized_en")).alias("exact_normalized_copy"),
        pl.col("normalized_pt")
        .map_elements(lambda value: stable_id("pt_duplicate", value), return_dtype=pl.String)
        .alias("pt_duplicate_group_id"),
        pl.col("normalized_en")
        .map_elements(lambda value: stable_id("en_duplicate", value), return_dtype=pl.String)
        .alias("en_duplicate_group_id"),
    )
    pt_counts = logical.group_by("pt_duplicate_group_id").len(name="pt_duplicate_group_size")
    en_counts = logical.group_by("en_duplicate_group_id").len(name="en_duplicate_group_size")
    logical = logical.join(pt_counts, on="pt_duplicate_group_id").join(en_counts, on="en_duplicate_group_id")

    paired = logical.select(
        "pair_id",
        "participant_id",
        "cohort_id",
        "year",
        "question_index",
        "prompt_id",
        "prompt_family",
        pl.col("source_text").alias("source_pt"),
        pl.col("translated_text").alias("translation_en"),
        "pt_word_count",
        "en_word_count",
        "exact_normalized_copy",
        "pt_duplicate_group_id",
        "pt_duplicate_group_size",
        "en_duplicate_group_id",
        "en_duplicate_group_size",
        "ingestion_record_count",
        "translation_variant_count",
    ).sort(["cohort_id", "participant_id", "question_index"])

    participants = (
        logical.select("participant_id", "cohort_id", "year").unique().sort(["cohort_id", "participant_id"])
    )
    prompts = (
        logical.select(
            "prompt_id",
            "cohort_id",
            "year",
            "question_index",
            "prompt_family",
            pl.col("question_text").alias("prompt_pt"),
        )
        .unique()
        .sort(["cohort_id", "question_index"])
    )
    if prompts.height != EXPECTED_PROMPTS:
        raise ValueError(f"Expected {EXPECTED_PROMPTS} prompts, found {prompts.height}")

    folds: list[pl.DataFrame] = []
    for test_cohort in sorted(cohort_counts):
        folds.append(
            participants.select(
                "participant_id",
                pl.lit(f"test_{test_cohort}").alias("fold_id"),
                pl.when(pl.col("cohort_id") == test_cohort)
                .then(pl.lit("test"))
                .otherwise(pl.lit("train"))
                .alias("role"),
            )
        )
    splits = pl.concat(folds).sort(["fold_id", "participant_id"])

    pair_lookup = logical.select(*dedup_keys, "pair_id")
    lineage = (
        selected.join(pair_lookup, on=dedup_keys, how="left")
        .select(
            "pair_id",
            "source_record_id",
            "survey_id",
            "legacy_survey_id",
            "survey_key_sha256",
            "respondent_key_sha256",
            "methodology",
            "sector",
        )
        .sort(["pair_id", "source_record_id"])
    )

    return {
        "paired_responses.parquet": paired,
        "participants.parquet": participants,
        "prompts.parquet": prompts,
        "splits.parquet": splits,
        "lineage.parquet": lineage,
    }


def build_curated_cohort(
    raw_snapshot: Path | None = None,
    destination: Path | None = None,
    sanitized_manifest_path: Path | None = None,
    private_selection_path: Path | None = None,
) -> dict[str, Any]:
    """Build once, validate on reuse, and never replace an existing cohort."""
    raw_snapshot = raw_snapshot or DATA_ROOT / "raw" / "snapshots" / SNAPSHOT_ID
    destination = destination or DATA_ROOT / "curated" / CURATED_VERSION
    sanitized_manifest_path = sanitized_manifest_path or DATA_ROOT / "manifests" / f"{CURATED_VERSION}.json"
    private_selection_path = private_selection_path or DATA_ROOT / "private" / "cohort_selection_v1.json"
    existing = _validated_existing(destination)
    if existing is not None:
        return existing

    staging = destination.parent / f".{CURATED_VERSION}.staging-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False, mode=0o750)
    try:
        selection = _load_private_selection(private_selection_path)
        selected, selection_qc = _load_selected(raw_snapshot, selection)
        tables = _prepare_tables(selected, selection)
        for filename, table in tables.items():
            table.write_parquet(staging / filename, compression="zstd", statistics=True)
        os.chmod(staging / "lineage.parquet", 0o600)

        paired = tables["paired_responses.parquet"]
        qc = {
            "curated_version": CURATED_VERSION,
            "selected_raw_rows": selected.height,
            "logical_pairs": paired.height,
            "participants": tables["participants.parquet"].height,
            "prompts": tables["prompts.parquet"].height,
            "exact_normalized_copy": int(paired["exact_normalized_copy"].sum()),
            "pt_two_words_or_fewer": paired.filter(pl.col("pt_word_count") <= 2).height,
            "pt_five_words_or_fewer": paired.filter(pl.col("pt_word_count") <= 5).height,
            "pt_word_count_median": float(cast(float, paired["pt_word_count"].median())),
            "pt_word_count_max": int(cast(int, paired["pt_word_count"].max())),
            "en_word_count_max": int(cast(int, paired["en_word_count"].max())),
            "duplicate_ingestion_rows_removed": selected.height - paired.height,
            "competing_translation_variants": 0,
            "automatic_language_verification": "pending",
            "exclusion_policy": "No short, exact-copy, or repeated responses were excluded",
            **selection_qc,
        }
        _write_json(staging / "qc_summary.json", qc)

        raw_manifest_path = raw_snapshot / "manifest.json"
        tracked_outputs = [*tables, "qc_summary.json"]
        output_hashes = {name: sha256_file(staging / name) for name in sorted(tracked_outputs)}
        manifest: dict[str, Any] = {
            "curated_version": CURATED_VERSION,
            "created_at_utc": _utc_now(),
            "source_snapshot_id": SNAPSHOT_ID,
            "source_manifest_sha256": sha256_file(raw_manifest_path),
            "normalization_version": NORMALIZATION_VERSION,
            "deduplication_keys": [
                "cohort_id",
                "respondent",
                "question_index",
                "normalized_pt",
                "normalized_en",
            ],
            "deduplication_representative": "smallest source_record_id",
            "cohort_source_keys": sorted(selection),
            "counts": qc,
            "output_sha256": output_hashes,
            "sensitive_local_only": ["lineage.parquet", "manifest.json"],
            "gpu_transfer_allowlist": [
                "paired_responses.parquet",
                "participants.parquet",
                "prompts.parquet",
                "splits.parquet",
                "qc_summary.json",
            ],
            "known_limitation": "Row-level translation model, timestamp, and job provenance are unavailable",
        }
        _write_json(staging / "manifest.json", manifest, mode=0o600)
        (staging / "_SUCCESS").touch(mode=0o440, exist_ok=False)
        os.rename(staging, destination)
        for path in destination.iterdir():
            if path.name == "lineage.parquet" or path.name == "manifest.json":
                os.chmod(path, 0o400)
            else:
                os.chmod(path, 0o440)
        os.chmod(destination, 0o550)

        sanitized = {
            "curated_version": CURATED_VERSION,
            "created_at_utc": manifest["created_at_utc"],
            "source_snapshot_id": SNAPSHOT_ID,
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "normalization_version": NORMALIZATION_VERSION,
            "counts": qc,
            "output_sha256": {key: value for key, value in output_hashes.items() if key != "lineage.parquet"},
            "gpu_transfer_allowlist": manifest["gpu_transfer_allowlist"],
            "known_limitation": manifest["known_limitation"],
        }
        sanitized_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if sanitized_manifest_path.exists():
            prior = json.loads(sanitized_manifest_path.read_text(encoding="utf-8"))
            if prior["output_sha256"] != sanitized["output_sha256"]:
                raise FileExistsError("The tracked sanitized manifest conflicts with the built cohort")
        else:
            _write_json(sanitized_manifest_path, sanitized, mode=0o644)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_transfer_cohort(
    curated_dir: Path,
    sanitized_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the minimal deidentified cohort copied to a compute host."""
    if not curated_dir.is_dir() or not sanitized_manifest_path.is_file():
        raise FileNotFoundError("The transferred cohort or sanitized manifest is missing")
    loaded: object = json.loads(sanitized_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("The sanitized cohort manifest must be a JSON object")
    manifest = cast(dict[str, Any], loaded)
    if manifest.get("curated_version") != CURATED_VERSION:
        raise ValueError("The transferred cohort version does not match the protocol")

    allowlist = manifest.get("gpu_transfer_allowlist")
    output_hashes = manifest.get("output_sha256")
    if not isinstance(allowlist, list) or not isinstance(output_hashes, dict):
        raise ValueError("The sanitized cohort manifest is incomplete")
    allowlist_values = cast(list[object], allowlist)
    output_hash_values = cast(dict[object, object], output_hashes)
    expected_allowlist = {
        "paired_responses.parquet",
        "participants.parquet",
        "prompts.parquet",
        "splits.parquet",
        "qc_summary.json",
    }
    if set(allowlist_values) != expected_allowlist or set(output_hash_values) != expected_allowlist:
        raise ValueError("The transferred cohort allowlist differs from the frozen contract")
    for filename in sorted(expected_allowlist):
        payload = curated_dir / filename
        expected_hash = output_hash_values.get(filename)
        if not payload.is_file() or not isinstance(expected_hash, str):
            raise FileNotFoundError(f"Transferred cohort file is missing: {filename}")
        if sha256_file(payload) != expected_hash:
            raise ValueError(f"Transferred cohort checksum mismatch: {filename}")

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Transferred cohort counts are missing")
    count_values = cast(dict[object, object], counts)
    expected_counts = {
        "logical_pairs": EXPECTED_PAIRS,
        "participants": EXPECTED_PARTICIPANTS,
        "prompts": EXPECTED_PROMPTS,
    }
    for key, expected in expected_counts.items():
        if count_values.get(key) != expected:
            raise ValueError(f"Transferred cohort count mismatch for {key}")
    return manifest


def main() -> None:
    """Build the default cohort while printing counts but never response text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-snapshot", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--private-selection", type=Path)
    args = parser.parse_args()
    manifest = build_curated_cohort(
        args.raw_snapshot, args.destination, private_selection_path=args.private_selection
    )
    counts = manifest["counts"]
    print(
        f"Validated {counts['participants']} participants and "
        f"{counts['logical_pairs']} paired responses in {CURATED_VERSION}."
    )


if __name__ == "__main__":
    main()
