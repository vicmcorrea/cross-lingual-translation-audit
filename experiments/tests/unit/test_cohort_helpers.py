from translation_audit.data.cohort import normalize_for_qc, stable_id


def test_normalization_is_nfkc_and_whitespace_only() -> None:
    assert normalize_for_qc("  Saúde\tmental\n") == "Saúde mental"
    assert normalize_for_qc("\uff21") == "A"
    assert normalize_for_qc("Casa") != normalize_for_qc("casa")


def test_stable_ids_are_namespaced_and_reproducible() -> None:
    assert stable_id("pair", "a", 1) == stable_id("pair", "a", 1)
    assert stable_id("pair", "a", 1) != stable_id("participant", "a", 1)
    assert len(stable_id("pair", "a", 1)) == 64
