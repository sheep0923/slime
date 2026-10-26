from slime.utils.weight_staleness import (
    collect_sample_weight_staleness,
    parse_weight_version,
    sample_weight_staleness,
)


def test_parse_weight_version_accepts_common_string_forms():
    assert parse_weight_version("1") == 1
    assert parse_weight_version("v000007") == 7
    assert parse_weight_version("engine-weight-12") == 12
    assert parse_weight_version("unknown") is None


def test_sample_staleness_uses_oldest_partial_rollout_version():
    assert sample_weight_staleness(["v3", "v5"], 8) == 5
    assert sample_weight_staleness([], 8) is None
    assert sample_weight_staleness(["unknown"], 8) is None


def test_collect_stats_counts_unknown_metadata_without_calling_it_fresh():
    stats = collect_sample_weight_staleness(
        [["1"], ["v3", "v4"], []],
        [0, 1, 2, 9],
        "5",
    )
    assert stats == (6.0, 2, 4, 2, 2)
