"""Helpers for measuring rollout-sample weight staleness.

Rollout engines report versions as strings (for example ``"1"`` or
``"v000001"``), while the trainer keeps the current version as an integer.
These helpers deliberately keep parsing and sample-level semantics separate
from distributed reduction so they can also be unit tested without Megatron.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


_VERSION_SUFFIX = re.compile(r"(\d+)\s*$")


def parse_weight_version(version: object) -> int | None:
    """Return the numeric part of a rollout weight version, if available."""

    if version is None:
        return None
    match = _VERSION_SUFFIX.search(str(version))
    return int(match.group(1)) if match else None


def sample_weight_staleness(weight_versions: Iterable[object] | None, current_version: object) -> int | None:
    """Compute staleness for one sample.

    A sample can be assembled from multiple generation requests during a
    partial rollout.  We use the *oldest* numeric version in that sample: any
    token produced by that version is potentially off-policy, so this is the
    conservative sample-level staleness.
    """

    current = parse_weight_version(current_version)
    if current is None:
        return None
    if weight_versions is None:
        return None
    versions = [parsed for version in weight_versions if (parsed := parse_weight_version(version)) is not None]
    if not versions:
        return None
    return current - min(versions)


def collect_sample_weight_staleness(
    weight_versions: Sequence[Sequence[object]] | None,
    sample_indices: Iterable[int],
    current_version: object,
) -> tuple[float, int, int | None, int | None, int]:
    """Collect local ``(sum, known, max, min, unknown)`` staleness stats.

    ``sample_indices`` are the local indices selected by the current training
    step's micro-batches.  Missing metadata is counted as unknown rather than
    silently treated as fresh data.
    """

    if weight_versions is None:
        indices = list(sample_indices)
        return 0.0, 0, None, None, len(indices)

    total = 0.0
    known = 0
    unknown = 0
    maximum = None
    minimum = None
    for index in sample_indices:
        versions = weight_versions[index] if 0 <= index < len(weight_versions) else ()
        staleness = sample_weight_staleness(versions, current_version)
        if staleness is None:
            unknown += 1
            continue
        total += staleness
        known += 1
        maximum = staleness if maximum is None else max(maximum, staleness)
        minimum = staleness if minimum is None else min(minimum, staleness)

    return total, known, maximum, minimum, unknown
