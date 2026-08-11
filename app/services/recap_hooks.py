"""Visual hook planning helpers for recap hook experiments."""

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Mapping

from app.models.schema import RecapHookStrategy


_OBSERVATION_FIELDS = (
    "timestamp",
    "evidence",
    "action",
    "expression",
    "shot_type",
    "readability",
    "suspense_score",
    "conflict_score",
    "emotion_score",
)
_SCORE_FIELDS = {
    RecapHookStrategy.suspense: "suspense_score",
    RecapHookStrategy.conflict: "conflict_score",
    RecapHookStrategy.emotion: "emotion_score",
}
_PLATFORM_METRIC_PLACEHOLDERS = {
    "impressions": None,
    "views": None,
    "likes": None,
    "comments": None,
    "shares": None,
    "watch_time_seconds": None,
    "completion_rate": None,
}


@dataclass(frozen=True)
class VisualObservation:
    timestamp: float
    evidence: str
    action: str
    expression: str
    shot_type: str
    readability: int
    suspense_score: int
    conflict_score: int
    emotion_score: int


@dataclass(frozen=True)
class HookCandidate:
    strategy: RecapHookStrategy
    start: float
    end: float
    observation: VisualObservation

    def as_range(self) -> tuple[float, float]:
        return self.start, self.end


class VisualObservationError(ValueError):
    """Raised when visual-analysis output is not a valid observation list."""


def parse_visual_observations(response: str) -> list[VisualObservation]:
    """Parse and validate the visual-analysis JSON array returned by a model."""
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VisualObservationError(
            "visual observations must be valid JSON array text"
        ) from None

    if not isinstance(payload, list):
        raise VisualObservationError("visual observations response must be a JSON array")

    observations = []
    timestamps = set()
    for index, item in enumerate(payload):
        label = f"visual observation at index {index}"
        if not isinstance(item, dict):
            raise VisualObservationError(f"{label} must be an object")

        missing = [field for field in _OBSERVATION_FIELDS if field not in item]
        if missing:
            raise VisualObservationError(
                f"{label} is missing required field(s): {', '.join(missing)}"
            )

        timestamp = item["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, Real)
            or not math.isfinite(float(timestamp))
            or timestamp < 0
        ):
            raise VisualObservationError(
                f"{label} timestamp must be a finite nonnegative number"
            )
        timestamp = float(timestamp)
        if timestamp in timestamps:
            raise VisualObservationError(
                f"{label} timestamp duplicates another visual observation"
            )
        timestamps.add(timestamp)

        text_values = {}
        for field in ("evidence", "action", "expression", "shot_type"):
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise VisualObservationError(
                    f"{label} {field} must be a nonblank string"
                )
            text_values[field] = value.strip()

        score_values = {}
        for field in (
            "readability",
            "suspense_score",
            "conflict_score",
            "emotion_score",
        ):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
                raise VisualObservationError(
                    f"{label} {field} must be an integer from 0 to 5"
                )
            score_values[field] = value

        observations.append(
            VisualObservation(timestamp=timestamp, **text_values, **score_values)
        )

    return sorted(observations, key=lambda observation: observation.timestamp)


def select_hook_candidates(
    observations,
    strategies,
    video_duration,
) -> tuple[dict[RecapHookStrategy, HookCandidate], dict[RecapHookStrategy, str]]:
    """Choose one non-overlapping visual source range for each requested strategy."""
    if (
        isinstance(video_duration, bool)
        or not isinstance(video_duration, Real)
        or not math.isfinite(float(video_duration))
        or video_duration <= 0
    ):
        raise ValueError("video_duration must be a positive finite number")
    duration = float(video_duration)

    requested_strategies = [_coerce_strategy(strategy) for strategy in strategies]
    if len(set(requested_strategies)) != len(requested_strategies):
        raise ValueError("strategies must not contain duplicates")

    selected = {}
    unavailable = {}
    assigned_ranges = []
    for strategy in requested_strategies:
        score_field = _SCORE_FIELDS[strategy]
        ranked = sorted(
            (
                observation
                for observation in observations
                if getattr(observation, score_field) > 0
            ),
            key=lambda observation: (
                -getattr(observation, score_field),
                -observation.readability,
                observation.timestamp,
            ),
        )
        if not ranked:
            unavailable[strategy] = (
                f"No observation has a positive {strategy.value} score."
            )
            continue

        overlaps_assigned_range = False
        has_in_bounds_window = False
        for observation in ranked:
            start = max(0.0, observation.timestamp - 1.5)
            end = min(duration, observation.timestamp + 1.5)
            if end <= start:
                continue
            has_in_bounds_window = True
            if any(_ranges_overlap((start, end), assigned) for assigned in assigned_ranges):
                overlaps_assigned_range = True
                continue

            candidate = HookCandidate(strategy, start, end, observation)
            selected[strategy] = candidate
            assigned_ranges.append(candidate.as_range())
            break
        else:
            if overlaps_assigned_range:
                unavailable[strategy] = (
                    f"All positive {strategy.value} candidates overlap a window "
                    "assigned to an earlier strategy."
                )
            elif not has_in_bounds_window:
                unavailable[strategy] = (
                    f"No positive {strategy.value} candidate falls within the video duration."
                )
            else:
                unavailable[strategy] = (
                    f"No usable positive {strategy.value} candidate is available."
                )

    return selected, unavailable


def build_hook_prompt(
    strategy: RecapHookStrategy,
    candidate: HookCandidate,
    transcript_context: str,
) -> str:
    """Build a source-grounded instruction for a strategy-specific opening."""
    strategy = _coerce_strategy(strategy)
    strategy_guidance = {
        RecapHookStrategy.suspense: (
            "Use the unresolved question in the evidence to create suspense."
        ),
        RecapHookStrategy.conflict: (
            "Use the direct clash in the evidence to create conflict."
        ),
        RecapHookStrategy.emotion: (
            "Use the emotional turning point in the evidence to create emotion."
        ),
    }[strategy]

    return (
        "Write a short recap opening for the selected visual moment.\n\n"
        f"Strategy: {strategy.value}\n"
        f"Strategy guidance: {strategy_guidance}\n"
        f"Source visual evidence [{candidate.start:.1f}-{candidate.end:.1f}s]: "
        f"{candidate.observation.evidence}\n"
        f"Nearby transcript: {transcript_context}\n\n"
        "The generated opening can only describe source-supported content from "
        "the visual evidence and nearby transcript. It cannot invent plot, "
        "violence, betrayal, or the final outcome. The hook must be paid off "
        "or explained within 3-8 seconds."
    )


def build_experiment_manifest(
    task_id,
    source_path,
    shared_body,
    analysis,
    variants,
) -> dict:
    """Create a JSON-serializable record of planned recap hook variants."""
    if not isinstance(shared_body, str):
        raise ValueError("shared_body must be a string")

    return {
        "version": 1,
        "task": {"id": _json_value(task_id)},
        "source": {"path": _json_value(source_path)},
        "shared_body_sha256": hashlib.sha256(
            shared_body.encode("utf-8")
        ).hexdigest(),
        "analysis": _json_value(analysis),
        "variants": _manifest_variants(variants),
        "platform_results": {},
    }


def _coerce_strategy(strategy) -> RecapHookStrategy:
    if isinstance(strategy, RecapHookStrategy):
        return strategy
    try:
        return RecapHookStrategy(strategy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown recap hook strategy: {strategy!r}") from None


def _ranges_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _manifest_variants(variants) -> dict[str, dict[str, Any]]:
    candidate_map = {}
    unavailable_map = {}
    variant_map = variants
    if isinstance(variants, tuple) and len(variants) == 2:
        candidate_map, unavailable_map = variants
        variant_map = {}

    if not isinstance(variant_map, Mapping):
        raise ValueError("variants must be a mapping or a candidate/reason tuple")
    if not isinstance(candidate_map, Mapping) or not isinstance(unavailable_map, Mapping):
        raise ValueError("candidate and unavailable strategy data must be mappings")

    strategies = []
    for raw_strategy in (
        list(variant_map) + list(candidate_map) + list(unavailable_map)
    ):
        strategy = _coerce_strategy(raw_strategy)
        if strategy not in strategies:
            strategies.append(strategy)

    result = {}
    for strategy in strategies:
        raw_variant = _mapping_value(variant_map, strategy)
        entry = _json_value(raw_variant) if isinstance(raw_variant, Mapping) else {}
        if not isinstance(entry, dict):
            entry = {"opening": _json_value(raw_variant)}

        candidate = entry.pop("candidate", _mapping_value(candidate_map, strategy))
        if isinstance(raw_variant, HookCandidate):
            candidate = raw_variant
        reason = entry.pop(
            "unavailable_reason", _mapping_value(unavailable_map, strategy)
        )
        entry["candidate"] = _json_value(candidate) if candidate is not None else None
        entry["unavailable_reason"] = _json_value(reason) if reason is not None else None
        entry["platform_metrics"] = dict(_PLATFORM_METRIC_PLACEHOLDERS)
        result[strategy.value] = entry

    return result


def _mapping_value(mapping: Mapping, strategy: RecapHookStrategy):
    if strategy in mapping:
        return mapping[strategy]
    return mapping.get(strategy.value)


def _json_value(value):
    if isinstance(value, HookCandidate):
        return {
            "strategy": value.strategy.value,
            "start": value.start,
            "end": value.end,
            "observation": _json_value(value.observation),
        }
    if isinstance(value, VisualObservation):
        return {
            "timestamp": value.timestamp,
            "evidence": value.evidence,
            "action": value.action,
            "expression": value.expression,
            "shot_type": value.shot_type,
            "readability": value.readability,
            "suspense_score": value.suspense_score,
            "conflict_score": value.conflict_score,
            "emotion_score": value.emotion_score,
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
