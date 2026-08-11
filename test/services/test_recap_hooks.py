import hashlib
import json
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

# Add project root to Python path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import RecapHookStrategy, VideoParams
from app.services.recap_hooks import (
    HookCandidate,
    VisualObservation,
    VisualObservationError,
    build_experiment_manifest,
    build_hook_prompt,
    parse_visual_observations,
    select_hook_candidates,
)


def _observation_payload(timestamp=5.0, **overrides):
    payload = {
        "timestamp": timestamp,
        "evidence": "A hand reaches toward a locked door.",
        "action": "reaches for the door",
        "expression": "worried",
        "shot_type": "close-up",
        "readability": 4,
        "suspense_score": 3,
        "conflict_score": 2,
        "emotion_score": 1,
    }
    payload.update(overrides)
    return payload


def _observation(timestamp=5.0, **overrides):
    return VisualObservation(**_observation_payload(timestamp, **overrides))


class TestRecapHookParams(unittest.TestCase):
    def test_defaults_to_all_recap_hook_strategies_in_enum_order(self):
        params = VideoParams(video_subject="A dramatic reunion")

        self.assertFalse(params.recap_hook_experiment_enabled)
        self.assertEqual(
            params.recap_hook_strategies,
            [
                RecapHookStrategy.suspense,
                RecapHookStrategy.conflict,
                RecapHookStrategy.emotion,
            ],
        )

    def test_rejects_empty_recap_hook_strategies(self):
        with self.assertRaisesRegex(
            ValidationError, "at least one recap hook strategy"
        ):
            VideoParams(
                video_subject="A dramatic reunion", recap_hook_strategies=[]
            )

    def test_rejects_duplicate_recap_hook_strategies(self):
        with self.assertRaisesRegex(
            ValidationError, "must not contain duplicate strategies"
        ):
            VideoParams(
                video_subject="A dramatic reunion",
                recap_hook_strategies=["suspense", "suspense"],
            )

    def test_accepts_a_selected_recap_hook_strategy_subset(self):
        params = VideoParams(
            video_subject="A dramatic reunion",
            recap_hook_experiment_enabled=True,
            recap_hook_strategies=["emotion", "conflict"],
        )

        self.assertTrue(params.recap_hook_experiment_enabled)
        self.assertEqual(
            params.recap_hook_strategies,
            [RecapHookStrategy.emotion, RecapHookStrategy.conflict],
        )


class TestVisualObservationParsing(unittest.TestCase):
    def test_parses_required_fields_and_sorts_by_timestamp(self):
        response = json.dumps(
            [
                _observation_payload(8.0, evidence="A character looks away."),
                _observation_payload(2.0, evidence="A letter is opened."),
            ]
        )

        observations = parse_visual_observations(response)

        self.assertEqual([item.timestamp for item in observations], [2.0, 8.0])
        self.assertEqual(observations[0].evidence, "A letter is opened.")
        self.assertEqual(observations[0].readability, 4)

    def test_rejects_malformed_or_invalid_observations_without_echoing_response(self):
        invalid_responses = [
            json.dumps(_observation_payload()),
            json.dumps([_observation_payload(evidence="   ")]),
            json.dumps([_observation_payload(timestamp=-0.1)]),
            json.dumps([_observation_payload(readability=True)]),
            json.dumps([_observation_payload(suspense_score=6)]),
            json.dumps([_observation_payload(), _observation_payload()]),
            json.dumps(
                [
                    _observation_payload(
                        evidence="response-secret-token", action="   "
                    )
                ]
            ),
        ]

        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(VisualObservationError) as context:
                    parse_visual_observations(response)
                self.assertNotIn("response-secret-token", str(context.exception))


class TestHookCandidateSelection(unittest.TestCase):
    def test_ranks_by_score_then_readability_then_timestamp(self):
        observations = [
            _observation(8.0, suspense_score=4, readability=5),
            _observation(7.0, suspense_score=5, readability=3),
            _observation(6.0, suspense_score=5, readability=5),
            _observation(4.0, suspense_score=5, readability=5),
        ]

        candidates, unavailable = select_hook_candidates(
            observations, [RecapHookStrategy.suspense], video_duration=20.0
        )

        candidate = candidates[RecapHookStrategy.suspense]
        self.assertEqual(candidate.observation.timestamp, 4.0)
        self.assertEqual(candidate.as_range(), (2.5, 5.5))
        self.assertEqual(unavailable, {})

    def test_prevents_overlap_and_explains_unavailable_strategies(self):
        observations = [
            _observation(
                5.0,
                suspense_score=5,
                conflict_score=0,
                emotion_score=0,
            ),
            _observation(
                6.0,
                suspense_score=0,
                conflict_score=5,
                emotion_score=0,
            ),
            _observation(
                11.0,
                suspense_score=0,
                conflict_score=0,
                emotion_score=0,
            ),
        ]

        candidates, unavailable = select_hook_candidates(
            observations,
            [
                RecapHookStrategy.suspense,
                RecapHookStrategy.conflict,
                RecapHookStrategy.emotion,
            ],
            video_duration=20.0,
        )

        self.assertEqual(set(candidates), {RecapHookStrategy.suspense})
        self.assertIn("overlap", unavailable[RecapHookStrategy.conflict].lower())
        self.assertIn("positive", unavailable[RecapHookStrategy.emotion].lower())

    def test_rejects_non_positive_or_non_finite_video_duration(self):
        observations = [_observation()]

        for duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    select_hook_candidates(
                        observations,
                        [RecapHookStrategy.suspense],
                        video_duration=duration,
                    )


class TestHookPromptAndManifest(unittest.TestCase):
    def test_prompt_grounds_each_strategy_in_evidence_and_transcript(self):
        observation = _observation(evidence="A hand reaches toward a locked door.")
        transcript_context = "[3.5-6.5s] The lock clicks, but nobody enters."
        strategy_words = {
            RecapHookStrategy.suspense: "unresolved question",
            RecapHookStrategy.conflict: "direct clash",
            RecapHookStrategy.emotion: "emotional turning point",
        }

        for strategy, expected_strategy_word in strategy_words.items():
            with self.subTest(strategy=strategy):
                candidate = HookCandidate(strategy, 3.5, 6.5, observation)
                prompt = build_hook_prompt(strategy, candidate, transcript_context)
                normalized = prompt.lower()

                self.assertIn(strategy.value, prompt)
                self.assertIn(expected_strategy_word, normalized)
                self.assertIn(observation.evidence, prompt)
                self.assertIn(transcript_context, prompt)
                self.assertIn("source-supported content", normalized)
                self.assertIn("cannot invent", normalized)
                self.assertIn("violence", normalized)
                self.assertIn("betrayal", normalized)
                self.assertIn("final outcome", normalized)
                self.assertIn("3-8 seconds", prompt)

    def test_manifest_hashes_shared_body_and_uses_none_metric_placeholders(self):
        observation = _observation()
        candidate = HookCandidate(
            RecapHookStrategy.suspense, 3.5, 6.5, observation
        )
        shared_body = "The story begins after the door opens."
        manifest = build_experiment_manifest(
            task_id="task-42",
            source_path="storage/source.mp4",
            shared_body=shared_body,
            analysis={"observations": [observation], "model": "vision-test"},
            variants={
                RecapHookStrategy.suspense: {
                    "candidate": candidate,
                    "opening": "Who is behind the door?",
                },
                RecapHookStrategy.conflict: {
                    "unavailable_reason": "No positive conflict score.",
                },
            },
        )

        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["task"]["id"], "task-42")
        self.assertEqual(manifest["source"]["path"], "storage/source.mp4")
        self.assertEqual(
            manifest["shared_body_sha256"],
            hashlib.sha256(shared_body.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn(shared_body, json.dumps(manifest))
        self.assertEqual(manifest["analysis"]["observations"][0]["timestamp"], 5.0)
        self.assertEqual(
            manifest["variants"]["suspense"]["candidate"]["start"], 3.5
        )
        self.assertIsNone(manifest["variants"]["conflict"]["candidate"])
        self.assertEqual(
            manifest["variants"]["conflict"]["unavailable_reason"],
            "No positive conflict score.",
        )
        self.assertIsNone(manifest["variants"]["suspense"]["platform_metrics"]["views"])
        self.assertEqual(manifest["platform_results"], {})
        json.dumps(manifest)
