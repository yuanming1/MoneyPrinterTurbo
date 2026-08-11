import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

# Add project root to Python path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import RecapHookStrategy, VideoParams


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
