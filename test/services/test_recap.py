import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import RecapHookStrategy, VideoParams
from app.services import recap, vision
from app.services.recap_hooks import HookCandidate, VisualObservation


def _observation_payload(timestamp, **overrides):
    payload = {
        "timestamp": timestamp,
        "evidence": "A character freezes at the doorway.",
        "action": "freezes",
        "expression": "afraid",
        "shot_type": "close-up",
        "readability": 4,
        "suspense_score": 4,
        "conflict_score": 0,
        "emotion_score": 0,
    }
    payload.update(overrides)
    return payload


def _candidate(strategy, start, end, timestamp):
    return HookCandidate(
        strategy=strategy,
        start=start,
        end=end,
        observation=VisualObservation(**_observation_payload(timestamp)),
    )


class TestFrameExtraction(unittest.TestCase):
    def test_extracts_regular_and_deduplicated_scene_frames_with_timestamps(self):
        self.assertTrue(
            hasattr(recap, "extract_analysis_frames"),
            "extract_analysis_frames is required for hook frame analysis",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "frames"

            def write_ffmpeg_outputs(command, **kwargs):
                command_text = " ".join(str(part) for part in command)
                if "select=" in command_text:
                    (output_dir / "scene_000001.jpg").parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    (output_dir / "scene_000001.jpg").write_bytes(b"scene-one")
                    (output_dir / "scene_000002.jpg").write_bytes(b"scene-two")
                    return SimpleNamespace(
                        stderr=(
                            b"[Parsed_showinfo_0] n:0 pts:1250 pts_time:1.25\n"
                            b"[Parsed_showinfo_0] n:1 pts:5000 pts_time:5.00\n"
                        )
                    )

                (output_dir / "regular_000001.jpg").parent.mkdir(
                    parents=True, exist_ok=True
                )
                (output_dir / "regular_000001.jpg").write_bytes(b"regular-zero")
                (output_dir / "regular_000002.jpg").write_bytes(b"regular-two")
                return SimpleNamespace(
                    stderr=(
                        b"[Parsed_showinfo_0] n:0 pts:250 pts_time:0.25\n"
                        b"[Parsed_showinfo_0] n:1 pts:2250 pts_time:2.25\n"
                    )
                )

            with patch.object(recap.video_service, "get_ffmpeg_binary", return_value="ffmpeg"):
                with patch.object(
                    recap.subprocess, "run", side_effect=write_ffmpeg_outputs
                ):
                    frames = recap.extract_analysis_frames("source.mp4", output_dir)

        self.assertEqual([frame.timestamp for frame in frames], [0.25, 1.25, 2.25, 5.0])
        self.assertEqual([frame.path.name for frame in frames], [
            "regular_000001.jpg",
            "scene_000001.jpg",
            "regular_000002.jpg",
            "scene_000002.jpg",
        ])

    def test_scene_timestamp_path_mismatch_raises_a_clear_error(self):
        self.assertTrue(hasattr(recap, "extract_analysis_frames"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "frames"

            def mismatched_scene_outputs(command, **kwargs):
                command_text = " ".join(str(part) for part in command)
                if "select=" in command_text:
                    (output_dir / "scene_000001.jpg").parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    (output_dir / "scene_000001.jpg").write_bytes(b"scene-one")
                    return SimpleNamespace(
                        stderr=(
                            b"pts_time:1.25\npts_time:5.00\n"
                        )
                    )
                (output_dir / "regular_000001.jpg").parent.mkdir(
                    parents=True, exist_ok=True
                )
                (output_dir / "regular_000001.jpg").write_bytes(b"regular")
                return SimpleNamespace(stderr=b"pts_time:0.00\n")

            with patch.object(recap.video_service, "get_ffmpeg_binary", return_value="ffmpeg"):
                with patch.object(
                    recap.subprocess, "run", side_effect=mismatched_scene_outputs
                ):
                    with self.assertRaisesRegex(ValueError, "scene-change frame metadata"):
                        recap.extract_analysis_frames("source.mp4", output_dir)


class TestHookExperimentAnalysis(unittest.TestCase):
    def test_validates_vision_before_attempting_audio_or_transcription(self):
        self.assertTrue(hasattr(recap, "analyze_hook_experiment"))
        params = VideoParams(video_subject="A recap")
        vision_error = vision.VisionConfigurationError("recap_vision_api_key missing")

        with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
            with patch.object(
                recap.vision, "load_recap_vision_config", side_effect=vision_error
            ):
                with patch.object(recap, "analyze_source_video") as analyze_source:
                    with self.assertRaisesRegex(vision.VisionConfigurationError, "api_key"):
                        recap.analyze_hook_experiment("task-1", params)

        analyze_source.assert_not_called()

    def test_analyzes_frame_batches_and_persists_source_grounded_results(self):
        self.assertTrue(hasattr(recap, "analyze_hook_experiment"))
        self.assertTrue(hasattr(recap, "FrameFile"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_dir = Path(temporary_directory) / "task-1"
            task_dir.mkdir()
            frame_dir = task_dir / "recap-analysis" / "frames"
            frame_dir.mkdir(parents=True)
            frame_type = recap.FrameFile
            frames = []
            for index in range(9):
                path = frame_dir / f"frame_{index}.jpg"
                path.write_bytes(f"frame-{index}".encode("ascii"))
                frames.append(frame_type(timestamp=float(index), path=path))

            transcript = [{"start": 0.0, "end": 2.0, "text": "A door opens."}]
            first_batch = json.dumps([_observation_payload(float(index)) for index in range(8)])
            second_batch = json.dumps([_observation_payload(8.0)])
            params = VideoParams(
                video_subject="A recap",
                recap_hook_strategies=[RecapHookStrategy.suspense],
            )
            vision_config = vision.VisionConfig("gemini", "test-key", "", "model")

            with patch.object(recap.utils, "task_dir", return_value=str(task_dir)):
                with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
                    with patch.object(
                        recap.vision, "load_recap_vision_config", return_value=vision_config
                    ):
                        with patch.object(recap, "_load_timeline", return_value=transcript):
                            with patch.object(
                                recap, "extract_analysis_frames", return_value=frames
                            ):
                                with patch.object(recap, "_get_video_duration", return_value=30.0):
                                    with patch.object(
                                        recap.vision,
                                        "analyze_frames",
                                        side_effect=[first_batch, second_batch],
                                    ) as analyze_frames:
                                        analysis = recap.analyze_hook_experiment("task-1", params)

            self.assertEqual(analysis.transcript, transcript)
            self.assertEqual([item.timestamp for item in analysis.observations], list(range(9)))
            self.assertEqual(
                analysis.candidates[RecapHookStrategy.suspense].observation.timestamp,
                0.0,
            )
            self.assertEqual(analysis.unavailable, {})
            self.assertEqual(
                analysis.observations_path,
                task_dir / "recap-analysis" / "visual-observations.json",
            )
            self.assertEqual(analyze_frames.call_count, 2)
            self.assertEqual(
                [len(call.args[2]) for call in analyze_frames.call_args_list], [8, 1]
            )
            self.assertEqual(
                analyze_frames.call_args_list[0].args[2][0],
                vision.FrameInput(0.0, b"frame-0", "image/jpeg"),
            )
            self.assertEqual(
                json.loads((task_dir / "recap-analysis" / "transcript-timeline.json").read_text("utf-8")),
                transcript,
            )
            self.assertEqual(
                json.loads(analysis.observations_path.read_text("utf-8"))[0]["timestamp"],
                0.0,
            )

    def test_rejects_empty_visual_observations(self):
        self.assertTrue(hasattr(recap, "analyze_hook_experiment"))
        params = VideoParams(video_subject="A recap")
        frame_type = getattr(recap, "FrameFile", None)
        self.assertIsNotNone(frame_type, "FrameFile is required for frame analysis")
        with tempfile.TemporaryDirectory() as temporary_directory:
            frame_path = Path(temporary_directory) / "frame.jpg"
            frame_path.write_bytes(b"frame")
            with patch.object(recap.utils, "task_dir", return_value=temporary_directory):
                with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
                    with patch.object(recap.vision, "load_recap_vision_config", return_value=MagicMock()):
                        with patch.object(recap, "_load_timeline", return_value=[{"start": 0, "end": 1, "text": "Hi"}]):
                            with patch.object(
                                recap,
                                "extract_analysis_frames",
                                return_value=[frame_type(0.0, frame_path)],
                            ):
                                with patch.object(recap.vision, "analyze_frames", return_value="[]"):
                                    with self.assertRaisesRegex(ValueError, "视觉观察"):
                                        recap.analyze_hook_experiment("task-1", params)

    def test_rejects_a_batch_missing_a_submitted_frame_timestamp(self):
        frame_type = recap.FrameFile
        with tempfile.TemporaryDirectory() as temporary_directory:
            frame_paths = []
            for index in range(2):
                path = Path(temporary_directory) / f"frame-{index}.jpg"
                path.write_bytes(b"frame")
                frame_paths.append(path)
            frames = [
                frame_type(timestamp=float(index), path=path)
                for index, path in enumerate(frame_paths)
            ]

            with patch.object(
                recap.vision,
                "analyze_frames",
                return_value=json.dumps([_observation_payload(0.0)]),
            ):
                with self.assertRaisesRegex(ValueError, "恰好覆盖"):
                    recap._analyze_visual_frame_batches(MagicMock(), frames)


class TestHookVariantMaterials(unittest.TestCase):
    def test_matches_shared_body_once_with_all_hook_ranges_excluded(self):
        self.assertTrue(hasattr(recap, "prepare_hook_variant_materials"))
        suspense = _candidate(RecapHookStrategy.suspense, 1.0, 4.0, 2.5)
        conflict = _candidate(RecapHookStrategy.conflict, 8.0, 11.0, 9.5)
        analysis = SimpleNamespace(
            candidates={
                RecapHookStrategy.suspense: suspense,
                RecapHookStrategy.conflict: conflict,
            },
            unavailable={RecapHookStrategy.emotion: "No positive emotion score."},
        )
        params = VideoParams(video_subject="A recap")

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(recap.utils, "task_dir", return_value=temporary_directory):
                with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
                    with patch.object(recap.video_service, "get_ffmpeg_binary", return_value="ffmpeg"):
                        with patch.object(recap, "_load_timeline", return_value=[{"start": 0, "end": 1, "text": "Hi"}]):
                            with patch.object(recap, "_get_video_duration", return_value=30.0):
                                with patch.object(
                                    recap, "_match_script_to_video", return_value=[(12.0, 15.0)]
                                ) as match_script:
                                    with patch.object(
                                        recap,
                                        "_cut_clips_by_ranges",
                                        side_effect=lambda source, output, binary, ranges: [
                                            str(Path(output) / "clip_0000.mp4")
                                        ],
                                    ) as cut_clips:
                                        variants = recap.prepare_hook_variant_materials(
                                            "task-1", params, analysis, "Shared body.", 12.0
                                        )

        match_script.assert_called_once_with(
            "Shared body.",
            [{"start": 0, "end": 1, "text": "Hi"}],
            params.video_clip_duration,
            30.0,
            excluded_ranges=[(1.0, 4.0), (8.0, 11.0)],
        )
        self.assertEqual(set(variants), {RecapHookStrategy.suspense, RecapHookStrategy.conflict})
        self.assertEqual(len(variants[RecapHookStrategy.suspense]), 2)
        self.assertTrue(any("shared-body" in str(call.args[1]) for call in cut_clips.call_args_list))
        self.assertTrue(any("suspense" in str(call.args[1]) for call in cut_clips.call_args_list))
        self.assertTrue(any("conflict" in str(call.args[1]) for call in cut_clips.call_args_list))

    def test_falls_back_after_the_latest_reserved_hook_when_body_matching_fails(self):
        self.assertTrue(hasattr(recap, "prepare_hook_variant_materials"))
        candidate = _candidate(RecapHookStrategy.suspense, 4.0, 7.0, 5.5)
        analysis = SimpleNamespace(candidates={RecapHookStrategy.suspense: candidate}, unavailable={})
        params = VideoParams(video_subject="A recap")

        with tempfile.TemporaryDirectory() as temporary_directory:
            body_ranges = []

            def cut_clips(source, output, binary, ranges):
                if "shared-body" in str(output):
                    body_ranges.extend(ranges)
                    return ["body.mp4"]
                return ["hook.mp4"]

            with patch.object(recap.utils, "task_dir", return_value=temporary_directory):
                with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
                    with patch.object(recap.video_service, "get_ffmpeg_binary", return_value="ffmpeg"):
                        with patch.object(recap, "_load_timeline", return_value=[]):
                            with patch.object(recap, "_get_video_duration", return_value=30.0):
                                with patch.object(recap, "_match_script_to_video", return_value=None) as match_script:
                                    with patch.object(
                                        recap, "_split_video_chronologically", return_value=["unexpected-body.mp4"]
                                    ) as split_video:
                                        with patch.object(
                                            recap, "_cut_clips_by_ranges", side_effect=cut_clips
                                        ):
                                            variants = recap.prepare_hook_variant_materials(
                                                "task-1", params, analysis, "Shared body.", 10.0
                                            )

        match_script.assert_called_once()
        split_video.assert_not_called()
        self.assertEqual(variants[RecapHookStrategy.suspense], ["hook.mp4", "body.mp4"])
        self.assertTrue(body_ranges)
        self.assertEqual(body_ranges[0][0], 7.0)

    def test_fallback_uses_earlier_unreserved_body_when_latest_hook_reaches_video_end(self):
        candidate = _candidate(RecapHookStrategy.suspense, 27.0, 30.0, 28.5)
        analysis = SimpleNamespace(candidates={RecapHookStrategy.suspense: candidate}, unavailable={})
        params = VideoParams(video_subject="A recap")

        with tempfile.TemporaryDirectory() as temporary_directory:
            body_ranges = []

            def cut_clips(source, output, binary, ranges):
                if "shared-body" in str(output):
                    body_ranges.extend(ranges)
                    return ["body.mp4"]
                return ["hook.mp4"]

            with patch.object(recap.utils, "task_dir", return_value=temporary_directory):
                with patch.object(recap, "_get_first_source_path", return_value="source.mp4"):
                    with patch.object(recap.video_service, "get_ffmpeg_binary", return_value="ffmpeg"):
                        with patch.object(recap, "_load_timeline", return_value=[]):
                            with patch.object(recap, "_get_video_duration", return_value=30.0):
                                with patch.object(recap, "_match_script_to_video", return_value=None):
                                    with patch.object(
                                        recap, "_split_video_chronologically", return_value=[]
                                    ) as split_video:
                                        with patch.object(
                                            recap, "_cut_clips_by_ranges", side_effect=cut_clips
                                        ):
                                            variants = recap.prepare_hook_variant_materials(
                                                "task-1", params, analysis, "Shared body.", 10.0
                                            )

        split_video.assert_not_called()
        self.assertEqual(variants[RecapHookStrategy.suspense], ["hook.mp4", "body.mp4"])
        self.assertTrue(body_ranges)
        self.assertTrue(all(end <= 27.0 for _, end in body_ranges))


class TestExcludedRecapMatches(unittest.TestCase):
    def test_discards_model_ranges_overlapping_reserved_hook_ranges(self):
        response = json.dumps(
            [
                {"start": 1.0, "end": 4.0},
                {"start": 4.0, "end": 7.0},
                {"start": 9.0, "end": 12.0},
            ]
        )

        with patch("app.services.llm._generate_response", return_value=response):
            ranges = recap._match_script_to_video(
                "A recap sentence.",
                [{"start": 0.0, "end": 12.0, "text": "Source dialogue."}],
                3.0,
                20.0,
                excluded_ranges=[(4.0, 8.0)],
            )

        self.assertEqual(ranges, [(1.0, 4.0), (9.0, 12.0)])


if __name__ == "__main__":
    unittest.main()
