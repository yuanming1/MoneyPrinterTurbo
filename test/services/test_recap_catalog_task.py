import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import RecapCatalogSelection, VideoParams
from app.services import task


class RecapCatalogTaskTests(unittest.TestCase):
    def test_rejects_unconfirmed_catalog_selection_before_recap_processing(self):
        params = VideoParams(
            video_subject="",
            video_source="recap",
            recap_catalog_selection=RecapCatalogSelection(
                catalog_id="qingque-kuaishou",
                item_id="qingque-kuaishou:001",
            ),
        )

        with patch.object(
            task.recap_catalog,
            "validate_selection",
            side_effect=ValueError("confirm source reuse rights before generating a recap"),
        ):
            error = task.prepare_recap_catalog_source_metadata("task-1", params)

        self.assertEqual(error, "confirm source reuse rights before generating a recap")

    def test_persists_validated_catalog_snapshot_before_recap_processing(self):
        params = VideoParams(
            video_subject="",
            video_source="recap",
            recap_catalog_selection=RecapCatalogSelection(
                catalog_id="qingque-kuaishou",
                item_id="qingque-kuaishou:001",
                rights_confirmed=True,
            ),
        )
        snapshot = {
            "catalog_id": "qingque-kuaishou",
            "item_id": "qingque-kuaishou:001",
            "title": "Alpha",
        }

        with patch.object(task.recap_catalog, "validate_selection", return_value=snapshot):
            with patch.object(task.task_artifacts, "write_recap_source_metadata") as write:
                error = task.prepare_recap_catalog_source_metadata("task-1", params)

        self.assertIsNone(error)
        write.assert_called_once_with("task-1", snapshot)

    def test_manual_recap_upload_does_not_need_a_catalog_selection(self):
        params = VideoParams(video_subject="manual", video_source="recap")

        with patch.object(task.task_artifacts, "write_recap_source_metadata") as write:
            error = task.prepare_recap_catalog_source_metadata("task-1", params)

        self.assertIsNone(error)
        write.assert_not_called()

    def test_pipeline_fails_preflight_before_transcribing_unconfirmed_catalog_source(self):
        params = VideoParams(
            video_subject="",
            video_source="recap",
            recap_catalog_selection=RecapCatalogSelection(
                catalog_id="qingque-kuaishou",
                item_id="qingque-kuaishou:001",
            ),
        )
        with patch.object(task.sm.state, "update_task"):
            with patch.object(
                task,
                "prepare_recap_catalog_source_metadata",
                return_value="confirm source reuse rights before generating a recap",
            ):
                with patch.object(
                    task,
                    "_mark_task_failed",
                    return_value={"state": "failed"},
                ) as mark_failed:
                    with patch("app.services.recap.enrich_recap_context") as enrich:
                        result = task._run_pipeline("task-1", params, stop_at="script")

        self.assertEqual(result, {"state": "failed"})
        mark_failed.assert_called_once_with(
            "task-1",
            "preflight",
            "confirm source reuse rights before generating a recap",
        )
        enrich.assert_not_called()


if __name__ == "__main__":
    unittest.main()
