import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import vision


class TestVisionConfiguration(unittest.TestCase):
    def test_load_config_requires_model(self):
        with self.assertRaisesRegex(vision.VisionConfigurationError, "recap_vision_model"):
            vision.load_recap_vision_config(
                {
                    "recap_vision_provider": "openai_compatible",
                    "recap_vision_api_key": "test-key",
                    "recap_vision_base_url": "https://example.com/v1",
                }
            )

    def test_load_config_rejects_unsupported_provider(self):
        with self.assertRaisesRegex(
            vision.VisionConfigurationError, "recap_vision_provider"
        ):
            vision.load_recap_vision_config(
                {
                    "recap_vision_provider": "unsupported",
                    "recap_vision_api_key": "test-key",
                    "recap_vision_model": "vision-model",
                }
            )


class TestVisionMessages(unittest.TestCase):
    def test_openai_messages_preserve_timestamp_mime_type_and_image_payload(self):
        frames = [
            vision.FrameInput(
                timestamp=4,
                image_bytes=b"first-frame",
                mime_type="image/jpeg",
            ),
            vision.FrameInput(
                timestamp=8.25,
                image_bytes=b"second-frame",
                mime_type="image/png",
            ),
        ]

        messages = vision.build_openai_messages("Describe these frames.", frames)

        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Describe these frames."})
        self.assertEqual(content[1], {"type": "text", "text": "t=4.00s"})
        self.assertEqual(
            content[2],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,Zmlyc3QtZnJhbWU="},
            },
        )
        self.assertEqual(content[3], {"type": "text", "text": "t=8.25s"})
        self.assertEqual(
            content[4],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,c2Vjb25kLWZyYW1l"},
            },
        )


class TestVisionAdapters(unittest.TestCase):
    def setUp(self):
        self.frames = [
            vision.FrameInput(
                timestamp=1.5,
                image_bytes=b"frame-bytes",
                mime_type="image/webp",
            )
        ]

    def test_openai_adapter_dispatches_and_normalizes_content(self):
        config = vision.VisionConfig(
            provider="openai_compatible",
            api_key="test-key",
            base_url="https://example.com/v1",
            model="vision-model",
        )
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="  [{\"scene\": 1}]  \n"))]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch.object(vision, "OpenAI", return_value=client) as openai:
            result = vision.analyze_frames(config, "Analyze.", self.frames)

        self.assertEqual(result, '[{"scene": 1}]')
        openai.assert_called_once_with(
            api_key="test-key", base_url="https://example.com/v1"
        )
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "vision-model")
        self.assertEqual(request["messages"][0]["content"][1]["text"], "t=1.50s")
        self.assertEqual(
            request["messages"][0]["content"][2]["image_url"]["url"],
            "data:image/webp;base64,ZnJhbWUtYnl0ZXM=",
        )

    def test_gemini_adapter_dispatches_with_image_parts(self):
        config = vision.VisionConfig(
            provider="gemini",
            api_key="test-key",
            base_url="https://gemini.example.com",
            model="gemini-vision-model",
        )
        fake_types = types.SimpleNamespace(
            Part=types.SimpleNamespace(from_bytes=MagicMock(return_value="image-part")),
            HttpOptions=MagicMock(return_value="http-options"),
        )
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = types.SimpleNamespace(
            text="  [{\"scene\": 1}]  "
        )
        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = MagicMock(return_value=client)
        fake_genai.types = fake_types
        fake_google = types.ModuleType("google")

        with patch.dict(
            sys.modules, {"google": fake_google, "google.genai": fake_genai}
        ):
            result = vision.analyze_frames(config, "Analyze.", self.frames)

        self.assertEqual(result, '[{"scene": 1}]')
        fake_types.HttpOptions.assert_called_once_with(
            base_url="https://gemini.example.com"
        )
        fake_genai.Client.assert_called_once_with(
            api_key="test-key", http_options="http-options"
        )
        fake_types.Part.from_bytes.assert_called_once_with(
            data=b"frame-bytes", mime_type="image/webp"
        )
        request = client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-vision-model")
        self.assertEqual(request["contents"], ["Analyze.", "t=1.50s", "image-part"])

    def test_empty_frames_bypass_provider_calls(self):
        config = vision.VisionConfig(
            provider="openai_compatible",
            api_key="test-key",
            base_url="https://example.com/v1",
            model="vision-model",
        )

        with patch.object(vision, "OpenAI") as openai:
            result = vision.analyze_frames(config, "Analyze.", [])

        self.assertEqual(result, "[]")
        openai.assert_not_called()

    def test_provider_errors_are_sanitized(self):
        config = vision.VisionConfig(
            provider="openai_compatible",
            api_key="secret-key",
            base_url="https://user:password@example.com/v1",
            model="vision-model",
        )

        with patch.object(
            vision, "OpenAI", side_effect=RuntimeError("secret-key password")
        ):
            with self.assertRaisesRegex(
                vision.VisionResponseError, "OpenAI-compatible vision request failed"
            ) as raised:
                vision.analyze_frames(config, "Analyze.", self.frames)

        self.assertNotIn("secret-key", str(raised.exception))
        self.assertNotIn("password", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
