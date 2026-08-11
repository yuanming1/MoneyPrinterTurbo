"""Provider-neutral visual analysis for recap generation."""

import base64
from dataclasses import dataclass
from typing import Mapping, Sequence

from openai import OpenAI

from app.config import config


_SUPPORTED_PROVIDERS = {"openai_compatible", "gemini"}


@dataclass(frozen=True)
class VisionConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class FrameInput:
    timestamp: float
    image_bytes: bytes
    mime_type: str


class VisionConfigurationError(ValueError):
    """Raised when recap visual analysis has incomplete provider settings."""


class VisionResponseError(RuntimeError):
    """Raised when a vision provider cannot produce usable response text."""


def _required_config_value(app_config: Mapping[str, object], setting: str) -> str:
    value = app_config.get(setting)
    if not isinstance(value, str) or not value.strip():
        raise VisionConfigurationError(f"{setting} must be configured for recap vision.")
    return value.strip()


def _optional_config_value(app_config: Mapping[str, object], setting: str) -> str:
    value = app_config.get(setting, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise VisionConfigurationError(f"{setting} must be a string when configured.")
    return value.strip()


def load_recap_vision_config(app_config=None) -> VisionConfig:
    """Load and validate the recap-specific vision provider settings."""
    runtime_config = app_config if app_config is not None else config.app
    provider = _required_config_value(runtime_config, "recap_vision_provider")
    if provider not in _SUPPORTED_PROVIDERS:
        raise VisionConfigurationError(
            "recap_vision_provider must be one of: openai_compatible, gemini."
        )

    api_key = _required_config_value(runtime_config, "recap_vision_api_key")
    model = _required_config_value(runtime_config, "recap_vision_model")
    base_url = _optional_config_value(runtime_config, "recap_vision_base_url")
    if provider == "openai_compatible" and not base_url:
        raise VisionConfigurationError(
            "recap_vision_base_url must be configured for openai_compatible."
        )

    return VisionConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def build_openai_messages(prompt: str, frames: Sequence[FrameInput]) -> list[dict]:
    """Build one OpenAI-compatible multimodal user message in frame order."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded_image = base64.b64encode(frame.image_bytes).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"t={frame.timestamp:.2f}s"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{frame.mime_type};base64,{encoded_image}"
                    },
                },
            ]
        )
    return [{"role": "user", "content": content}]


def _normalize_response_text(value: object) -> str:
    if not isinstance(value, str):
        raise VisionResponseError("Vision provider returned invalid text content.")

    text = value.strip()
    if not text:
        raise VisionResponseError("Vision provider returned empty text content.")
    return text


def _analyze_openai_compatible(
    vision_config: VisionConfig, prompt: str, frames: Sequence[FrameInput]
) -> str:
    try:
        client = OpenAI(
            api_key=vision_config.api_key,
            base_url=vision_config.base_url,
        )
        response = client.chat.completions.create(
            model=vision_config.model,
            messages=build_openai_messages(prompt, frames),
        )
        return _normalize_response_text(response.choices[0].message.content)
    except VisionResponseError:
        raise
    except Exception:
        raise VisionResponseError("OpenAI-compatible vision request failed.") from None


def _analyze_gemini(
    vision_config: VisionConfig, prompt: str, frames: Sequence[FrameInput]
) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception:
        raise VisionResponseError("Gemini vision SDK is unavailable.") from None

    try:
        http_options = (
            types.HttpOptions(base_url=vision_config.base_url)
            if vision_config.base_url
            else None
        )
        client_arguments = {"api_key": vision_config.api_key}
        if http_options is not None:
            client_arguments["http_options"] = http_options

        contents: list[object] = [prompt]
        for frame in frames:
            contents.extend(
                [
                    f"t={frame.timestamp:.2f}s",
                    types.Part.from_bytes(
                        data=frame.image_bytes,
                        mime_type=frame.mime_type,
                    ),
                ]
            )

        with genai.Client(**client_arguments) as client:
            response = client.models.generate_content(
                model=vision_config.model,
                contents=contents,
            )
        return _normalize_response_text(response.text)
    except VisionResponseError:
        raise
    except Exception:
        raise VisionResponseError("Gemini vision request failed.") from None


def analyze_frames(
    vision_config: VisionConfig, prompt: str, frames: Sequence[FrameInput]
) -> str:
    """Analyze sampled frames using the configured vision provider."""
    if not frames:
        return "[]"
    if vision_config.provider == "openai_compatible":
        return _analyze_openai_compatible(vision_config, prompt, frames)
    if vision_config.provider == "gemini":
        return _analyze_gemini(vision_config, prompt, frames)
    raise VisionConfigurationError(
        "recap_vision_provider must be one of: openai_compatible, gemini."
    )
