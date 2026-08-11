"""
漫剧二创模块

三阶段智能二创流水线：
  1. AI 理解视频：提取音频 → Whisper 转写原片台词（带时间戳）
  2. AI 扩写解说：将转写内容注入 LLM 上下文，基于真实剧情写解说
  3. 画面智能匹配：LLM 将解说每段映射到视频中最相关的时间段，精准切片

所有 LLM/Whisper 调用都有优雅降级：失败时回退到顺序切片，不会中断流程。
"""

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import RecapHookStrategy, VideoConcatMode
from app.services import video as video_service
from app.services import recap_hooks, vision
from app.utils import file_security, utils


@dataclass(frozen=True)
class FrameFile:
    """One extracted JPEG frame and its source-video timestamp."""

    timestamp: float
    path: Path


@dataclass
class RecapHookAnalysis:
    """Source-grounded inputs and selected openings for recap hook variants."""

    transcript: list[dict]
    observations: list[recap_hooks.VisualObservation]
    candidates: dict[RecapHookStrategy, recap_hooks.HookCandidate]
    unavailable: dict[RecapHookStrategy, str]
    observations_path: Path


# ============================================================================
# 阶段 1：AI 理解视频 —— 提取音频 + Whisper 转写
# ============================================================================

def analyze_source_video(task_id, source_path):
    """
    分析源视频：提取音频并用 Whisper 转写，返回带时间戳的内容时间线。

    Returns:
        时间线列表，每项 {"text": "...", "start": 0.0, "end": 3.5}
        转写失败时返回空列表。
    """
    audio_path = os.path.join(utils.task_dir(task_id), "source_audio.wav")
    _extract_audio(source_path, audio_path)
    if not os.path.exists(audio_path):
        logger.warning("recap: audio extraction failed, skipping analysis")
        return []

    segments = _transcribe_with_whisper(task_id, audio_path)

    # 保存时间线供阶段 3 使用
    timeline_path = os.path.join(utils.task_dir(task_id), "source_timeline.json")
    try:
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"recap: failed to save timeline: {exc}")

    logger.info(f"recap: analyzed source video, {len(segments)} transcript segments")
    return segments


def _extract_audio(source_path, output_path):
    """用 FFmpeg 从视频中提取 16kHz 单声道 WAV（Whisper 推荐格式）。"""
    ffmpeg_binary = video_service.get_ffmpeg_binary()
    cmd = [
        ffmpeg_binary, "-y",
        "-i", source_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        logger.info(f"recap: extracted audio to {output_path}")
    except Exception as exc:
        logger.warning(f"recap: audio extraction failed: {exc}")


def _transcribe_with_whisper(task_id, audio_path):
    """
    用 Whisper 转写音频，返回带时间戳的段落列表。

    独立加载模型，不依赖 subtitle 服务的配置，避免 large-v3 未下载时影响二创。
    默认使用 small 模型（已验证可下载），兼顾速度和中文识别准确度。
    """
    # 国内网络下载 HuggingFace 模型经常超时，默认使用镜像。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("recap: faster-whisper not installed, skipping transcription")
        return []

    # 优先使用本地已下载的模型，按优先级尝试
    model_candidates = ["small", "medium", "large-v3-turbo", "large-v3"]
    local_models_dir = os.path.join(utils.root_dir(), "models")

    model = None
    used_model_name = None
    for candidate in model_candidates:
        local_path = os.path.join(local_models_dir, f"whisper-{candidate}")
        if os.path.isdir(local_path) and os.path.isfile(os.path.join(local_path, "model.bin")):
            try:
                model = WhisperModel(local_path, device="cpu", compute_type="int8")
                used_model_name = candidate
                break
            except Exception:
                continue

    # 没有本地模型，尝试在线下载 small（最小最快）
    if model is None:
        try:
            logger.info("recap: no local whisper model, downloading 'small'...")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            used_model_name = "small"
        except Exception as exc:
            logger.warning(f"recap: failed to load whisper model: {exc}")
            return []

    logger.info(f"recap: transcribing with model '{used_model_name}'...")
    try:
        segments_iter, info = model.transcribe(
            audio_path, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append({
                    "text": text,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                })
        logger.info(
            f"recap: transcription done, {len(segments)} segments, "
            f"language={info.language}"
        )
        return segments
    except Exception as exc:
        logger.warning(f"recap: transcription failed: {exc}")
        return []


# ============================================================================
# Hook experiment analysis -- source frames + configured vision provider
# ============================================================================

_PTS_TIME_PATTERN = re.compile(
    r"pts_time:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
_VISUAL_ANALYSIS_PROMPT = """Analyze the supplied recap source frames. For every
frame, return exactly one JSON object in a JSON array. Each object must use the
timestamp supplied with that frame and contain exactly these fields:
timestamp, evidence, action, expression, shot_type, readability,
suspense_score, conflict_score, emotion_score.

Use scores from 0 to 5. Describe only visible source evidence. Do not infer
plot facts that are not visible in the supplied frame."""


def extract_analysis_frames(source_path, output_dir) -> list[FrameFile]:
    """Extract regular and scene-change JPEG samples for visual hook analysis."""
    frame_dir = Path(output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_analysis_frames(frame_dir)
    ffmpeg_binary = video_service.get_ffmpeg_binary()

    regular_template = frame_dir / "regular_%06d.jpg"
    regular_command = [
        ffmpeg_binary, "-y",
        "-i", str(source_path),
        "-vf", "fps=1/2,showinfo,scale=640:-2",
        "-q:v", "2",
        str(regular_template),
    ]
    try:
        regular_result = subprocess.run(
            regular_command, check=True, capture_output=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("recap: regular analysis-frame extraction failed")
        return []

    regular_frames = _regular_frame_files(frame_dir, regular_result.stderr)
    scene_template = frame_dir / "scene_%06d.jpg"
    scene_command = [
        ffmpeg_binary, "-y",
        "-i", str(source_path),
        "-vf", "select='gt(scene,0.40)',showinfo,scale=640:-2",
        "-vsync", "vfr",
        "-q:v", "2",
        str(scene_template),
    ]
    try:
        scene_result = subprocess.run(
            scene_command, check=True, capture_output=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("recap: scene-change analysis-frame extraction failed")
        return regular_frames

    scene_paths = _nonempty_jpegs(frame_dir, "scene_*.jpg")
    scene_times = _showinfo_timestamps(scene_result.stderr)
    if len(scene_paths) != len(scene_times):
        raise ValueError(
            "FFmpeg scene-change frame metadata did not match extracted frames."
        )

    scene_frames = [
        FrameFile(timestamp=timestamp, path=path)
        for timestamp, path in zip(scene_times, scene_paths)
        if all(abs(timestamp - regular.timestamp) >= 0.75 for regular in regular_frames)
    ]
    return sorted(regular_frames + scene_frames, key=lambda frame: frame.timestamp)


def analyze_hook_experiment(task_id, params) -> RecapHookAnalysis:
    """Create source-grounded hook candidates from transcript and video frames."""
    source_path = _get_first_source_path(params)
    if not source_path:
        raise ValueError("无法定位二创实验的源视频文件")

    # Validate credentials before audio extraction or Whisper model work begins.
    vision_config = vision.load_recap_vision_config()
    task_path = Path(utils.task_dir(task_id))
    analysis_dir = task_path / "recap-analysis"
    timeline = _load_timeline(str(task_path))
    if not timeline:
        timeline = analyze_source_video(task_id, source_path)
    if not timeline:
        raise ValueError("二创实验需要可用的源视频台词转写")

    _write_recap_json(analysis_dir / "transcript-timeline.json", timeline, "timeline")
    frames = extract_analysis_frames(source_path, analysis_dir / "frames")
    if not frames:
        raise ValueError("二创实验未能提取可供视觉分析的视频帧")

    observations = _analyze_visual_frame_batches(vision_config, frames)
    if not observations:
        raise ValueError("二创实验没有得到可用的视觉观察结果")

    observations_path = analysis_dir / "visual-observations.json"
    _write_recap_json(
        observations_path,
        [_visual_observation_json(observation) for observation in observations],
        "visual observations",
    )
    duration = _get_video_duration(source_path)
    if duration <= 0:
        raise ValueError("无法读取二创实验源视频的时长")
    candidates, unavailable = recap_hooks.select_hook_candidates(
        observations, params.recap_hook_strategies, duration
    )
    return RecapHookAnalysis(
        transcript=timeline,
        observations=observations,
        candidates=candidates,
        unavailable=unavailable,
        observations_path=observations_path,
    )


def _remove_previous_analysis_frames(frame_dir: Path):
    for pattern in ("regular_*.jpg", "scene_*.jpg"):
        for path in frame_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                logger.warning("recap: could not clear a previous analysis frame")


def _regular_frame_files(frame_dir: Path, stderr) -> list[FrameFile]:
    paths = _nonempty_jpegs(frame_dir, "regular_*.jpg")
    timestamps = _showinfo_timestamps(stderr)
    if len(paths) != len(timestamps):
        raise ValueError(
            "FFmpeg regular frame metadata did not match extracted frames."
        )
    return [
        FrameFile(timestamp=timestamp, path=path)
        for timestamp, path in zip(timestamps, paths)
    ]


def _nonempty_jpegs(frame_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path for path in frame_dir.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    )


def _showinfo_timestamps(stderr) -> list[float]:
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        return []

    timestamps = []
    for match in _PTS_TIME_PATTERN.finditer(text):
        try:
            timestamp = float(match.group(1))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            timestamps.append(timestamp)
    return timestamps


def _analyze_visual_frame_batches(vision_config, frames) -> list[recap_hooks.VisualObservation]:
    observations = []
    observed_timestamps = set()
    for first_index in range(0, len(frames), 8):
        frame_batch = frames[first_index:first_index + 8]
        inputs = []
        for frame in frame_batch:
            try:
                image_bytes = frame.path.read_bytes()
            except OSError:
                raise ValueError("无法读取用于二创实验的视频帧") from None
            if not image_bytes:
                raise ValueError("二创实验的视频帧为空")
            inputs.append(vision.FrameInput(frame.timestamp, image_bytes, "image/jpeg"))

        response = vision.analyze_frames(vision_config, _VISUAL_ANALYSIS_PROMPT, inputs)
        batch_observations = recap_hooks.parse_visual_observations(response)
        frame_timestamps = {frame.timestamp for frame in frame_batch}
        observation_timestamps = {
            observation.timestamp for observation in batch_observations
        }
        if observation_timestamps != frame_timestamps:
            raise ValueError(
                "视觉观察结果必须恰好覆盖提交的视频帧时间戳"
            )
        for observation in batch_observations:
            if observation.timestamp in observed_timestamps:
                raise ValueError("视觉观察结果重复了视频帧时间戳")
            observed_timestamps.add(observation.timestamp)
            observations.append(observation)
    return sorted(observations, key=lambda observation: observation.timestamp)


def _write_recap_json(path: Path, payload, label: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, TypeError, ValueError, OverflowError):
        raise ValueError(f"无法保存二创实验{label}数据") from None


def _visual_observation_json(observation: recap_hooks.VisualObservation) -> dict:
    return {
        "timestamp": observation.timestamp,
        "evidence": observation.evidence,
        "action": observation.action,
        "expression": observation.expression,
        "shot_type": observation.shot_type,
        "readability": observation.readability,
        "suspense_score": observation.suspense_score,
        "conflict_score": observation.conflict_score,
        "emotion_score": observation.emotion_score,
    }


# ============================================================================
# 阶段 2：AI 扩写解说 —— 注入视频内容到脚本生成上下文
# ============================================================================

RECAP_SCRIPT_SYSTEM_PROMPT = """你是一位专业的短视频解说创作者。

用户会提供一段视频的台词转写文本（来自语音识别，可能有少量错误）。你需要基于这些真实内容创作一段解说文案。

核心要求：
1. 解说内容必须紧扣转写文本中的真实剧情和对话，不要编造不存在的内容
2. 解说要有观点、有情感、有节奏，是对原片的解读和再创作，不是简单复述
3. 语言要口语化、有感染力，适合配音朗读
4. 只返回解说文案正文，不要加标题、序号或任何 markdown 格式标记
5. 不要在段落开头加"旁白"、"解说"等角色标记
6. 必须使用与视频台词转写相同的语言撰写解说，如果 Initialization 中指定了 language 参数，以该参数为准
""".strip()


_SUBJECT_PROMPT_TEMPLATE = """根据以下视频台词转写，生成一个简短有吸引力的视频主题（10-20个字）。
用与台词相同的语言生成主题。只返回主题文字，不要加引号或其他标记。

视频台词：
{transcript}
"""


def _detect_language(text):
    """
    从文本中简单检测语言：通过中文字符占比判断。
    返回语言代码（"zh"、"en" 等），供脚本生成使用。
    """
    if not text:
        return ""
    # 统计中文字符数量
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    total_chars = len(text)
    if total_chars == 0:
        return ""
    # 中文字符占比超过 15% 就判定为中文（转写文本会混入标点和空格）
    if chinese_chars / total_chars > 0.15:
        return "zh"
    return "en"


def _generate_subject_from_transcript(transcript_text):
    """
    根据视频转写内容自动生成一个简短的视频主题。
    失败时返回空字符串，由调用方决定回退策略。
    """
    from app.services import llm

    # 截取前 1000 字，足够理解主题
    text = transcript_text[:1000]
    prompt = _SUBJECT_PROMPT_TEMPLATE.format(transcript=text)

    try:
        subject = llm._generate_response(prompt)
        subject = subject.strip().strip("\"'""")
        # 限制长度
        if len(subject) > 50:
            subject = subject[:50]
        return subject
    except Exception as exc:
        logger.warning(f"recap: auto subject generation failed: {exc}")
        return ""


def enrich_recap_context(task_id, params):
    """
    在脚本生成前调用：转写源视频内容，注入到 LLM 的脚本生成上下文。

    修改 params 的 video_script_prompt 和 custom_system_prompt，
    使后续 generate_script 调用能基于真实视频内容写解说。
    """
    source_path = _get_first_source_path(params)
    if not source_path:
        logger.warning("recap: no source video found, skip context enrichment")
        return

    logger.info("recap: phase 1 — analyzing source video...")
    segments = analyze_source_video(task_id, source_path)

    if not segments:
        # 转写失败时，如果用户也没填主题，无法继续智能二创。
        # 如果用户填了主题，降级为用主题生成（不注入视频内容）。
        if not params.video_subject.strip():
            raise ValueError(
                "视频内容分析失败（Whisper 转写为空）。"
                "可能原因：1) Whisper 模型未下载成功，请检查网络或配置 HF 镜像；"
                "2) 视频可能没有音轨。请手动填写视频主题后重试。"
            )
        logger.warning(
            "recap: no transcript available, script will use subject only"
        )
        return

    # 拼接转写文本，保留时间信息帮助 LLM 理解剧情节奏
    transcript_lines = []
    for seg in segments:
        transcript_lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}s] {seg['text']}")
    transcript_text = "\n".join(transcript_lines)

    # 限制总长度，避免超出 token 限制（保留开头和结尾的剧情）
    max_chars = 3000
    if len(transcript_text) > max_chars:
        half = max_chars // 2
        transcript_text = (
            transcript_text[:half]
            + "\n...(中间部分省略)...\n"
            + transcript_text[-half:]
        )

    # 用户未填主题时，根据视频转写内容自动生成一个
    if not params.video_subject.strip():
        auto_subject = _generate_subject_from_transcript(
            " ".join(s["text"] for s in segments)
        )
        if auto_subject:
            params.video_subject = auto_subject
            logger.info(f"recap: auto-generated subject: {auto_subject}")

    # 从转写内容检测语言，确保脚本生成使用正确语言
    full_transcript = " ".join(s["text"] for s in segments)
    detected_lang = _detect_language(full_transcript)
    if detected_lang and not params.video_language:
        params.video_language = detected_lang
        logger.info(f"recap: detected language: {detected_lang}")

    # 将转写内容注入到脚本生成的提示词中
    context_block = (
        f"\n\n# 视频原始内容（AI 转写台词，带时间戳）\n{transcript_text}"
    )
    if params.video_script_prompt:
        params.video_script_prompt += context_block
    else:
        params.video_script_prompt = context_block.strip()

    # 使用二创专用系统提示词（用户未自定义时才覆盖）
    if not params.custom_system_prompt:
        params.custom_system_prompt = RECAP_SCRIPT_SYSTEM_PROMPT

    logger.info(
        f"recap: phase 2 — enriched script context with "
        f"{len(segments)} segments, {len(transcript_text)} chars"
    )


# ============================================================================
# 阶段 3：画面智能匹配 —— LLM 将解说映射到视频时间段
# ============================================================================

_MATCH_PROMPT_TEMPLATE = """你是专业的视频剪辑助手。请根据视频台词时间线，为解说文案的每个段落选择视频中画面最匹配的时间段。

## 视频台词时间线
{timeline}

## 解说文案（已分段）
{script_segments}

## 任务
为每个解说段落选择视频中画面最相关的{clip_duration}秒片段。
- 选择的片段应该在视频总时长 {video_duration:.1f} 秒以内
- 相邻段落尽量选择不同的时间段，避免画面重复
- 如果某个段落无法精确匹配，选择语义最接近的时间段

严格只输出 JSON 数组，不要加 markdown 标记或解释：
[{{"start": 0.0, "end": 5.0}}, {{"start": 12.5, "end": 17.5}}]
"""


def prepare_recap_materials(task_id, params, audio_duration):
    """
    二创素材准备：基于解说文案智能匹配源视频片段。

    流程：
      1. 加载之前转写的时间线和生成的脚本
      2. 调用 LLM 将脚本段落映射到视频时间段
      3. 用 FFmpeg 从匹配的时间段切片
      4. LLM 匹配失败时回退到顺序切片
    """
    if not params.video_materials:
        raise ValueError("二创模式需要至少提供一个源视频")

    # 强制顺序拼接，保持叙事连贯性。
    params.video_concat_mode = VideoConcatMode.sequential
    params.match_materials_to_script = True

    max_clip_duration = params.video_clip_duration or 5
    needed_segments = math.ceil(audio_duration / max_clip_duration) + 2

    source_path = _get_first_source_path(params)
    if not source_path:
        raise ValueError("无法定位源视频文件")

    task_dir = utils.task_dir(task_id)
    clips_dir = os.path.join(task_dir, "recap_clips")
    os.makedirs(clips_dir, exist_ok=True)
    ffmpeg_binary = video_service.get_ffmpeg_binary()

    # --- 尝试智能匹配 ---
    clip_ranges = None
    timeline = _load_timeline(task_dir)
    script_text = _load_script_text(task_dir)

    if timeline and script_text:
        logger.info("recap: phase 3 — intelligent clip matching...")
        video_duration = _get_video_duration(source_path)
        clip_ranges = _match_script_to_video(
            script_text, timeline, max_clip_duration, video_duration
        )
    else:
        logger.info("recap: no timeline or script, falling back to chronological cutting")

    # --- 根据匹配结果切片 ---
    if clip_ranges:
        segment_paths = _cut_clips_by_ranges(
            source_path, clips_dir, ffmpeg_binary, clip_ranges
        )
        logger.info(f"recap: smart-matched {len(segment_paths)} clips")
    else:
        # 回退：顺序切片
        segment_paths = _split_video_chronologically(
            source_path=source_path,
            output_dir=clips_dir,
            ffmpeg_binary=ffmpeg_binary,
            segment_duration=max_clip_duration,
            max_segments=needed_segments,
        )
        logger.info(f"recap: chronological fallback, {len(segment_paths)} clips")

    if not segment_paths:
        raise ValueError("无法从源视频切出有效片段，请检查视频格式和路径")

    # 源视频不够长时循环补齐
    if len(segment_paths) < needed_segments:
        logger.info(
            f"recap: only {len(segment_paths)} clips available, "
            f"recycling to fill {needed_segments}"
        )
        original_count = len(segment_paths)
        idx = 0
        while len(segment_paths) < needed_segments:
            segment_paths.append(segment_paths[idx % original_count])
            idx += 1

    logger.info(f"recap: prepared {len(segment_paths)} clips for composition")
    return segment_paths[:needed_segments]


def prepare_hook_variant_materials(
    task_id, params, analysis: RecapHookAnalysis, shared_body, max_audio_duration
) -> dict[RecapHookStrategy, list[str]]:
    """Cut one shared body and strategy-specific source-grounded hook openings."""
    if not isinstance(shared_body, str) or not shared_body.strip():
        raise ValueError("二创实验需要非空的共享解说正文")

    source_path = _get_first_source_path(params)
    if not source_path:
        raise ValueError("无法定位二创实验的源视频文件")

    task_path = Path(utils.task_dir(task_id))
    ffmpeg_binary = video_service.get_ffmpeg_binary()
    clip_duration = params.video_clip_duration or 5
    video_duration = _get_video_duration(source_path)
    if video_duration <= 0:
        raise ValueError("无法读取二创实验源视频的时长")

    selected = sorted(
        analysis.candidates.items(),
        key=lambda item: (item[1].start, item[1].end, item[0].value),
    )
    reserved_ranges = [candidate.as_range() for _, candidate in selected]
    timeline = _load_timeline(str(task_path))
    shared_body_ranges = _match_script_to_video(
        shared_body,
        timeline,
        clip_duration,
        video_duration,
        excluded_ranges=reserved_ranges,
    )
    if shared_body_ranges:
        shared_body_ranges = [
            time_range for time_range in shared_body_ranges
            if not _range_overlaps_any(time_range, reserved_ranges)
        ]

    shared_clips_dir = task_path / "experiments" / "hook" / "shared-body" / "clips"
    if shared_body_ranges:
        shared_body_clips = _cut_clips_by_ranges(
            source_path, str(shared_clips_dir), ffmpeg_binary, shared_body_ranges
        )
    else:
        latest_reserved_end = max((end for _, end in reserved_ranges), default=0.0)
        needed_segments = math.ceil(max_audio_duration / clip_duration) + 2
        fallback_ranges = _chronological_ranges_excluding(
            video_duration,
            clip_duration,
            needed_segments,
            reserved_ranges,
            start_time=latest_reserved_end,
        )
        shared_body_clips = _cut_clips_by_ranges(
            source_path, str(shared_clips_dir), ffmpeg_binary, fallback_ranges
        )

    if not shared_body_clips:
        raise ValueError("无法为二创实验切出共享正文片段")

    variants = {}
    for strategy, candidate in selected:
        hook_clips_dir = task_path / "experiments" / "hook" / strategy.value / "clips"
        hook_clips = _cut_clips_by_ranges(
            source_path,
            str(hook_clips_dir),
            ffmpeg_binary,
            [candidate.as_range()],
        )
        if not hook_clips:
            logger.warning(f"recap: no hook clips cut for {strategy.value} strategy")
            continue
        variants[strategy] = hook_clips + shared_body_clips
    return variants


def _match_script_to_video(
    script_text,
    timeline,
    clip_duration,
    video_duration,
    excluded_ranges: Sequence[tuple[float, float]] = (),
):
    """
    调用 LLM 将解说文案段落映射到视频中最相关的时间段。

    Returns:
        时间范围列表 [(start, end), ...]，失败时返回 None。
    """
    from app.services import llm

    # 将脚本分成段落
    script_segments = _split_script_into_segments(script_text)
    if not script_segments:
        return None

    # 构建时间线文本
    timeline_lines = []
    for seg in timeline:
        timeline_lines.append(f"[{seg['start']:.1f}-{seg['end']:.1f}s] {seg['text']}")
    timeline_text = "\n".join(timeline_lines)

    # 构建分段文本
    script_lines = []
    for i, seg in enumerate(script_segments):
        script_lines.append(f"段落{i + 1}: {seg}")
    script_segment_text = "\n".join(script_lines)

    prompt = _MATCH_PROMPT_TEMPLATE.format(
        timeline=timeline_text,
        script_segments=script_segment_text,
        clip_duration=clip_duration,
        video_duration=video_duration,
    )

    try:
        response = llm._generate_response(prompt)
        ranges = _parse_match_response(response, clip_duration, video_duration)
        if ranges:
            ranges = [
                candidate for candidate in ranges
                if not _range_overlaps_any(candidate, excluded_ranges)
            ]
        if ranges:
            logger.info(
                f"recap: LLM matched {len(ranges)} clip ranges: {ranges[:3]}..."
            )
            return ranges
        logger.warning("recap: LLM returned no valid non-excluded ranges")
        return None
    except Exception as exc:
        logger.warning(f"recap: LLM clip matching failed: {exc}")
        return None


def _split_script_into_segments(script_text):
    """将脚本文案按句号、问号、感叹号分成段落。"""
    sentences = re.split(r"[。！？\n.!?]", script_text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def _parse_match_response(response, clip_duration, video_duration):
    """
    解析 LLM 返回的 JSON 时间范围列表。

    容错处理：提取 JSON 数组、校验时间范围合法性。
    """
    if not response:
        return None

    # 尝试从响应中提取 JSON 数组
    json_match = re.search(r"\[.*?\]", response, re.DOTALL)
    if not json_match:
        return None

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    ranges = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = float(item.get("start", 0))
        end = float(item.get("end", start + clip_duration))

        # 校验时间范围合法性
        start = max(0, start)
        end = min(video_duration, end)
        if end <= start:
            end = min(video_duration, start + clip_duration)
        if end > start:
            ranges.append((start, end))

    return ranges if ranges else None


def _range_overlaps_any(
    candidate: tuple[float, float], excluded_ranges: Sequence[tuple[float, float]]
) -> bool:
    candidate_start, candidate_end = candidate
    for excluded_start, excluded_end in excluded_ranges:
        if candidate_start < excluded_end and excluded_start < candidate_end:
            return True
    return False


def _chronological_ranges_excluding(
    video_duration: float,
    clip_duration: float,
    max_segments: int,
    excluded_ranges: Sequence[tuple[float, float]],
    start_time: float = 0.0,
) -> list[tuple[float, float]]:
    """Build chronological clip ranges, preferring after start_time then wrapping."""
    if video_duration <= 0 or clip_duration <= 0 or max_segments <= 0:
        return []

    normalized_exclusions = []
    for excluded_start, excluded_end in excluded_ranges:
        start = max(0.0, min(video_duration, excluded_start))
        end = max(0.0, min(video_duration, excluded_end))
        if end > start:
            normalized_exclusions.append((start, end))
    normalized_exclusions.sort()

    merged_exclusions = []
    for start, end in normalized_exclusions:
        if merged_exclusions and start <= merged_exclusions[-1][1]:
            merged_exclusions[-1] = (merged_exclusions[-1][0], max(end, merged_exclusions[-1][1]))
        else:
            merged_exclusions.append((start, end))

    preferred_start = max(0.0, min(video_duration, start_time))
    windows = [(preferred_start, video_duration)]
    if preferred_start > 0:
        windows.append((0.0, preferred_start))

    ranges = []
    for window_start, window_end in windows:
        cursor = window_start
        for excluded_start, excluded_end in merged_exclusions:
            if excluded_end <= cursor:
                continue
            if excluded_start >= window_end:
                break
            if cursor < excluded_start:
                cursor = _append_chronological_ranges(
                    ranges,
                    cursor,
                    min(excluded_start, window_end),
                    clip_duration,
                    max_segments,
                )
            cursor = max(cursor, excluded_end)
            if len(ranges) >= max_segments or cursor >= window_end:
                break
        if len(ranges) >= max_segments:
            break
        if cursor < window_end:
            _append_chronological_ranges(
                ranges, cursor, window_end, clip_duration, max_segments
            )
        if len(ranges) >= max_segments:
            break
    return ranges


def _append_chronological_ranges(
    ranges: list[tuple[float, float]],
    start: float,
    end: float,
    clip_duration: float,
    max_segments: int,
) -> float:
    while start < end and len(ranges) < max_segments:
        next_end = min(start + clip_duration, end)
        ranges.append((start, next_end))
        start = next_end
    return start


def _cut_clips_by_ranges(source_path, output_dir, ffmpeg_binary, clip_ranges):
    """根据时间范围列表用 FFmpeg 流拷贝切割视频片段。"""
    segments = []
    for idx, (start, end) in enumerate(clip_ranges):
        duration = end - start
        if duration <= 0:
            continue
        output_path = os.path.join(output_dir, f"clip_{idx:04d}.mp4")
        cmd = [
            ffmpeg_binary, "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", source_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                segments.append(output_path)
                logger.debug(
                    f"recap: cut clip {idx} "
                    f"[{start:.1f}-{end:.1f}s, {duration:.1f}s]"
                )
        except Exception as exc:
            logger.warning(f"recap: cut failed at {start:.1f}s: {exc}")

    return segments


# ============================================================================
# 工具函数
# ============================================================================

def _get_first_source_path(params):
    """安全解析第一个可用的源视频绝对路径。"""
    local_videos_dir = utils.storage_dir("local_videos")
    for material in params.video_materials or []:
        if not material.url:
            continue
        try:
            return file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError:
            continue
    return None


def _load_timeline(task_dir):
    """加载阶段 1 保存的视频转写时间线。"""
    timeline_path = os.path.join(task_dir, "source_timeline.json")
    if not os.path.exists(timeline_path):
        return []
    try:
        with open(timeline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_script_text(task_dir):
    """从任务记录中加载生成的脚本文案。"""
    script_path = os.path.join(task_dir, "script.json")
    if not os.path.exists(script_path):
        return ""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("script", "")
    except Exception:
        return ""


def _split_video_chronologically(
    source_path, output_dir, ffmpeg_binary,
    segment_duration, max_segments, start_index=0, start_time=0.0,
):
    """
    顺序切片回退方案：从视频开头按固定时长逐段切割。
    """
    duration = _get_video_duration(source_path)
    if duration <= 0:
        logger.warning(f"recap: cannot read duration, skip: {source_path}")
        return []

    segments = []
    start_time = max(0.0, float(start_time))
    idx = start_index

    while start_time < duration and len(segments) < max_segments:
        output_path = os.path.join(output_dir, f"clip_{idx:04d}.mp4")
        cmd = [
            ffmpeg_binary, "-y",
            "-ss", f"{start_time:.3f}",
            "-t", f"{segment_duration:.3f}",
            "-i", source_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                segments.append(output_path)
        except Exception:
            pass

        start_time += segment_duration
        idx += 1

    return segments


def _get_video_duration(video_path):
    """用 moviepy 读取视频时长。"""
    clip = None
    try:
        clip = VideoFileClip(video_path)
        return clip.duration
    except Exception:
        return 0.0
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
