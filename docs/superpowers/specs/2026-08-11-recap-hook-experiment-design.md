# Manga Drama Recap Hook Experiment Design

## Status

Approved direction: visual-scored hook generation. Each recap source produces three
platform-ready variants: `suspense`, `conflict`, and `emotion`. Publishing remains a
human action. Platform metrics are manually returned for later analysis.

## Problem Statement

The current recap flow transcribes the source with Whisper, asks a text LLM to map
script sentences to transcript timestamps, and cuts the corresponding clips. When a
match fails it cuts the source chronologically. This establishes basic semantic
relevance, but it cannot establish whether a selected frame contains a revealing
action, an expressive reaction, or a visually legible conflict. It also creates a
single linear script and does not reserve a high-attention opening.

As a result, generated recap videos can accurately paraphrase the source while
still looking like fixed-length source clips attached to narration rather than a
short-form drama edit designed for the first few seconds of attention.

## Goals

- Generate all three hook strategies from one source video in one recap task.
- Keep the body of the story materially equivalent so the hook is the primary
  experimental variable.
- Ground each hook in source frames and transcript timestamps; do not invent plot
  events, character actions, or the ending.
- Export self-contained video variants and a machine-readable experiment manifest.
- Preserve enough metadata for the user to backfill platform performance and for
  later analysis to compare strategy results across source videos.
- Fail early when a required vision model is not configured. Do not silently claim
  that visual analysis occurred when the system fell back to text-only matching.

## Non-Goals

- Automatic publishing, scheduling, or account management for third-party
  platforms.
- Real-time retention prediction before publishing.
- Replacing the existing general material-download workflow.
- Recreating a full nonlinear video editor in this change.

## Chosen Approach

Use a hybrid visual-scored pipeline. FFmpeg extracts sampled source frames with
timestamps. A configured vision-capable LLM returns structured observations and
scores for each frame. The recap service combines those observations with the
Whisper timeline to choose source-grounded candidate windows for three hook
strategies. A shared narrative body is generated once; each strategy receives a
short, source-grounded opening and its own selected source clips.

This is preferred over text-only hooks because the current text LLM cannot know
whether its selected timestamp visibly contains the described moment. It is
substantially smaller and safer than a full shot-detection, character-tracking,
and automatic-publishing system.

## User Experience

The `recap` video source gains an opt-in "Hook Experiment" control.

- The strategy selector exposes `Suspense`, `Conflict`, and `Emotion` separately.
  All three are selected by default and can remain independently selectable for
  future non-experiment use.
- With Hook Experiment enabled and all strategies selected, one source submission
  produces three completed videos. The normal `video_count` setting is not reused;
  it has unrelated semantics and must not control experiment variants.
- The task result shows each variant, its opening copy, selected source timestamps,
  and a downloadable experiment manifest.
- The generated task does not publish anything to a platform. The user reviews and
  uploads an approved variant through their own account.

## Processing Pipeline

1. Validate recap input and a dedicated vision-model configuration before invoking
   Whisper, TTS, or FFmpeg composition.
2. Produce the existing Whisper transcript timeline and generate source thumbnails
   every two seconds. Include scene-boundary thumbnails when a simple FFmpeg scene
   threshold finds an additional cut.
3. Submit small batches of timestamped thumbnails to the vision model. Each response
   must be valid JSON with visual evidence, visible characters, action, expression,
   shot type, readability, and three integer scores from zero to five:
   `suspense_score`, `conflict_score`, and `emotion_score`.
4. Join visual observations to nearby transcript entries and turn the strongest
   observations into 1.5-3.5 second candidate windows. Reject black frames,
   transition frames, invalid ranges, and duplicate windows.
5. Generate a single factual narrative body from the transcript and one shared body
   clip plan. The body starts after the hook and does not restate the selected
   opening.
6. For each selected strategy, choose a distinct highest-scoring candidate window
   and generate a 6-8 second opening. The prompt includes only the candidate's
   visual evidence and nearby transcript. It requires a payoff or explanation in
   the next 3-8 seconds of the shared body.
7. Compose an independent audio track, subtitles, selected source clips, and final
   MP4 for every strategy. Every variant uses the same body audio and body clip plan;
   only its opening copy, opening audio, and opening source clips vary. The existing
   recap clip matcher receives all reserved hook ranges so it cannot immediately
   repeat an opening shot in the shared body.
8. Write an experiment manifest before marking the task complete.

## Hook Rules

| Strategy | Selects | Opening promise | Prohibited behavior |
| --- | --- | --- | --- |
| `suspense` | An unusual state, reveal setup, or unanswered reaction | Ask why or how without revealing the final answer | Claiming an unseen reveal or ending |
| `conflict` | A confrontation, decisive action, or explicit opposition | State the visible event first, then explain the relationship | Describing violence, betrayal, or an action absent from the source |
| `emotion` | A legible facial reaction, vulnerable moment, or emotional dialogue | Establish the character's feeling and stakes | Assigning a feeling that is not supported by the visual evidence or dialogue |

If a strategy has no eligible candidate after scoring, the task still emits a
manifest entry with `status: "unavailable"` and an explicit reason. It does not
substitute an unrelated clip. A task with zero eligible strategies fails before
audio generation.

## Data Contracts

Add an explicit strategy enum and experiment fields to `VideoParams` rather than
overloading free-form prompts:

```python
recap_hook_experiment_enabled: bool = False
recap_hook_strategies: list[RecapHookStrategy] = [
    RecapHookStrategy.suspense,
    RecapHookStrategy.conflict,
    RecapHookStrategy.emotion,
]
```

The vision service is a small provider-neutral interface. The first implementation
supports configured OpenAI-compatible and Gemini multimodal models. Its runtime
configuration is separate from `VideoParams` and has explicit
`recap_vision_provider`, `recap_vision_model`, `recap_vision_base_url`, and
`recap_vision_api_key` entries. Unsupported providers fail validation with a direct
configuration error. A text-only LLM must never be silently treated as a vision
model.

Task artifacts are stored under the existing task directory:

```text
tasks/<task_id>/
  recap-analysis/
    transcript-timeline.json
    visual-observations.json
    frames/
  experiments/hook/
    experiment.json
    suspense/{script.txt, hook-plan.json, audio.mp3, final.mp4}
    conflict/{script.txt, hook-plan.json, audio.mp3, final.mp4}
    emotion/{script.txt, hook-plan.json, audio.mp3, final.mp4}
```

`experiment.json` contains source identity, shared-body hash, vision model name,
candidate windows, per-strategy status, final files, and an empty `platform_results`
array. A platform-result record has the following shape:

```json
{
  "strategy": "conflict",
  "platform": "douyin",
  "published_at": "2026-08-11T12:00:00+08:00",
  "views": 0,
  "avg_watch_seconds": null,
  "avg_watch_percent": null,
  "completion_rate": null,
  "three_second_retention": null,
  "likes": 0,
  "comments": 0,
  "shares": 0,
  "favorites": 0,
  "notes": ""
}
```

The user can return this data as JSON or CSV for analysis. Missing metrics remain
`null`; zero is used only for a metric explicitly reported as zero.

## Experiment Discipline

Generating three variants provides creative alternatives, but posting all three
near-identical videos to the same account can create duplicate-content risk and
does not produce clean causal evidence. For platform analysis:

- Generate all three variants for every source video.
- Review the variants, then randomly assign one strategy to publish for each source
  video in the main experiment.
- Balance strategy assignment by genre, source length, account, and posting window.
- Collect at least 30 published, distinct source videos per strategy before making a
  directional decision. Compare metrics only within comparable platform/account
  cohorts.
- Use three-second retention or the closest platform-provided early-retention metric
  as the primary measure. Use average watch percentage, completion rate, and
  interactions per 1,000 views as secondary measures.

The remaining two variants are retained for editorial review and controlled,
small-scale follow-up tests where the platform policy and account strategy permit
them.

## Error Handling

- No vision model configuration: fail during preflight with setup guidance.
- Vision response invalid or incomplete: retry the batch once, then fail the task
  before script/audio work if no valid observations remain.
- A single strategy has no valid visual candidate: mark that strategy unavailable;
  continue only when another requested strategy is valid.
- FFmpeg frame extraction or composition fails: retain analysis artifacts and mark
  the affected variant failed in the manifest.
- Existing non-experiment recap tasks retain their exact current behavior.

## Testing

- Unit-test strategy prompt construction, structured vision-response parsing,
  score validation, candidate-window selection, duplicate prevention, and hook
  grounding constraints.
- Add task-service tests with mocked Whisper, vision LLM, TTS, and FFmpeg outputs to
  verify three variant manifests and independent artifact paths.
- Add validation tests for missing/unsupported vision-model configuration and a
  regression test proving non-experiment recap requests still take the old path.
- Add WebUI tests for the experiment toggle, three selected strategies, and result
  links/statuses.
- Run one fixture-based end-to-end task with a short local source video and verify
  that each completed variant has a readable MP4, unique opening range, and manifest
  record.

## Acceptance Criteria

- A supported vision configuration and one source video produce three independently
  playable recap variants for the three default strategies.
- Every generated hook is traceable to stored visual evidence and transcript
  timestamps.
- The body script, body audio, and body clip plan remain shared; no hook strategy
  reuses another strategy's opening window.
- Missing visual capability fails clearly before chargeable downstream generation.
- The manifest can accept manually returned platform metrics without ambiguous
  zero-versus-missing values.
- Existing recap and non-recap video generation tests continue to pass.
