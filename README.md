# Script2Voice

Generate narration WAV files and matching SRT subtitles from a tagged script.

The main workflow is:

```text
tagged script -> Qwen3-TTS longform WAV -> Qwen3 ForcedAligner -> SRT
```

Each `[tag]` block is treated as one narration module. Script2Voice generates one WAV and one SRT per block, then also exports a stitched `audio_full.wav` and `SRT_FULL.srt`.

## Input

Tags must be on their own line:

```text
[summary]
(Show project homepage)
大家好，今天介绍一个本地 AI 视频知识管理工具。

[settings]
第一次使用时，进入左侧的控制中心进行配置。
```

Lines like `(Show project homepage)` are visual notes. They are saved to `blocks.json`, but they are not sent to TTS and do not appear in subtitles.

## Output

```text
output/
  audio_blocks/
    001_summary.wav
    002_settings.wav
  subtitle_blocks/
    001_summary.srt
    002_settings.srt
  audio_full.wav
  SRT_FULL.srt
  blocks.json
```

Per-block SRT files start at `00:00:00,000`. `SRT_FULL.srt` is offset to match `audio_full.wav`.

## Usage

Preview subtitle sentence splitting without loading models:

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_preview" `
  --preview-srt-only `
  --ref-audio "E:\voices\ref_audio.wav"
```

Generate WAV and SRT:

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_voice" `
  --ref-audio "E:\voices\ref_audio.wav" `
  --ref-text "The exact transcript spoken in the reference audio." `
  --temperature 0.6 `
  --repetition-penalty 1.1
```

Use a voice preset:

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "script.txt" `
  --output-dir "out" `
  --voice-presets "E:\voices\voice_presets.json" `
  --voice-preset default
```

See available options:

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --help
```

## Dependencies

The wrapper uses:

```text
E:\conda-envs\ai-cu128\python.exe
```

Required Python packages in that environment:

```powershell
python -m pip install qwen-tts qwen-asr soundfile numpy
```

Models are not stored in this repository. Recommended locations:

```text
E:\gittools\models\Qwen3-TTS-12Hz-1.7B-Base
E:\gittools\models\hf_cache
```

## Notes

Script2Voice no longer generates TTS sentence by sentence. That old path caused unstable emotion and inconsistent delivery. The current path generates each tag as longform audio first, then uses forced alignment to derive sentence-level subtitle timing.
