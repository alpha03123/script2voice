# Script2Voice

Generate narration WAV files and matching SRT subtitles from a tagged script.

```text
tagged script -> Qwen3-TTS longform WAV -> Qwen3 ForcedAligner -> SRT
```

Each `[tag]` block becomes one narration module. Script2Voice exports one WAV/SRT pair per module and also writes a stitched `audio_full.wav` with `SRT_FULL.srt`.

## Input

Tags and visual notes must be on their own lines:

```text
[summary]
(Show project homepage)
大家好，今天介绍一个本地 AI 视频知识管理工具。

[settings]
第一次使用时，进入左侧的控制中心进行配置。
```

`[summary]` starts a module. `(Show project homepage)` is saved as a visual note in `blocks.json`, but it is not sent to TTS and does not appear in subtitles.

## Configuration

Machine-level settings live in `settings.toml`:

```toml
[tts]
model = "models/Qwen3-TTS-12Hz-1.7B-Base"
device = "cuda"
dtype = "bf16"
max_seq_len = 8192
max_new_tokens = 4096

[aligner]
model = "Qwen/Qwen3-ForcedAligner-0.6B"
device_map = "cuda:0"
dtype = "bf16"
```

Voice and style presets live in `presets/`:

```toml
[voice]
ref_audio = "voices/tutorial/ref.wav"
ref_text_file = "voices/tutorial/ref.txt"

[style]
instruct = "请用自然、清晰、稳定的中文教程讲解语气朗读。"
temperature = 0.6
top_k = 50
repetition_penalty = 1.1
```

Put the reference WAV and its exact transcript in the matching `voices/<name>/` folder. Real voice files and transcripts are ignored by git.

## Usage

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_voice" `
  --preset tutorial
```

Output:

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

## Dependencies

The wrapper uses:

```text
E:\conda-envs\ai-cu128\python.exe
```

Required Python packages in that environment:

```powershell
python -m pip install qwen-tts qwen-asr soundfile numpy
```

Models are not stored in this repository. By default, place them under `models/` or edit `settings.toml`.

## Notes

Script2Voice does not generate TTS sentence by sentence. It generates each tag as longform audio first, then uses forced alignment to derive sentence-level subtitle timing.
