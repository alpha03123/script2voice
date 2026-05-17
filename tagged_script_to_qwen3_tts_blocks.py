#!/usr/bin/env python
"""Generate WAV and SRT files from a tagged narration script."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = PROJECT_ROOT / "settings.toml"
DEFAULT_PRESETS_DIR = PROJECT_ROOT / "presets"
DEFAULT_PRESET = "tutorial"
DEFAULT_INSTRUCT = (
    "请用自然、清晰、稳定的中文教程讲解语气朗读。"
    "语速适中，情绪克制但有亲和力，重点术语读清楚，"
    "不要使用夸张、播音腔或过度营销的语气。"
)

TAG_LINE_RE = re.compile(r"^\s*\[([^\[\]\r\n]+)\]\s*$")
VISUAL_NOTE_LINE_RE = re.compile(r"^\s*\(([^()\r\n]+)\)\s*$")
SENTENCE_RE = re.compile(r"([^。！？!?\n]+[。！？!?]?)")
CAPTION_BOUNDARY_PUNCTUATION = "，,。.!！?？；;、：:“”\"'‘’「」『』《》()（）[]【】"
CAPTION_BOUNDARY_CHARS = CAPTION_BOUNDARY_PUNCTUATION + " \t\r\n"
SRT_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")


@dataclass(frozen=True)
class ScriptBlock:
    index: int
    tag: str
    text: str
    visual_notes: list[str]


@dataclass(frozen=True)
class SentenceAudio:
    text: str
    duration: float
    start: float = 0.0
    end: float = 0.0


@dataclass(frozen=True)
class OutputBlock:
    index: int
    tag: str
    text: str
    visual_notes: list[str]
    wav: str
    srt: str
    duration: float
    sample_rate: int
    sentences: list[SentenceAudio]


def read_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def resolve_project_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def get_nested(config: dict, keys: tuple[str, ...], default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_settings(path: Path = DEFAULT_SETTINGS) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Settings file does not exist: {path}")
    return read_toml(path)


def load_preset(name: str, presets_dir: Path = DEFAULT_PRESETS_DIR) -> dict:
    if not name:
        raise ValueError("--preset is required")
    preset_path = presets_dir / f"{name}.toml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset does not exist: {preset_path}")
    return read_toml(preset_path)


def apply_config(args, settings: dict, preset: dict) -> None:
    args.model = resolve_project_path(get_nested(settings, ("tts", "model")))
    args.device = get_nested(settings, ("tts", "device"), "cuda")
    args.dtype = get_nested(settings, ("tts", "dtype"), "bf16")
    args.max_seq_len = get_nested(settings, ("tts", "max_seq_len"), 8192)
    args.max_new_tokens = get_nested(settings, ("tts", "max_new_tokens"), 4096)
    args.hf_hub_cache = resolve_project_path(get_nested(settings, ("paths", "hf_hub_cache"), "cache/hf"))
    args.block_gap_seconds = get_nested(settings, ("output", "block_gap_seconds"), 0.4)
    args.dry_run_chars_per_second = get_nested(settings, ("output", "dry_run_chars_per_second"), 6.0)

    args.aligner_model = get_nested(settings, ("aligner", "model"), "Qwen/Qwen3-ForcedAligner-0.6B")
    args.aligner_device_map = get_nested(settings, ("aligner", "device_map"), "cuda:0")
    args.aligner_dtype = get_nested(settings, ("aligner", "dtype"), "bf16")

    args.ref_audio = resolve_project_path(get_nested(preset, ("voice", "ref_audio")))
    ref_text_file = resolve_project_path(get_nested(preset, ("voice", "ref_text_file")))
    if ref_text_file is None:
        raise ValueError(f"Preset {args.preset!r} is missing voice.ref_text_file")
    args.ref_text = Path(ref_text_file).read_text(encoding="utf-8").strip()
    if not args.ref_text:
        raise ValueError(f"Preset {args.preset!r} has an empty ref_text_file: {ref_text_file}")

    args.instruct = get_nested(preset, ("style", "instruct"), DEFAULT_INSTRUCT)
    args.temperature = get_nested(preset, ("style", "temperature"), 0.8)
    args.top_k = get_nested(preset, ("style", "top_k"), 50)
    args.repetition_penalty = get_nested(preset, ("style", "repetition_penalty"), 1.05)
    args.greedy = get_nested(preset, ("style", "greedy"), False)
    args.xvec_only = get_nested(preset, ("style", "xvec_only"), True)
    args.non_streaming_mode = get_nested(preset, ("style", "non_streaming_mode"), False)

    if args.model is None:
        raise ValueError("settings.toml is missing tts.model")
    if args.ref_audio is None:
        raise ValueError(f"Preset {args.preset!r} is missing voice.ref_audio")


def normalize_text(lines: list[str]) -> str:
    paragraphs = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return "\n".join(paragraphs).strip()


def parse_tagged_script(content: str) -> list[ScriptBlock]:
    blocks: list[ScriptBlock] = []
    current_tag: str | None = None
    current_lines: list[str] = []
    current_visual_notes: list[str] = []

    def flush() -> None:
        nonlocal current_tag, current_lines, current_visual_notes
        if current_tag is None:
            return
        text = normalize_text(current_lines)
        if not text:
            raise ValueError(f"Tag [{current_tag}] has no text")
        blocks.append(
            ScriptBlock(
                index=len(blocks) + 1,
                tag=current_tag,
                text=text,
                visual_notes=current_visual_notes,
            )
        )
        current_lines = []
        current_visual_notes = []

    for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = TAG_LINE_RE.match(raw_line)
        if match:
            flush()
            current_tag = match.group(1).strip()
            if not current_tag:
                raise ValueError("Empty tag is not allowed")
            continue

        visual_match = VISUAL_NOTE_LINE_RE.match(raw_line)
        if visual_match:
            if current_tag is None:
                raise ValueError("Visual note must appear inside a [tag] block")
            current_visual_notes.append(visual_match.group(1).strip())
            continue

        if current_tag is None and raw_line.strip():
            raise ValueError("Script text must start with a [tag] line")
        current_lines.append(raw_line)

    flush()
    if not blocks:
        raise ValueError("No [tag] blocks found")
    return blocks


def safe_stem(value: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._ ")
    if not stem:
        raise ValueError("Tag produced an empty file name")
    return stem


def format_srt_time(seconds: float) -> str:
    millis_total = max(0, round(seconds * 1000))
    millis = millis_total % 1000
    seconds_total = millis_total // 1000
    sec = seconds_total % 60
    minutes_total = seconds_total // 60
    minute = minutes_total % 60
    hour = minutes_total // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"


def parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def offset_srt_times(srt: str, offset_seconds: float) -> str:
    def replace(match: re.Match[str]) -> str:
        start = parse_srt_time(match.group(1)) + offset_seconds
        end = parse_srt_time(match.group(2)) + offset_seconds
        return f"{format_srt_time(start)} --> {format_srt_time(end)}"

    return SRT_TIME_RE.sub(replace, srt)


def clean_caption(caption: str) -> str:
    return caption.strip(CAPTION_BOUNDARY_CHARS)


def normalize_for_match(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text, flags=re.UNICODE)


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    pieces = [clean_caption(match.group(1)) for match in SENTENCE_RE.finditer(compact) if match.group(1)]
    pieces = [piece for piece in pieces if piece]
    return pieces or [clean_caption(text)]


def make_local_srt(sentences: list[SentenceAudio]) -> str:
    if not sentences:
        raise ValueError("Cannot create SRT for empty text")

    entries = []
    for index, sentence in enumerate(sentences, start=1):
        caption = clean_caption(sentence.text)
        if caption:
            entries.append((index, sentence.start, sentence.end, caption))

    return "\n\n".join(
        f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{caption}"
        for index, start, end, caption in entries
    ) + "\n"


def as_float32_audio(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def load_qwen3_model(args):
    if args.repo:
        repo = Path(args.repo).resolve()
        sys.path.insert(0, str(repo))

    from faster_qwen3_tts import FasterQwen3TTS

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation="sdpa",
        max_seq_len=args.max_seq_len,
    )


def load_aligner(args):
    os.environ.setdefault("HF_HOME", args.hf_hub_cache)
    os.environ.setdefault("HF_HUB_CACHE", args.hf_hub_cache)

    from qwen_asr import Qwen3ForcedAligner

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return Qwen3ForcedAligner.from_pretrained(
        args.aligner_model,
        dtype=dtype_map[args.aligner_dtype],
        device_map=args.aligner_device_map,
        cache_dir=args.hf_hub_cache,
    )


def synthesize_text(model, text: str, args) -> tuple[np.ndarray, int]:
    audio_list, sample_rate = model.generate_voice_clone(
        text=text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        repetition_penalty=args.repetition_penalty,
        xvec_only=args.xvec_only,
        non_streaming_mode=args.non_streaming_mode,
        instruct=args.instruct or None,
    )
    return as_float32_audio(audio_list[0]), sample_rate


def item_to_dict(item) -> dict:
    return {
        "text": item.text,
        "start_time": item.start_time,
        "end_time": item.end_time,
    }


def build_sentence_cues(items: list[dict], text: str) -> list[SentenceAudio]:
    sentences = split_sentences(text)
    cues: list[SentenceAudio] = []
    item_index = 0

    for sentence in sentences:
        target_len = len(normalize_for_match(sentence))
        if target_len == 0:
            continue

        collected = []
        start_index = item_index
        while item_index < len(items) and len(normalize_for_match("".join(collected))) < target_len:
            token = items[item_index]["text"]
            if normalize_for_match(token):
                collected.append(token)
            item_index += 1

        if not collected or len(normalize_for_match("".join(collected))) < target_len:
            raise RuntimeError(f"alignment tokens ended before sentence was matched: {sentence}")

        token_items = items[start_index:item_index]
        timed_items = [item for item in token_items if item["start_time"] is not None and item["end_time"] is not None]
        if not timed_items:
            raise RuntimeError(f"sentence has no timed tokens: {sentence}")

        start = float(timed_items[0]["start_time"])
        end = float(timed_items[-1]["end_time"])
        cues.append(SentenceAudio(text=sentence, duration=end - start, start=start, end=end))

    if not cues:
        raise RuntimeError("Forced aligner did not return any usable cue")
    return cues


def align_wav(alignment_model, wav_path: Path, text: str, args) -> list[SentenceAudio]:
    result = alignment_model.align(audio=str(wav_path), text=text, language=args.language)[0]
    items = [item_to_dict(item) for item in result.items]
    return build_sentence_cues(items, text)


def fake_sentence_infos(text: str, chars_per_second: float) -> list[SentenceAudio]:
    cursor = 0.0
    sentences = []
    for sentence in split_sentences(text):
        duration = max(0.5, len(sentence) / chars_per_second)
        sentences.append(SentenceAudio(text=sentence, duration=duration, start=cursor, end=cursor + duration))
        cursor += duration
    return sentences


def write_outputs(blocks: list[ScriptBlock], args) -> list[OutputBlock]:
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "audio_blocks"
    subtitle_dir = output_dir / "subtitle_blocks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    tts_model = None if args.dry_run else load_qwen3_model(args)
    aligner = None if args.dry_run else load_aligner(args)
    output_blocks: list[OutputBlock] = []

    for block in blocks:
        stem = f"{block.index:03d}_{safe_stem(block.tag)}"
        wav_path = audio_dir / f"{stem}.wav"
        srt_path = subtitle_dir / f"{stem}.srt"

        if args.dry_run:
            sample_rate = 24000
            sentence_infos = fake_sentence_infos(block.text, args.dry_run_chars_per_second)
            duration = sentence_infos[-1].end
            audio = np.zeros(round(duration * sample_rate), dtype=np.float32)
        else:
            assert tts_model is not None
            assert aligner is not None
            print(f"[{block.index}/{len(blocks)}] TTS [{block.tag}]", flush=True)
            audio, sample_rate = synthesize_text(tts_model, block.text, args)
            duration = len(audio) / sample_rate
            sf.write(wav_path, audio, sample_rate)

            print(f"[{block.index}/{len(blocks)}] Align [{block.tag}]", flush=True)
            sentence_infos = align_wav(aligner, wav_path, block.text, args)

        if args.dry_run:
            sf.write(wav_path, audio, sample_rate)
        srt_path.write_text(make_local_srt(sentence_infos), encoding="utf-8")

        output_blocks.append(
            OutputBlock(
                index=block.index,
                tag=block.tag,
                text=block.text,
                visual_notes=block.visual_notes,
                wav=str(wav_path),
                srt=str(srt_path),
                duration=duration,
                sample_rate=sample_rate,
                sentences=sentence_infos,
            )
        )

    blocks_json = output_dir / "blocks.json"
    blocks_json.write_text(
        json.dumps([asdict(block) for block in output_blocks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_full_outputs(output_blocks, output_dir, args)
    return output_blocks


def renumber_srt_entries(srt: str, start_index: int) -> tuple[str, int]:
    blocks = [block for block in re.split(r"\n\s*\n", srt.strip()) if block.strip()]
    output = []
    index = start_index
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines[0] = str(index)
        else:
            lines.insert(0, str(index))
        output.append("\n".join(lines))
        index += 1
    return "\n\n".join(output), index


def write_full_outputs(output_blocks: list[OutputBlock], output_dir: Path, args) -> None:
    if not output_blocks:
        return

    audio_parts = []
    sample_rate = None
    full_srt_parts = []
    cursor = 0.0
    next_srt_index = 1

    for block_index, block in enumerate(output_blocks, start=1):
        audio, sr = sf.read(block.wav, dtype="float32")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(f"Sample rate changed from {sample_rate} to {sr}")

        audio_parts.append(audio)
        local_srt = Path(block.srt).read_text(encoding="utf-8")
        shifted_srt = offset_srt_times(local_srt, cursor)
        renumbered_srt, next_srt_index = renumber_srt_entries(shifted_srt, next_srt_index)
        if renumbered_srt:
            full_srt_parts.append(renumbered_srt)

        cursor += block.duration
        if block_index != len(output_blocks) and args.block_gap_seconds > 0:
            gap = np.zeros(round(args.block_gap_seconds * sr), dtype=np.float32)
            audio_parts.append(gap)
            cursor += args.block_gap_seconds

    assert sample_rate is not None
    sf.write(output_dir / "audio_full.wav", np.concatenate(audio_parts), sample_rate)
    (output_dir / "SRT_FULL.srt").write_text("\n\n".join(full_srt_parts) + "\n", encoding="utf-8")


def write_srt_preview(blocks: list[ScriptBlock], args) -> None:
    output_dir = Path(args.output_dir).resolve()
    subtitle_dir = output_dir / "subtitle_blocks"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    preview_blocks = []
    full_srt_parts = []
    cursor = 0.0
    next_srt_index = 1

    for block_index, block in enumerate(blocks, start=1):
        stem = f"{block.index:03d}_{safe_stem(block.tag)}"
        srt_path = subtitle_dir / f"{stem}.srt"
        sentence_infos = fake_sentence_infos(block.text, args.dry_run_chars_per_second)
        local_srt = make_local_srt(sentence_infos)
        srt_path.write_text(local_srt, encoding="utf-8")

        shifted_srt = offset_srt_times(local_srt, cursor)
        renumbered_srt, next_srt_index = renumber_srt_entries(shifted_srt, next_srt_index)
        if renumbered_srt:
            full_srt_parts.append(renumbered_srt)

        duration = sentence_infos[-1].end
        preview_blocks.append(
            {
                "index": block.index,
                "tag": block.tag,
                "srt": str(srt_path),
                "fake_duration": duration,
                "sentences": [asdict(sentence) for sentence in sentence_infos],
            }
        )
        cursor += duration
        if block_index != len(blocks):
            cursor += args.block_gap_seconds

    (output_dir / "SRT_FULL.srt").write_text("\n\n".join(full_srt_parts) + "\n", encoding="utf-8")
    (output_dir / "srt_preview.json").write_text(
        json.dumps(preview_blocks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate WAV/SRT files from a tagged narration script.")
    parser.add_argument("--script", required=True, help="Input tagged script path")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="Preset name from presets/<name>.toml")
    parser.add_argument("--preview-srt-only", action="store_true", help="Only write fake-timed SRT files for caption review")
    parser.add_argument("--dry-run", action="store_true", help="Validate parsing and write silent WAV/SRT placeholders")
    parser.set_defaults(repo="", language="Chinese")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_config(args, load_settings(), load_preset(args.preset))
    script_path = Path(args.script).resolve()
    blocks = parse_tagged_script(script_path.read_text(encoding="utf-8-sig"))
    if args.preview_srt_only:
        write_srt_preview(blocks, args)
        print(f"Wrote SRT preview to {Path(args.output_dir).resolve()}")
        return
    output_blocks = write_outputs(blocks, args)
    print(f"Wrote {len(output_blocks)} block(s) to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
