#!/usr/bin/env python
"""Generate per-tag TTS WAV and local SRT blocks from a tagged script."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


DEFAULT_REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time "
    "are bullish on scaling up reinforcement learning atop LLMs. If we're actually "
    "close to a human-like learner, then this whole approach of training on verifiable "
    "outcomes is doomed."
)
DEFAULT_INSTRUCT = (
    "请用自然、清晰、稳定的中文教程讲解语气朗读。"
    "语速适中，情绪克制但有亲和力，重点术语读清楚，"
    "不要使用夸张、播音腔或过度营销的语气。"
)

TAG_LINE_RE = re.compile(r"^\s*\[([^\[\]\r\n]+)\]\s*$")
VISUAL_NOTE_LINE_RE = re.compile(r"^\s*\(([^()\r\n]+)\)\s*$")
SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")
CAPTION_BOUNDARY_PUNCTUATION = "，,。.!！?？；;、：:“”\"'‘’「」『』《》()（）[]【】"
CAPTION_BOUNDARY_CHARS = CAPTION_BOUNDARY_PUNCTUATION + " \t\r\n"
SRT_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
DEFAULT_VOICE_PRESETS = Path(__file__).resolve().with_name("voice_presets.json")
DEFAULT_INDEXTTS2_MODEL = r"E:\gittools\models\IndexTTS-2"
DEFAULT_HF_HUB_CACHE = r"E:\gittools\models\hf_cache"


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


def split_caption_text(text: str, max_chars: int) -> list[str]:
    caption = clean_caption(re.sub(r"\s+", " ", text).strip())
    return [caption] if caption else []


def split_sentences(text: str, max_chars: int) -> list[str]:
    pieces = [
        clean_caption(match.group(0))
        for match in SENTENCE_RE.finditer(text.replace("\n", " "))
        if match.group(0).strip()
    ]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        pieces = [clean_caption(text)]

    return pieces


def clean_caption(caption: str) -> str:
    return caption.strip(CAPTION_BOUNDARY_CHARS)


def make_local_srt(sentences: list[SentenceAudio], gap_seconds: float, max_caption_chars: int = 24) -> str:
    if not sentences:
        raise ValueError("Cannot create SRT for empty text")

    entries = []
    cursor = 0.0
    for index, sentence in enumerate(sentences, start=1):
        caption = clean_caption(sentence.text)
        if not caption:
            continue
        start = cursor
        end = start + sentence.duration
        entries.append((index, start, end, caption))
        cursor = end + gap_seconds

    return "\n\n".join(
        f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{caption}"
        for index, start, end, caption in entries
    ) + "\n"


def as_float32_audio(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def load_model(args):
    if getattr(args, "backend", "qwen3") == "indextts2":
        return load_indextts2_model(args)
    return load_qwen3_model(args)


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


def load_indextts2_model(args):
    os.environ.setdefault("HF_HUB_CACHE", args.hf_hub_cache)
    model_dir = Path(args.index_model).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"IndexTTS2 model directory does not exist: {model_dir}")
    cfg_path = model_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"IndexTTS2 config does not exist: {cfg_path}")

    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        model_dir=str(model_dir),
        cfg_path=str(cfg_path),
        use_fp16=args.index_fp16,
        use_cuda_kernel=args.index_cuda_kernel,
        use_deepspeed=False,
        use_torch_compile=args.index_torch_compile,
    )


def apply_voice_preset(args) -> None:
    presets_path = Path(args.voice_presets).resolve()
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    if args.voice_preset not in presets:
        available = ", ".join(sorted(presets))
        raise ValueError(f"Voice preset {args.voice_preset!r} not found. Available presets: {available}")

    preset = presets[args.voice_preset]
    if args.ref_audio is None:
        ref_audio = preset.get("ref_audio")
        if not ref_audio:
            raise ValueError(f"Voice preset {args.voice_preset!r} is missing ref_audio")
        ref_audio_path = Path(ref_audio)
        if not ref_audio_path.is_absolute():
            ref_audio_path = presets_path.parent / ref_audio_path
        args.ref_audio = str(ref_audio_path.resolve())

    if args.ref_text is None:
        ref_text = preset.get("ref_text")
        if not ref_text:
            raise ValueError(f"Voice preset {args.voice_preset!r} is missing ref_text")
        args.ref_text = ref_text


def parse_emo_vector(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 8:
        raise ValueError("--index-emo-vector must contain exactly 8 comma-separated numbers")
    return [float(part) for part in parts]


def synthesize_text(model, text: str, args) -> tuple[np.ndarray, int]:
    if getattr(args, "backend", "qwen3") == "indextts2":
        return synthesize_text_indextts2(model, text, args)
    return synthesize_text_qwen3(model, text, args)


def synthesize_text_qwen3(model, text: str, args) -> tuple[np.ndarray, int]:
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


def synthesize_text_indextts2(model, text: str, args) -> tuple[np.ndarray, int]:
    result = model.infer(
        spk_audio_prompt=args.ref_audio,
        text=text,
        output_path=None,
        emo_vector=parse_emo_vector(args.index_emo_vector),
        emo_alpha=args.index_emo_alpha,
        use_random=args.index_use_random,
        verbose=False,
        diffusion_steps=args.index_diffusion_steps,
        inference_cfg_rate=args.index_cfg_rate,
    )
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"IndexTTS2 returned {len(result)} outputs for one sentence")
        result = result[0]
    sample_rate, audio = result
    audio = as_float32_audio(audio)
    if np.issubdtype(audio.dtype, np.integer):
        raise RuntimeError("Internal error: integer audio was converted before normalization")
    if np.max(np.abs(audio), initial=0.0) > 1.0:
        audio = audio / 32767.0
    return audio, int(sample_rate)


def synthesize_block_sentences(model, block: ScriptBlock, tmp_dir: Path, args) -> tuple[np.ndarray, int, list[SentenceAudio]]:
    sentence_texts = split_sentences(block.text, args.max_caption_chars)
    if not sentence_texts:
        raise ValueError(f"Tag [{block.tag}] has no sentence text")

    gap = np.zeros(round(args.sentence_gap_seconds * 24000), dtype=np.float32)
    parts: list[np.ndarray] = []
    sentence_infos: list[SentenceAudio] = []
    sample_rate = None
    cursor = 0.0

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for sentence_index, sentence_text in enumerate(sentence_texts, start=1):
        print(f"  ({sentence_index}/{len(sentence_texts)}) {sentence_text}", flush=True)
        audio, sr = synthesize_text(model, sentence_text, args)
        if sample_rate is None:
            sample_rate = sr
            gap = np.zeros(round(args.sentence_gap_seconds * sr), dtype=np.float32)
        elif sample_rate != sr:
            raise RuntimeError(f"Sample rate changed from {sample_rate} to {sr}")

        sentence_tmp_path = tmp_dir / f"{sentence_index:03d}.wav"
        sf.write(sentence_tmp_path, audio, sr)

        duration = len(audio) / sr
        start = cursor
        end = start + duration
        sentence_infos.append(SentenceAudio(text=sentence_text, duration=duration, start=start, end=end))
        parts.append(audio)
        cursor = end
        if sentence_index != len(sentence_texts) and len(gap) > 0:
            parts.append(gap)
            cursor += args.sentence_gap_seconds

    assert sample_rate is not None
    combined = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    return combined, sample_rate, sentence_infos


def write_outputs(blocks: list[ScriptBlock], args) -> list[OutputBlock]:
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "audio_blocks"
    subtitle_dir = output_dir / "subtitle_blocks"
    tmp_root = output_dir / "tmp_sentence_wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    model = None if args.dry_run else load_model(args)
    output_blocks: list[OutputBlock] = []

    for block in blocks:
        stem = f"{block.index:03d}_{safe_stem(block.tag)}"
        wav_path = audio_dir / f"{stem}.wav"
        srt_path = subtitle_dir / f"{stem}.srt"
        tmp_dir = tmp_root / stem

        if args.dry_run:
            sample_rate = 24000
            sentence_infos = []
            parts = []
            cursor = 0.0
            sentence_texts = split_sentences(block.text, args.max_caption_chars)
            for sentence_text in sentence_texts:
                duration = max(0.5, len(sentence_text) / args.dry_run_chars_per_second)
                audio_part = np.zeros(round(duration * sample_rate), dtype=np.float32)
                sentence_infos.append(
                    SentenceAudio(
                        text=sentence_text,
                        duration=duration,
                        start=cursor,
                        end=cursor + duration,
                    )
                )
                parts.append(audio_part)
                cursor += duration
                if sentence_text != sentence_texts[-1] and args.sentence_gap_seconds > 0:
                    parts.append(np.zeros(round(args.sentence_gap_seconds * sample_rate), dtype=np.float32))
                    cursor += args.sentence_gap_seconds
            audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
            duration = len(audio) / sample_rate
        else:
            assert model is not None
            print(f"[{block.index}/{len(blocks)}] [{block.tag}] {block.text}", flush=True)
            audio, sample_rate, sentence_infos = synthesize_block_sentences(model, block, tmp_dir, args)
            duration = len(audio) / sample_rate
            if not args.keep_tmp:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        sf.write(wav_path, audio, sample_rate)
        srt_path.write_text(
            make_local_srt(sentence_infos, args.sentence_gap_seconds, args.max_caption_chars),
            encoding="utf-8",
        )

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

    full_audio_parts: list[np.ndarray] = []
    full_srt_parts: list[str] = []
    cursor = 0.0
    next_srt_index = 1
    sample_rate = None

    for block_index, block in enumerate(output_blocks, start=1):
        audio, sr = sf.read(block.wav, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(f"Sample rate changed from {sample_rate} to {sr}")

        full_audio_parts.append(audio)
        local_srt = Path(block.srt).read_text(encoding="utf-8")
        shifted_srt = offset_srt_times(local_srt, cursor)
        renumbered_srt, next_srt_index = renumber_srt_entries(shifted_srt, next_srt_index)
        if renumbered_srt:
            full_srt_parts.append(renumbered_srt)

        cursor += len(audio) / sr
        if block_index != len(output_blocks) and args.block_gap_seconds > 0:
            gap = np.zeros(round(args.block_gap_seconds * sr), dtype=np.float32)
            full_audio_parts.append(gap)
            cursor += args.block_gap_seconds

    assert sample_rate is not None
    full_audio = np.concatenate(full_audio_parts) if full_audio_parts else np.zeros(1, dtype=np.float32)
    sf.write(output_dir / "audio_full.wav", full_audio, sample_rate)
    (output_dir / "SRT_FULL.srt").write_text("\n\n".join(full_srt_parts) + "\n", encoding="utf-8")


def make_fake_sentence_infos(block: ScriptBlock, args) -> list[SentenceAudio]:
    sentence_infos = []
    cursor = 0.0
    for sentence_text in split_sentences(block.text, args.max_caption_chars):
        duration = max(0.5, len(sentence_text) / args.dry_run_chars_per_second)
        sentence_infos.append(
            SentenceAudio(
                text=sentence_text,
                duration=duration,
                start=cursor,
                end=cursor + duration,
            )
        )
        cursor += duration + args.sentence_gap_seconds
    return sentence_infos


def write_srt_preview(blocks: list[ScriptBlock], args) -> None:
    output_dir = Path(args.output_dir).resolve()
    subtitle_dir = output_dir / "subtitle_blocks"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    full_srt_parts = []
    cursor = 0.0
    next_srt_index = 1
    preview_blocks = []
    for block_index, block in enumerate(blocks, start=1):
        stem = f"{block.index:03d}_{safe_stem(block.tag)}"
        srt_path = subtitle_dir / f"{stem}.srt"
        sentence_infos = make_fake_sentence_infos(block, args)
        local_srt = make_local_srt(sentence_infos, args.sentence_gap_seconds, args.max_caption_chars)
        srt_path.write_text(local_srt, encoding="utf-8")

        shifted_srt = offset_srt_times(local_srt, cursor)
        renumbered_srt, next_srt_index = renumber_srt_entries(shifted_srt, next_srt_index)
        if renumbered_srt:
            full_srt_parts.append(renumbered_srt)

        duration = 0.0
        if sentence_infos:
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
    parser = argparse.ArgumentParser(description="Generate per-tag WAV/SRT blocks from a tagged script.")
    parser.add_argument("--script", required=True, help="Input tagged script path")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--backend", default="qwen3", choices=["qwen3", "indextts2"], help="TTS backend")
    parser.add_argument("--repo", default="", help="Optional faster-qwen3-tts repo path; normally unnecessary when installed in the environment")
    parser.add_argument("--model", default=r"E:\gittools\models\Qwen3-TTS-12Hz-1.7B-Base", help="Model id or local path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--language", default="Chinese", help="TTS language hint")
    parser.add_argument("--voice-presets", default=str(DEFAULT_VOICE_PRESETS), help="JSON file containing voice presets")
    parser.add_argument("--voice-preset", default="default", help="Voice preset name from --voice-presets")
    parser.add_argument("--ref-audio", default=None, help="Manual reference audio override")
    parser.add_argument("--ref-text", default=None, help="Manual reference transcript override")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT, help="TTS style instruction; pass an empty string to disable")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--icl", dest="xvec_only", action="store_false", help="Use full ICL cloning instead of x-vector-only")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--index-model", default=DEFAULT_INDEXTTS2_MODEL, help="IndexTTS2 model directory")
    parser.add_argument("--hf-hub-cache", default=DEFAULT_HF_HUB_CACHE, help="Shared HuggingFace cache directory")
    parser.add_argument("--index-fp16", action=argparse.BooleanOptionalAction, default=True, help="Use FP16 for IndexTTS2")
    parser.add_argument("--index-cuda-kernel", action="store_true", help="Try BigVGAN custom CUDA kernel for IndexTTS2")
    parser.add_argument("--index-torch-compile", action="store_true", help="Enable torch.compile for IndexTTS2 s2mel")
    parser.add_argument("--index-diffusion-steps", type=int, default=25, help="IndexTTS2 s2mel diffusion steps")
    parser.add_argument("--index-cfg-rate", type=float, default=0.7, help="IndexTTS2 classifier-free guidance rate")
    parser.add_argument("--index-emo-vector", default="0,0,0,0,0,0,0,0", help="IndexTTS2 emotion vector: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm")
    parser.add_argument("--index-emo-alpha", type=float, default=1.0, help="IndexTTS2 emotion strength")
    parser.add_argument("--index-use-random", action="store_true", help="Enable IndexTTS2 random sampling")
    parser.add_argument("--max-caption-chars", type=int, default=24, help="Maximum characters per SRT cue")
    parser.add_argument("--sentence-gap-seconds", type=float, default=0.2, help="Silence inserted between sentence WAVs")
    parser.add_argument("--block-gap-seconds", type=float, default=0.4, help="Silence inserted between full block WAVs")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary per-sentence WAV files")
    parser.add_argument("--preview-srt-only", action="store_true", help="Only write fake-timed SRT files for caption review")
    parser.add_argument("--dry-run", action="store_true", help="Validate parsing and write silent WAV/SRT placeholders")
    parser.add_argument("--dry-run-chars-per-second", type=float, default=6.0)
    parser.set_defaults(xvec_only=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    apply_voice_preset(args)
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
