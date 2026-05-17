#!/usr/bin/env python
"""Render an SRT file to a single WAV with FasterQwen3TTS."""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def clean_text(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        try:
            index = int(lines[0].strip())
            timing_line = lines[1]
            text_lines = lines[2:]
        except ValueError:
            index = len(cues) + 1
            timing_line = lines[0]
            text_lines = lines[1:]

        if "-->" not in timing_line:
            raise ValueError(f"Cue {index} has no timing line")

        start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
        text = clean_text(text_lines)
        if text:
            cues.append(Cue(index=index, start=parse_timestamp(start_text), end=parse_timestamp(end_text), text=text))

    if not cues:
        raise ValueError(f"No subtitle cues found in {path}")
    return cues


def as_float32_audio(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def append_with_timeline(parts: list[np.ndarray], cursor: int, target_start: int, audio: np.ndarray) -> int:
    if target_start > cursor:
        parts.append(np.zeros(target_start - cursor, dtype=np.float32))
        cursor = target_start
    parts.append(audio)
    return cursor + len(audio)


def max_tokens_for_cue(cue: Cue, ceiling: int) -> int:
    duration = max(cue.end - cue.start, 0.5)
    return min(ceiling, max(32, round(duration * 16) + 16))


def fit_to_cue(audio: np.ndarray, cue: Cue, sample_rate: int) -> np.ndarray:
    target_len = max(1, round((cue.end - cue.start) * sample_rate))
    if len(audio) <= target_len:
        return audio

    clipped = audio[:target_len].copy()
    fade_len = min(round(0.05 * sample_rate), len(clipped))
    if fade_len > 1:
        clipped[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return clipped


def load_model(args):
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


def synthesize(args) -> None:
    srt_path = Path(args.srt).resolve()
    output_path = Path(args.output).resolve()
    cues = parse_srt(srt_path)
    model = load_model(args)

    parts: list[np.ndarray] = []
    cursor = 0
    sample_rate = None

    for cue in cues:
        print(f"[{cue.index}/{cues[-1].index}] {cue.start:.3f}s -> {cue.end:.3f}s {cue.text}", flush=True)
        audio_list, sr = model.generate_voice_clone(
            text=cue.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            max_new_tokens=max_tokens_for_cue(cue, args.max_new_tokens)
            if args.limit_tokens_to_cues
            else args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.greedy,
            repetition_penalty=args.repetition_penalty,
            xvec_only=args.xvec_only,
            non_streaming_mode=args.non_streaming_mode,
        )
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(f"Sample rate changed from {sample_rate} to {sr}")

        audio = as_float32_audio(audio_list[0])
        if args.fit_to_cues:
            audio = fit_to_cue(audio, cue, sr)
        cursor = append_with_timeline(parts, cursor, round(cue.start * sr), audio)

    assert sample_rate is not None
    final_end = round(max(cue.end for cue in cues) * sample_rate)
    if final_end > cursor:
        parts.append(np.zeros(final_end - cursor, dtype=np.float32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    sf.write(output_path, combined, sample_rate)
    print(f"Wrote {output_path} ({len(combined) / sample_rate:.2f}s, {sample_rate} Hz)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SRT subtitles to one WAV using FasterQwen3TTS.")
    parser.add_argument("--srt", required=True, help="Input .srt path")
    parser.add_argument("--output", required=True, help="Output .wav path")
    parser.add_argument("--repo", default=r"E:\gittools\githubs\faster-qwen3-tts", help="faster-qwen3-tts repo path")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model id or local path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--language", default="Chinese", help="TTS language hint")
    parser.add_argument("--ref-audio", default=r"E:\gittools\githubs\faster-qwen3-tts\ref_audio.wav")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--icl", dest="xvec_only", action="store_false", help="Use full ICL cloning instead of x-vector-only")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--fit-to-cues", action="store_true", help="Trim each generated segment to its subtitle window")
    parser.add_argument("--limit-tokens-to-cues", action="store_true", help="Reduce max_new_tokens based on cue duration")
    parser.set_defaults(xvec_only=True)
    return parser


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    synthesize(build_parser().parse_args())
