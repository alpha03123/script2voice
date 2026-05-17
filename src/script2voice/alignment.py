from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .models import SentenceAudio
from .subtitles import normalize_for_match, split_sentences


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


def get_wav_duration(wav_path: Path) -> float:
    info = sf.info(wav_path)
    return info.frames / info.samplerate


def split_text_for_alignment(text: str, audio_duration: float, max_chunk_seconds: float) -> list[tuple[str, float, float]]:
    sentences = split_sentences(text)
    total_chars = sum(max(1, len(normalize_for_match(sentence))) for sentence in sentences)
    if total_chars == 0:
        raise ValueError("Cannot align empty text")

    chunks = []
    current_sentences = []
    current_chars = 0
    cursor_chars = 0
    max_chars = max(1, int(total_chars * max_chunk_seconds / audio_duration))

    for sentence in sentences:
        sentence_chars = max(1, len(normalize_for_match(sentence)))
        if current_sentences and current_chars + sentence_chars > max_chars:
            start = audio_duration * cursor_chars / total_chars
            cursor_chars += current_chars
            end = audio_duration * cursor_chars / total_chars
            chunks.append((join_alignment_sentences(current_sentences), start, end))
            current_sentences = []
            current_chars = 0

        current_sentences.append(sentence)
        current_chars += sentence_chars

    if current_sentences:
        start = audio_duration * cursor_chars / total_chars
        chunks.append((join_alignment_sentences(current_sentences), start, audio_duration))

    return chunks


def join_alignment_sentences(sentences: list[str]) -> str:
    return "。".join(sentences) + "。"


def slice_wav(source: Path, destination: Path, start_seconds: float, end_seconds: float) -> None:
    audio, sample_rate = sf.read(source, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    start_frame = max(0, round(start_seconds * sample_rate))
    end_frame = min(len(audio), round(end_seconds * sample_rate))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid alignment chunk range: {start_seconds:.3f}-{end_seconds:.3f}")
    sf.write(destination, audio[start_frame:end_frame], sample_rate)


def offset_cues(cues: list[SentenceAudio], offset_seconds: float) -> list[SentenceAudio]:
    return [
        SentenceAudio(
            text=cue.text,
            duration=cue.duration,
            start=cue.start + offset_seconds,
            end=cue.end + offset_seconds,
        )
        for cue in cues
    ]


def align_short_wav(alignment_model, wav_path: Path, text: str, args) -> list[SentenceAudio]:
    result = alignment_model.align(audio=str(wav_path), text=text, language=args.language)[0]
    items = [item_to_dict(item) for item in result.items]
    return build_sentence_cues(items, text)


def align_wav(alignment_model, wav_path: Path, text: str, args) -> list[SentenceAudio]:
    duration = get_wav_duration(wav_path)
    max_chunk_seconds = args.aligner_max_chunk_seconds
    if duration <= max_chunk_seconds:
        return align_short_wav(alignment_model, wav_path, text, args)

    chunks = split_text_for_alignment(text, duration, max_chunk_seconds)
    temp_root = wav_path.parent.parent / "temp" / "alignment_chunks"
    temp_root.mkdir(parents=True, exist_ok=True)
    combined: list[SentenceAudio] = []

    with tempfile.TemporaryDirectory(prefix=f"{wav_path.stem}_", dir=temp_root) as temp_dir:
        temp_path = Path(temp_dir)
        for index, (chunk_text, start, end) in enumerate(chunks, start=1):
            chunk_wav = temp_path / f"{index:03d}.wav"
            print(
                f"  align chunk {index}/{len(chunks)} "
                f"{start:.1f}s-{end:.1f}s",
                flush=True,
            )
            slice_wav(wav_path, chunk_wav, start, end)
            combined.extend(offset_cues(align_short_wav(alignment_model, chunk_wav, chunk_text, args), start))

    if not combined:
        raise RuntimeError("Forced aligner did not return any usable cue")
    return combined
