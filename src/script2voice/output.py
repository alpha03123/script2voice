from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .alignment import align_wav, load_aligner
from .models import OutputBlock, ScriptBlock, SentenceAudio
from .subtitles import make_local_srt, offset_srt_times, renumber_srt_entries, split_sentences
from .tts import load_qwen3_model, synthesize_text


def safe_stem(value: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._ ")
    if not stem:
        raise ValueError("Tag produced an empty file name")
    return stem


def fake_sentence_infos(text: str, chars_per_second: float) -> list[SentenceAudio]:
    cursor = 0.0
    sentences = []
    for sentence in split_sentences(text):
        duration = max(0.5, len(sentence) / chars_per_second)
        sentences.append(SentenceAudio(text=sentence, duration=duration, start=cursor, end=cursor + duration))
        cursor += duration
    return sentences


def select_tail_sentences(sentences: list[SentenceAudio], tail_seconds: float) -> list[SentenceAudio]:
    if not sentences:
        return []
    end_time = sentences[-1].end
    cutoff = max(0.0, end_time - tail_seconds)
    selected = [sentence for sentence in sentences if sentence.end > cutoff]
    return selected or [sentences[-1]]


def write_rolling_reference(
    source_wav: Path,
    sentences: list[SentenceAudio],
    output_path: Path,
    tail_seconds: float,
    min_seconds: float,
) -> tuple[str, str] | None:
    selected = select_tail_sentences(sentences, tail_seconds)
    if not selected:
        return None

    start = selected[0].start
    end = selected[-1].end
    if end - start < min_seconds:
        return None

    audio, sample_rate = sf.read(source_wav, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    start_frame = max(0, round(start * sample_rate))
    end_frame = min(len(audio), round(end * sample_rate))
    if end_frame <= start_frame:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio[start_frame:end_frame], sample_rate)
    ref_text = " ".join(sentence.text for sentence in selected)
    return str(output_path), ref_text


def write_outputs(blocks: list[ScriptBlock], args) -> list[OutputBlock]:
    output_dir = Path(args.output_dir).resolve()
    audio_dir = output_dir / "audio_blocks"
    subtitle_dir = output_dir / "subtitle_blocks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    tts_model = None if args.dry_run else load_qwen3_model(args)
    aligner = None if args.dry_run else load_aligner(args)
    output_blocks: list[OutputBlock] = []
    base_ref_audio = args.ref_audio
    base_ref_text = args.ref_text
    rolling_ref: tuple[str, str] | None = None
    rolling_ref_dir = output_dir / "temp" / "rolling_refs"

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
            args.ref_audio, args.ref_text = rolling_ref or (base_ref_audio, base_ref_text)
            print(f"[{block.index}/{len(blocks)}] TTS [{block.tag}]", flush=True)
            audio, sample_rate = synthesize_text(tts_model, block.text, args)
            duration = len(audio) / sample_rate
            sf.write(wav_path, audio, sample_rate)

            print(f"[{block.index}/{len(blocks)}] Align [{block.tag}]", flush=True)
            sentence_infos = align_wav(aligner, wav_path, block.text, args)
            if args.continuity_enabled:
                rolling_ref = write_rolling_reference(
                    wav_path,
                    sentence_infos,
                    rolling_ref_dir / f"{stem}.wav",
                    args.continuity_ref_tail_seconds,
                    args.continuity_min_ref_seconds,
                )

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
