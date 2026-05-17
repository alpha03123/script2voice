from __future__ import annotations

import os
from pathlib import Path

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


def align_wav(alignment_model, wav_path: Path, text: str, args) -> list[SentenceAudio]:
    result = alignment_model.align(audio=str(wav_path), text=text, language=args.language)[0]
    items = [item_to_dict(item) for item in result.items]
    return build_sentence_cues(items, text)
