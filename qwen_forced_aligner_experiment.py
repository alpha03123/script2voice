import argparse
import json
import os
import re
from pathlib import Path

import torch
from qwen_asr import Qwen3ForcedAligner


SENTENCE_END_RE = re.compile(r"([^。！？!?]+[。！？!?]?)")
CAPTION_BOUNDARY_PUNCTUATION = "，,。.!！?？；;、：:“”\"'‘’「」『』《》()（）[]【】"
CAPTION_BOUNDARY_CHARS = CAPTION_BOUNDARY_PUNCTUATION + " \t\r\n"


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def normalize_for_match(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text, flags=re.UNICODE)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", "", text)
    return [match.group(1) for match in SENTENCE_END_RE.finditer(text) if match.group(1)]


def clean_caption(caption: str) -> str:
    return caption.strip(CAPTION_BOUNDARY_CHARS)


def item_to_dict(item) -> dict:
    return {
        "text": item.text,
        "start_time": item.start_time,
        "end_time": item.end_time,
    }


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_srt(path: Path, cues: list[dict]) -> None:
    lines = []
    for index, cue in enumerate(cues, start=1):
        lines.extend(
            [
                str(index),
                f"{format_srt_time(cue['start_time'])} --> {format_srt_time(cue['end_time'])}",
                cue["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sentence_cues(items: list[dict], text: str) -> list[dict]:
    sentences = split_sentences(text)
    cues = []
    item_index = 0

    for sentence in sentences:
        target_len = len(normalize_for_match(sentence))
        collected = []
        start_index = item_index

        while item_index < len(items) and len(normalize_for_match("".join(collected))) < target_len:
            token = items[item_index]["text"]
            if normalize_for_match(token):
                collected.append(token)
            item_index += 1

        matched = normalize_for_match("".join(collected))
        if not collected or len(matched) < target_len:
            raise RuntimeError(f"alignment tokens ended before sentence was matched: {sentence}")

        token_items = items[start_index:item_index]
        timed_items = [item for item in token_items if item["start_time"] is not None and item["end_time"] is not None]
        if not timed_items:
            raise RuntimeError(f"sentence has no timed tokens: {sentence}")

        cues.append(
            {
                "text": clean_caption(sentence),
                "start_time": float(timed_items[0]["start_time"]),
                "end_time": float(timed_items[-1]["end_time"]),
                "aligned_text": "".join(collected),
            }
        )

    return cues


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen3 forced alignment on one WAV/text pair.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cache-dir", default=r"E:\gittools\models\hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", args.cache_dir)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = Path(args.text).read_text(encoding="utf-8")

    model = Qwen3ForcedAligner.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )
    result = model.align(audio=args.audio, text=text, language=args.language)[0]
    items = [item_to_dict(item) for item in result.items]
    stem = Path(args.audio).stem
    write_json(output_dir / f"{stem}.alignment.raw.json", {"items": items})
    cues = build_sentence_cues(items, text)

    write_json(output_dir / f"{stem}.alignment.json", {"items": items, "cues": cues})
    write_srt(output_dir / f"{stem}.aligned.srt", cues)
    print(f"items={len(items)} cues={len(cues)}")
    print(output_dir / f"{stem}.alignment.json")
    print(output_dir / f"{stem}.aligned.srt")


if __name__ == "__main__":
    main()
