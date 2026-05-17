#!/usr/bin/env python
"""Experiment: synthesize one tagged block as a continuous Qwen3-TTS WAV."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


TOOL_PATH = Path(__file__).resolve().with_name("tagged_script_to_qwen3_tts_blocks.py")


def load_tagged_tool():
    spec = importlib.util.spec_from_file_location("tagged_script_tool", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one [tag] as a continuous Qwen3-TTS WAV.")
    parser.add_argument("--script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="", help="Tag name to synthesize; defaults to first block")
    parser.add_argument("--model", default=r"E:\gittools\models\Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--voice-presets", default=str(Path(__file__).resolve().with_name("voice_presets.json")))
    parser.add_argument("--voice-preset", default="default")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--icl", dest="xvec_only", action="store_false")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--fake-srt-chars-per-second", type=float, default=6.0)
    parser.set_defaults(xvec_only=True)
    return parser


def select_block(blocks, tag: str):
    if not tag:
        return blocks[0]
    for block in blocks:
        if block.tag == tag:
            return block
    available = ", ".join(block.tag for block in blocks)
    raise ValueError(f"Tag {tag!r} not found. Available tags: {available}")


def main() -> None:
    tool = load_tagged_tool()
    args = build_parser().parse_args()
    if args.instruct is None:
        args.instruct = tool.DEFAULT_INSTRUCT
    tool.apply_voice_preset(args)

    script_path = Path(args.script).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = tool.parse_tagged_script(script_path.read_text(encoding="utf-8-sig"))
    block = select_block(blocks, args.tag)
    stem = f"{block.index:03d}_{tool.safe_stem(block.tag)}"

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation="sdpa",
        max_seq_len=args.max_seq_len,
    )

    print(f"[tag] {block.tag}")
    print(f"[chars] {len(block.text)}")
    audio_list, sample_rate = model.generate_voice_clone(
        text=block.text,
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

    audio = tool.as_float32_audio(audio_list[0])
    wav_path = output_dir / f"{stem}.wav"
    text_path = output_dir / f"{stem}.txt"
    fake_srt_path = output_dir / f"{stem}.fake.srt"
    meta_path = output_dir / f"{stem}.json"

    sf.write(wav_path, audio, sample_rate)
    text_path.write_text(block.text, encoding="utf-8")

    sentence_infos = []
    cursor = 0.0
    for sentence_text in tool.split_sentences(block.text, 999999):
        duration = max(0.5, len(sentence_text) / args.fake_srt_chars_per_second)
        sentence_infos.append(tool.SentenceAudio(sentence_text, duration, cursor, cursor + duration))
        cursor += duration
    fake_srt_path.write_text(tool.make_local_srt(sentence_infos, 0.0), encoding="utf-8")

    duration = len(audio) / sample_rate
    meta_path.write_text(
        json.dumps(
            {
                "tag": block.tag,
                "chars": len(block.text),
                "wav": str(wav_path),
                "text": str(text_path),
                "fake_srt": str(fake_srt_path),
                "sample_rate": sample_rate,
                "duration": duration,
                "settings": {
                    "max_seq_len": args.max_seq_len,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                    "xvec_only": args.xvec_only,
                    "non_streaming_mode": args.non_streaming_mode,
                },
                "sentences": [asdict(sentence) for sentence in sentence_infos],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[wav] {wav_path}")
    print(f"[duration] {duration:.3f}s")
    print(f"[text] {text_path}")
    print(f"[fake_srt] {fake_srt_path}")


if __name__ == "__main__":
    main()
