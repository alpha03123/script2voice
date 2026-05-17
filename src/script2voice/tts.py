from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


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
