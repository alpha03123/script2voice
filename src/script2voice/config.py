from __future__ import annotations

import os
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "settings.toml"
DEFAULT_PRESETS_DIR = PROJECT_ROOT / "presets"
DEFAULT_PRESET = "tutorial"
DEFAULT_INSTRUCT = (
    "请用自然、清晰、稳定的中文教程讲解语气朗读。"
    "语速适中，情绪克制但有亲和力，重点术语读清楚，"
    "不要使用夸张、播音腔或过度营销的语气。"
)


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
    args.model = os.environ.get("SCRIPT2VOICE_TTS_MODEL") or resolve_project_path(get_nested(settings, ("tts", "model")))
    args.device = get_nested(settings, ("tts", "device"), "cuda")
    args.dtype = get_nested(settings, ("tts", "dtype"), "bf16")
    args.max_seq_len = get_nested(settings, ("tts", "max_seq_len"), 8192)
    args.max_new_tokens = get_nested(settings, ("tts", "max_new_tokens"), 4096)
    args.hf_hub_cache = os.environ.get("SCRIPT2VOICE_HF_HUB_CACHE") or resolve_project_path(
        get_nested(settings, ("paths", "hf_hub_cache"), "cache/hf")
    )
    args.block_gap_seconds = get_nested(settings, ("output", "block_gap_seconds"), 0.4)
    args.dry_run_chars_per_second = get_nested(settings, ("output", "dry_run_chars_per_second"), 6.0)

    args.aligner_model = get_nested(settings, ("aligner", "model"), "Qwen/Qwen3-ForcedAligner-0.6B")
    args.aligner_device_map = get_nested(settings, ("aligner", "device_map"), "cuda:0")
    args.aligner_dtype = get_nested(settings, ("aligner", "dtype"), "bf16")
    args.aligner_max_chunk_seconds = get_nested(settings, ("aligner", "max_chunk_seconds"), 180.0)

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
    args.continuity_enabled = get_nested(preset, ("continuity", "enabled"), False)
    args.continuity_ref_tail_seconds = get_nested(preset, ("continuity", "ref_tail_seconds"), 30.0)
    args.continuity_min_ref_seconds = get_nested(preset, ("continuity", "min_ref_seconds"), 8.0)

    if args.model is None:
        raise ValueError("settings.toml is missing tts.model")
    if args.ref_audio is None:
        raise ValueError(f"Preset {args.preset!r} is missing voice.ref_audio")
