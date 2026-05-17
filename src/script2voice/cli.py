from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import DEFAULT_PRESET, apply_config, load_preset, load_settings
from .models import ScriptBlock
from .output import write_outputs, write_srt_preview
from .script_parser import parse_tagged_script


FULL_MAX_NEW_TOKENS = 25600


def merge_blocks_for_full_mode(blocks: list[ScriptBlock]) -> list[ScriptBlock]:
    if not blocks:
        return []
    text = "\n".join(block.text for block in blocks)
    visual_notes = []
    for block in blocks:
        visual_notes.extend(block.visual_notes)
    return [ScriptBlock(index=1, tag="full", text=text, visual_notes=visual_notes)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate WAV/SRT files from a tagged narration script.")
    parser.add_argument("--script", required=True, help="Input tagged script path")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="Preset name from presets/<name>.toml")
    parser.add_argument("--full", action="store_true", help="Merge all tags into one full-length TTS block")
    parser.add_argument("--preview-srt-only", action="store_true", help="Only write fake-timed SRT files for caption review")
    parser.add_argument("--dry-run", action="store_true", help="Validate parsing and write silent WAV/SRT placeholders")
    parser.set_defaults(repo="", language="Chinese")
    return parser


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args()
    apply_config(args, load_settings(), load_preset(args.preset))
    script_path = Path(args.script).resolve()
    blocks = parse_tagged_script(script_path.read_text(encoding="utf-8-sig"))
    if args.full:
        args.max_new_tokens = FULL_MAX_NEW_TOKENS
        args.continuity_enabled = False
        blocks = merge_blocks_for_full_mode(blocks)
    if args.preview_srt_only:
        write_srt_preview(blocks, args)
        print(f"Wrote SRT preview to {Path(args.output_dir).resolve()}")
        return
    output_blocks = write_outputs(blocks, args)
    print(f"Wrote {len(output_blocks)} block(s) to {Path(args.output_dir).resolve()}")
