from __future__ import annotations

import re

from .models import ScriptBlock


TAG_LINE_RE = re.compile(r"^\s*\[([^\[\]\r\n]+)\]\s*$")
VISUAL_NOTE_LINE_RE = re.compile(r"^\s*\(([^()\r\n]+)\)\s*$")


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
