from __future__ import annotations

import re

from .models import SentenceAudio


SENTENCE_RE = re.compile(r"([^。！？!?\n]+[。！？!?]?)")
CAPTION_BOUNDARY_PUNCTUATION = "，,。.!！?？；;、：:“”\"'‘’「」『』《》()（）[]【】"
CAPTION_BOUNDARY_CHARS = CAPTION_BOUNDARY_PUNCTUATION + " \t\r\n"
SRT_TIME_RE = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")


def format_srt_time(seconds: float) -> str:
    millis_total = max(0, round(seconds * 1000))
    millis = millis_total % 1000
    seconds_total = millis_total // 1000
    sec = seconds_total % 60
    minutes_total = seconds_total // 60
    minute = minutes_total % 60
    hour = minutes_total // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{millis:03d}"


def parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def offset_srt_times(srt: str, offset_seconds: float) -> str:
    def replace(match: re.Match[str]) -> str:
        start = parse_srt_time(match.group(1)) + offset_seconds
        end = parse_srt_time(match.group(2)) + offset_seconds
        return f"{format_srt_time(start)} --> {format_srt_time(end)}"

    return SRT_TIME_RE.sub(replace, srt)


def clean_caption(caption: str) -> str:
    return caption.strip(CAPTION_BOUNDARY_CHARS)


def normalize_for_match(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text, flags=re.UNICODE)


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    pieces = [clean_caption(match.group(1)) for match in SENTENCE_RE.finditer(compact) if match.group(1)]
    pieces = [piece for piece in pieces if piece]
    return pieces or [clean_caption(text)]


def make_local_srt(sentences: list[SentenceAudio]) -> str:
    if not sentences:
        raise ValueError("Cannot create SRT for empty text")

    entries = []
    for index, sentence in enumerate(sentences, start=1):
        caption = clean_caption(sentence.text)
        if caption:
            entries.append((index, sentence.start, sentence.end, caption))

    return "\n\n".join(
        f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{caption}"
        for index, start, end, caption in entries
    ) + "\n"


def renumber_srt_entries(srt: str, start_index: int) -> tuple[str, int]:
    blocks = [block for block in re.split(r"\n\s*\n", srt.strip()) if block.strip()]
    output = []
    index = start_index
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines[0] = str(index)
        else:
            lines.insert(0, str(index))
        output.append("\n".join(lines))
        index += 1
    return "\n\n".join(output), index
