"""
SRT file parser and writer. Preserves all timing, formatting, and structure.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SRTBlock:
    index: int
    start: str
    end: str
    text: str          # original text (may be multi-line)
    translated: str = ""

    @property
    def display_text(self) -> str:
        """Return translated text if available, else original."""
        return self.translated if self.translated else self.text

    def to_srt(self, use_translation: bool = True) -> str:
        content = self.translated if (use_translation and self.translated) else self.text
        return f"{self.index}\n{self.start} --> {self.end}\n{content}"


def parse_srt(path: str | Path) -> List[SRTBlock]:
    """Parse an SRT file into a list of SRTBlock objects."""
    text = Path(path).read_text(encoding="utf-8-sig")  # handle BOM
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks: List[SRTBlock] = []
    # Split on blank lines between blocks
    raw_blocks = re.split(r"\n{2,}", text.strip())

    for raw in raw_blocks:
        lines = raw.strip().splitlines()
        if len(lines) < 3:
            continue

        # Index line
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Timecode line
        timecode_pattern = r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
        m = re.match(timecode_pattern, lines[1].strip())
        if not m:
            continue

        start, end = m.group(1), m.group(2)
        # Normalize separator to comma
        start = start.replace(".", ",")
        end = end.replace(".", ",")

        # Text content (everything after the timecode line)
        text_content = "\n".join(lines[2:]).strip()

        blocks.append(SRTBlock(index=index, start=start, end=end, text=text_content))

    return blocks


def write_srt(blocks: List[SRTBlock], path: str | Path, use_translation: bool = True) -> None:
    """Write SRTBlock list to an SRT file."""
    out = "\n\n".join(block.to_srt(use_translation=use_translation) for block in blocks)
    Path(path).write_text(out + "\n", encoding="utf-8")


def strip_html_tags(text: str) -> str:
    """Remove HTML formatting tags from subtitle text for translation."""
    return re.sub(r"<[^>]+>", "", text)


def restore_html_tags(original: str, translated: str) -> str:
    """
    Re-apply leading/trailing HTML tags from the original text to the translation.
    e.g. if original is "<i>Hello</i>", wrap translated in <i>...</i>.
    """
    # Find opening tags at start
    leading = re.match(r"^(<[^>]+>)+", original)
    trailing = re.search(r"(<\/[^>]+>)+$", original)

    result = translated
    if leading:
        result = leading.group(0) + result
    if trailing:
        result = result + trailing.group(0)
    return result
