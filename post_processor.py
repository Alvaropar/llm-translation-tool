"""
Post-processing checks enforced programmatically after translation.

Implements the mechanical rules from the style guide that the LLM cannot
reliably self-police across a full file:

  §2-2  Max 28 chars per display line  (≤56 chars total for 2-line cards)
  §2-5  CPS rule: if duration ≤ 1 second, translation should be ≤ 10 chars

Returns structured SubtitleIssue objects so they can be written to the
字幕轴问题 section of the Excel glossary sheet.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from srt_parser import SRTBlock

logger = logging.getLogger(__name__)

MAX_CHARS_PER_LINE = 28
MAX_LINES_PER_CARD = 2
MAX_TOTAL_CHARS = MAX_CHARS_PER_LINE * MAX_LINES_PER_CARD  # 56
SHORT_DURATION_MS = 1000
SHORT_DURATION_MAX_CHARS = 10


@dataclass
class SubtitleIssue:
    block_index: int
    start: str
    end: str
    original_text: str
    translated_text: str
    description: str       # written to column Q (问题)


def _timecode_to_ms(tc: str) -> int:
    tc = tc.replace(",", ".")
    h, m, rest = tc.split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(rest)) * 1000)


def _duration_ms(block: SRTBlock) -> int:
    return _timecode_to_ms(block.end) - _timecode_to_ms(block.start)


def check_and_warn(blocks: List[SRTBlock]) -> List[SubtitleIssue]:
    """
    Run all post-translation checks.
    Logs warnings for human review and returns a list of SubtitleIssue
    objects for writing to the Excel glossary sheet.
    """
    issues: List[SubtitleIssue] = []

    for block in blocks:
        text = block.translated or block.text
        clean = re.sub(r"<[^>]+>", "", text)
        char_count = len(clean)
        duration = _duration_ms(block)

        problems = []

        if char_count > MAX_TOTAL_CHARS:
            problems.append(
                f"Demasiado largo ({char_count} chars > {MAX_TOTAL_CHARS} máx). "
                f"Considerar simplificar o dividir con subtítulo adyacente."
            )

        if duration <= SHORT_DURATION_MS and char_count > SHORT_DURATION_MAX_CHARS:
            problems.append(
                f"Duración corta ({duration}ms ≤ {SHORT_DURATION_MS}ms) "
                f"pero {char_count} chars (recomendado ≤{SHORT_DURATION_MAX_CHARS}). "
                f"Considerar fusionar con subtítulo adyacente."
            )

        if problems:
            description = " | ".join(problems)
            issues.append(SubtitleIssue(
                block_index=block.index,
                start=block.start,
                end=block.end,
                original_text=block.text,
                translated_text=text,
                description=description,
            ))
            logger.warning(
                "[%d] %s --> %s | %s",
                block.index, block.start, block.end, description,
            )
            logger.warning("     Text: %r", text)

    if issues:
        logger.warning("%d subtitle(s) flagged for review.", len(issues))
    else:
        logger.info("Post-processing: all entries within display limits.")

    return issues
