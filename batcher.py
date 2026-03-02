"""
Context-aware batching engine.

Groups subtitle blocks into batches that:
  - Stay within token/character limits
  - Include overlap (surrounding context) so the model understands scene flow
  - Respect scene boundaries (large time gaps = likely scene change)
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Generator, List, Tuple

from srt_parser import SRTBlock

logger = logging.getLogger(__name__)


def timecode_to_ms(tc: str) -> int:
    """Convert HH:MM:SS,mmm to milliseconds."""
    tc = tc.replace(",", ".")
    parts = tc.split(":")
    h, m, rest = int(parts[0]), int(parts[1]), float(parts[2])
    return int((h * 3600 + m * 60 + rest) * 1000)


def is_scene_break(prev: SRTBlock, current: SRTBlock, gap_ms: int = 3000) -> bool:
    """Return True if the gap between two blocks suggests a scene change."""
    prev_end = timecode_to_ms(prev.end)
    cur_start = timecode_to_ms(current.start)
    return (cur_start - prev_end) >= gap_ms


def make_batches(
    blocks: List[SRTBlock],
    config: Dict[str, Any],
) -> Generator[List[Tuple[int, str]], None, None]:
    """
    Yield batches of (block_index, text) tuples for translation.

    Config keys used:
      batch_size         : max subtitle entries per batch (default 30)
      max_chars_per_batch: max total characters per batch (default 3000)
      context_overlap    : how many entries from the previous batch to prepend
                           as read-only context (default 2)
      scene_break_gap_ms : millisecond gap that signals a scene change (default 3000)
    """
    batch_size = config.get("batch_size", 30)
    max_chars = config.get("max_chars_per_batch", 3000)
    overlap = config.get("context_overlap", 2)
    gap_ms = config.get("scene_break_gap_ms", 3000)

    current_batch: List[Tuple[int, str]] = []
    current_chars = 0
    prev_block: SRTBlock | None = None

    def flush(batch: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        return batch

    for i, block in enumerate(blocks):
        # Check for forced flush on scene break
        scene_break = (
            prev_block is not None and is_scene_break(prev_block, block, gap_ms)
        )

        block_chars = len(block.text)

        should_flush = (
            current_batch
            and (
                len(current_batch) >= batch_size
                or current_chars + block_chars > max_chars
                or scene_break
            )
        )

        if should_flush:
            yield flush(current_batch)
            # Keep overlap entries for context
            if overlap > 0:
                current_batch = current_batch[-overlap:]
                current_chars = sum(len(t) for _, t in current_batch)
            else:
                current_batch = []
                current_chars = 0

        current_batch.append((block.index, block.text))
        current_chars += block_chars
        prev_block = block

    if current_batch:
        yield flush(current_batch)


def build_index_map(blocks: List[SRTBlock]) -> Dict[int, SRTBlock]:
    """Map block.index → block for fast lookup."""
    return {b.index: b for b in blocks}
