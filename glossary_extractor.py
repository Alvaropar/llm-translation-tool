"""
Glossary extractor: runs a single LLM call after translation to extract
terminology entries for the 术语表 Excel sheet.

Extracted categories:
  person_names    → columns A–D (人名&称呼)
  org_place_names → columns E–G (家族名&产业名&团队&地名)
  other_terms     → columns H–J (其他&特殊物品本地化)

First-occurrence timecodes are found by scanning the actual SRT blocks
rather than asking the LLM, which is more reliable and costs no extra tokens.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from srt_parser import SRTBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TermEntry:
    original: str
    translation: str
    notes: str = ""
    timecode: str = ""   # "EP{n} HH:MM:SS,mmm --> HH:MM:SS,mmm"


@dataclass
class GlossaryData:
    person_names: List[TermEntry] = field(default_factory=list)
    org_place_names: List[TermEntry] = field(default_factory=list)
    other_terms: List[TermEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# First-occurrence timecode finder
# ---------------------------------------------------------------------------

def _find_first_timecode(term: str, blocks: List[SRTBlock], episode_label: str) -> str:
    """
    Scan original block texts for the first occurrence of `term` (case-insensitive).
    Returns a formatted timecode string or empty string if not found.
    """
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for block in blocks:
        if pattern.search(block.text):
            return f"{episode_label} {block.start} --> {block.end}"
    return ""


def _attach_timecodes(
    entries: List[TermEntry],
    blocks: List[SRTBlock],
    episode_label: str,
) -> None:
    """Fill in first-occurrence timecodes for a list of entries in-place."""
    for entry in entries:
        if not entry.timecode and entry.original:
            entry.timecode = _find_first_timecode(entry.original, blocks, episode_label)


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """You are a professional subtitle terminology extractor.
Your job is to analyze English → Spanish translated subtitle pairs and extract proper nouns and localized terms into a structured glossary.

Return ONLY valid JSON — no markdown, no explanation, no code fences.

JSON schema:
{
  "person_names": [
    {"original": "...", "translation": "...", "notes": "..."}
  ],
  "org_place_names": [
    {"original": "...", "translation": "..."}
  ],
  "other_terms": [
    {"original": "...", "translation": "...", "notes": "..."}
  ]
}

Rules:
- person_names: character names, nicknames, honorifics (Mr., Dr., etc.).
  In "notes": include alternate forms or shortened names used later (e.g. "Sr. Gómez / informal: Diego").
- org_place_names: company names, family names, team names, club names, place names, countries, cities, regions.
- other_terms: currency conversions, luxury brand replacements (e.g. "Rolls-Royce" → "auto de lujo"),
  special items, made-up in-universe terms, fantasy rankings, dish localizations.
- Only include terms that actually appear in the provided subtitle pairs.
- Skip terms that are identical in both languages UNLESS they are proper nouns worth tracking.
- If a term was NOT localized (kept as-is), still include it if it is a key proper noun.
- Omit common words, verbs, adjectives — only proper nouns and localized special terms."""


def _build_extraction_prompt(blocks: List[SRTBlock]) -> str:
    lines = []
    for block in blocks:
        # Use the translated text if available, otherwise skip
        if not block.translated:
            continue
        original_flat = block.text.replace("\n", " ").strip()
        translated_flat = block.translated.replace("\n", " ").strip()
        lines.append(f"[{block.index}] EN: {original_flat} | ES: {translated_flat}")

    if not lines:
        return ""

    return "Extract terminology from these subtitle pairs:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing with graceful fallback
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse LLM JSON response, stripping any accidental markdown fences."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")

    return json.loads(text[start:end])


def _entries_from_raw(raw_list: Any, has_notes: bool = False) -> List[TermEntry]:
    entries = []
    if not isinstance(raw_list, list):
        return entries
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original", "")).strip()
        translation = str(item.get("translation", "")).strip()
        notes = str(item.get("notes", "")).strip() if has_notes else ""
        if original and translation:
            entries.append(TermEntry(original=original, translation=translation, notes=notes))
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_terminology(
    blocks: List[SRTBlock],
    episode_label: str,
    backend: Any,          # LocalCausalLMBackend instance
    config: Dict[str, Any],
) -> GlossaryData:
    """
    Run a single LLM extraction call and return structured GlossaryData.
    Falls back to an empty GlossaryData on any failure.
    """
    prompt = _build_extraction_prompt(blocks)
    if not prompt:
        logger.warning("No translated blocks available for glossary extraction.")
        return GlossaryData()

    logger.info("Running terminology extraction for %s…", episode_label)

    # Build the message list using the backend's already-loaded tokenizer
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        import torch

        backend._load(config)
        tokenizer = backend._tokenizer
        model = backend._model

        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = f"{_EXTRACTION_SYSTEM}\n\n{prompt}"

        inputs = tokenizer(rendered, return_tensors="pt")
        first_device = next(model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        prompt_len = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(
            generation[0][prompt_len:], skip_special_tokens=True
        )

        raw = _parse_json_response(output_text)
        data = GlossaryData(
            person_names=_entries_from_raw(raw.get("person_names", []), has_notes=True),
            org_place_names=_entries_from_raw(raw.get("org_place_names", []), has_notes=False),
            other_terms=_entries_from_raw(raw.get("other_terms", []), has_notes=True),
        )

        # Attach first-occurrence timecodes by scanning original blocks
        _attach_timecodes(data.person_names, blocks, episode_label)
        _attach_timecodes(data.org_place_names, blocks, episode_label)
        _attach_timecodes(data.other_terms, blocks, episode_label)

        logger.info(
            "Extracted: %d names, %d org/places, %d other terms",
            len(data.person_names),
            len(data.org_place_names),
            len(data.other_terms),
        )
        return data

    except Exception as exc:
        logger.error("Glossary extraction failed: %s", exc)
        return GlossaryData()


def episode_label_from_path(path) -> str:
    """
    Derive an episode label from a filename.
    '1.srt' → 'EP1', 'episode_03.srt' → 'EP3', 'foo.srt' → 'EP?'
    """
    from pathlib import Path
    stem = Path(path).stem
    m = re.search(r"\d+", stem)
    return f"EP{int(m.group())}" if m else "EP?"
