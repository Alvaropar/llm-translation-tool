"""
Translation backend: local HuggingFace causal LM loaded in int4.

Design decisions for performance:
- System prompt (guidelines) is built once and cached per backend instance.
- User message per batch contains only the subtitle entries — no repeated boilerplate.
- Multi-line subtitle text is flattened to a single line before the model sees it;
  the SRT player handles word-wrapping (per guideline §2-1).
- Model is loaded in NF4 int4 via BitsAndBytesConfig for fast inference.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TranslationBackend:
    name: str = "base"

    def translate_batch(
        self,
        items: List[Tuple[int, str]],
        config: Dict[str, Any],
    ) -> List[Tuple[int, str]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Compact system prompt built from the translation guidelines.
# Kept short on purpose: every token in the system message costs inference
# time on every batch. Only rules that differ from default LLM behavior are
# included; basics like "be accurate" are assumed.
# ---------------------------------------------------------------------------

def build_system_prompt(config: Dict[str, Any]) -> str:
    src = config.get("source_language", "English")
    tgt = config.get("target_language", "Spanish")
    formality = config.get("formality", "neutral")
    style_notes = config.get("style_notes", "").strip()
    glossary: Dict[str, str] = config.get("glossary", {}) or {}

    if formality == "formal":
        pronoun_rule = "Use 'usted' for singular address; 'ustedes' for plural (never 'vosotros')."
    elif formality == "informal":
        pronoun_rule = "Use 'tú' for singular address; 'ustedes' for plural (never 'vosotros')."
    else:
        pronoun_rule = (
            "Use 'tú' for close/familiar relationships, 'usted' for strangers or hierarchical contexts; "
            "'ustedes' for all second-person plural (never 'vosotros')."
        )

    glossary_block = ""
    if glossary:
        entries = "\n".join(f"  {k} → {v}" for k, v in glossary.items())
        glossary_block = f"\nFixed glossary — always use these translations:\n{entries}\n"

    style_block = f"\nAdditional style note: {style_notes}" if style_notes else ""

    return f"""You are a professional subtitle translator for short dramas, translating from {src} to Latin American {tgt}.

LANGUAGE & STYLE
- Produce natural, colloquial Latin American Spanish. Never translate word-for-word.
- Match the character's emotion, register, and tone precisely.
- Localize idioms and slang to the closest Spanish equivalent; do not transliterate.
- {pronoun_rule}
- Use standard modern grammar (e.g. 'solo' not 'sólo', 'este' not 'éste').

CONCISENESS (subtitle display constraints)
- Each subtitle must fit in at most 2 display lines of 28 characters each (≤56 chars total).
- Be as concise as possible without losing the core meaning or emotion.
- If a subtitle duration is ≤1 second, keep the translation under 10 characters.
- Do NOT insert line breaks — the player handles word-wrap automatically.

PROPER NOUNS
- Modern currency (USD, CNY, KRW, etc.): convert to US Dollars, rounding to clean amounts.
  Example: 2,000万 RMB → "tres millones de dólares". Ancient/fictional currency: keep the original unit name.
- Chinese character names: localize to Spanish names (meaning or phonetic similarity). English names: use phonetically similar Spanish name.
- Fictional place names: translate meaning into Spanish (e.g. 江城 → "Ciudad del Río"). Real Chinese cities → invent a Spanish fictional name.
- Well-known luxury brands: replace with a generic descriptor (luxury car, designer watch, etc.).
- After a character relationship is established, use the character's name in subsequent lines instead of repeating the relationship title.

SENSITIVE CONTENT
- Swear words / coarse language: localize to the culturally equivalent Spanish expression or soften appropriately.
{glossary_block}{style_block}
OUTPUT FORMAT
- Respond with ONLY the translated subtitle entries.
- Keep the exact [N] index prefix on every line.
- Output each subtitle as a single line of text (no line breaks inside a subtitle).
- Do not add commentary, notes, or blank lines between entries."""


# ---------------------------------------------------------------------------
# Local causal LM backend
# ---------------------------------------------------------------------------

class LocalCausalLMBackend(TranslationBackend):
    name = "local_causal_lm"

    def __init__(self, model_path: str, device_map: str = "auto"):
        self.model_path = model_path
        self.device_map = device_map
        self._tokenizer = None
        self._model = None
        self._system_prompt: Optional[str] = None

    def _load(self, config: Dict[str, Any]) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading model in int4 from: %s", self.model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bits/param
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            quantization_config=bnb_config,
        )
        self._model.eval()

        # Build and cache the system prompt once for this config
        self._system_prompt = build_system_prompt(config)
        logger.info("Model loaded. System prompt: %d chars", len(self._system_prompt))

    def _build_user_message(self, items: List[Tuple[int, str]]) -> str:
        """
        Minimal user message: just the numbered subtitle entries.
        Multi-line subtitle text is joined with a space (single line per entry)
        because the player handles word-wrapping (guideline §2-1).
        """
        lines = []
        for idx, text in items:
            flat = text.replace("\n", " ").strip()
            lines.append(f"[{idx}] {flat}")
        return "Translate:\n" + "\n".join(lines)

    def _parse_response(
        self,
        response_text: str,
        expected_indices: List[int],
    ) -> Dict[int, str]:
        results: Dict[int, str] = {}
        # Match [N] followed by text up to the next [N] or end of string
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.DOTALL)
        expected_set = set(expected_indices)
        for match in pattern.finditer(response_text):
            idx = int(match.group(1))
            text = match.group(2).strip()
            if idx in expected_set:
                results[idx] = text
        return results

    def translate_batch(
        self,
        items: List[Tuple[int, str]],
        config: Dict[str, Any],
    ) -> List[Tuple[int, str]]:
        import torch

        self._load(config)
        assert self._tokenizer is not None and self._model is not None

        expected = [idx for idx, _ in items]
        user_message = self._build_user_message(items)
        retries = config.get("retries", 3)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"{self._system_prompt}\n\n{user_message}"

        for attempt in range(retries):
            try:
                inputs = self._tokenizer(rendered, return_tensors="pt")
                # Move to the first device of a possibly multi-GPU model
                first_device = next(self._model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generation = self._model.generate(
                        **inputs,
                        max_new_tokens=config.get("max_tokens", 1024),
                        do_sample=False,        # greedy — fast and deterministic for translation
                        temperature=None,       # must be None when do_sample=False
                        top_p=None,
                        repetition_penalty=1.05,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                output_tokens = generation[0][prompt_len:]
                output_text = self._tokenizer.decode(output_tokens, skip_special_tokens=True)

                parsed = self._parse_response(output_text, expected)
                missing = [idx for idx in expected if idx not in parsed]
                if missing:
                    logger.warning(
                        "Attempt %d: missing indices %s, retrying…", attempt + 1, missing
                    )
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue

                return [(idx, parsed.get(idx, original)) for idx, original in items]

            except Exception as exc:
                logger.error("Generation failed (attempt %d): %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise

        return items  # fallback: return originals untranslated
