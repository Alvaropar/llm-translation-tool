#!/usr/bin/env python3
"""
SRT Translation Pipeline — English → Spanish (or any configured pair)

Usage:
    python translate.py input.srt                        # uses config.yaml
    python translate.py input.srt -o output.srt
    python translate.py input.srt --model-path /home/coder/models/Llama-3.3-70B-Instruct
    python translate.py *.srt --batch-dir ./translated/  # translate multiple files
    python translate.py --folder ./original-srt/         # translate all SRTs in a folder
    python translate.py --config my_config.yaml input.srt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from srt_parser import SRTBlock, parse_srt, write_srt, strip_html_tags, restore_html_tags
from batcher import make_batches, build_index_map
from backends import LocalCausalLMBackend, TranslationBackend
from post_processor import check_and_warn
from glossary_extractor import extract_terminology, episode_label_from_path
from excel_writer import append_to_glossary

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Language pair
    "source_language": "English",
    "target_language": "Spanish",

    # Local model settings
    "model_path": "/home/coder/models/Llama-3.3-70B-Instruct",
    "device_map": "auto",

    # Translation quality
    "temperature": 0.1,      # low = more deterministic (better for MT)
    "max_tokens": 4096,
    "retries": 3,

    # Batching
    "batch_size": 30,
    "max_chars_per_batch": 3000,
    "context_overlap": 2,     # subtitle entries from prev batch to include as context
    "scene_break_gap_ms": 3000,

    # Style / content customization
    "formality": "neutral",   # neutral | formal | informal
    "style_notes": "",        # e.g. "This is a sports broadcast; use energetic language."
    "glossary": {},           # {"original term": "translation to use"}

    # Output
    "output_suffix": "_es",   # appended before .srt extension when no -o given
    "output_dir": None,       # if set, output files go here
    "preserve_formatting": True,  # re-apply <i>/<b> tags after translation

    # Glossary Excel
    "glossary_xlsx": "术语表&画面字.xlsx",  # path to the Excel file; null to disable
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        path = Path(config_path)
        if not path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def build_backend(config: Dict[str, Any], cli_overrides: Dict[str, Any]) -> TranslationBackend:
    model_path = cli_overrides.get("model_path") or config["model_path"]
    device_map = cli_overrides.get("device_map") or config.get("device_map", "auto")
    logger.info("Using local backend — model path: %s  device_map: %s", model_path, device_map)
    return LocalCausalLMBackend(model_path=model_path, device_map=device_map)


# ---------------------------------------------------------------------------
# Core translation function
# ---------------------------------------------------------------------------

def translate_file(
    input_path: Path,
    output_path: Path,
    backend: "LocalCausalLMBackend",
    config: Dict[str, Any],
) -> None:
    logger.info("Parsing: %s", input_path)
    blocks = parse_srt(input_path)
    if not blocks:
        logger.warning("No subtitle blocks found in %s", input_path)
        return

    logger.info("Found %d subtitle blocks", len(blocks))
    index_map = build_index_map(blocks)

    # Prepare text for translation:
    # 1. Save originals (with HTML tags) for tag restoration.
    # 2. Strip HTML tags so the model only sees plain text.
    # 3. Flatten multi-line subtitle text to a single line — per style guide §2-1
    #    the player handles word-wrapping; translators should not insert line breaks.
    preserve_fmt = config.get("preserve_formatting", True)
    originals: Dict[int, str] = {}
    for block in blocks:
        originals[block.index] = block.text
        clean = strip_html_tags(block.text) if preserve_fmt else block.text
        block.text = clean.replace("\n", " ").strip()

    # Process batches
    batches = list(make_batches(blocks, config))
    logger.info("Processing %d batches…", len(batches))

    translated_count = 0
    progress = tqdm(batches, desc=input_path.name, unit="batch", dynamic_ncols=True)
    for batch in progress:
        progress.set_postfix(entries=f"{batch[0][0]}–{batch[-1][0]}")

        t0 = time.time()
        results = backend.translate_batch(batch, config)
        elapsed = time.time() - t0
        progress.set_postfix(entries=f"{batch[0][0]}–{batch[-1][0]}", took=f"{elapsed:.1f}s")

        for idx, translated_text in results:
            if idx in index_map:
                block = index_map[idx]
                # Re-apply HTML formatting tags if needed
                if preserve_fmt and idx in originals:
                    translated_text = restore_html_tags(originals[idx], translated_text)
                block.translated = translated_text
                translated_count += 1

    # Restore original text (with HTML tags) for blocks that failed translation
    for block in blocks:
        if not block.translated:
            block.text = originals.get(block.index, block.text)

    logger.info("Translated %d/%d blocks", translated_count, len(blocks))

    # Post-processing: char limit + CPS checks; returns structured issues for Excel
    issues = check_and_warn(blocks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(blocks, output_path)
    logger.info("Written: %s", output_path)

    # Glossary extraction + Excel update
    xlsx_path = config.get("glossary_xlsx", "术语表&画面字.xlsx")
    if xlsx_path:
        episode_label = episode_label_from_path(input_path)
        glossary = extract_terminology(blocks, episode_label, backend, config)
        append_to_glossary(xlsx_path, glossary, issues, episode_label)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def resolve_output_path(
    input_path: Path,
    output_arg: Optional[str],
    config: Dict[str, Any],
) -> Path:
    if output_arg:
        return Path(output_arg)

    suffix = config.get("output_suffix", "_es")
    out_name = input_path.stem + suffix + input_path.suffix

    out_dir = config.get("output_dir")
    if out_dir:
        return Path(out_dir) / out_name

    return input_path.parent / out_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate SRT subtitle files using LLM or MT models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("inputs", nargs="*", metavar="INPUT.srt", help="Input SRT file(s)")
    p.add_argument("--folder", metavar="DIR",
                   help="Translate all *.srt files in DIR; output goes to DIR/spanish-srt/")
    p.add_argument("-o", "--output", metavar="OUTPUT.srt",
                   help="Output path (single file only)")
    p.add_argument("--batch-dir", metavar="DIR",
                   help="Directory for output files when translating multiple inputs")
    p.add_argument("--config", metavar="FILE", default="config.yaml",
                   help="YAML config file (default: config.yaml)")
    p.add_argument("--model-path", metavar="PATH",
                   help="Override local model path from config")
    p.add_argument("--device-map", metavar="MAP",
                   help="Override transformers device_map (default: auto)")
    p.add_argument("--source-lang", metavar="LANG",
                   help="Source language (default: English)")
    p.add_argument("--target-lang", metavar="LANG",
                   help="Target language (default: Spanish)")
    p.add_argument("--formality", choices=["neutral", "formal", "informal"])
    p.add_argument("--style-notes", metavar="TEXT",
                   help="Free-text style instructions passed to the LLM")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config_path = args.config if Path(args.config).exists() else None
    config = load_config(config_path)

    # Apply CLI overrides to config
    if args.source_lang:
        config["source_language"] = args.source_lang
    if args.target_lang:
        config["target_language"] = args.target_lang
    if args.formality:
        config["formality"] = args.formality
    if args.style_notes:
        config["style_notes"] = args.style_notes
    if args.batch_dir:
        config["output_dir"] = args.batch_dir

    # Resolve input file list
    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            logger.error("--folder path is not a directory: %s", args.folder)
            sys.exit(1)
        if args.inputs or args.output:
            logger.error("--folder cannot be combined with positional inputs or -o/--output.")
            sys.exit(1)
        input_paths = sorted(folder.glob("*.srt"))
        if not input_paths:
            logger.error("No .srt files found in: %s", folder)
            sys.exit(1)
        # Output goes to <folder>/spanish-srt/, filenames unchanged
        config["output_dir"] = str(folder / "spanish-srt")
        config["output_suffix"] = ""
        logger.info("Folder mode: %d file(s) → %s", len(input_paths), config["output_dir"])
    else:
        if not args.inputs:
            logger.error("Provide input file(s) or use --folder.")
            sys.exit(1)
        input_paths = [Path(p) for p in args.inputs]
        if len(input_paths) > 1 and args.output:
            logger.error("Cannot use -o/--output with multiple input files; use --batch-dir instead.")
            sys.exit(1)

    cli_overrides = {
        "model_path": args.model_path,
        "device_map": args.device_map,
    }

    # Build backend (loads model only once for all files)
    backend = build_backend(config, cli_overrides)

    for input_path in input_paths:
        if not input_path.exists():
            logger.error("File not found: %s", input_path)
            continue
        output_path = resolve_output_path(input_path, args.output if len(input_paths) == 1 and not args.folder else None, config)
        translate_file(input_path, output_path, backend, config)

    logger.info("Done.")


if __name__ == "__main__":
    main()
