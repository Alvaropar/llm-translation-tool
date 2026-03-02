# SRT Translation Pipeline

Translates subtitle files (`.srt`) from English to Latin American Spanish using a local LLM loaded directly on GPU. Built around the professional translation style guide in `translation_guidelines.txt`. Automatically populates `术语表&画面字.xlsx` with extracted terminology and flagged issues.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 48GB+ VRAM for 70B models in int4)
- A downloaded HuggingFace model (see [Models](#models))

```bash
pip install -r requirements.txt
```

## Models

The pipeline loads any HuggingFace causal LM directly on GPU in **int4** (NF4 + double quantization via bitsandbytes). Set the path in `config.yaml` or pass `--model-path`.

| Model | VRAM (int4) | Notes |
|---|---|---|
| `meta-llama/Llama-3.3-70B-Instruct` | ~40GB | Recommended — best overall quality |
| `Qwen/Qwen2.5-72B-Instruct` | ~41GB | Strong multilingual alternative |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | ~24GB | Good quality on a single 24GB GPU |
| `facebook/nllb-200-3.3B` | ~13GB | Fastest; purpose-built MT, less contextual |

Download a model:
```bash
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
```

## Usage

```bash
# Translate one file (reads config.yaml)
python translate.py original-srt/1.srt

# Specify output path
python translate.py original-srt/1.srt -o translated/1.srt

# Translate all SRTs in a folder → original-srt/spanish-srt/
python translate.py --folder original-srt/

# Translate multiple files into a custom output directory
python translate.py original-srt/*.srt --batch-dir ./translated/

# Override model path at runtime
python translate.py original-srt/1.srt --model-path /path/to/model

# Use a different config file
python translate.py original-srt/1.srt --config my_config.yaml
```

### `--folder` mode

When `--folder DIR` is used, the pipeline:
- Finds all `*.srt` files inside `DIR` (sorted by name)
- Writes output to `DIR/spanish-srt/` with the **original filename unchanged** (no `_es` suffix)
- Loads the model once and processes all files sequentially

## Configuration

All settings live in `config.yaml`:

```yaml
model_path: "/home/coder/models/Llama-3.3-70B-Instruct"
device_map: "auto"          # "auto" = GPU if available, else CPU

source_language: "English"
target_language: "Spanish"

formality: "neutral"        # neutral | formal | informal
style_notes: ""             # free-text style instructions for the LLM
glossary:                   # terms that must always be translated a specific way
  # "NFL": "NFL"

batch_size: 30              # subtitle entries per LLM call
context_overlap: 2          # entries from previous batch kept for scene context
output_suffix: "_es"        # appended to filename when not using --folder
output_dir: null            # null = same folder as input

glossary_xlsx: "术语表&画面字.xlsx"  # set to null to disable Excel output
```

### CLI overrides

| Flag | Description |
|---|---|
| `--folder DIR` | Translate all `*.srt` files in DIR; output → `DIR/spanish-srt/` |
| `--model-path PATH` | Override `model_path` from config |
| `--device-map MAP` | Override `device_map` (e.g. `cuda:0`) |
| `--source-lang LANG` | Source language |
| `--target-lang LANG` | Target language |
| `--formality` | `neutral` / `formal` / `informal` |
| `--style-notes TEXT` | Free-text style instruction |
| `--batch-dir DIR` | Output directory for multi-file runs |
| `-v` | Verbose logging |

## File Structure

```
translation-tool/
├── translate.py             # Entry point and pipeline orchestration
├── backends.py              # Model loading (int4) and LLM inference
├── srt_parser.py            # SRT parsing and writing
├── batcher.py               # Context-aware batch grouping
├── post_processor.py        # Post-translation quality checks
├── glossary_extractor.py    # LLM-based terminology extraction
├── excel_writer.py          # Writes glossary + issues to Excel
├── config.yaml              # All user-facing settings
├── requirements.txt
├── translation_guidelines.txt
├── 术语表&画面字.xlsx        # Terminology + issue log (auto-updated)
└── original-srt/            # Input SRT files
    └── spanish-srt/         # Output when using --folder
```

## How It Works

### 1. Parsing
`srt_parser.py` reads the `.srt` file, normalises line endings and timecodes, and produces a list of `SRTBlock` objects. HTML formatting tags (`<i>`, `<b>`) are stripped before translation and restored afterwards. Multi-line subtitle text is flattened to a single line — the SRT player handles word-wrapping (per style guide §2-1).

### 2. Batching
`batcher.py` groups blocks into batches respecting:
- **Batch size / character limits** — avoids exceeding the model's context
- **Scene breaks** — large time gaps (default: >3s) trigger a new batch so the model doesn't carry context across unrelated scenes
- **Overlap** — the last N entries of each batch are prepended to the next, giving the model scene continuity

### 3. Translation
`backends.py` loads the model once in **NF4 int4** (via `BitsAndBytesConfig`) and uses **greedy decoding** for fast, deterministic output. A progress bar shows batch progress with subtitle index range and per-batch timing.

The translation guidelines are compiled into a single **system prompt** built once per session — only the subtitle entries change per batch call. Key rules encoded:
- Natural, colloquial Latin American Spanish — no literal/word-for-word translation
- Pronoun register: `tú`/`usted` by context, always `ustedes` for plural
- Modern currency converted to USD; ancient/fictional currency kept as-is
- Chinese character and place names localised to Spanish equivalents
- Luxury brands replaced with generic descriptors
- Character relationship titles replaced by names after first introduction
- Conciseness: translations must fit within 2 × 28-character display lines

### 4. Post-processing
`post_processor.py` checks every translated block against the style guide's mechanical rules and returns structured `SubtitleIssue` objects:
- **Too long** — translation exceeds 56 chars (2 × 28-char display lines)
- **Short duration** — subtitle lasts ≤1s but translation exceeds 10 chars

Issues are logged as warnings for human review and written to the Excel sheet.

### 5. Glossary extraction & Excel update
After each file is fully translated, `glossary_extractor.py` runs a single LLM call over all translated pairs to extract terminology into JSON:

```
person_names     → columns A–D  (人名&称呼)
org_place_names  → columns E–G  (家族名&产业名&团队&地名)
other_terms      → columns H–J  (其他&特殊物品本地化)
subtitle issues  → columns O–S  (字幕轴问题)
```

First-occurrence timecodes are found by scanning the actual SRT blocks (not the LLM), which is faster and more reliable. `excel_writer.py` then appends all entries below existing data, preserving the sheet's formatting and any manually entered rows. The `画面字` section (K–N) is left for manual entry as it requires visual inspection of the video.

Set `glossary_xlsx: null` in `config.yaml` to skip the Excel step entirely.

## Translation Guidelines

Full guidelines are in [`translation_guidelines.txt`](translation_guidelines.txt). They cover accuracy, natural flow, subtitle formatting, proper noun handling, currency, character names, place names, pronouns, and sensitive content.
