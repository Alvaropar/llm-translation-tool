# SRT Translation Pipeline

Translates subtitle files (`.srt`) from English to Latin American Spanish using a local LLM loaded directly on GPU. Built around the professional translation style guide in `translation_guidelines.txt`.

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
python translate.py original-srt/1.srt -o translated/1_es.srt

# Override model path at runtime
python translate.py original-srt/1.srt --model-path /path/to/model

# Translate all SRTs into a directory
python translate.py original-srt/*.srt --batch-dir ./translated/

# Use a different config file
python translate.py original-srt/1.srt --config my_config.yaml
```

Output files are named `<input>_es.srt` by default (configurable via `output_suffix`).

## Configuration

All settings live in `config.yaml`:

```yaml
model_path: "/home/coder/models/Llama-3.3-70B-Instruct"
device_map: "auto"          # "auto" = GPU if available, else CPU

source_language: "English"
target_language: "Spanish"

formality: "neutral"        # neutral | formal | informal
style_notes: ""             # free-text instructions, e.g. "sports broadcast, energetic tone"
glossary:                   # terms that must always be translated a specific way
  # "NFL": "NFL"

batch_size: 30              # subtitle entries per LLM call
context_overlap: 2          # entries from previous batch kept for scene context
output_suffix: "_es"
output_dir: null            # null = same folder as input
```

### CLI overrides

| Flag | Description |
|---|---|
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
├── translate.py           # Entry point and pipeline orchestration
├── backends.py            # Model loading (int4) and LLM inference
├── srt_parser.py          # SRT parsing and writing
├── batcher.py             # Context-aware batch grouping
├── post_processor.py      # Post-translation quality checks
├── config.yaml            # All user-facing settings
├── requirements.txt
├── translation_guidelines.txt
└── original-srt/          # Input SRT files
```

## How It Works

### 1. Parsing
`srt_parser.py` reads the `.srt` file, normalises line endings and timecodes, and produces a list of `SRTBlock` objects. HTML formatting tags (`<i>`, `<b>`) are stripped before translation and restored afterwards.

### 2. Batching
`batcher.py` groups blocks into batches respecting:
- **Batch size / character limits** — avoids exceeding the model's context
- **Scene breaks** — large time gaps (default: >3s) trigger a new batch so the model doesn't carry context across unrelated scenes
- **Overlap** — the last N entries of each batch are prepended to the next one, giving the model continuity

### 3. Translation
`backends.py` loads the model once in **NF4 int4** (via `BitsAndBytesConfig`) and uses **greedy decoding** (`do_sample=False`) for fast, deterministic output.

The translation guidelines are compiled into a single **system prompt** that is built once and reused across all batches — only the subtitle entries change per call. Multi-line subtitle text is flattened to a single line before translation; the SRT player handles word-wrapping.

Key rules encoded in the system prompt (from `translation_guidelines.txt`):
- Natural, colloquial Latin American Spanish — no literal/word-for-word translation
- Pronoun register: `tú`/`usted` by context, always `ustedes` for plural
- Modern currency converted to USD; ancient/fictional currency kept as-is
- Chinese character and place names localised to Spanish equivalents
- Luxury brands replaced with generic descriptors
- Character relationship titles replaced by names after first introduction
- Conciseness: translations must fit within 2 × 28-character display lines

### 4. Post-processing
`post_processor.py` checks every translated block and logs warnings for:
- **Too long** — translation exceeds 56 chars (2 × 28-char lines)
- **Short duration** — subtitle lasts ≤1s but translation exceeds 10 chars

No content is silently dropped; warnings are for human review.

## Translation Guidelines

Full guidelines are in [`translation_guidelines.txt`](translation_guidelines.txt). They cover accuracy, natural flow, subtitle formatting, proper noun handling, currency, character names, place names, pronouns, and sensitive content.
