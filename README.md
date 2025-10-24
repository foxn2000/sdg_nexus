# SDG‑Nexus (Scalable Data Generator — Nexus)

[日本語 README](README.JA.md)

SDG‑Nexus is a CLI tool and Python library for scalable, YAML‑driven data generation and transformation with Large Language Models (LLMs). It executes AI agent blueprints over JSONL/CSV datasets and talks to any OpenAI‑compatible endpoint (OpenAI, vLLM, SGLang, etc.) via the `/v1/chat/completions` API. An adaptive concurrency controller automatically tunes parallelism to balance throughput and stability.

Note: This implementation supports a subset of the YAML spec compatible with MABEL Studio v1.1 (models / blocks / connections, etc.). See the example and schema notes below.

## Highlights

- YAML blueprint loading and validation (focused on `models` and `blocks`)
- Block engine for `ai`, `logic`, `python`, and `end`
- Works with OpenAI‑compatible Chat Completions (`/v1/chat/completions`)
- Adaptive concurrency: adjusts batch size based on latency and error rate
- Processes JSONL/CSV inputs record‑by‑record and saves final outputs as JSONL
- Optional intermediate outputs for debugging (`--save-intermediate`)

## Installation

Requirements: Python >= 3.10

Option A: Local editable install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Option B: Standard install (no local editing)
```bash
pip install -U pip
pip install -r requirements.txt
pip install .
```

The CLI entry point `sdg` will be installed via `pyproject.toml`.

## Quickstart

1) Set credentials

- Using OpenAI:
  - Export `OPENAI_API_KEY` in your shell.
- Using an OpenAI‑compatible server (vLLM/SGLang/etc.):
  - Set `models[].base_url` to your server URL.
  - Provide an API key via `models[].api_key`. If you write `${ENV.SOMETHING}`, the current implementation ignores the variable name and reads `os.environ['OPENAI_API_KEY']`.

2) Run the example
```bash
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output out/output.jsonl \
  --max-batch 8 --min-batch 1
```

- The example reads JSONL records (each having `UserInput`), executes an `ai -> ai -> logic -> python -> end` pipeline, and writes the final JSONL to `out/output.jsonl`.

3) CSV input (alternative)
```bash
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.csv \
  --output out/out.jsonl
```

## CLI reference

Subcommand style (preferred):
```text
sdg run --yaml PATH --input PATH --output PATH [options]
```

Options:
- `--yaml` (required): YAML blueprint path
- `--input` (required): Input dataset (.jsonl or .csv)
- `--output` (required): Output JSONL file
- `--max-batch` (int, default 8): Max concurrent requests
- `--min-batch` (int, default 1): Min concurrent requests
- `--target-latency-ms` (int, default 3000): Target average latency per request
- `--save-intermediate` (flag): Save intermediate outputs for debugging

Legacy mode is also supported for backward compatibility:
```bash
sdg --yaml ... --input ... --output ...
```

## Input and output formats

- JSONL input: one JSON object per line. Each key becomes a variable available to templates in subsequent blocks.
  - Example line: `{"UserInput": "What is the capital of Japan?"}`

- CSV input: header row is used as field names; each row is converted to a dict with string values.

- Output: by default, for each input record SDG‑Nexus writes a single JSON line composed of fields specified in the `end` block’s `final` list.

- Intermediate outputs: when `--save-intermediate` is given, SDG‑Nexus also stores block‑level values in the per‑record result as keys named `_{exec}_{name}` (e.g., `_2_ShortAnswer`).

## YAML blueprint (supported subset)

Top level:
```yaml
mabel: { version: "1.0" }     # optional
models: [ ... ]               # required
blocks:  [ ... ]              # required
connections: [ ... ]          # parsed, reserved for future use
```

### models

A model definition is referenced by name from `ai` blocks.

Fields:
- `name` (str): Identifier used by blocks.
- `api_model` (str): The model name for the backend (e.g., `gpt-4o-mini`, `phi4-mini`, etc.).
- `api_key` (str): API key literal or `${ENV.SOMETHING}`. Note: current implementation ignores the variable name and reads `OPENAI_API_KEY` from environment when this form is used.
- `base_url` (str, optional): Base URL; `/v1` is appended automatically if missing (e.g., `http://127.0.0.1:8000` becomes `http://127.0.0.1:8000/v1`).
- `organization` (str, optional): OpenAI organization ID (if applicable).
- `headers` (map, optional): Extra vendor‑specific headers (the SDK manages standard headers).
- `request_defaults` (map, optional): Default request parameters merged into each `ai` call.
  - Common keys: `temperature`, `top_p`, `max_tokens`, `timeout_sec`, `retry`
  - `retry` structure:
    ```yaml
    retry:
      max_attempts: 3
      backoff:
        initial_ms: 250
        factor: 2.0
    ```

Example:
```yaml
models:
  - name: writer
    api_model: gpt-4o-mini
    api_key: "${ENV.OPENAI_API_KEY}"
    base_url: https://api.openai.com
    headers:
      HTTP-Referer: "https://your-app.example"
    request_defaults:
      temperature: 0.2
      top_p: 0.95
      max_tokens: 400
      timeout_sec: 60
      retry:
        max_attempts: 3
        backoff: { initial_ms: 250, factor: 2.0 }
```

Parameter precedence: `request_defaults` (model) are merged with `params` (block); block `params` take priority.

### blocks

Blocks are executed in ascending order of `exec`. Each block supports:
- `exec` (int): Execution order
- `run_if` (object or JSON string, optional): Conditional; if false, the block is skipped for that record
- `on_error` (`fail` | `continue`, default `fail`): Whether to continue pipeline when the block fails

Supported block types:

#### ai

Fields:
- `type: ai`
- `model` (str): Name of a model in `models`
- `system_prompt` (str, optional): Rendered with the record context
- `prompts` (list[str]): Rendered and joined into a user message
- `outputs` (list[OutputDef], optional): How to extract fields from the LLM response
  - `name` (str): Output variable name
  - `select` (`full` | `tag` | `regex`): Extraction mode
  - `tag` (str, required for `tag`): Capture content inside `<tag> ... </tag>` (case‑insensitive, multiline)
  - `regex` (str, required for `regex`): Python regex; if it has a capture group `( ... )`, that group is returned; otherwise the full match
  - `join_with` (str, optional): If multiple values are extracted, join them into one string
- `params` (map, optional): Overrides for per‑request parameters (e.g., `temperature`, `max_tokens`, `timeout_sec`, `retry`)

Default output when `outputs` is omitted:
```yaml
outputs:
  - name: full
    select: full
```

#### logic

Fields:
- `type: logic`
- `name` (str, optional)
- `op` (`if` | `and` | `or` | `not` | `for`)
- `cond` (object): For `if`, a JSON object supporting:
  - `equals`, `not_equals`, `contains`, `is_empty`
  - `gt`, `lt`, `gte`, `lte`
  - logical composition: `and`, `or`, `not`
  - All operands are rendered as templates before comparison.
- `then`, `else` (str, optional): Text outputs for `if`
- `operands` (list): For `and`/`or`/`not`, list of conditions
- For loops (`op: for`):
  - `list` (str): Name of the list (or comma‑separated string) in the context
  - `parse` (str, optional: `"regex"` for regex parse; defaults to split by comma)
  - `regex_pattern` (str, when `parse: regex`)
  - `var` (str, optional): Loop variable name, default `"item"`
  - `drop_empty` (bool, optional): Filter empty items
  - `where` (object): Condition applied per item (same operators as `cond`)
  - `map` (str): Template applied to each item (e.g., `"* {item}"`)
- `outputs` (list of maps): How to expose results to the context, per op:
  - For `if`:  
    - `name`: output key  
    - `from`: `boolean` | `text` | `source`  
      - `boolean` → the evaluated truth value  
      - `text` → `then`/`else` output  
      - `source` → copy a named field from the context (`source: FieldName`)
  - For `and`/`or`/`not`:  
    - `name`: output key, value is the boolean result
  - For `for`:  
    - `name`: output key  
    - `join_with` (str, optional): Join list into a string  
    - `limit` (int, optional), `offset` (int, optional)

#### python

Fields:
- `type: python`
- `name` (str, optional)
- `function` (str, required): Name of a function inside `code_path`
- `inputs` (list[str]): Names of fields from context to pass as positional args
- `code_path` (str): Python file path to load dynamically
- `venv_path` (ignored): Use the active Python environment instead
- `outputs` (list[str]): Names to map returned values into the context  
  - Function return shape:
    - dict → pick keys specified in `outputs`
    - non‑dict → mapped positionally to names in `outputs` (single value or list/tuple)

#### end

Fields:
- `type: end`
- `final` (list of `{ name, value }`): Templates evaluated and written to the final JSONL
- `exit_code` (str, optional)
- `reason` (str, optional)

### Templates and extraction helpers

- Templates: any string may reference `{VarName}` or dotted keys like `{foo.bar}`; missing keys resolve to empty strings.
- Tag extraction: `<tag> ... </tag>` (case‑insensitive, multiline).
- Regex extraction: if a capturing group exists, group(1) is returned; otherwise the full match.

## Adaptive concurrency (batch optimizer)

At each `ai` round that runs across multiple records, SDG‑Nexus measures the average latency and error count, then adjusts the concurrency:
- If errors occur or the average latency exceeds `--target-latency-ms`, it reduces concurrency down to `--min-batch`.
- If requests are stable and fast, it increases concurrency up to `--max-batch`.

This increases throughput while being resilient to rate limits and server slowdowns.

## Backends (OpenAI‑compatible)

- Uses the OpenAI Python SDK (>= 1.40) with a custom `base_url`.
- If your `base_url` does not end with `/v1`, SDG‑Nexus appends `/v1` automatically.
- Extra vendor headers can be set via `models[].headers`.

Examples:
- OpenAI: `base_url: https://api.openai.com`
- vLLM proxy: `base_url: http://127.0.0.1:8000`
- SGLang: `base_url: http://127.0.0.1:30000`

Note: This project optimizes concurrency with standard HTTP chat calls; it does not use the OpenAI Batches API.

## Programmatic usage (Python)

```python
from sdg.runner import run

run(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="out/output.jsonl",
    max_batch=8,
    min_batch=1,
    target_latency_ms=3000,
    save_intermediate=False,
)
```

## Example blueprint

See `examples/sdg_demo.yaml` for a complete, runnable pipeline demonstrating:
- Two `ai` blocks (`planner` then `writer`)
- A `logic` guard checking the presence of a short answer
- A `python` block (`examples/helpers.py`) to post‑process the answer
- An `end` block that selects the final fields

## Error handling

- Block-level errors:
  - Default behavior is `on_error: fail`, which stops the pipeline and raises the exception.
  - If you set `on_error: continue`, the pipeline proceeds and the record context receives an error string under the key `error_block_{exec}` (e.g., `error_block_2`).
- Conditional execution:
  - If a block's `run_if` evaluates to false for a given record, that block is skipped for that record.

## Troubleshooting

- 401 Unauthorized: Check `OPENAI_API_KEY` or your server/API key configuration.
- Rate limits / slowdowns: lower `--max-batch` or the model `request_defaults.max_tokens`, raise `--target-latency-ms`.
- Empty outputs:
  - For `regex`, ensure the pattern and capture groups are correct.
  - For `tag`, confirm your LLM output actually wraps content in the chosen tag.
- CSV inputs are strings; cast to numbers in `python` blocks if needed.

## License

MIT
