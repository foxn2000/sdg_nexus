# SDG (Scalable Data Generator) Usage Guide

This document explains how to run SDG pipelines using existing YAML files (MABEL format).

## Table of Contents

1. [Overview](#overview)
2. [CLI Usage](#cli-usage)
3. [Python API Usage](#python-api-usage)
4. [Parsers](#parsers)
5. [Input/Output Data Formats](#inputoutput-data-formats)

---

## Overview

SDG is a tool that processes input datasets according to pipelines defined in YAML (MABEL format), combining LLM calls and Python code execution to generate output data.

Basic workflow:
1. Prepare a YAML blueprint file
2. Prepare an input dataset (JSONL or CSV)
3. Execute the pipeline via CLI or Python API
4. Retrieve the output data (JSONL)

---

## CLI Usage

### Basic Command

```bash
# Basic format
sdg run --yaml <YAML file> --input <input file> --output <output file>

# Example
sdg run --yaml examples/sdg_demo.yaml --input examples/data/input.jsonl --output output/result.jsonl
```

### Execution Modes

SDG has two execution modes:

#### 1. Streaming Mode (Default)

Processes each data row in parallel and writes to the output file as soon as each row completes.

```bash
# Streaming mode (default)
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl

# Specify concurrency
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --max-concurrent 16

# Disable progress display
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --no-progress
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--max-concurrent` | 8 | Maximum number of concurrent rows |
| `--no-progress` | false | Disable progress display |

**Features:**
- Intermediate results are less likely to be lost (real-time writing)
- Memory efficient
- Output order is completion order (may differ from input order)

> **Note:** If you need the original order, sort by the `_row_index` field in the output.

#### 2. Batch Mode

Processes data in blocks. Enable with the `--batch-mode` flag.

```bash
# Enable batch mode
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --batch-mode

# Specify batch sizes
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --batch-mode --max-batch 16 --min-batch 2 --target-latency-ms 5000
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--batch-mode` | false | Enable batch mode |
| `--max-batch` | 8 | Maximum concurrent requests per block |
| `--min-batch` | 1 | Minimum concurrent requests per block |
| `--target-latency-ms` | 3000 | Target average latency (milliseconds) |

### Common Options

| Option | Description |
|--------|-------------|
| `--save-intermediate` | Save intermediate results |

### Legacy Mode

For backward compatibility, execution without subcommands is also supported:

```bash
# Legacy format (backward compatible)
sdg --yaml pipeline.yaml --input data.jsonl --output result.jsonl
```

---

## Python API Usage

### Basic Usage

```python
import asyncio
from sdg.config import load_config
from sdg.executors import run_pipeline

async def main():
    # 1. Load configuration
    cfg = load_config("examples/sdg_demo.yaml")
    
    # 2. Prepare dataset
    dataset = [
        {"UserInput": "What is AI?"},
        {"UserInput": "Explain machine learning"},
    ]
    
    # 3. Execute pipeline
    results = await run_pipeline(
        cfg,
        dataset,
        max_batch=4,
        min_batch=1,
        target_latency_ms=3000,
        save_intermediate=False,
    )
    
    # 4. Process results
    for result in results:
        print(result)

asyncio.run(main())
```

### Streaming Execution

```python
from sdg.runner import run_streaming

# Execute in streaming mode (synchronous function)
run_streaming(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=8,
    save_intermediate=False,
    show_progress=True,
)
```

### Batch Execution

```python
from sdg.runner import run

# Execute in batch mode (synchronous function)
run(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_batch=8,
    min_batch=1,
    target_latency_ms=3000,
    save_intermediate=False,
)
```

### Loading and Processing JSONL Files

```python
import asyncio
import json
from sdg.config import load_config
from sdg.executors import run_pipeline

def load_jsonl(file_path: str) -> list:
    """Load JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: list, file_path: str) -> None:
    """Save to JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

async def main():
    cfg = load_config("examples/sdg_demo.yaml")
    dataset = load_jsonl("examples/data/input.jsonl")
    
    results = await run_pipeline(cfg, dataset)
    
    save_jsonl(results, "output/result.jsonl")

asyncio.run(main())
```

### Inspecting Configuration

```python
from sdg.config import load_config

cfg = load_config("examples/sdg_demo_v2.yaml")

# Check version
print(f"MABEL version: {cfg.get_version()}")  # "1.0" or "2.0"
print(f"Is v2: {cfg.is_v2()}")  # True or False

# Global variables (v2 only)
print(f"Constants: {cfg.globals_.const}")
print(f"Variables: {cfg.globals_.vars}")

# Model information
for m in cfg.models:
    print(f"Model: {m.name} -> {m.api_model}")

# Block list
for b in cfg.blocks:
    print(f"exec={b.exec}, type={b.type}, name={b.name or '(unnamed)'}")
```

---

## Parsers

SDG provides parsers for processing LLM outputs and input data.

### AI Output Parsers (`select` Option)

Used in the `outputs` section of AI blocks. Parses response text from LLMs to extract required parts.

```yaml
outputs:
  - name: FullResponse
    select: full          # Get entire response

  - name: ExtractedTag
    select: tag           # Extract content within specific tags
    tag: answer           # Content of <answer>...</answer>

  - name: FirstLine
    select: regex         # Extract matched portions using regex
    regex: "^(.+?)$"      # Regex pattern

  - name: JsonField
    select: jsonpath      # Extract value using JSONPath expression (v2)
    path: "$.result.value"
```

| select value | Description | Required options |
|-------------|-------------|------------------|
| `full` | Get entire response (default) | None |
| `tag` | Extract content within XML tags | `tag` |
| `regex` | Extract matched portions using regex | `regex` |
| `jsonpath` | Extract value using JSONPath (v2) | `path` |

### Logic Parsers (`parse` Option)

Used in `for` loops within `logic` blocks. Converts text data into lists.

```yaml
- type: logic
  exec: 2
  op: for
  list: "{TextData}"
  parse: lines           # Specify parse method
  var: item
  outputs:
    - name: ProcessedItems
      from: list
```

| parse value | Description | Example |
|-------------|-------------|---------|
| `lines` | Split by newlines | `"a\nb\nc"` → `["a", "b", "c"]` |
| `csv` | Parse as CSV | `"a,b,c"` → `["a", "b", "c"]` |
| `json` | Parse as JSON | `"[1,2,3]"` → `[1, 2, 3]` |
| `regex` | Extract matches using regex | Results depend on pattern |

```yaml
# Regex parsing example
- type: logic
  op: for
  list: "{TextData}"
  parse: regex
  regex_pattern: "\\d+"   # Extract all numbers
  var: number
  outputs:
    - name: Numbers
      from: list
```

### Additional Options

```yaml
- type: logic
  op: for
  list: "{TextData}"
  parse: lines
  drop_empty: true        # Exclude empty lines
  where:                  # Filter condition (MEX expression)
    ne: ["{item}", ""]
  map: "{item} processed" # Mapping (transformation)
  var: item
```

---

## Input/Output Data Formats

### Input Data Formats

SDG supports the following input formats:

#### JSONL Format (Recommended)

```jsonl
{"UserInput": "What is AI?", "Category": "tech"}
{"UserInput": "Tell me about the weather", "Category": "general"}
```

#### CSV Format

```csv
UserInput,Category
What is AI?,tech
Tell me about the weather,general
```

### Output Data Format

Output is always in JSONL format. The `final` fields defined in the `end` block of the pipeline are output.

```jsonl
{"answer": "AI is...", "status": "success", "_row_index": 0}
{"answer": "Weather is...", "status": "success", "_row_index": 1}
```

> **Note:** `_row_index` indicates the original row number in the input data. In streaming mode, output order may differ from input order, so you can use this field to restore the original order.

---

## Examples

### Example 1: Simple Q&A Pipeline

```bash
# Using examples/sdg_demo.yaml
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output output/qa_result.jsonl
```

### Example 2: Pipeline Using v2 Features

```bash
# Using examples/sdg_demo_v2.yaml (global variables, MEX expressions, while loops, etc.)
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/v2_result.jsonl \
  --max-concurrent 4
```

### Example 3: Processing Large Datasets

```bash
# Stream processing large datasets
sdg run \
  --yaml pipeline.yaml \
  --input large_dataset.jsonl \
  --output output/large_result.jsonl \
  --max-concurrent 16
```

---

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Check the `api_key` field in the YAML file
   - When using environment variables, set it as `api_key: "${OPENAI_API_KEY}"`

2. **Input File Not Found**
   - Verify the file path is correct
   - Relative paths are relative to the execution directory

3. **Output Order Differs from Input**
   - In streaming mode, output is in completion order
   - Sort by the `_row_index` field to restore the original order

4. **Out of Memory**
   - Reduce the `--max-concurrent` value
   - Use streaming mode (default)