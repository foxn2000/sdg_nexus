# SDG (Scalable Data Generator) Usage Guide

This document explains how to run SDG pipelines using existing YAML files (MABEL format).

## Table of Contents

1. [Overview](#overview)
2. [CLI Usage](#cli-usage)
3. [Python API Usage](#python-api-usage)
4. [Advanced Optimization](#advanced-optimization)
5. [Parsers](#parsers)
6. [Input/Output Data Formats](#inputoutput-data-formats)

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

## Advanced Optimization

SDG Nexus provides advanced optimization features for high-throughput LLM inference workloads. These features are particularly useful when working with vLLM or SGLang backends.

### Adaptive Concurrency Control

The [`AdaptiveController`](../sdg/adaptive/controller.py:28) automatically adjusts concurrency levels based on observed latencies and backend metrics using an AIMD (Additive Increase Multiplicative Decrease) algorithm.

**Key Features:**
- Automatically increases concurrency when latencies are low
- Quickly decreases concurrency when errors occur or latencies spike
- Monitors backend queue depth and cache usage (vLLM/SGLang)
- Maintains target latency bounds

**Basic Usage:**

```python
from sdg.adaptive.controller import AdaptiveController

controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=64,
    target_latency_ms=2000.0,  # Target P95 latency
    target_queue_depth=32,
)

# Get current concurrency limit
limit = controller.current_concurrency

# Record completed request
controller.record_latency(latency_ms=150.0, is_error=False)

# Get statistics
stats = controller.get_stats()
print(f"Current concurrency: {stats['current_concurrency']}")
print(f"P95 latency: {stats['p95_latency_ms']}ms")
```

**Advanced Configuration:**

```python
controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=128,
    target_latency_ms=2000.0,
    target_queue_depth=32,
    # AIMD parameters
    increase_step=2,              # Additive increase per cycle
    decrease_factor=0.5,          # Multiplicative decrease (50%)
    # Sensitivity
    latency_tolerance=1.5,        # Decrease if latency > target * 1.5
    error_rate_threshold=0.05,    # Decrease if error rate > 5%
    # Timing
    adjustment_interval_ms=1000,  # Adjust every 1 second
    window_size=50,               # Track last 50 samples
)
```

### Metrics Collection

The [`MetricsCollector`](../sdg/adaptive/metrics.py:75) polls backend metrics endpoints to gather real-time information about queue depth, cache usage, and throughput.

**Supported Backends:**
- **vLLM**: Prometheus metrics at `/metrics` endpoint
- **SGLang**: Prometheus metrics at `/metrics` endpoint

**Basic Usage:**

```python
from sdg.adaptive.metrics import MetricsCollector, MetricsType

collector = MetricsCollector(
    base_url="http://localhost:8000",
    metrics_type=MetricsType.VLLM,
    poll_interval_ms=500,
)

await collector.start()

# Get latest metrics
metrics = collector.get_latest()
if metrics and metrics.is_valid:
    print(f"Queue depth: {metrics.queue_depth}")
    print(f"Cache usage: {metrics.cache_usage_percent}%")
    print(f"Running requests: {metrics.num_requests_running}")

await collector.stop()
```

**Available Metrics:**

| Metric | Description | vLLM | SGLang |
|--------|-------------|------|--------|
| `num_requests_waiting` | Requests in queue | ✓ | ✓ |
| `num_requests_running` | Currently processing | ✓ | ✓ |
| `queue_depth` | Total queue depth (waiting + running) | ✓ | ✓ |
| `cache_usage_percent` | KV cache utilization | ✓ | ✓ |
| `prompt_tokens_total` | Total input tokens | ✓ | ✓ |
| `generation_tokens_total` | Total output tokens | ✓ | ✓ |

### Request Batching

The [`RequestBatcher`](../sdg/adaptive/batcher.py:38) groups multiple requests together before submission to maximize continuous batching benefits.

**Key Features:**
- Dynamic batch sizing based on queue state
- Maximum wait time to ensure latency bounds
- Token-aware batching (optional)
- Priority queue support

**Basic Usage:**

```python
from sdg.adaptive.batcher import RequestBatcher

async def batch_processor(payloads):
    # Your batch processing logic
    results = await client.batch_chat(payloads)
    return results

batcher = RequestBatcher(
    batch_processor=batch_processor,
    max_batch_size=64,
    max_wait_ms=50,
)

async with batcher:
    # Submit requests
    result = await batcher.submit({
        "messages": [{"role": "user", "content": "Hello"}]
    })
```

**Advanced Configuration:**

```python
batcher = RequestBatcher(
    batch_processor=batch_processor,
    max_batch_size=64,
    max_wait_ms=50,
    max_tokens_per_batch=8192,  # Limit total tokens
    token_estimator=custom_estimator,  # Custom token counter
    enabled=True,
)

# Get statistics
stats = batcher.get_stats()
print(f"Average batch size: {stats['avg_batch_size']}")
print(f"Pending requests: {stats['pending_count']}")
```

### Integrated Optimization

The [`AdaptiveConcurrencyManager`](../sdg/adaptive/controller.py:293) combines all optimization features for turnkey high-performance inference.

**Full Example:**

```python
from sdg.adaptive.controller import AdaptiveConcurrencyManager
from sdg.adaptive.metrics import MetricsType

async def main():
    manager = AdaptiveConcurrencyManager(
        base_url="http://localhost:8000",
        metrics_type=MetricsType.VLLM,
        min_concurrency=1,
        max_concurrency=64,
        target_latency_ms=2000.0,
        target_queue_depth=32,
        enabled=True,
    )
    
    async with manager:
        # Execute requests with automatic concurrency control
        async with manager.acquire():
            start = time.time()
            result = await execute_request()
            latency = (time.time() - start) * 1000
            manager.record_latency(latency)
        
        # Monitor statistics
        stats = manager.get_stats()
        print(f"Current concurrency: {stats['current_concurrency']}")
        print(f"Backend queue: {stats.get('backend_queue_depth', 'N/A')}")

asyncio.run(main())
```

**CLI Integration:**

The optimization features are automatically enabled when using batch mode:

```bash
# Adaptive concurrency with vLLM backend
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --batch-mode \
  --max-batch 64 \
  --min-batch 1 \
  --target-latency 2000
```

### Multi-Backend Support

For load-balanced deployments with multiple backend instances:

```python
from sdg.adaptive.metrics import MultiBackendMetricsCollector, MetricsType

collector = MultiBackendMetricsCollector(
    backends={
        "http://backend1:8000": MetricsType.VLLM,
        "http://backend2:8000": MetricsType.VLLM,
        "http://backend3:8000": MetricsType.VLLM,
    },
    poll_interval_ms=500,
)

await collector.start()

# Get aggregated metrics across all backends
metrics = collector.get_aggregated_metrics()
print(f"Total queue depth: {collector.get_total_queue_depth()}")

await collector.stop()
```

### Best Practices

1. **Start Conservative**: Begin with lower concurrency limits and let the controller increase it automatically
2. **Monitor Metrics**: Use [`get_stats()`](../sdg/adaptive/controller.py:422) to track performance
3. **Tune for Your Workload**: Adjust `target_latency_ms` based on your requirements
4. **Backend Metrics**: Enable metrics collection for vLLM/SGLang for better optimization
5. **Batching**: Use request batching for workloads with predictable request patterns

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