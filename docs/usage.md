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

# Using Hugging Face Datasets
sdg run --yaml examples/sdg_demo.yaml --dataset squad --split validation --output output/result.jsonl
```

### Execution Modes

SDG has two execution modes for streaming data processing:

#### 1. Streaming Mode with Fixed Concurrency (Default)

Processes each data row in parallel with a fixed concurrency level and writes to the output file as soon as each row completes.

```bash
# Streaming mode (default)
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl

# Specify fixed concurrency
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --max-concurrent 16

# Disable progress display
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl --no-progress
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--max-concurrent` | 8 | Maximum number of concurrent rows (fixed) |
| `--no-progress` | false | Disable progress display |
| `--dataset` | - | Hugging Face dataset name |
| `--subset` | - | Dataset subset name |
| `--split` | train | Dataset split |
| `--mapping` | - | Key mapping (`orig:new` format, can be used multiple times) |

**Features:**
- Fixed concurrency level throughout execution
- Intermediate results are less likely to be lost (real-time writing)
- Memory efficient
- Output order is completion order (may differ from input order)

> **Note:** If you need the original order, sort by the `_row_index` field in the output.

#### 2. Streaming Mode with Adaptive Concurrency

Dynamically adjusts concurrency based on observed latencies and optional backend metrics. Enable with the `--adaptive` flag.

```bash
# Enable adaptive concurrency control
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --min-batch 1 --max-batch 32 --target-latency-ms 2000

# With vLLM backend metrics
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics --min-batch 1 --max-batch 64

# With request batching for higher throughput
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics --enable-request-batching
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--adaptive` | false | Enable adaptive concurrency control |
| `--min-batch` | 1 | Minimum concurrency level |
| `--max-batch` | 64 | Maximum concurrency level |
| `--target-latency-ms` | 3000 | Target P95 latency (milliseconds) |
| `--target-queue-depth` | 32 | Target backend queue depth |
| `--use-vllm-metrics` | false | Use vLLM Prometheus metrics |
| `--use-sglang-metrics` | false | Use SGLang Prometheus metrics |
| `--enable-request-batching` | false | Enable request batching |
| `--max-batch-size` | 32 | Max requests per batch |
| `--max-wait-ms` | 50 | Max wait time for batch formation (ms) |

**Features:**
- Automatically adjusts concurrency between `--min-batch` and `--max-batch`
- Increases concurrency when latencies are low
- Decreases concurrency when errors occur or latencies spike
- Monitors backend metrics (vLLM/SGLang) for better optimization
- Optional request batching for maximum throughput

**How It Works:**
The adaptive controller uses an algorithm inspired by TCP congestion control (Vegas/Reno/BBR):
- **Slow Start**: Doubles concurrency when P95 latency < target × 0.7 (exponential increase)
- **Congestion Avoidance**: Linear increase (+2) after reaching ssthresh
- **Graduated Decrease**: 15%-50% reduction based on congestion severity (mild congestion ignored)
- **EMA Smoothing**: Filters noise and detects trends
- **Monitoring**: Polls backend metrics every 500ms (if enabled)
- **Adjustment**: Evaluates every 1 second based on last 50 samples

### Common Options

| Option | Description |
|--------|-------------|
| `--save-intermediate` | Save intermediate results |

### Optimization Options

SDG provides optimization options for improved performance and resource efficiency:

| Option | Default | Description |
|--------|---------|-------------|
| `--use-shared-transport` | false | Use shared HTTP transport (connection pooling) |
| `--no-http2` | false | Disable HTTP/2 (enabled by default) |

**Optimization Options Details:**

- **`--use-shared-transport`**: Shares HTTP connection pools across multiple requests. This reduces the overhead of establishing new connections and improves performance, especially when processing many short requests.

- **`--no-http2`**: By default, HTTP/2 is enabled. Use this flag to fall back to HTTP/1.1 if you have compatibility issues with certain backends or proxies.

**Usage Examples:**

```bash
# Use shared transport for better connection efficiency
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --use-shared-transport

# Disable HTTP/2 to use HTTP/1.1
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --no-http2

# Combine adaptive concurrency control with optimization options
sdg run --yaml pipeline.yaml --input data.jsonl --output result.jsonl \
  --adaptive --use-vllm-metrics \
  --use-shared-transport
```

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

# Streaming execution with optimization options
run_streaming(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=8,
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,  # Use shared HTTP transport
    http2=True,                  # Enable HTTP/2 (default)
)
```

### Adaptive Concurrency Execution

```python
from sdg.runner import run_streaming_adaptive

# Execute with adaptive concurrency control
run_streaming_adaptive(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=64,
    min_concurrent=1,
    target_latency_ms=2000,
    target_queue_depth=32,
    metrics_type="vllm",  # "none", "vllm", "sglang"
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,
    http2=True,
)
```

### Request Batching Execution

```python
from sdg.runner import run_streaming_adaptive_batched

# Execute with request batching enabled
run_streaming_adaptive_batched(
    yaml_path="examples/sdg_demo.yaml",
    input_path="examples/data/input.jsonl",
    output_path="output/result.jsonl",
    max_concurrent=64,
    min_concurrent=1,
    target_latency_ms=2000,
    target_queue_depth=32,
    metrics_type="vllm",
    max_batch_size=32,
    max_wait_ms=50,
    save_intermediate=False,
    show_progress=True,
    use_shared_transport=True,
    http2=True,
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

The [`AdaptiveController`](../sdg/adaptive/controller.py:286) implements advanced control logic inspired by TCP congestion control algorithms (Vegas, Reno, BBR) to automatically adjust concurrency levels based on observed latencies and backend metrics.

**Key Features:**
- **Control Phases**: Two-phase control with Slow Start (exponential increase) and Congestion Avoidance (linear increase)
- **EMA-based Smoothing**: Noise reduction and trend detection using Exponential Moving Average
- **Vegas-style Congestion Detection**: Proactive RTT-based congestion detection
- **Graduated Decrease Logic**: Immediate decrease for errors, gentle decrease for latency issues
- Monitors backend queue depth and cache usage (vLLM/SGLang)
- Maintains target latency bounds

**Algorithm Details:**

1. **Slow Start Phase**:
   - When concurrency is below ssthresh (slow start threshold) and latency is below 70% of target
   - Exponentially increases concurrency (doubles)
   - Quickly converges to optimal concurrency

2. **Congestion Avoidance Phase**:
   - Linear increase (Additive Increase) after reaching ssthresh
   - Cautiously explores the upper limit

3. **Graduated Decrease Logic**:
   - On errors: Immediate Multiplicative Decrease
   - Severe congestion: Multiplicative decrease (50%)
   - Moderate congestion: Gentle decrease (15%)
   - Mild congestion: Linear decrease only after 3 consecutive detections

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

# Get current control phase
phase = controller.phase  # ControlPhase.SLOW_START or ControlPhase.CONGESTION_AVOIDANCE

# Record completed request
controller.record_latency(latency_ms=150.0, is_error=False)

# Get statistics
stats = controller.get_stats()
print(f"Current concurrency: {stats['current_concurrency']}")
print(f"P95 latency: {stats['p95_latency_ms']}ms")
print(f"Control phase: {stats['phase']}")
print(f"EMA latency: {stats['ema_latency_ms']}ms")
print(f"Congestion signal: {stats['vegas_congestion_signal']}")
```

**Advanced Configuration:**

```python
controller = AdaptiveController(
    min_concurrency=1,
    max_concurrency=128,
    target_latency_ms=2000.0,
    target_queue_depth=32,
    # Basic AIMD parameters
    increase_step=2,              # Additive increase per cycle in CA phase
    decrease_factor=0.5,          # Multiplicative decrease for errors (50%)
    # Sensitivity
    latency_tolerance=1.5,        # Decrease if latency > target * 1.5
    error_rate_threshold=0.05,    # Decrease if error rate > 5%
    # Timing
    adjustment_interval_ms=1000,  # Adjust every 1 second
    window_size=50,               # Track last 50 samples
    # Advanced parameters
    ema_alpha=0.3,                # EMA smoothing factor (0-1, higher = more reactive)
    slow_start_threshold=32,      # Initial ssthresh
    vegas_alpha=2.0,              # Vegas lower threshold
    vegas_beta=4.0,               # Vegas upper threshold
    mild_decrease_factor=0.85,    # Decrease factor for mild congestion (15% reduction)
    trend_sensitivity=0.1,        # Trend detection sensitivity
)
```

**Getting EMA and Congestion Statistics:**

```python
# EMA statistics (noise-filtered latency)
ema_stats = controller.get_ema_stats()
print(f"Smoothed latency: {ema_stats['latency']['value']}ms")
print(f"Latency trend: {ema_stats['latency']['trend']}")  # positive=increasing, negative=decreasing
print(f"Latency variance: {ema_stats['latency']['variance']}")

# Congestion detection statistics (Vegas-style)
congestion_stats = controller.get_congestion_stats()
print(f"Base latency: {congestion_stats['base_latency_ms']}ms")
print(f"Congestion level: {congestion_stats['congestion_level']}")  # none/mild/moderate/severe
print(f"Congestion signal: {congestion_stats['congestion_signal']}")
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

Enable adaptive concurrency control in streaming mode with the `--adaptive` flag:

```bash
# Basic adaptive concurrency
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --min-batch 1 \
  --max-batch 32 \
  --target-latency-ms 2000

# With vLLM backend metrics
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64

# With request batching
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --enable-request-batching \
  --max-batch-size 32
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

## Phase 2 Optimization

SDG Nexus Phase 2 introduces advanced optimizations for scalability and memory efficiency. These features are **opt-in** and disabled by default, ensuring backward compatibility.

### Hierarchical Task Scheduling

Efficiently processes large datasets by dividing data into chunks and progressively generating/executing tasks.

**Features:**
- **Chunk-based data division**: Splits datasets into manageable chunks
- **Limited pending tasks**: Controls memory usage by limiting queued tasks
- **Minimized startup delay**: Processes begin immediately without generating all tasks upfront
- **Efficient coordination**: Uses Python `Condition` for cooperative control

**CLI Usage:**

```bash
# Enable hierarchical scheduling
sdg run \
  --yaml pipeline.yaml \
  --input large_data.jsonl \
  --output result.jsonl \
  --enable-scheduling \
  --max-pending-tasks 100 \
  --chunk-size 50
```

**Python API Usage:**

```python
import asyncio
from sdg.config import load_config
from sdg.executors import run_pipeline_streaming

async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i, "text": f"sample_{i}"} for i in range(10000)]
    
    # Enable hierarchical scheduling
    async for result in run_pipeline_streaming(
        cfg,
        dataset,
        max_concurrent=16,
        enable_scheduling=True,          # Enable scheduling
        max_pending_tasks=100,            # Max pending tasks
        chunk_size=50,                    # Chunk size
    ):
        if result.error:
            print(f"Error in row {result.row_index}: {result.error}")
        else:
            print(f"Completed row {result.row_index}")

asyncio.run(main())
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_scheduling` | `False` | Enable hierarchical scheduling |
| `max_pending_tasks` | `1000` | Maximum number of pending tasks (memory control) |
| `chunk_size` | `100` | Dataset chunk size |

### Memory Optimization

#### Streaming Context Manager

Manages contexts with LRU cache, automatically releasing completed contexts.

**Features:**
- **LRU cache**: Automatically evicts least recently used contexts
- **Automatic memory release**: Clears contexts upon completion
- **Reference counting**: Safe memory deallocation
- **Optional memory monitoring**: Monitor memory usage with psutil

**CLI Usage:**

```bash
# Enable memory optimization
sdg run \
  --yaml pipeline.yaml \
  --input large_data.jsonl \
  --output result.jsonl \
  --enable-memory-optimization \
  --max-cache-size 500 \
  --enable-memory-monitoring
```

**Python API Usage:**

```python
async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(50000)]
    
    # Enable memory optimization
    async for result in run_pipeline_streaming(
        cfg,
        dataset,
        max_concurrent=32,
        enable_memory_optimization=True,  # Enable memory optimization
        max_cache_size=500,               # Cache size
        enable_memory_monitoring=True,    # Enable memory monitoring
    ):
        # Process...
        pass
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_memory_optimization` | `False` | Enable memory optimization |
| `max_cache_size` | `500` | Maximum context cache size |
| `enable_memory_monitoring` | `False` | Enable memory usage monitoring |

#### Batch Mode Memory Release

Progressive memory release is also available for traditional `run_pipeline` (batch mode).

**Python API Usage:**

```python
async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(10000)]
    
    # Enable memory optimization in batch mode
    results = await run_pipeline(
        cfg,
        dataset,
        enable_memory_optimization=True,  # Enable memory optimization
        gc_interval=100,                  # GC execution interval
        memory_threshold_mb=1024,         # Memory warning threshold (MB)
    )
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_memory_optimization` | `False` | Enable memory optimization |
| `enable_memory_monitoring` | `False` | Enable memory usage monitoring |
| `gc_interval` | `100` | Garbage collection interval (processed rows) |
| `memory_threshold_mb` | `1024` | Memory usage warning threshold (MB) |

### Integration with Adaptive Pipeline

Hierarchical scheduling and memory optimization can be combined with adaptive concurrency control.

**CLI Usage:**

```bash
# Enable all optimizations
sdg run \
  --yaml pipeline.yaml \
  --input huge_data.jsonl \
  --output result.jsonl \
  --adaptive \
  --min-batch 4 \
  --max-batch 64 \
  --target-latency-ms 2000 \
  --use-vllm-metrics \
  --enable-scheduling \
  --max-pending-tasks 200 \
  --chunk-size 100 \
  --enable-memory-optimization \
  --max-cache-size 1000
```

**Python API Usage:**

```python
from sdg.executors import run_pipeline_streaming_adaptive

async def main():
    cfg = load_config("config.yaml")
    dataset = [{"id": i} for i in range(100000)]
    
    # Enable all optimizations
    async for result in run_pipeline_streaming_adaptive(
        cfg,
        dataset,
        # Adaptive control
        max_concurrent=64,
        min_concurrent=4,
        target_latency_ms=2000,
        metrics_type="vllm",
        # Phase 2: Scheduling
        enable_scheduling=True,
        max_pending_tasks=200,
        chunk_size=100,
        # Phase 2: Memory optimization
        enable_memory_optimization=True,
        max_cache_size=1000,
        enable_memory_monitoring=True,
    ):
        # Process...
        pass
```

### Performance Guidelines

#### Small Datasets (< 1,000 rows)

```python
# Scheduling: Disabled (overhead too high)
enable_scheduling=False

# Memory optimization: Disabled (not needed)
enable_memory_optimization=False
```

#### Medium Datasets (1,000 - 10,000 rows)

```python
# Scheduling: Enabled
enable_scheduling=True
max_pending_tasks=100
chunk_size=50

# Memory optimization: Enabled
enable_memory_optimization=True
max_cache_size=500
```

#### Large Datasets (> 10,000 rows)

```python
# Scheduling: Enabled
enable_scheduling=True
max_pending_tasks=500
chunk_size=200

# Memory optimization: Enabled
enable_memory_optimization=True
max_cache_size=1000
enable_memory_monitoring=True  # Enable memory monitoring
gc_interval=100
```

### Tuning Tips

1. **max_pending_tasks**:
   - Too small: Throughput decreases
   - Too large: Memory usage increases
   - Recommended: 5-10x of `max_concurrent`

2. **chunk_size**:
   - Too small: Scheduling overhead increases
   - Too large: Memory usage increases
   - Recommended: 20-50% of `max_pending_tasks`

3. **max_cache_size**:
   - LRU cache size
   - Recommended: 5-10% of dataset size

### Troubleshooting

**High Memory Usage:**
1. Set `enable_memory_optimization=True`
2. Reduce `max_cache_size`
3. Reduce `gc_interval` (more frequent GC)
4. Enable `enable_memory_monitoring=True` to monitor usage

**Slow Processing:**
1. Increase `max_concurrent`
2. Increase `max_pending_tasks`
3. Increase `chunk_size`
4. Try `enable_scheduling=False` (for small datasets)

**Out of Memory Errors:**
1. Reduce `max_pending_tasks`
2. Reduce `chunk_size`
3. Reduce `max_cache_size`
4. Set `enable_memory_optimization=True`

For detailed implementation and API reference, see the [Phase 2 Optimization Guide](phase2_optimization.md).

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

#### Hugging Face Datasets

You can directly load datasets from Hugging Face Hub.

**Basic Usage:**

```bash
# Use the validation split of the squad dataset
sdg run --yaml pipeline.yaml --dataset squad --split validation --output result.jsonl

# Specify a subset
sdg run --yaml pipeline.yaml --dataset glue --subset mrpc --split train --output result.jsonl
```

**Key Mapping Feature:**

When the dataset's key names differ from the input keys expected by your pipeline, you can use the `--mapping` option to map keys.

```bash
# Example: Map dataset's "context" to "text" and "question" to "query"
sdg run --yaml pipeline.yaml \
  --dataset squad \
  --mapping context:text \
  --mapping question:query \
  --output result.jsonl

# Multiple key mappings
sdg run --yaml pipeline.yaml \
  --dataset my_dataset \
  --mapping original_field:UserInput \
  --mapping label:Category \
  --output result.jsonl
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | - | Hugging Face dataset name (required) |
| `--subset` | - | Dataset subset name (optional) |
| `--split` | train | Dataset split (train/validation/test, etc.) |
| `--mapping` | - | Key mapping (`orig:new` format, can be used multiple times) |

**Notes:**

- When `--dataset` is specified, `--input` cannot be used
- Key mappings are applied to each row in the dataset
- Mapped keys can be used in your pipeline YAML file

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
# Using examples/sdg_demo.yaml with fixed concurrency
sdg run \
  --yaml examples/sdg_demo.yaml \
  --input examples/data/input.jsonl \
  --output output/qa_result.jsonl \
  --max-concurrent 16
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

### Example 3: Processing Large Datasets with Fixed Concurrency

```bash
# Stream processing large datasets with fixed concurrency
sdg run \
  --yaml pipeline.yaml \
  --input large_dataset.jsonl \
  --output output/large_result.jsonl \
  --max-concurrent 16
```

### Example 4: Adaptive Concurrency for vLLM Backend

```bash
# Adaptive concurrency that adjusts based on latency
sdg run \
  --yaml examples/question_generator_agent_v2.yaml \
  --input examples/data/question_generator_input.jsonl \
  --output output/generated_questions_v2.jsonl \
  --adaptive \
  --min-batch 1 \
  --max-batch 32 \
  --target-latency-ms 2000
```

### Example 5: vLLM with Backend Metrics Optimization

```bash
# Use vLLM's Prometheus metrics for optimal concurrency control
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64 \
  --target-latency-ms 2000
```

### Example 6: Maximum Throughput with Request Batching

```bash
# Enable request batching for maximum throughput
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --enable-request-batching \
  --max-batch 64 \
  --max-batch-size 32
```

### Example 7: High-Speed Execution with Optimization Options

```bash
# Maximize connection efficiency with shared HTTP transport
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-concurrent 16 \
  --use-shared-transport

# Combine adaptive concurrency control with optimization options
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --adaptive \
  --use-vllm-metrics \
  --min-batch 1 \
  --max-batch 64 \
  --use-shared-transport

# Disable HTTP/2 for compatibility
sdg run \
  --yaml pipeline.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-concurrent 8 \
  --no-http2
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