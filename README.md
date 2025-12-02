# SDG Nexus

## Overview

**SDG-Nexus (Scalable Data Generator Nexus)** is a framework designed to efficiently generate synthetic datasets for LLMs (Large Language Models) and perform large-scale data analysis using AI agents. It is specifically designed for use cases that require parallel operation of numerous AI agents and high-speed batch processing, achieving significant improvements in processing capacity and flexibility compared to traditional methods.

By adopting the latest **MABEL (Model And Blocks Expansion Language) v2.0**, it enables highly descriptive and flexible structured agent programs. Additionally, it allows simultaneous operation of different LLM models, making load balancing and performance optimization easy. This makes it highly effective for tasks such as large-scale data analysis using LLMs, data augmentation, real-time inference, and synthetic data generation.

Furthermore, by incorporating adaptive batch processing and error handling mechanisms internally, stable operation is possible even in situations where request volumes fluctuate. It is particularly optimized for workloads involving high-frequency and large-scale inference, such as Natural Language Processing (NLP), generative AI applications, and AI agent-based automation systems.

This framework is designed with a focus on large-scale, high-speed, and stable utilization of AI agents, making it an ideal tool for users who need to efficiently scale up advanced tasks using LLMs.

---

## Features

* **MABEL v2.0 Support**
  * Turing-complete expression language (MEX)
  * Advanced control structures (`while`, `recurse`, `reduce`, `call`, `let`)
  * Inline Python functions
  * Global variable support
* **MABEL v1.x Backward Compatibility**
  * Automatic version detection
* **Advanced Concurrent Processing**
  * Automatically optimized adaptive batch processing
* **Multi-Model Support**
  * Define and operate multiple LLM models simultaneously
* **Flexible I/O Support**
  * JSONL and CSV format support in streaming and batch modes
* **Robust Error Handling**
  * Flexible error handling with retry mechanisms

---

## Requirements

* Python `>= 3.10`
* PyYAML `>= 6.0.1`
* openai `>= 1.40.0`
* tqdm `>= 4.66.0`

---

## Installation

Examples of installation using multiple environment management methods are provided.

### Standard pip Installation

```bash
pip install -e .
```

### Installation with pyenv

```bash
# Python version management
pyenv install 3.12.0
pyenv local 3.12.0

# Set up venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Installation with conda

```bash
# Create and activate environment
conda create -n sdg python=3.12
conda activate sdg

# Install
pip install -e .
```

### Fast Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

uv pip install -e .
```

---

## Quick Start

Minimal configuration example:

```yaml
mabel:
  version: "2.0"

models:
  - name: gpt4
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}

blocks:
  - type: ai
    exec: 1
    model: gpt4
    prompts:
      - "Summarize: {UserInput}"
    outputs:
      - name: Summary
        select: full
  
  - type: end
    exec: 2
    final:
      - name: answer
        value: "{Summary}"
```

For detailed specifications, please refer to:

* **[MABEL v2 Specification](docs/mabel/mabel_v2.md)** - Detailed feature descriptions, samples, and specifications

---

## Usage

### Command Line (CLI) Execution

Basic JSONL processing:

```bash
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/result.jsonl
```

Execution with custom batch settings:

```bash
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-batch 16 \
  --min-batch 2 \
  --target-latency 2000
```

### Using Python API

```python
from sdg.config import load_config
from sdg.executors import run_pipeline
import asyncio

# Load configuration
cfg = load_config("pipeline.yaml")

# Prepare dataset
dataset = [
    {"UserInput": "What is AI?"},
    {"UserInput": "Explain machine learning"}
]

# Run pipeline asynchronously
results = asyncio.run(run_pipeline(cfg, dataset))

for result in results:
    print(result)
```

---

## Detailed Documentation 📖

* **[Usage Guide](docs/usage.md)** - Detailed usage of CLI and Python API
* **[MABEL v2 Complete Specification](docs/mabel/mabel_v2.md)** - MABEL grammar and feature details

---

## Examples

Sample code and data are provided in the following directory.

* **`examples/`**
  * `sdg_demo.yaml` : Basic usage example
  * `sdg_demo_v2.yaml` : Advanced MABEL v2 sample
  * `sdg_comprehensive_v2.yaml` : Comprehensive v2 feature sample
  * `helpers.py` : External Python function usage example
  * `data/` : Sample input/output datasets

---

## License 📝

This project is provided under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## Contributing 🤝

Contributions to SDG-Nexus are welcome!
When submitting pull requests, please ensure:

* MABEL v1 compatibility is maintained
* MABEL v2 features comply with the latest specifications
* All existing samples pass tests
* Appropriate documentation is provided

---

## Support 🛠️

For bug reports and feature requests, please use [GitHub Issues](https://github.com/your-repository/issues).

---
