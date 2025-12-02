# SDG Nexus

MABEL (Model And Blocks Expansion Language) based AI Agent System supporting **v2.0 specification**.

## Features

- **MABEL v2.0 Support**: Turing-complete expression language (MEX), advanced control structures (`while`, `recurse`, `reduce`, `call`, `let`), inline Python functions, and global variables
- **MABEL v1.x Compatibility**: Full backward compatibility with automatic version detection
- **Batch Processing**: Optimized concurrent AI API calls with adaptive batching
- **Multi-Model Support**: Define and use multiple LLM models
- **Flexible I/O**: JSONL and CSV support with streaming and batch modes
- **Error Handling**: Configurable error handling with retry mechanisms

## Installation

```bash
pip install -e .
```

## Requirements

- Python >= 3.10
- PyYAML >= 6.0.1
- openai >= 1.40.0
- tqdm >= 4.66.0

## Quick Start

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

For detailed MABEL syntax and advanced features, see:
- **[MABEL v2 Quick Reference (English)](docs/mabel/mabel_v2_en.md)** - Quick reference guide for v2.0 features
- **[MABEL v2 Complete Specification (日本語)](docs/mabel/mabel_v2.md)** - Complete specification with all features, examples, and implementation status

## Usage

### Command Line

```bash
# Process JSONL input
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input examples/data/input.jsonl \
  --output output/result.jsonl

# With custom batch settings
sdg run \
  --yaml examples/sdg_demo_v2.yaml \
  --input data.jsonl \
  --output result.jsonl \
  --max-batch 16 \
  --min-batch 2 \
  --target-latency 2000
```

### Python API

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

# Run pipeline
results = asyncio.run(run_pipeline(cfg, dataset))

for result in results:
    print(result)
```

📖 **Documentation:**
- **[Usage Guide](docs/usage.md)** - How to run SDG pipelines (CLI and Python API)
- **[MABEL v2 Quick Reference (English)](docs/mabel/mabel_v2_en.md)** - Quick reference for MABEL v2.0
- **[MABEL v2 Complete Specification (日本語)](docs/mabel/mabel_v2.md)** - Complete specification with all features and examples

## Examples

See `examples/` directory for sample YAML files and data:
- `sdg_demo.yaml` / `sdg_demo_v2.yaml` - Basic and advanced examples
- `sdg_comprehensive_v2.yaml` - Comprehensive v2.0 example with all features
- `helpers.py` - External Python functions example
- `data/` - Sample input/output data files

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please ensure:
- v1 compatibility is maintained
- v2 features follow MABEL 2.0 specification
- Tests pass for both v1 and v2 examples
- Code is well-documented

## Support

For issues and feature requests, please use the GitHub issue tracker.
