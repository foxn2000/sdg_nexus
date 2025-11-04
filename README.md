# SDG Nexus

MABEL (Model And Blocks Expansion Language) based AI Agent System supporting **v2.0 specification**.

## Features

### MABEL v2.0 Support
- **MEX Expression Language**: Turing-complete expression evaluation
- **Global Variables**: Constants and mutable variables with `globals.const` and `globals.vars`
- **Advanced Logic Operators**:
  - `set`: Variable assignment with MEX expressions
  - `while`: Conditional loops with budget controls
  - `emit`: Value collection in loops
  - Full MEX operators: arithmetic, comparison, string, collection, regex, etc.
- **Inline Python Functions**: Define Python code directly in YAML with `function_code`
- **Enhanced Python Integration**: Context object (`ctx`) with `vars`, `get`, `set`, `log`
- **Budget Controls**: Loop/recursion limits with `budgets` configuration
- **Enhanced AI Outputs**:
  - JSONPath support (`select: jsonpath`)
  - Type hints (`type_hint: number|boolean|json`)
  - Variable saving (`save_to.vars`)

### MABEL v1.x Compatibility
- Full backward compatibility maintained
- Automatic version detection from `mabel.version`
- v1.0 YAML files work without modification

### Core Features
- **Batch Processing**: Optimized concurrent AI API calls
- **Adaptive Batching**: Dynamic batch size adjustment based on latency
- **Multi-Model Support**: Define and use multiple LLM models
- **Flexible I/O**: JSONL and CSV support
- **Error Handling**: Configurable error handling (`fail`, `continue`, `retry`)

## Installation

```bash
pip install -e .
```

## Quick Start

### v2.0 Example

```yaml
mabel:
  version: "2.0"

globals:
  const:
    APP_NAME: "My Agent"
  vars:
    counter: 0

budgets:
  loops:
    max_iters: 100
    on_exceed: "error"

models:
  - name: gpt4
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults:
      temperature: 0.0
      max_tokens: 500

blocks:
  # Set variable with MEX expression
  - type: logic
    exec: 1
    op: set
    var: counter
    value: {"add": [{"var": "counter"}, 1]}
  
  # While loop
  - type: logic
    exec: 2
    op: while
    init:
      - op: set
        var: i
        value: 0
    cond:
      lt:
        - {"var": "i"}
        - 5
    step:
      - op: set
        var: i
        value: {"add": [{"var": "i"}, 1]}
      - op: emit
        value: {"var": "i"}
    outputs:
      - name: Numbers
        from: list
  
  # Inline Python
  - type: python
    exec: 3
    entrypoint: process
    function_code: |
      def process(ctx, numbers: list) -> dict:
          ctx.log("info", f"Processing {len(numbers)} numbers")
          total = sum(numbers)
          return {"Sum": total}
    inputs:
      numbers: "{Numbers}"
    outputs: [Sum]
  
  - type: end
    exec: 100
    final:
      - name: result
        value: "{Sum}"
    include_vars:
      - counter
```

### v1.0 Example (Still Supported)

```yaml
mabel:
  version: "1.0"

models:
  - name: planner
    api_model: gpt-4o-mini
    api_key: ${ENV.OPENAI_API_KEY}
    request_defaults:
      temperature: 0.2

blocks:
  - type: ai
    exec: 1
    model: planner
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

## MABEL v2 Architecture

### MEX Expression Language

MEX provides a safe, Turing-complete expression language:

```yaml
# Arithmetic
{"add": [1, 2, 3]}  # 6
{"mul": [{"var": "x"}, 2]}  # x * 2

# Comparison
{"gt": [{"var": "count"}, 10]}  # count > 10
{"eq": ["{Status}", "ok"]}     # Status == "ok"

# Logic
{"and": [
  {"gt": [{"var": "score"}, 80]},
  {"lt": [{"var": "errors"}, 5]}
]}

# String operations
{"concat": ["Hello, ", {"var": "name"}]}
{"replace": ["{text}", "old", "new"]}

# Collections
{"map": {"list": [1,2,3], "fn": {"mul": [{"var": "item"}, 2]}}}
{"filter": {"list": [1,2,3,4], "fn": {"gt": [{"var": "item"}, 2]}}}

# Control flow
{"if": {
  "cond": {"gt": [{"var": "x"}, 0]},
  "then": "positive",
  "else": "non-positive"
}}
```

### Block Types

#### AI Block
```yaml
- type: ai
  exec: 1
  model: gpt4
  system_prompt: "You are a helpful assistant."
  prompts:
    - "Question: {UserInput}"
  mode: json  # v2: json mode
  outputs:
    - name: Answer
      select: jsonpath  # v2: JSONPath
      path: "$.response.text"
      type_hint: string
  save_to:  # v2: save to global vars
    vars:
      last_answer: Answer
```

#### Logic Block
```yaml
# v2: set
- type: logic
  exec: 1
  op: set
  var: total
  value: {"add": [{"var": "total"}, 10]}

# v2: while
- type: logic
  exec: 2
  op: while
  init:
    - op: set
      var: i
      value: 0
  cond:
    lt: [{"var": "i"}, 10]
  step:
    - op: set
      var: i
      value: {"add": [{"var": "i"}, 1]}
    - op: emit
      value: {"var": "i"}
  outputs:
    - name: Numbers
      from: list

# v1: for (still supported)
- type: logic
  exec: 3
  op: for
  list: "{Lines}"
  parse: lines
  var: line
  map: "- {line}"
  outputs:
    - name: Formatted
      from: join
      join_with: "\n"
```

#### Python Block
```yaml
# v2: inline function
- type: python
  exec: 1
  entrypoint: process
  function_code: |
    def process(ctx, data: dict) -> dict:
        # ctx.vars: global variables
        # ctx.get(path): get nested value
        # ctx.set(path, val): set global variable
        # ctx.log(level, msg): logging
        
        ctx.log("info", f"Processing {len(data)} items")
        result = {"processed": len(data)}
        return result
  inputs:
    data: "{InputData}"
  outputs: [processed]

# v1: external file (still supported)
- type: python
  exec: 2
  function: my_function
  code_path: ./helper.py
  inputs: [Input1, Input2]
  outputs: [Output1]
```

#### End Block
```yaml
- type: end
  exec: 100
  final:
    - name: answer
      value: "{Result}"
    - name: metadata
      value: "{Meta}"
  include_vars:  # v2: include global vars
    - counter
    - timestamp
```

## Configuration

### Runtime (v2)
```yaml
runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"
    requirements:
      - "numpy>=1.24"
      - "pandas>=2.0"
    allow_network: false
```

### Budgets (v2)
```yaml
budgets:
  loops:
    max_iters: 1000
    on_exceed: "error"  # error | truncate | continue
  recursion:
    max_depth: 128
    on_exceed: "error"
  wall_time_ms: 300000
  ai:
    max_calls: 100
    max_tokens: 500000
```

### Global Variables (v2)
```yaml
globals:
  const:  # Read-only
    APP_VERSION: "1.0"
    MAX_RETRIES: 3
  vars:   # Mutable
    counter: 0
    state: "init"
    results: []
```

## Migration from v1 to v2

v1 YAML files work without changes. To leverage v2 features:

1. Update version:
```yaml
mabel:
  version: "2.0"  # was "1.0"
```

2. Add global variables (optional):
```yaml
globals:
  vars:
    my_var: 0
```

3. Use MEX expressions in conditions:
```yaml
# v1 (JSON string, still works)
run_if: "{\"equals\":[\"{ Status}\",\"ok\"]}"

# v2 (native MEX, recommended)
run_if:
  eq: ["{Status}", "ok"]
```

4. Use inline Python for simple functions:
```yaml
# v1 (external file)
- type: python
  function: helper
  code_path: ./helper.py

# v2 (inline)
- type: python
  entrypoint: helper
  function_code: |
    def helper(ctx, x):
        return {"result": x * 2}
```

## Examples

See `examples/` directory:
- `sdg_demo.yaml` - v1.0 compatible example
- `sdg_demo_v2.yaml` - v2.0 feature showcase
- `helpers.py` - External Python functions example

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please ensure:
- v1 compatibility is maintained
- v2 features follow MABEL 2.0 specification
- Tests pass for both v1 and v2 examples
