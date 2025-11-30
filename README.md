# SDG Nexus

MABEL (Model And Blocks Expansion Language) based AI Agent System supporting **v2.0 specification**.

## Features

### MABEL v2.0 Support
- **MEX Expression Language**: Turing-complete expression evaluation engine
- **Global Variables**: Constants and mutable variables with `globals.const` and `globals.vars`
- **Advanced Logic Operators**:
  - `set`: Variable assignment with MEX expressions
  - `let`: Local variable bindings with scoped execution
  - `while`: Conditional loops with budget controls
  - `emit`: Value collection in loops
  - `reduce`: List fold operations with accumulator
  - `call`: User-defined logic function calls
  - `recurse`: Recursive function execution with base case
  - Full MEX operators: arithmetic, comparison, string, collection, regex, logic
- **Inline Python Functions**: Define Python code directly in YAML with `function_code`
- **Enhanced Python Integration**: Context object (`ctx`) with `vars`, `get`, `set`, `log`, `emit`
- **Budget Controls**: Loop/recursion/AI call limits with `budgets` configuration
- **Enhanced AI Outputs**:
  - JSONPath support (`select: jsonpath`)
  - Type hints (`type_hint: number|boolean|json|string`)
  - Variable saving (`save_to.vars`)
  - JSON mode (`mode: json`)
- **User-Defined Functions**: Define reusable logic and Python functions

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

## Requirements

- Python >= 3.10
- PyYAML >= 6.0.1
- openai >= 1.40.0
- tqdm >= 4.66.0

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
  
  # While loop with emit
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
          ctx.set("total_sum", total)
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
      - total_sum
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

📖 **For detailed usage guide, see [docs/usage.md](docs/usage.md)**

## MABEL v2 Architecture

### MEX Expression Language

MEX provides a safe, Turing-complete expression language:

```yaml
# Arithmetic
{"add": [1, 2, 3]}  # 6
{"mul": [{"var": "x"}, 2]}  # x * 2
{"sub": [10, 3]}  # 7
{"div": [10, 2]}  # 5
{"mod": [10, 3]}  # 1

# Comparison
{"gt": [{"var": "count"}, 10]}  # count > 10
{"lt": [{"var": "score"}, 50]}  # score < 50
{"gte": [{"var": "x"}, 0]}  # x >= 0
{"lte": [{"var": "y"}, 100]}  # y <= 100
{"eq": ["{Status}", "ok"]}  # Status == "ok"
{"ne": ["{Status}", "error"]}  # Status != "error"

# Logic
{"and": [
  {"gt": [{"var": "score"}, 80]},
  {"lt": [{"var": "errors"}, 5]}
]}
{"or": [
  {"eq": ["{Status}", "ok"]},
  {"eq": ["{Status}", "pending"]}
]}
{"not": {"eq": ["{Status}", "failed"]}}

# String operations
{"concat": ["Hello, ", {"var": "name"}]}
{"replace": ["{text}", "old", "new"]}
{"length": ["{message}"]}
{"upper": ["{text}"]}
{"lower": ["{TEXT}"]}
{"trim": ["  spaced  "]}
{"split": ["{csv}", ","]}
{"join": [["a", "b", "c"], "_"]}

# Collections
{"map": {"list": [1,2,3], "fn": {"mul": [{"var": "item"}, 2]}}}
{"filter": {"list": [1,2,3,4], "fn": {"gt": [{"var": "item"}, 2]}}}
{"reduce": {"list": [1,2,3], "fn": {"add": [{"var": "acc"}, {"var": "item"}]}, "init": 0}}
{"get": {"dict": {"a": 1, "b": 2}, "key": "a"}}
{"keys": [{"a": 1, "b": 2}]}
{"values": [{"a": 1, "b": 2}]}
{"length": [[1, 2, 3]]}

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
  mode: json  # v2: json mode for structured output
  outputs:
    - name: Answer
      select: jsonpath  # v2: JSONPath extraction
      path: "$.response.text"
      type_hint: string  # v2: Type conversion
  save_to:  # v2: Save to global variables
    vars:
      last_answer: Answer
  on_error: continue  # v2: Error handling
  retry:  # v2: Retry configuration
    max_attempts: 3
    backoff_ms: 1000
```

#### Logic Block

##### set - Variable Assignment
```yaml
- type: logic
  exec: 1
  op: set
  var: total
  value: {"add": [{"var": "total"}, 10]}
```

##### let - Local Bindings
```yaml
- type: logic
  exec: 1
  op: let
  bindings:
    x: 10
    y: {"mul": [{"var": "x"}, 2]}
  body:
    - op: set
      var: result
      value: {"add": [{"var": "x"}, {"var": "y"}]}
  outputs:
    - name: Result
      from: var
      var: result
```

##### while - Conditional Loop
```yaml
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
```

##### reduce - List Fold
```yaml
- type: logic
  exec: 1
  op: reduce
  list: "{Items}"
  var: item
  value: 0  # Initial accumulator value
  body:
    - op: set
      var: accumulator
      value: {"add": [{"var": "accumulator"}, {"var": "item"}]}
  outputs:
    - name: Total
      from: accumulator
```

##### call - User-Defined Function
```yaml
# Define function
functions:
  logic:
    - name: double
      args: [x]
      returns: [result]
      body:
        - op: set
          var: result
          value: {"mul": [{"var": "x"}, 2]}

# Call function
blocks:
  - type: logic
    exec: 1
    op: call
    function: double
    with:
      x: 5
    outputs:
      - name: Doubled
        from: var
        var: result
```

##### recurse - Recursive Function
```yaml
# Factorial function using recursion
- type: logic
  exec: 1
  op: recurse
  name: factorial
  function:
    args: [n]
    returns: [result]
    base_case:
      cond:
        lte: [{"var": "n"}, 1]
      value: [1]
    body:
      - op: set
        var: n_minus_1
        value: {"sub": [{"var": "n"}, 1]}
      - op: call
        name: factorial
        with:
          n: {"var": "n_minus_1"}
        returns: [sub_result]
      - op: set
        var: result
        value: {"mul": [{"var": "n"}, {"var": "sub_result"}]}
  with:
    n: 5
  outputs:
    - name: Factorial
      from: value
```

##### for - List Iteration (v1 compatible)
```yaml
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

##### v2: Inline Function
```yaml
- type: python
  exec: 1
  entrypoint: process
  function_code: |
    def process(ctx, data: dict) -> dict:
        # ctx.vars: Global variables dictionary
        # ctx.get(path): Get nested value from context
        # ctx.set(path, val): Set global variable
        # ctx.log(level, msg): Log message (info, warning, error)
        # ctx.emit(name, value): Emit value to collector
        
        ctx.log("info", f"Processing {len(data)} items")
        
        # Access global variables
        counter = ctx.vars.get("counter", 0)
        ctx.set("counter", counter + 1)
        
        result = {"processed": len(data)}
        return result
  inputs:
    data: "{InputData}"
  outputs: [processed]
  timeout_ms: 5000  # v2: Execution timeout
```

##### v1: External File (Still Supported)
```yaml
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
      value: '{"status": "complete", "count": {counter}}'
  include_vars:  # v2: Include global variables in output
    - counter
    - timestamp
  final_mode: map  # v2: Output mode (map | list)
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
  const:  # Read-only constants
    APP_VERSION: "1.0"
    MAX_RETRIES: 3
    API_ENDPOINT: "https://api.example.com"
  vars:   # Mutable variables
    counter: 0
    state: "init"
    results: []
```

### User-Defined Functions (v2)
```yaml
functions:
  logic:
    - name: calculate_sum
      args: [a, b]
      returns: [sum]
      body:
        - op: set
          var: sum
          value: {"add": [{"var": "a"}, {"var": "b"}]}
  
  python:
    - name: custom_transform
      args: [data]
      returns: [transformed]
      # Implementation details...
```

## Migration from v1 to v2

v1 YAML files work without changes. To leverage v2 features:

### 1. Update Version
```yaml
mabel:
  version: "2.0"  # was "1.0"
```

### 2. Add Global Variables (Optional)
```yaml
globals:
  vars:
    my_var: 0
```

### 3. Use MEX Expressions
```yaml
# v1 (JSON string, still works)
run_if: "{\"equals\":[\"{ Status}\",\"ok\"]}"

# v2 (native MEX, recommended)
run_if:
  eq: ["{Status}", "ok"]
```

### 4. Use Inline Python
```yaml
# v1 (external file)
- type: python
  function: helper
  code_path: ./helper.py

# v2 (inline, recommended for simple functions)
- type: python
  entrypoint: helper
  function_code: |
    def helper(ctx, x):
        return {"result": x * 2}
```

### 5. Leverage New Logic Operators
```yaml
# Use set, let, while, reduce, call, recurse
- type: logic
  exec: 1
  op: set
  var: counter
  value: {"add": [{"var": "counter"}, 1]}
```

## Examples

See `examples/` directory:
- `sdg_demo.yaml` - v1.0 compatible example
- `sdg_demo_v2.yaml` - v2.0 feature showcase
- `sdg_comprehensive_v2.yaml` - Comprehensive v2.0 example with all features
- `helpers.py` - External Python functions example
- `data/` - Sample input/output data files

## Advanced Features

### Recursive Functions
```yaml
# Fibonacci using recursion
- type: logic
  exec: 1
  op: recurse
  name: fib
  function:
    args: [n]
    returns: [result]
    base_case:
      cond:
        lte: [{"var": "n"}, 1]
      value: [{"var": "n"}]
    body:
      - op: set
        var: n1
        value: {"sub": [{"var": "n"}, 1]}
      - op: set
        var: n2
        value: {"sub": [{"var": "n"}, 2]}
      - op: call
        name: fib
        with: {n: {"var": "n1"}}
        returns: [fib1]
      - op: call
        name: fib
        with: {n: {"var": "n2"}}
        returns: [fib2]
      - op: set
        var: result
        value: {"add": [{"var": "fib1"}, {"var": "fib2"}]}
  with: {n: 10}
  outputs:
    - name: Result
      from: value
```

### Complex MEX Expressions
```yaml
# Nested conditionals and operations
- type: logic
  exec: 1
  op: set
  var: grade
  value:
    if:
      cond: {"gte": [{"var": "score"}, 90]}
      then: "A"
      else:
        if:
          cond: {"gte": [{"var": "score"}, 80]}
          then: "B"
          else:
            if:
              cond: {"gte": [{"var": "score"}, 70]}
              then: "C"
              else: "F"
```

### AI with JSONPath
```yaml
- type: ai
  exec: 1
  model: gpt4
  mode: json
  prompts:
    - "Generate a JSON object with fields: name, age, email"
  outputs:
    - name: Name
      select: jsonpath
      path: "$.name"
      type_hint: string
    - name: Age
      select: jsonpath
      path: "$.age"
      type_hint: number
    - name: Email
      select: jsonpath
      path: "$.email"
      type_hint: string
```

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
