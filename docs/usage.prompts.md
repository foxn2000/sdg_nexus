# MABEL Flow Creation Guide: Prompts for AI Coding Agents

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Basic Prompt Template](#basic-prompt-template)
4. [Usage](#usage)
5. [Customizing Prompts](#customizing-prompts)
6. [Implementation Examples](#implementation-examples)
7. [Testing and Validation](#testing-and-validation)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This document explains how to efficiently create MABEL flows (YAML-formatted AI agent definition files) using **AI coding agents** such as Roo Cline, Cline, Claude Code, and Codex.

**What is MABEL?**

MABEL (Model And Blocks Expansion Language) is a language specification that allows you to declaratively describe AI agent processing flows in YAML format. Used by SDG Nexus, it enables you to define complex AI workflows in a structured manner.

For more details, refer to:
- [MABEL v2 Complete Specification](./mabel/mabel_v2_en.md)
- [SDG Usage Guide](./usage.md)

---

## Prerequisites

### Required Tools

- **AI Coding Agent** (one of the following)
  - Roo Cline
  - Cline
  - Claude Code
  - GitHub Copilot
  - Cursor
  - Other AI assistants

### Recommended Models

To generate high-quality MABEL flows, we recommend using high-performance models such as:

- **Claude 3.5 Sonnet** (recommended)
- **GPT-4 Turbo / GPT-4o**
- **Gemini 1.5 Pro**
- Other latest frontier models

### Required Knowledge

- Basic YAML syntax
- Basic concepts of AI agents
- (Recommended) Basic MABEL specifications

---

## Basic Prompt Template

Send the following prompt to your AI coding agent to generate MABEL flows.

```markdown
## Prompt Objective

Referring to the currently operational AI agent system (SDG Nexus) implementation, create a clear and operationally guaranteed YAML file in MABEL flow format based on the specified logic.

## Tasks to Perform

You are a logic engineer responsible for designing logic for AI agents. Please follow these steps to complete the task.

### Step 1: Review Existing Specifications

Review the following files to understand MABEL specifications and conventions:

1. **Source Code**: Python code under the `sdg/` directory
2. **Sample YAML**: Each YAML file stored in the `examples/` directory
3. **Documentation**: Documents under `docs/`, with special focus on:
   - `docs/mabel/mabel_v2_en.md` - MABEL v2 Complete Specification
   - `docs/usage.md` - Usage Guide

### Step 2: Design and Implement MABEL Flow

Based on the above research, design and implement the logic I present as a concrete MABEL flow.

**Implementation Requirements:**

- Must be written in executable YAML format
- Must comply with MABEL v2.0 specifications
- Reference the format of YAML files in existing `examples/`

### Step 3: File Placement

Place the completed files as follows:

- **YAML Files**: Store under `/flows/yaml/` directory
  - Name files to indicate processing content (e.g., `data_analysis_flow.yaml`)
- **Helper Scripts**: If supplementary code like Python is needed
  - Create as Python files under `/flows/codes/` directory
  - Configure appropriately for reference from YAML

## Output Guidelines

### Quality Requirements

1. **Operational Guarantee**
   - Comply with the format and specifications of YAML files in existing `examples/`
   - Must be in a fully operational state (no syntax errors)

2. **Code Quality**
   - Organize all implementation with clear and consistent structure
   - Add appropriate comments
   - Follow understandable naming conventions for variables

3. **Error Handling**
   - Include appropriate error handling (utilize `on_error`, `retry`)
   - Configure budget controls (`budgets`) appropriately

4. **Documentation**
   - Include explanatory comments in YAML files
   - Add supplementary explanations for complex processes

### Pre-Implementation Confirmation

If there are unclear points or deficiencies, explicitly point them out before implementation and present improvement proposals:

- Unclear specifications
- Required additional information
- Alternative implementation proposals
- Risks or constraints

## Logic to Implement

Below is the logic I want implemented.

<Insert logic description in text format here>

---

## Expected Deliverables

1. **YAML Files** (`/flows/yaml/*.yaml`)
   - Operational MABEL flow definition
   - With appropriate comments

2. **Helper Scripts** (`/flows/codes/*.py`) *only if needed*
   - Python function implementations
   - With appropriate docstrings

3. **README** (`/flows/README.md`) *recommended*
   - Flow overview
   - Usage instructions
   - Input data format
   - Output data format

## Post-Implementation Testing

After implementation is complete, perform the following tests:

1. **Syntax Check**
   ```bash
   # YAML syntax check
   python -c "import yaml; yaml.safe_load(open('flows/yaml/your_flow.yaml'))"
   ```

2. **Execution Test**
   ```bash
   # Run with small test data
   sdg run --yaml flows/yaml/your_flow.yaml \
           --input test_data.jsonl \
           --output test_output.jsonl
   ```

3. **Result Verification**
   - Verify output file is generated correctly
   - Check for errors
   - Confirm expected results are obtained
```

---

## Usage

### Basic Usage

1. **Copy the Prompt**
   - Copy the "Basic Prompt Template" above

2. **Describe the Logic**
   - Replace `<Insert logic description in text format here>` with the logic you want to implement

3. **Send to AI Agent**
   - Send the copied prompt to your AI coding agent

4. **Review Generated Results**
   - Review the generated YAML file and modify as needed

### Example: Generating a Simple Q&A Flow

```markdown
## Logic to Implement

Create a simple Q&A flow that receives questions from users and generates answers using an AI model.

**Requirements:**
- Input: Question in `UserInput` field
- Processing: Generate concise answer using GPT-4o-mini
- Output: Store answer in `Answer` field
- System prompt: "You are a helpful assistant that provides concise answers"
```

---

## Customizing Prompts

### For Advanced Features

When using MABEL v2 advanced features (recursion, loops, conditionals, etc.), add the following to your prompt:

```markdown
## Additional Requirements

Please utilize the following MABEL v2 advanced features:

- **Recursive Processing** (`op: recurse`): [Describe use case]
- **While Loops** (`op: while`): [Describe use case]
- **User-Defined Functions** (`functions.logic`): [Describe use case]
- **MEX Expression Language**: Use for conditional checks and value calculations

Reference: See §6.2 (Logic Blocks) in `docs/mabel/mabel_v2_en.md`.
```

### For Python Integration

When generating flows that include Python code:

```markdown
## Python Integration Requirements

Please implement the following Python features:

- **Inline Python** (`function_code`): [Implementation details]
- **External Python Files** (`code_path`): [Libraries to use and processing details]
- **Integrated Virtual Environment**: Specify required packages in `runtime.python`

Required libraries:
- numpy
- pandas
- Others [add as needed]
```

---

## Implementation Examples

### Example 1: Simple Data Processing Flow

**Logic Description:**

```
Create a flow that reads CSV data, processes numerical data in each row, and outputs statistical information.

Input: CSV string of numbers
Processing:
1. Parse CSV into list of numbers
2. Calculate sum, average, max, min
3. Calculate standard deviation using Python function
Output: Statistical information in JSON format
```

**Expected Generated Output:**

```yaml
mabel:
  version: "2.0"
  name: "Data Statistics Flow"
  description: "Statistical processing of CSV data"

runtime:
  python:
    interpreter: "python>=3.11,<3.13"
    venv: ".venv"

globals:
  vars:
    numbers: []
    stats: {}

blocks:
  - type: logic
    exec: 1
    name: "Parse CSV"
    op: for
    list: "{InputCSV}"
    parse: csv
    var: num
    map: {"to_number": "{num}"}
    outputs:
      - name: Numbers
        from: list

  - type: logic
    exec: 2
    name: "Calculate Sum"
    op: reduce
    list: "{Numbers}"
    value: 0
    var: item
    accumulator: sum
    body:
      - op: set
        var: sum
        value: {"add": [{"var": "sum"}, {"var": "item"}]}
    outputs:
      - name: Total
        from: accumulator

  # ... additional statistical calculation blocks ...

  - type: end
    exec: 100
    final:
      - name: statistics
        value: "{stats}"
```

### Example 2: AI-Driven Iterative Processing

**Logic Description:**

```
Create a flow that summarizes a document and iteratively improves the summary until it's sufficiently concise.

Requirements:
- Generate initial summary
- Repeat summarization until character count is 200 or less
- Maximum 5 attempts
- Save history of each iteration
```

---

## Testing and Validation

### 1. Syntax Validation

Verify YAML file syntax is correct:

```bash
# Syntax check with Python
python -c "import yaml; print(yaml.safe_load(open('flows/yaml/your_flow.yaml')))"

# Or check with SDG config loading
python -c "from sdg.config import load_config; load_config('flows/yaml/your_flow.yaml')"
```

### 2. Test Execution

Run with small test data:

```bash
# Prepare test data
echo '{"UserInput": "This is a test"}' > test_input.jsonl

# Execute
sdg run --yaml flows/yaml/your_flow.yaml \
        --input test_input.jsonl \
        --output test_output.jsonl \
        --max-concurrent 1

# Check results
cat test_output.jsonl
```

### 3. Debug Mode

If there are issues, enable detailed logging:

```bash
# Set log level with environment variable
export PYTHONPATH=.
python -m sdg run --yaml flows/yaml/your_flow.yaml \
                  --input test_input.jsonl \
                  --output test_output.jsonl
```

---

## Best Practices

### 1. Incremental Implementation

- Start with a simple flow
- Add features after verifying operation
- Run tests at each stage

### 2. Clear Naming

- Give blocks understandable `name`s
- Set unique `id`s for clear references
- Use output names that reflect processing content

### 3. Error Handling

```yaml
blocks:
  - type: ai
    exec: 1
    model: gpt4
    prompts: ["..."]
    on_error: "retry"  # Retry on error
    retry:
      max_attempts: 3
      backoff:
        type: "exponential"
        base_ms: 1000
    outputs:
      - name: Answer
        select: full
```

### 4. Budget Control

```yaml
budgets:
  loops:
    max_iters: 1000
    on_exceed: "error"
  recursion:
    max_depth: 128
    on_exceed: "error"
  wall_time_ms: 300000  # 5 minutes
  ai:
    max_calls: 100
    max_tokens: 100000
```

### 5. Documentation

- Add comments in YAML files
- Explain usage in README file
- Provide sample input data

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: YAML Syntax Error

**Error Message:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution:**
- Check indentation (watch for mixing spaces and tabs)
- Verify space after colon (:)
- Check quote escaping

#### Issue 2: Block Execution Order Issues

**Symptom:**
Output is used before it's referenced

**Solution:**
```yaml
# Check exec values and set according to dependencies
blocks:
  - type: ai
    exec: 1  # Execute first
    outputs:
      - name: Answer
        select: full
  
  - type: logic
    exec: 2  # After Answer becomes available
    op: if
    cond: {"ne": ["{Answer}", ""]}
    # ...
```

#### Issue 3: API Key Not Found

**Error Message:**
```
Error: Missing API key
```

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or specify directly in YAML (not recommended)
models:
  - name: gpt4
    api_key: "sk-..."  # Use environment variables in production
```

#### Issue 4: Python Function Not Found

**Error Message:**
```
ModuleNotFoundError: No module named 'helpers'
```

**Solution:**
```yaml
# Set code_path correctly
- type: python
  exec: 10
  code_path: ./flows/codes/helpers.py  # Check relative path
  function: process_data
  # ...
```

#### Issue 5: Out of Memory

**Symptom:**
Memory errors during large-scale data processing

**Solution:**
```bash
# Reduce concurrency
sdg run --yaml flow.yaml \
        --input large_data.jsonl \
        --output result.jsonl \
        --max-concurrent 4  # Reduced from default 8

# Or enable Phase 2 optimization
sdg run --yaml flow.yaml \
        --input large_data.jsonl \
        --output result.jsonl \
        --enable-memory-optimization \
        --max-cache-size 500
```

---

## Related Resources

- **[MABEL v2 Complete Specification](./mabel/mabel_v2_en.md)** - Detailed language specifications
- **[SDG Usage Guide](./usage.md)** - CLI and Python API usage
- **[Sample Collection](../examples/)** - Examples of working MABEL flows
- **[Phase 2 Optimization Guide](./others/phase2_optimization.md)** - Large-scale data processing optimization

---

## Summary

By using the prompt template introduced in this guide, you can efficiently create MABEL flows leveraging AI coding agents.

**Key Points:**

1. Reference existing specifications and samples
2. Communicate clear requirements
3. Implement incrementally
4. Always perform operational verification
5. Leave documentation

Create high-quality MABEL flows and maximize the potential of your AI agent systems.