# LLM Agent Benchmark

This project provides a framework for benchmarking different types of Large Language Model (LLM) agents:

1. **Standard Agent**: Uses direct LLM calls with unstructured text responses
2. **Structured Agent**: Uses LLM calls with structured outputs through Pydantic models

## Components

- `agents.py`: Implementation of both agent types
- `benchmark.py`: Benchmarking framework to compare performance
- `kshot_examples.json`: K-shot examples for different tasks to improve agent performance
- `run_example.py`: Example script to run benchmark and K-shot examples

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install litellm tenacity pydantic matplotlib numpy
```

## Usage

### Running the Benchmark

```bash
python run_example.py
```

This will:
1. Create a test cases file from the K-shot examples
2. Run the benchmark comparing both agent types
3. Generate visualizations of the results
4. Run examples of both agents with K-shot prompting

### Configuration

You can modify the following parameters:
- `BENCHMARK_MOCK`: Environment variable to toggle between mock responses and actual LLM calls
- Test cases in `kshot_examples.json`
- Model name and other parameters in the benchmark setup

## K-shot Examples

The `kshot_examples.json` file contains various test cases with example inputs and outputs that can be used as few-shot examples for the agents. Each test case includes:

- `id`: Unique identifier
- `query`: The question or task
- `expected_answer`: Expected response (if applicable)
- `examples`: List of example queries and responses for K-shot learning

## Visualizations

The benchmark generates two visualizations:
1. Response times by test case
2. Overall average response times by agent type

## Extending the Framework

### Adding New Tasks

To add new benchmark tasks:
1. Add new test cases to `kshot_examples.json`
2. Include representative examples for K-shot learning

### Adding New Agent Types

To implement a new agent type:
1. Create a new class that inherits from `BaseAgent` in `agents.py`
2. Implement the required methods
3. Update the benchmark to include the new agent type

## Implementation Details

### Standard Agent

Uses the `call_llm` method to get unstructured text responses from the LLM.

### Structured Agent

Uses the `call_llm_structured_output` method to get structured responses based on defined Pydantic models.

## Performance Considerations

- Structured output may take longer to generate but provides more consistent responses
- K-shot examples significantly improve performance on specialized tasks
- The mock implementation provides a way to test the framework without incurring API costs