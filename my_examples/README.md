# Custom Agent Examples

## Overview

This directory contains custom implementations of OpenAI agents using the Agents Python SDK. These examples demonstrate various patterns, capabilities, and best practices for building AI agents for different use cases. The implementations showcase how to structure agent code, manage context, implement handoffs between specialized agents, and utilize advanced features of the SDK.

## Directory Structure

```
my_examples/
├── README.md                 # This documentation file
├── __init__.py               # Package initialization
├── demo_agents/              # Complete agent demonstrations
│   ├── __init__.py
│   ├── agent_features_demo.py  # Showcases advanced SDK features
│   └── core_concepts_demo.py   # Basic concepts implementation
├── shared/                   # Shared utilities and components
│   ├── __init__.py
│   └── utils.py              # Common helper functions
└── utils/                    # Utility functions and tools
    └── __init__.py
```

## Component Descriptions

### Demo Agents

- **core_concepts_demo.py**: Demonstrates fundamental SDK concepts including:
  - Basic agent configuration
  - Tool implementation
  - Tracing
  - Guardrails
  - Simple workflows

- **agent_features_demo.py**: Showcases advanced features including:
  - Custom context management
  - Structured output with Pydantic
  - Agent handoffs
  - Dynamic instructions
  - Lifecycle hooks
  - Complex guardrails

### Shared Components

- **utils.py**: Contains shared helper functions used across different examples, including:
  - Common data processing utilities
  - Standardized logging functions
  - Helper methods for context management

## Running the Examples

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install openai pydantic openai-agents
```

### Running a Demo

1. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Run a specific example:
```bash
python -m my_examples.demo_agents.core_concepts_demo
# or
python -m my_examples.demo_agents.agent_features_demo
```

3. Follow the interactive prompts to see the agents in action

## Dependencies and Setup Requirements

### Required Packages

- `openai`: For API access
- `pydantic`: For data validation and structured outputs
- `openai-agents`: The core Agents SDK

### Optional Packages

- `logfire`: For enhanced logging
- `agentops`: For agent operations monitoring
- `braintrust`: For evaluation and testing

### Environment Setup

1. Python 3.9 or higher is recommended
2. An OpenAI API key with access to the required models
3. Sufficient API quota for running the examples
4. Recommended: Set up a virtual environment to isolate dependencies

---

For more details on the OpenAI Agents SDK, refer to the [official documentation](https://platform.openai.com/docs/agents/).

