# OpenAI Agents Python SDK Examples

This directory contains examples demonstrating the features and capabilities of the OpenAI Agents Python SDK.

## Overview

The examples in this directory showcase various aspects of the SDK, from basic agent configuration to complex workflows with multiple agents, guardrails, and custom tools.

## Examples

### Core Concepts

- **[core_concepts_demo.py](core_concepts_demo.py)**: Demonstrates the key concepts of the SDK including agents, handoffs, guardrails, and tracing. Uses a coordinator agent that delegates to specialized research, analysis, and writing agents.

### Agent Features

- **[agent_features_demo.py](agent_features_demo.py)**: A comprehensive implementation of a customer support system showcasing basic agent configuration, custom context, structured output types, specialized agents with handoffs, dynamic instructions, lifecycle hooks, and guardrails.

### Agent Patterns

Examples showcasing different agent interaction patterns will be added here.

### Tools

Examples focusing on tool implementation and usage will be added here.

## Shared Utilities

The `shared/` directory contains common utilities used across multiple examples, including:

- Logging setup
- Helper functions for displaying results
- Common context types and tools

## Running the Examples

To run any example, navigate to the project root directory and run:

```bash
python -m examples.example_name
```

For instance:

```bash
python -m examples.core_concepts_demo
```

## Creating Your Own Examples

When creating your own examples:

1. Follow the patterns established in existing examples
2. Use the shared utilities from `examples/shared/` when appropriate
3. Add comprehensive documentation and comments
4. Add your example to this README.md file

## Requirements

All examples assume you have installed the required dependencies and have proper authentication set up for the OpenAI API.

