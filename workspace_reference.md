# OpenAI Agents SDK Workspace Reference

## Introduction

This reference document provides a comprehensive overview of the OpenAI Agents SDK project structure, key components, and file relationships. It is designed to help developers quickly understand the codebase organization and locate important files and directories. This document will be maintained by an agent to ensure it remains up-to-date as the project evolves.

The OpenAI Agents SDK is a lightweight yet powerful framework for building multi-agent workflows, focusing on a small set of core primitives: Agents, Handoffs, and Guardrails, along with built-in tracing capabilities.

## Main Directories

### src/agents/

The primary source code directory containing the core implementation of the Agents SDK.

- **Agent Implementation**:
  - Contains the core `Agent` class implementation
  - Includes the agent loop logic for running agents
  - Implements tool handling and function calling mechanisms
  
- **Key Components**:
  - Model interfaces and implementations (supporting OpenAI Chat Completions API format)
  - Tracing infrastructure for monitoring and debugging agent runs
  - Handoff mechanisms for agent-to-agent delegation
  - Guardrails for input and output validation
  - Tool integration frameworks

### docs/

Documentation files for the SDK, organized by topic and API reference.

- **Main Documentation Pages**:
  - `index.md`: Introduction to the SDK and core concepts
  - `quickstart.md`: Getting started guide
  - Various feature documentation (agents, tools, handoffs, tracing, etc.)
  
- **API Reference**:
  - Detailed API documentation for all components
  - Organized into sections for Agents, Tracing, and Extensions

### examples/

Sample implementations demonstrating different use cases and patterns for the SDK.

- **basic/**: Simple examples including hello world implementations
  - Contains basic examples like hello world for regular Python and Jupyter notebooks
  
- **agent_patterns/**: Examples of common agent workflow patterns
  - Demonstrates different architectural approaches for agent systems
  
- **research_bot/**: A multi-agent research system example
  - Implements a research flow with planner, search, and writer agents
  - Shows how to coordinate multiple agents to accomplish a complex task
  
- **customer_service/**: Customer service agent implementation
  
- **tool_examples/**: Examples of different tool implementations and usage patterns

### tests/

The test suite directory containing unit and integration tests for the SDK.

- Organized by component (agents, tools, tracing, etc.)
- Includes test fixtures and utilities
- Ensures SDK functionality and reliability

### workspace_master/

Contains CLI and computer-related implementations for the SDK.

## Key Files

### Project Configuration Files

- **pyproject.toml**: Python project configuration including dependencies and build settings
- **Makefile**: Development, testing, and build commands for the project
- **mkdocs.yml**: Documentation generation configuration
  - Configures site structure, navigation, and theme settings
  - Defines documentation organization and sections

### Documentation Files

- **README.md**: Project overview, installation instructions, and basic examples
  - Introduces core concepts (Agents, Handoffs, Guardrails, Tracing)
  - Provides getting started instructions and basic examples
  - References additional examples and documentation

- **docs/index.md**: Detailed introduction to the SDK
  - Explains core principles and features
  - Provides installation instructions
  - Shows basic examples

### Utility Files

- **tool_keeper.py**: Tool management implementation

## File Relationships and Dependencies

### Core Component Relationships

1. **Agent and Runner**: 
   - The `Agent` class defines agent capabilities and configuration
   - The `Runner` class executes the agent loop and manages communication with models

2. **Tools and Function Tools**:
   - Tool definitions are used by agents during execution
   - Function tools provide a decorator-based approach to convert Python functions to agent tools

3. **Tracing Infrastructure**:
   - Integrated throughout the SDK to provide visibility into agent operations
   - Supports various output destinations and extensions

4. **Handoff Mechanism**:
   - Enables agents to transfer control to other agents
   - Implemented as a special kind of tool call

### Documentation Structure

The documentation is organized hierarchically:
1. Introduction and Quickstart
2. Feature documentation
3. API Reference

MkDocs generates the documentation site based on the configuration in `mkdocs.yml`, pulling content from the `docs/` directory and API documentation directly from code docstrings.

## Development Workflow

The project uses a modern Python development workflow:
- **uv**: For dependency management
- **make**: For common development tasks (tests, linting, etc.)
- **MkDocs**: For documentation generation

Key development commands are defined in the Makefile:
- `make sync`: Install dependencies
- `make tests`: Run the test suite
- `make mypy`: Run type checker
- `make lint`: Run code linter

## Getting Started

To get started with the codebase:
1. Install dependencies using `make sync`
2. Set the `OPENAI_API_KEY` environment variable
3. Explore the examples to understand usage patterns
4. Refer to the documentation for detailed API information

