"""
Utility functions for examples.
"""

import logging
from typing import Any, Dict, List, Optional

# Configure logging
def setup_logger(name: str = "examples") -> logging.Logger:
    """Set up and return a logger with standard configuration."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

# Helper for pretty printing
def pretty_print_json(data: Dict[str, Any], indent: int = 2) -> None:
    """Print a dictionary as formatted JSON."""
    import json
    print(json.dumps(data, indent=indent, default=str))

# Helper for displaying agent responses
def display_agent_response(agent_name: str, response: Any) -> None:
    """Display an agent response in a formatted way."""
    import json
    
    print(f"\n{'='*40}")
    print(f"Response from {agent_name}:")
    print(f"{'='*40}")
    
    if isinstance(response, str):
        print(response)
    elif hasattr(response, 'model_dump'):
        # For pydantic models
        print(json.dumps(response.model_dump(), indent=2))
    elif isinstance(response, dict):
        pretty_print_json(response)
    else:
        print(response)
    
    print(f"{'='*40}\n")

