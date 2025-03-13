"""Track usage metrics."""
from dataclasses import dataclass

@dataclass
class Usage:
    """Usage metrics for a run."""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: 'Usage') -> None:
        """Add usage metrics from another Usage object."""
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
