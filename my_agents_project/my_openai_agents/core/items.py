"""Items generated during a run."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Union
from dataclasses import dataclass, field
from .usage import Usage

TResponseInputItem = TypeVar('TResponseInputItem', bound='ResponseInputItem')

class ResponseInputItem(ABC):
    """Base class for items that can be used as input to a response."""
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the item to a dictionary."""
        pass

class RunItem(ResponseInputItem):
    """An item generated during a run."""
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "run_item"}

    def to_input_item(self) -> 'ResponseInputItem':
        """Convert this item to an input item."""
        return self

@dataclass
class ModelResponse:
    """A response from a model."""
    output: str
    usage: Usage = field(default_factory=Usage)
    referenceable_id: str | None = None

class ItemHelpers:
    """Helper methods for working with items."""
    @staticmethod
    def input_to_new_input_list(
        input: Union[str, List[TResponseInputItem]]
    ) -> List[TResponseInputItem]:
        """Convert an input to a list of input items."""
        if isinstance(input, str):
            return [TextMessage(content=input)]  # type: ignore
        return list(input)

@dataclass
class TextMessage(ResponseInputItem):
    """A text message."""
    content: str
    role: str = "user"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "message",
            "role": self.role,
            "content": self.content
        }
