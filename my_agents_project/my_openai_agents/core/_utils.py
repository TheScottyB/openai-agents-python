"""Internal utilities for the agents package."""
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Awaitable

from pydantic import BaseModel

T = TypeVar("T")
MaybeAwaitable = Union[T, Awaitable[T]]

def ensure_list(value: Union[T, List[T], None]) -> List[T]:
    """Ensure a value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary."""
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj

def get_class_name(cls: Type[Any]) -> str:
    """Get the name of a class."""
    return cls.__name__
