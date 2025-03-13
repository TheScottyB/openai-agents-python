"""Base interface for computer environments that can execute actions."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Literal

Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]

class Computer(ABC):
    """A computer implemented with sync operations."""

    @property
    @abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def screenshot(self) -> str:
        pass

    @abstractmethod
    def click(self, x: int, y: int, button: Button = "left") -> None:
        """Click at the specified coordinates."""
        pass

    @abstractmethod
    def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        pass

    @abstractmethod
    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates."""
        pass

    @abstractmethod
    def type(self, text: str) -> None:
        """Type the specified text."""
        pass

    @abstractmethod
    def wait(self, ms: int = 1000) -> None:
        """Wait for the specified number of milliseconds."""
        pass

    @abstractmethod
    def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        pass

    @abstractmethod
    def keypress(self, keys: List[str]) -> None:
        """Press the specified keys."""
        pass

    @abstractmethod
    def drag(self, path: List[Tuple[int, int]]) -> None:
        """Drag along the specified path of coordinates."""
        pass

    def cleanup(self) -> None:
        """Clean up any resources. Override if needed."""
        pass


class AsyncComputer(ABC):
    """A computer implemented with async operations."""

    @property
    @abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abstractmethod
    async def screenshot(self) -> str:
        pass

    @abstractmethod
    async def click(self, x: int, y: int, button: Button = "left") -> None:
        pass

    @abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        pass

    @abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        pass

    @abstractmethod
    async def type(self, text: str) -> None:
        pass

    @abstractmethod
    async def wait(self, ms: int = 1000) -> None:
        pass

    @abstractmethod
    async def move(self, x: int, y: int) -> None:
        pass

    @abstractmethod
    async def keypress(self, keys: List[str]) -> None:
        pass

    @abstractmethod
    async def drag(self, path: List[Tuple[int, int]]) -> None:
        pass

    async def cleanup(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
