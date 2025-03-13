from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class Computer(ABC):
    """Base interface for computer environments that can execute CUA actions."""
    
    @abstractmethod
    async def setup(self) -> None:
        """Initialize the computer environment."""
        pass

    @abstractmethod
    async def click(self, x: int, y: int, button: str = "left") -> None:
        """Click at the specified coordinates."""
        pass

    @abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        pass

    @abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates."""
        pass

    @abstractmethod
    async def type(self, text: str) -> None:
        """Type the specified text."""
        pass

    @abstractmethod
    async def wait(self, ms: int = 1000) -> None:
        """Wait for the specified number of milliseconds."""
        pass

    @abstractmethod
    async def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        pass

    @abstractmethod
    async def keypress(self, keys: List[str]) -> None:
        """Press the specified keys."""
        pass

    @abstractmethod
    async def drag(self, path: List[Tuple[int, int]]) -> None:
        """Drag along the specified path of coordinates."""
        pass

    async def cleanup(self) -> None:
        """Clean up any resources. Override if needed."""
        pass

    async def get_screenshot(self) -> Optional[bytes]:
        """Get the current screenshot. Override if supported."""
        return None
