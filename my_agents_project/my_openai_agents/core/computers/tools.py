"""Tools for controlling computer actions."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from my_openai_agents.core.tool import Tool, ToolCall

class NavigateParams(BaseModel):
    """Parameters for navigation."""
    url: str

class ClickParams(BaseModel):
    """Parameters for clicking."""
    x: int
    y: int

class TypeParams(BaseModel):
    """Parameters for typing."""
    text: str

class SearchParams(BaseModel):
    """Parameters for searching."""
    query: str

def create_computer_tools(computer: Any) -> List[Tool]:
    """Create tools for controlling computer actions."""
    async def navigate(params: NavigateParams) -> Dict[str, Any]:
        return await computer.navigate_to(params.url)

    async def click(params: ClickParams) -> Dict[str, Any]:
        return await computer.click_at(params.x, params.y)

    async def type_text(params: TypeParams) -> Dict[str, Any]:
        return await computer.type_text(params.text)

    async def take_screenshot() -> Dict[str, Any]:
        return await computer.take_screenshot()

    async def search(params: SearchParams) -> Dict[str, Any]:
        return await computer.search(params.query)

    return [
        Tool(
            name="navigate_to",
            description="Navigate to a specified URL",
            parameters=NavigateParams,
            callback=navigate
        ),
        Tool(
            name="click_at",
            description="Click at specific coordinates",
            parameters=ClickParams,
            callback=click
        ),
        Tool(
            name="type_text",
            description="Type text at the current position",
            parameters=TypeParams,
            callback=type_text
        ),
        Tool(
            name="take_screenshot",
            description="Take a screenshot of the current page",
            parameters=None,
            callback=take_screenshot
        ),
        Tool(
            name="search",
            description="Perform a search action",
            parameters=SearchParams,
            callback=search
        )
    ]
