from playwright.async_api import async_playwright, TimeoutError, Error as PlaywrightError
from typing import List, Tuple, Optional
from urllib.parse import urlparse
from .computer import Computer
import asyncio
import logging

class LocalPlaywright(Computer):
    """Local browser implementation using Playwright."""
    
    def __init__(self, start_url: str = "https://bing.com"):
        self.start_url = start_url
        self.playwright = None
        self.browser = None
        self.page = None
        
    async def setup(self):
        """Initialize the browser asynchronously."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=False)
            self.page = await self.browser.new_page()
            await self.page.goto(self.start_url)
    
    # Standard Computer interface methods
    async def click(self, x: int, y: int, button: str = "left") -> None:
        await self.page.mouse.click(x, y, button=button)
    
    async def double_click(self, x: int, y: int) -> None:
        await self.page.mouse.dblclick(x, y)
    
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self.page.mouse.move(x, y)
        await self.page.mouse.wheel(delta_x=scroll_x, delta_y=scroll_y)
    
    async def type(self, text: str) -> None:
        await self.page.keyboard.type(text)
    
    async def wait(self, ms: int = 1000) -> None:
        await self.page.wait_for_timeout(ms)
    
    async def move(self, x: int, y: int) -> None:
        await self.page.mouse.move(x, y)
    
    async def keypress(self, keys: List[str]) -> None:
        for key in keys:
            await self.page.keyboard.press(key)
    
    async def drag(self, path: List[Tuple[int, int]]) -> None:
        if not path:
            return
        
        # Move to start position
        start_x, start_y = path[0]
        await self.page.mouse.move(start_x, start_y)
        await self.page.mouse.down()
        
        # Move through path
        for x, y in path[1:]:
            await self.page.mouse.move(x, y)
        
        # Release at final position
        await self.page.mouse.up()
    
    # Additional navigation methods
    async def goto(self, url: str) -> None:
        """Navigate to a specific URL.
        
        Args:
            url: The URL to navigate to
            
        Raises:
            ValueError: If the URL is invalid
            PlaywrightError: If navigation fails
            TimeoutError: If the page load times out
        """
        try:
            # Validate URL format
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid URL format: {url}")
            
            # Check if browser/page is ready
            if not self.page:
                await self.setup()
            
            # Navigate with timeout and wait until network is idle
            await self.page.goto(
                url,
                wait_until="networkidle",
                timeout=30000  # 30 seconds timeout
            )
            
        except ValueError as e:
            logging.error(f"URL validation error: {str(e)}")
            raise
        except TimeoutError:
            logging.error(f"Navigation timeout for URL: {url}")
            raise TimeoutError(f"Navigation to {url} timed out after 30 seconds")
        except PlaywrightError as e:
            logging.error(f"Navigation error for URL {url}: {str(e)}")
            raise

    async def back(self) -> None:
        """Go back to the previous page."""
        await self.page.go_back()

    async def forward(self) -> None:
        """Go forward to the next page."""
        await self.page.go_forward()

    async def refresh(self) -> None:
        """Refresh the current page."""
        await self.page.reload()

    async def click_element(self, selector: str) -> None:
        """Click an element using a CSS selector."""
        await self.page.click(selector)

    async def type_into(self, selector: str, text: str) -> None:
        """Type text into a specific element."""
        await self.page.fill(selector, text)

    async def wait_for_element(self, selector: str, timeout: int = 5000) -> None:
        """Wait for an element to appear on the page."""
        await self.page.wait_for_selector(selector, timeout=timeout)

    async def get_current_url(self) -> str:
        """Get the current page URL."""
        return self.page.url

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def get_screenshot(self) -> Optional[bytes]:
        """Take a screenshot of the current page."""
        return await self.page.screenshot()
