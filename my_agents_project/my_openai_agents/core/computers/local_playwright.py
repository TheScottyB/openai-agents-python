"""Local Playwright implementation for computer control."""
from typing import Optional, Tuple, Any, Dict
import asyncio
from playwright.async_api import async_playwright, Browser, Page, BrowserType

class LocalPlaywright:
    """Local Playwright implementation for computer control."""
    def __init__(self, start_url: str = "https://www.bing.com"):
        """Initialize the Playwright browser."""
        self.start_url = start_url
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.current_url: Optional[str] = None

    async def setup(self) -> None:
        """Set up the browser environment."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=['--start-maximized']
            )
            self.page = await self.browser.new_page(
                viewport={"width": 1920, "height": 1080}
            )
            await self.navigate_to(self.start_url)
            print("Browser initialized successfully")
        except Exception as e:
            print(f"Error setting up browser: {str(e)}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    async def navigate_to(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            await self.page.goto(url)
            self.current_url = url
            return {
                "success": True,
                "message": f"Successfully navigated to {url}",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to navigate: {str(e)}",
                "data": None
            }

    async def click_at(self, x: int, y: int) -> Dict[str, Any]:
        """Click at specific coordinates."""
        try:
            await self.page.mouse.click(x, y)
            return {
                "success": True,
                "message": f"Clicked at coordinates ({x}, {y})",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to click: {str(e)}",
                "data": None
            }

    async def type_text(self, text: str) -> Dict[str, Any]:
        """Type text at the current position."""
        try:
            await self.page.keyboard.type(text)
            return {
                "success": True,
                "message": f"Typed text: {text}",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to type text: {str(e)}",
                "data": None
            }

    async def take_screenshot(self) -> Dict[str, Any]:
        """Take a screenshot of the current page."""
        try:
            screenshot = await self.page.screenshot()
            return {
                "success": True,
                "message": "Screenshot taken",
                "data": screenshot.decode('utf-8')
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to take screenshot: {str(e)}",
                "data": None
            }

    async def search(self, query: str) -> Dict[str, Any]:
        """Perform a search action."""
        try:
            # If on Bing, use the search box
            if "bing.com" in self.current_url:
                search_box = await self.page.wait_for_selector("#sb_form_q")
                await search_box.click()
                await search_box.fill(query)
                await search_box.press("Enter")
            # If on Google, use the search box
            elif "google.com" in self.current_url:
                search_box = await self.page.wait_for_selector('input[name="q"]')
                await search_box.click()
                await search_box.fill(query)
                await search_box.press("Enter")
            else:
                # For other sites, just type and press enter
                await self.type_text(query)
                await self.page.keyboard.press("Enter")

            return {
                "success": True,
                "message": f"Searched for: {query}",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to perform search: {str(e)}",
                "data": None
            }
