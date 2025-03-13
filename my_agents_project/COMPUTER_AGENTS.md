# Computer Control Agents

This documentation describes the computer control functionality in the project, which allows agents to interact with computers and browsers.

## Architecture

The system consists of several key components:

1. Computer Interface
    - Base abstract classes: `Computer` and `AsyncComputer`
    - Define standard operations for computer control
    - Support both synchronous and asynchronous implementations

2. Implementations
    - `LocalPlaywright`: Browser control using Playwright
    - Support for various environments (browser, mac, windows, ubuntu)

3. Tools
    - Computer control tools for agents
    - Standard actions: click, type, navigate, etc.
    - Error handling and result reporting

## Usage

### Browser Automation

```python
from my_openai_agents.core.agent import Agent
from my_openai_agents.core.computers.local_playwright import LocalPlaywright
from my_openai_agents.core.computers.tools import create_computer_tools

# Initialize the browser
computer = LocalPlaywright(start_url="https://www.bing.com")
await computer.setup()

# Create tools for the agent
tools = create_computer_tools(computer)

# Create the browser automation agent
agent = Agent(
    name="browser_automation",
    instructions="""You are a browser automation agent that can control a web browser.""",
    tools=tools,
    output_type=BrowserResult,
    model=model
)

# Run tasks
result = await Runner.run(agent, "Search for 'OpenAI news' on Bing")
```

### Available Operations

#### Basic Mouse Operations
- `click(x, y, button="left")`: Click at coordinates
- `double_click(x, y)`: Double click at coordinates
- `move(x, y)`: Move cursor to coordinates
- `drag(path)`: Drag along a path of coordinates

#### Keyboard Operations
- `type(text)`: Type text
- `keypress(keys)`: Press specific keys

#### Browser Specific
- `goto(url)`: Navigate to URL
- `screenshot()`: Take screenshot
- `scroll(x, y, scroll_x, scroll_y)`: Scroll at position

### Error Handling

All operations return a `ComputerAction` result:
```python
class ComputerAction(BaseModel):
    success: bool
    message: str
```

### Dependencies

Required packages:
- playwright
- asyncio
- pydantic

Install Playwright browsers:
```bash
playwright install chromium
```

## Examples

1. Simple Web Search
```python
await Runner.run(agent, "Search for 'Python programming' on Bing")
```

2. Complex Navigation
```python
await Runner.run(agent, "Go to GitHub, search for 'OpenAI agents', and open the first result")
```

3. Form Filling
```python
await Runner.run(agent, "Fill out the contact form at example.com with test data")
```

## Best Practices

1. Always use `try/finally` to ensure cleanup:
```python
try:
    await computer.setup()
    # Your code here
finally:
    await computer.cleanup()
```

2. Handle errors appropriately in tasks
3. Use appropriate timeouts for operations
4. Validate URLs before navigation
5. Clean up resources properly

## Future Improvements

Planned features:
- Additional computer environments (mac, windows, ubuntu)
- Enhanced element selection
- Better error recovery
- More sophisticated action sequences
- Recording and playback capabilities
