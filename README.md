# Claude Desktop MCP Server Documentation

This README documents the setup and configuration of the Model Context Protocol (MCP) server for Claude Desktop application.

## Version Information

This setup was initially created for:
- **Claude Desktop**: v0.8.1
- **Node.js**: v20.16.0
- **npm**: 10.8.1
- **Volta**: 1.1.1
12|- **MCP Server**: @modelcontextprotocol/server-filesystem

## Package Versions and Dependencies

Exact versions used in this setup:

### Node.js Versions
- **Project-specific Node.js**: v20.19.0 (managed by Volta)
- **System default Node.js**: v20.16.0

### Volta Configuration
This project uses Volta to manage Node.js versions at the project level. The Volta configuration in package.json specifies:
```json
"volta": {
  "node": "20.19.0"
}
```

### MCP Server Version
- **@modelcontextprotocol/server-filesystem**: v2025.1.14

### NPM Dependencies
All dependencies are managed through package.json to ensure consistent behavior across environments.

## Overview

The Claude Desktop application uses a Model Context Protocol (MCP) server to facilitate file system access. This server allows Claude to access specific directories on your system (by default, Desktop and Downloads folders).

## Installation Directory

This custom MCP server installation is located at:
```
~/claude-mcp/
```

## Node.js Management

This installation uses Volta to manage Node.js at the workspace level, avoiding global Node.js installations:
```
volta pin node@20
```

## Important Paths

### Claude Application Support

Claude stores its configuration, logs, and other data at:
```
/Users/scottybe/Library/Application Support/Claude/
```

### Configuration Files

1. **Main Configuration**:
   ```
   /Users/scottybe/Library/Application Support/Claude/config.json
   ```

2. **Desktop-specific Configuration**:
   ```
   /Users/scottybe/Library/Application Support/Claude/claude_desktop_config.json
   ```

3. **Configuration Backup** (created during setup):
   ```
   /Users/scottybe/Library/Application Support/Claude/claude_desktop_config.json.backup
   ```

### Binary and Executable Locations

1. **Claude Desktop Application**:
   ```
   /Applications/Claude.app
   ```

2. **MCP Server Executable**:
   ```
   /Users/scottybe/claude-mcp/node_modules/.bin/mcp-server-filesystem
   ```

3. **Node.js (managed by Volta)**:
   ```
   ~/.volta/bin/node
   ```

4. **npm (managed by Volta)**:
   ```
   ~/.volta/bin/npm
   ```

### Log Files and Diagnostics

1. **Sentry Logs** (error reporting):
   ```
   /Users/scottybe/Library/Application Support/Claude/sentry/
   ```
   - Contains `scope_v3.json` and `session.json`

2. **Session Storage**:
   ```
   /Users/scottybe/Library/Application Support/Claude/Session Storage/
   ```

### Python Virtual Environment

Claude also maintains a Python virtual environment at:
```
/Users/scottybe/Library/Application Support/Claude/.venv/
```

## MCP Server Configuration

The MCP server is configured to:

1. Use the local installation of `@modelcontextprotocol/server-filesystem` instead of trying to use `npx`
2. Watch specific directories:
   - `/Users/scottybe/Desktop`
   - `/Users/scottybe/Downloads`

## MCP Server Command

The modified command path used to launch the MCP server (from the configuration files):
```
/Users/scottybe/claude-mcp/node_modules/.bin/mcp-server-filesystem
```

## Troubleshooting

If you encounter issues with the MCP server:

1. Check if the MCP server process is running:
   ```
   ps aux | grep mcp-server-filesystem
   ```

2. Verify configuration files are correctly pointing to the local installation
   ```
   cat "/Users/scottybe/Library/Application Support/Claude/config.json"
   cat "/Users/scottybe/Library/Application Support/Claude/claude_desktop_config.json"
   ```

3. If needed, restart the Claude application:
   ```
   pkill -x Claude && sleep 2 && open -a Claude
   ```

4. Check for error logs in the sentry directory:
   ```
   cat "/Users/scottybe/Library/Application Support/Claude/sentry/scope_v3.json"
   ```

## MCP Server Dependencies

This local installation includes the following npm package:
- `@modelcontextprotocol/server-filesystem` (managed via npm at 10.8.1)

## Updating the MCP Server

To update the MCP server in the future, run:
```
cd ~/claude-mcp
npm update @modelcontextprotocol/server-filesystem
```

## Restoring Default Configuration

If you need to restore the default configuration, you can:
1. Use the backup configuration file:
   ```
   cp "/Users/scottybe/Library/Application Support/Claude/claude_desktop_config.json.backup" "/Users/scottybe/Library/Application Support/Claude/claude_desktop_config.json"
   ```
2. Or simply uninstall and reinstall the Claude Desktop application

185|
## Agent Coding Chat

When working with Claude in this codebase, you can leverage Claude's code-writing capabilities by following these guidelines:

1. **Clear and Specific Instructions**: When asking Claude to write or modify code, be as specific as possible about what you want to achieve.

2. **Provide Context**: Share relevant files and code snippets to help Claude understand the codebase structure.

3. **Iterative Development**: Break down complex tasks into smaller steps, reviewing Claude's output at each stage.

4. **Error Correction**: If Claude makes a mistake, explain the error clearly and provide additional context if needed.

5. **Leverage Claude's Strengths**: Claude excels at:
   - Explaining code concepts
   - Refactoring existing code
   - Generating boilerplate code
   - Documenting code

## Context Priming

To help Claude understand this codebase and provide better assistance:

1. **Essential Files**:
   - `README.md`: Contains documentation about the Claude Desktop MCP server setup
   - `CLAUDE.md`: If present, contains specific instructions for Claude
   - `ai_docs/*`: Contains additional documentation and guides for AI assistants

2. **Codebase Exploration**:
   - Run `git ls-files` to see all tracked files in the repository
   - Share key configuration files from `/Users/scottybe/Library/Application Support/Claude/`
   - For specific issues, provide relevant log files from the paths listed in this README

3. **Optimal Context Order**:
   - Start with high-level documentation (README.md, CLAUDE.md)
   - Include relevant implementation files for specific tasks
   - Add error logs or terminal output for debugging issues

4. **Working with Configuration**:
   - When modifying configuration files, always create backups
   - Test changes by restarting the Claude application
   - Verify the MCP server is running with `ps aux | grep mcp-server-filesystem`

Following these practices will help Claude provide more accurate and helpful assistance with this codebase.
