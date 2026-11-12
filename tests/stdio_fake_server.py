"""A tiny stdio MCP server standing in for the filesystem server in SSE tests."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem", log_level="WARNING")


@mcp.tool(description="List a directory.")
def list_directory(path: str) -> str:
    """path: directory."""
    return "[FILE] a.txt\n[FILE] b.txt"


if __name__ == "__main__":
    mcp.run(transport="stdio")
