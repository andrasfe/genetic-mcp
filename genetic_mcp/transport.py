"""HTTP/SSE transport implementation for Genetic MCP."""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """MCP request format."""
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None
    id: int | None = None


class MCPResponse(BaseModel):
    """MCP response format."""
    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict[str, Any] | None = None
    id: int | None = None


def create_http_app(mcp_server: Any,
                   on_startup: Callable | None = None,
                   on_shutdown: Callable | None = None) -> FastAPI:
    """Create FastAPI app for HTTP/SSE transport."""

    app = FastAPI(title="Genetic MCP Server", version="0.1.0")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store active SSE connections
    active_connections: dict[str, asyncio.Queue] = {}

    @app.on_event("startup")
    async def startup_event():
        """Initialize server on startup."""
        if on_startup:
            await on_startup()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        if on_shutdown:
            await on_shutdown()

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "genetic-mcp",
            "version": "0.1.0",
            "description": "MCP server for genetic algorithm-based idea generation"
        }

    @app.post("/")
    async def handle_mcp_request(request: MCPRequest):
        """Handle MCP requests."""
        try:
            # Extract method and params
            method = request.method
            params = request.params or {}

            # Map MCP methods to server tools
            if method == "tools/list":
                # Return available tools
                tools = []
                for tool_name, tool_func in mcp_server._tools.items():
                    tools.append({
                        "name": tool_name,
                        "description": tool_func.__doc__ or "",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},  # Would need to extract from function signature
                            "required": []
                        }
                    })

                return MCPResponse(
                    id=request.id,
                    result={"tools": tools}
                )

            elif method == "tools/call":
                # Call a tool
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                if tool_name not in mcp_server._tools:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32601,
                            "message": f"Tool not found: {tool_name}"
                        }
                    )

                # Call the tool
                result = await mcp_server._tools[tool_name](**tool_args)

                return MCPResponse(
                    id=request.id,
                    result={"content": [{"type": "text", "text": json.dumps(result)}]}
                )

            else:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                )

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e)
                }
            )

    @app.get("/sse/{session_id}")
    async def sse_endpoint(session_id: str, request: Request):
        """SSE endpoint for streaming progress updates."""

        async def event_generator():
            """Generate SSE events."""
            # Create queue for this connection
            queue = asyncio.Queue()
            active_connections[session_id] = queue

            try:
                # Send initial connection event
                yield {
                    "event": "connected",
                    "data": json.dumps({"session_id": session_id})
                }

                # Stream progress updates
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break

                    try:
                        # Get progress update (with timeout to allow disconnect check)
                        progress = await asyncio.wait_for(queue.get(), timeout=1.0)

                        yield {
                            "event": "progress",
                            "data": json.dumps(progress)
                        }

                    except asyncio.TimeoutError:
                        # Send heartbeat
                        yield {
                            "event": "heartbeat",
                            "data": json.dumps({"timestamp": asyncio.get_event_loop().time()})
                        }

            finally:
                # Remove connection
                if session_id in active_connections:
                    del active_connections[session_id]

        return EventSourceResponse(event_generator())

    @app.post("/progress/{session_id}")
    async def send_progress(session_id: str, progress: dict[str, Any]):
        """Send progress update to SSE clients."""
        if session_id in active_connections:
            await active_connections[session_id].put(progress)
            return {"status": "sent"}
        return {"status": "no_active_connection"}

    return app


class HTTPTransport:
    """HTTP client for testing the server."""

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        import httpx
        self.session = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool via HTTP."""
        request = MCPRequest(
            method="tools/call",
            params={"name": name, "arguments": arguments},
            id=1
        )

        response = await self.session.post(
            self.base_url,
            json=request.dict()
        )
        response.raise_for_status()

        mcp_response = MCPResponse(**response.json())
        if mcp_response.error:
            raise Exception(f"MCP Error: {mcp_response.error}")

        # Extract result
        if mcp_response.result and "content" in mcp_response.result:
            content = mcp_response.result["content"][0]["text"]
            return json.loads(content)

        return mcp_response.result
