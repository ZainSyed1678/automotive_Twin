"""
MCP Server Package for Automotive Digital Twin Fault Diagnosis.
Exposes read-only diagnostic, simulation, and retrieval tools.
"""

from .server import create_mcp_server, server

__all__ = ["create_mcp_server", "server"]
