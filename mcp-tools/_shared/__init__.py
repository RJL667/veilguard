"""Shared helpers for Veilguard MCP servers.

Each MCP server lives in its own directory (mcp-tools/web/,
mcp-tools/filesystem/, mcp-tools/sub-agents/, etc.) so they can
ship independent dependency lists and run as separate processes.
This directory holds the small set of utilities that benefit from
being identical across servers — caching, result offloading, error
hint enrichment, cost telemetry.

Usage from a server's server.py:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "_shared"))
    from tool_caching import cached
    from result_offload import maybe_offload
    from error_hints import enrich_error
    from tool_cost import record_cost, format_cost_hint

The shim adds the shared dir to sys.path; the imports are flat (no
package prefix) so each helper is one import. Helpers use only the
Python stdlib — adding to a server's requirements.txt is not needed.
"""
