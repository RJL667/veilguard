"""Veilguard agent-runtime — embedded Claude Agent SDK as the multi-agent loop.

This service owns the LLM→tool→LLM loop for Anthropic-bound conversations.
The pii-proxy continues to handle redaction, audit, and multi-provider
routing; it forwards Anthropic calls to agent-runtime instead of going
direct.  agent-runtime then:

  1. Pulls the TCMM blob via tcmm-service (HTTP) and pins it as a
     byte-stable system prefix.
  2. Loads the agent registry (Director + ICs + Critic-split + consultants)
     from agents/*.md markdown personas.
  3. Invokes the Claude Agent SDK's query() with our PreToolUse hook for
     the approval gate, our subagent definitions, and our MCP server list.
  4. Captures token usage + tool-call breadcrumbs and writes to the same
     pii_audit Lance table the proxy uses.
  5. Streams back as SSE so LibreChat can render assistant turns + tool
     calls + status pings inline.

The harness is intentionally small.  The SDK does the agent loop; we
supply the middleware that makes it multi-tenant + provenance-aware +
audit-friendly + Veilguard-shaped.
"""

__version__ = "0.1.0"
