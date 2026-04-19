"""Veilguard Sub-Agents: Configuration constants."""

import os
from pathlib import Path

# ── API Keys ─────────────────────────────────────────────────────────────────

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _line.startswith("GOOGLE_API_KEY=") and not GOOGLE_API_KEY:
            GOOGLE_API_KEY = _line.split("=", 1)[1].strip()
        elif _line.startswith("ANTHROPIC_API_KEY=") and not ANTHROPIC_API_KEY:
            ANTHROPIC_API_KEY = _line.split("=", 1)[1].strip()

# ── Backend Routing ──────────────────────────────────────────────────────────

TIMEOUT = int(os.environ.get("SUB_AGENT_TIMEOUT", "120"))
DEFAULT_BACKEND = os.environ.get("SUB_AGENT_BACKEND", "claude")

BACKENDS = {
    "gemini": {
        "url": os.environ.get("SUB_AGENT_URL", "http://localhost:4000/gemini/v1beta/openai/chat/completions"),
        "key": GOOGLE_API_KEY,
        "default_model": "gemini-2.5-flash",
        "format": "openai",
    },
    "claude": {
        "url": os.environ.get("CLAUDE_AGENT_URL", "http://localhost:4000/anthropic/v1/messages"),
        "key": ANTHROPIC_API_KEY,
        "default_model": "claude-sonnet-4-6",
        "format": "anthropic",
    },
}

DEFAULT_MODEL = BACKENDS[DEFAULT_BACKEND]["default_model"]
PLAN_MODELS = {"gemini": "gemini-2.5-pro", "claude": "claude-opus-4"}

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
SCRATCHPAD_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "scratchpad"
OFFLOAD_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "results"
SESSION_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "sessions"
TRANSCRIPT_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "transcripts"
DAEMON_DATA_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "daemons"
SKILLS_DIR = Path(__file__).parent / "skills"
PLAYBOOKS_DIR = Path(__file__).parent / "playbooks"

# ── TCMM ─────────────────────────────────────────────────────────────────────

TCMM_URL = os.environ.get("TCMM_URL", "http://localhost:8811")

# ── Bounds ───────────────────────────────────────────────────────────────────

MAX_TASKS = 500
MAX_NOTIFICATIONS = 200
MAX_MAILBOX_PER_AGENT = 100
MAX_MANAGED_TASKS = 200
OFFLOAD_THRESHOLD = 8000
COMPACT_THRESHOLD = 12000
AFK_THRESHOLD = 300  # seconds

# ── Roles ────────────────────────────────────────────────────────────────────

ROLES = {
    "analyst": {
        "name": "Phishing Analyst",
        "system": (
            "You are a senior phishing analyst at Phishield. You analyze emails, URLs, "
            "and attachments for phishing indicators. You identify social engineering tactics, "
            "check sender reputation, analyze URL patterns, and assess threat level. "
            "Always provide a confidence score (0-100) and clear reasoning."
        ),
    },
    "threat": {
        "name": "Threat Intelligence Analyst",
        "system": (
            "You are a threat intelligence analyst at Phishield. You research threat actors, "
            "TTPs (Tactics, Techniques, Procedures), IOCs (Indicators of Compromise), and "
            "map findings to MITRE ATT&CK framework. Provide structured intelligence reports."
        ),
    },
    "writer": {
        "name": "Report Writer",
        "system": (
            "You are a professional security report writer at Phishield. You take raw analysis "
            "and findings and produce clear, executive-ready reports. Use proper formatting, "
            "include risk ratings, recommendations, and action items. Write for both technical "
            "and non-technical audiences."
        ),
    },
    "coder": {
        "name": "Code Assistant",
        "system": (
            "You are a Python/JavaScript developer. Write clean, well-structured code. "
            "Include error handling and comments. When asked to fix code, explain what was wrong."
        ),
    },
    "researcher": {
        "name": "Research Assistant",
        "system": (
            "You are a thorough research assistant. Gather information, synthesize findings, "
            "and present them in a structured format with sources and confidence levels."
        ),
    },
    "critic": {
        "name": "Quality Reviewer",
        "system": (
            "You are a critical reviewer. Evaluate the provided work for accuracy, completeness, "
            "clarity, and quality. Identify specific issues and suggest concrete improvements. "
            "Rate overall quality as: PASS, NEEDS_REVISION, or FAIL. "
            "If PASS, respond with just 'PASS: [brief comment]'. "
            "If not, list specific issues to fix."
        ),
    },
    "verify": {
        "name": "Verification Specialist",
        "system": (
            "You are an adversarial verification specialist at Phishield. Your job is to:\n"
            "1. Check every factual claim against known information\n"
            "2. Identify logical inconsistencies and unsupported assumptions\n"
            "3. Flag missing evidence or weak reasoning\n"
            "4. Test edge cases the original analysis missed\n"
            "5. Check for confirmation bias\n\n"
            "Be skeptical. Assume nothing. Challenge everything.\n"
            "Rate: VERIFIED / PARTIALLY_VERIFIED / UNVERIFIED / CONTRADICTED\n"
            "For each claim, state your confidence and reasoning."
        ),
    },
    "security": {
        "name": "Security Reviewer",
        "system": (
            "You are a security code reviewer at Phishield. Analyze for:\n"
            "- Injection vulnerabilities (SQL, XSS, command injection, SSTI)\n"
            "- Authentication/authorization flaws\n"
            "- Secrets exposure (API keys, passwords, tokens in code)\n"
            "- Insecure configurations (default creds, open ports, weak TLS)\n"
            "- OWASP Top 10 issues\n"
            "- Insecure deserialization, SSRF, path traversal\n\n"
            "Rate: SECURE / ISSUES_FOUND / CRITICAL_ISSUES\n"
            "List every finding with: severity (Critical/High/Medium/Low), "
            "location, description, and recommended fix."
        ),
    },
    "explore": {
        "name": "Explorer",
        "system": (
            "You are a fast exploration agent. Your job is to gather information "
            "and answer questions quickly. You CANNOT modify anything — only observe "
            "and report. Be concise: bullet points, short sentences, no filler. "
            "If you don't know something, say so. Don't speculate."
        ),
    },
    "planner": {
        "name": "Strategic Planner",
        "system": (
            "You are a strategic planner at Phishield. Given a goal:\n"
            "1. Analyze the problem space and constraints\n"
            "2. Consider 2-3 approaches with trade-offs\n"
            "3. Recommend the best approach with concrete steps\n"
            "4. Identify dependencies, risks, and mitigation strategies\n"
            "5. Estimate effort and timeline\n\n"
            "You plan but do NOT execute. Output a structured, actionable plan."
        ),
    },
}
