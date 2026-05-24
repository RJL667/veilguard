"""agents/*.md persona file loader.

Format decision (spec §0.3 / Phase 0.0.2): the existing personas
(phishing-analyst, threat-analyst, report-writer) use bold-Markdown KV
frontmatter, NOT YAML.  The parser handles both:

    # Phishing Analyst                                ← display name (ignored)

    **Model:** claude-sonnet-4-6                      ← scalar value
    **Tools:** filesystem (read_file), web (fetch)   ← grouped tools

    Or extended (new personas added by multi-agent platform):

    **Agent ID:** researcher
    **Role:** ic
    **Manager:** director
    **Team:** core
    **Tool Allow List:** web_search, fetch, recall, observe
    **Schema Version:** 1

    ## System Prompt

    <body>

Rules:
  * Filename stem is the agent_id (lowercase, hyphenated) UNLESS the
    `**Agent ID:**` line overrides.  Existing personas auto-derive.
  * Files without `**Role:**` default to `consultant` (backward compat
    for the 3 existing personas).
  * Files without `**Manager:**` / `**Team:**` default to None.
  * `**Tools:**` accepts both grouped form `category (t1, t2)` and bare
    comma-list `t1, t2, t3`.  Grouped values are flattened for the
    SDK's allow_list, but the grouping is preserved in `tool_groups`
    for future UI / display use.
  * `**Model:**` accepts scalar `claude-sonnet-4-6` OR mapped form
    `reactive=claude-sonnet-4-7, rank_pass=claude-haiku-4-5` (Director's
    dual-model spec §0.3).
  * Files NOT matching `[a-z][a-z0-9-]*\\.md` are skipped (so
    CONSTITUTION.md, PROMPTS.md, README.md don't fail to parse).
  * Files lacking `**Model:**` are skipped (not personas).

Output:
  * `PersonaSpec` dataclass with all parsed fields plus a content
    SHA-256 for cache-stability (used by the prompt-cache memoization
    key — same content → same cache slot).
  * `PersonaRegistry` for lookup by agent_id.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .text_sanitize import sanitize_bytes, Issue

logger = logging.getLogger("agent-runtime.personas")


class PersonaSpecError(ValueError):
    """Raised when an `agents/*.md` file is structurally malformed."""


# Bold-KV grammar: `**Key Name:** value`.  Capture group `key` is the
# label (case-insensitive lookup downstream); `val` is everything after
# the colon up to end-of-line.
_BOLD_KV_RE = re.compile(
    r"^\*\*(?P<key>[A-Za-z][\w ]*?):\*\*\s*(?P<val>.+?)\s*$",
    re.MULTILINE,
)

# Tools grammar:
#   - Grouped:   `filesystem (read_file, list_directory), web (fetch)`
#   - Flat:      `read_file, list_directory, fetch`
# The grouped form has parentheses; flat does not.
_TOOLS_GROUP_RE = re.compile(r"(?P<grp>[a-z][\w-]*)\s*\((?P<inner>[^)]+)\)")

# Mapped-model grammar: `reactive=X, rank_pass=Y`
_MODEL_MAP_RE = re.compile(r"(?P<role>[a-z_]+)\s*=\s*(?P<model>[a-zA-Z0-9._-]+)")

_SYSPROMPT_HDR_RE = re.compile(r"^##\s+System\s+Prompt\s*$", re.MULTILINE | re.IGNORECASE)

# Filename whitelist — must be lowercase + hyphens + .md.  Excludes
# CONSTITUTION.md, PROMPTS.md, README.md by uppercase first letter.
_PERSONA_FILE_RE = re.compile(r"^[a-z][a-z0-9-]*\.md$")


@dataclass(frozen=True)
class PersonaSpec:
    """One agent persona, parsed from its markdown file.

    Frozen so accidental mutation can't sneak in between request handlers
    and the SDK call.  Identity (agent_id, source_path) plus the parsed
    values + a content hash for cache memoization.
    """

    agent_id: str
    role: str                                 # director | ic | consultant
    manager_id: Optional[str]
    team_id: Optional[str]
    model: str                                # scalar form, or "default" model from mapping
    model_map: dict[str, str] = field(default_factory=dict)  # {"reactive": "...", "rank_pass": "..."}
    tools: list[str] = field(default_factory=list)
    tool_groups: dict[str, list[str]] = field(default_factory=dict)
    system_prompt: str = ""
    display_name: str = ""
    source_path: Optional[Path] = None
    content_sha256: str = ""
    schema_version: int = 1

    def model_for(self, role: str = "default") -> str:
        """Return the right model for a given internal role.

        Director uses `model_map["reactive"]` for reactive turns and
        `model_map["rank_pass"]` for the Haiku proposal-ranking call.
        ICs / consultants use the scalar `model` field for everything.
        """
        if self.model_map and role in self.model_map:
            return self.model_map[role]
        return self.model


class PersonaRegistry:
    """In-memory persona lookup, populated at service startup.

    Hot-reload support is a future addition (watch agents/ dir for
    changes); for now, persona definitions are read once on startup and
    the service must restart to pick up edits.  This is fine for now
    because personas change rarely (~weekly at most).
    """

    def __init__(self, by_id: dict[str, PersonaSpec]):
        self._by_id = by_id

    def get(self, agent_id: str) -> Optional[PersonaSpec]:
        return self._by_id.get(agent_id)

    def require(self, agent_id: str) -> PersonaSpec:
        p = self.get(agent_id)
        if p is None:
            raise KeyError(f"unknown agent_id: {agent_id}")
        return p

    def all(self) -> Iterable[PersonaSpec]:
        return self._by_id.values()

    def by_role(self, role: str) -> list[PersonaSpec]:
        return [p for p in self._by_id.values() if p.role == role]

    def consultants(self) -> list[PersonaSpec]:
        return self.by_role("consultant")

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._by_id


# ── Parser ───────────────────────────────────────────────────────────────


def _parse_tools(raw: str) -> tuple[list[str], dict[str, list[str]]]:
    """Parse the `**Tools:**` value into (flat list, group dict).

    Grouped form:  "filesystem (read_file, list_directory), web (fetch)"
    Flat form:     "read_file, list_directory, fetch"
    """
    raw = raw.strip()
    if not raw:
        return [], {}

    flat: list[str] = []
    groups: dict[str, list[str]] = {}

    if "(" in raw:
        for m in _TOOLS_GROUP_RE.finditer(raw):
            grp = m["grp"]
            items = [t.strip() for t in m["inner"].split(",") if t.strip()]
            groups[grp] = items
            flat.extend(items)
    else:
        flat = [t.strip() for t in raw.split(",") if t.strip()]

    return flat, groups


def _parse_model(raw: str) -> tuple[str, dict[str, str]]:
    """Parse the `**Model:**` value into (default_model, mapping).

    Scalar:  "claude-sonnet-4-6"          → ("claude-sonnet-4-6", {})
    Mapped:  "reactive=X, rank_pass=Y"    → (X, {"reactive": X, "rank_pass": Y})

    For mapped form, the default model is whatever's tagged `reactive`
    (the user-facing turn), or the first entry if no `reactive` key.
    """
    raw = raw.strip()
    if "=" in raw:
        mapping: dict[str, str] = {}
        for m in _MODEL_MAP_RE.finditer(raw):
            mapping[m["role"]] = m["model"]
        default = mapping.get("reactive") or (
            next(iter(mapping.values())) if mapping else raw
        )
        return default, mapping
    return raw, {}


def _split_header_body(text: str) -> tuple[str, str]:
    """Split markdown content at `## System Prompt`.

    Returns (header_block, body).  If the heading is absent, the body
    is empty and ALL of text is treated as header (parser will still
    extract bold-KV lines).  Personas without a system prompt aren't
    usable but we don't fail at parse time — that's a runtime concern.
    """
    parts = _SYSPROMPT_HDR_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1].strip()
    return text, ""


def _parse_one(path: Path) -> Optional[PersonaSpec]:
    """Parse one markdown file into a PersonaSpec, or return None if
    the file isn't a persona (no `**Model:**` line).

    Raises PersonaSpecError on structural problems (e.g., empty model
    value, invalid role, malformed tools).
    """
    raw_bytes = path.read_bytes()
    # System prompts may contain prose Unicode; use text mode so we
    # preserve em-dashes etc.
    text, issues = sanitize_bytes(raw_bytes, mode="text")
    for issue in issues:
        logger.warning(
            f"[personas] {path.name}: sanitized {issue.code} — {issue.detail}"
        )

    header, body = _split_header_body(text)

    # Extract bold-KV pairs from header.
    kv: dict[str, str] = {}
    for m in _BOLD_KV_RE.finditer(header):
        key = m["key"].strip().lower()
        kv[key] = m["val"].strip()

    # Display name = first `# <Name>` line, if present.
    display = ""
    for line in header.split("\n"):
        s = line.strip()
        if s.startswith("# ") and not s.startswith("## "):
            display = s[2:].strip()
            break

    # `model` is required to even count as a persona.  Missing = skip.
    if "model" not in kv:
        return None

    default_model, model_map = _parse_model(kv["model"])
    if not default_model:
        raise PersonaSpecError(f"{path.name}: empty model value")

    tools_raw = kv.get("tools", "")
    flat_tools, tool_groups = _parse_tools(tools_raw)

    # agent_id from `**Agent ID:**` line, else filename stem.
    agent_id = kv.get("agent id", "").strip() or path.stem
    if not re.match(r"^[a-z][a-z0-9-]*$", agent_id):
        raise PersonaSpecError(
            f"{path.name}: agent_id {agent_id!r} must match [a-z][a-z0-9-]*"
        )

    # role from `**Role:**` line, else default `consultant`.
    role = kv.get("role", "consultant").strip().lower()
    if role not in ("director", "ic", "consultant"):
        raise PersonaSpecError(
            f"{path.name}: role {role!r} must be director|ic|consultant"
        )

    manager_id = kv.get("manager", "").strip().lower() or None
    if manager_id in ("none", ""):
        manager_id = None

    team_id = kv.get("team", "").strip().lower() or None
    if team_id in ("none", ""):
        team_id = None

    schema_version = 1
    try:
        if "schema version" in kv:
            schema_version = int(kv["schema version"].strip())
    except ValueError:
        raise PersonaSpecError(
            f"{path.name}: schema_version must be int, got {kv['schema version']!r}"
        )

    # Cache memoization key — same content → same SDK config → same
    # Anthropic prompt-cache hit.  Strip volatile fields (path, mtime)
    # so only semantic content contributes.
    canon = "\n".join([
        f"agent_id={agent_id}",
        f"role={role}",
        f"manager={manager_id or ''}",
        f"team={team_id or ''}",
        f"model={default_model}",
        f"model_map={sorted(model_map.items())}",
        f"tools={','.join(sorted(flat_tools))}",
        f"prompt={body}",
    ])
    content_sha256 = hashlib.sha256(canon.encode("utf-8")).hexdigest()

    return PersonaSpec(
        agent_id=agent_id,
        role=role,
        manager_id=manager_id,
        team_id=team_id,
        model=default_model,
        model_map=model_map,
        tools=flat_tools,
        tool_groups=tool_groups,
        system_prompt=body,
        display_name=display or agent_id,
        source_path=path,
        content_sha256=content_sha256,
        schema_version=schema_version,
    )


def load_personas(agents_dir: Path) -> PersonaRegistry:
    """Load every `agents/*.md` file under `agents_dir` into a registry.

    Files that don't match the persona filename pattern (uppercase
    starting letter, e.g. CONSTITUTION.md / PROMPTS.md / README.md) are
    silently skipped.

    Files that match the pattern but lack `**Model:**` are also skipped
    (probably a doc file under agents/ that isn't a persona).

    Files that match AND have `**Model:**` but fail structural checks
    raise PersonaSpecError — startup logs the error and continues with
    the rest.  One bad file shouldn't take down the whole service.
    """
    by_id: dict[str, PersonaSpec] = {}
    if not agents_dir.exists():
        logger.warning(f"[personas] dir not found: {agents_dir}")
        return PersonaRegistry({})

    for path in sorted(agents_dir.iterdir()):
        if not path.is_file():
            continue
        if not _PERSONA_FILE_RE.match(path.name):
            logger.debug(f"[personas] skipping non-persona file: {path.name}")
            continue

        try:
            spec = _parse_one(path)
        except PersonaSpecError as e:
            logger.error(f"[personas] {path.name}: PARSE ERROR — {e}")
            continue
        except Exception as e:
            logger.exception(f"[personas] {path.name}: UNEXPECTED — {e}")
            continue

        if spec is None:
            logger.debug(f"[personas] {path.name}: no **Model:** line, skipping")
            continue

        if spec.agent_id in by_id:
            other = by_id[spec.agent_id]
            logger.error(
                f"[personas] duplicate agent_id {spec.agent_id!r}: "
                f"{path.name} vs {other.source_path.name if other.source_path else '?'} — "
                "second occurrence ignored"
            )
            continue

        by_id[spec.agent_id] = spec

    return PersonaRegistry(by_id)


__all__ = [
    "PersonaSpec",
    "PersonaRegistry",
    "PersonaSpecError",
    "load_personas",
]
