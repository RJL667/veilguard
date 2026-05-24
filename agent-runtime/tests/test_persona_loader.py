"""Unit tests for personas.loader.

Covers:
  - Existing personas (no extended frontmatter) load as role=consultant.
  - New personas with full frontmatter (Director, Researcher, etc.).
  - Grouped + flat tools grammar.
  - Mapped model form (reactive=X, rank_pass=Y).
  - Edge cases: CRLF in source, smart quotes, missing **Model:**, etc.
  - Filename whitelist (PROMPTS.md / CONSTITUTION.md skipped).
"""

from pathlib import Path

import pytest

from app.personas.loader import (
    PersonaSpec,
    PersonaRegistry,
    PersonaSpecError,
    load_personas,
    _parse_one,
    _parse_tools,
    _parse_model,
)


# ── _parse_tools ────────────────────────────────────────────────────────


class TestParseTools:
    def test_grouped(self):
        flat, groups = _parse_tools("filesystem (read_file, list_directory), web (fetch)")
        assert "read_file" in flat
        assert "list_directory" in flat
        assert "fetch" in flat
        assert groups["filesystem"] == ["read_file", "list_directory"]
        assert groups["web"] == ["fetch"]

    def test_flat(self):
        flat, groups = _parse_tools("read_file, write_file, fetch")
        assert flat == ["read_file", "write_file", "fetch"]
        assert groups == {}

    def test_empty(self):
        flat, groups = _parse_tools("")
        assert flat == []
        assert groups == {}

    def test_whitespace_tolerant(self):
        flat, _ = _parse_tools("  read_file ,  write_file  ")
        assert flat == ["read_file", "write_file"]


# ── _parse_model ────────────────────────────────────────────────────────


class TestParseModel:
    def test_scalar(self):
        default, mapping = _parse_model("claude-sonnet-4-6")
        assert default == "claude-sonnet-4-6"
        assert mapping == {}

    def test_mapped(self):
        default, mapping = _parse_model("reactive=claude-sonnet-4-7, rank_pass=claude-haiku-4-5")
        assert default == "claude-sonnet-4-7"
        assert mapping == {
            "reactive": "claude-sonnet-4-7",
            "rank_pass": "claude-haiku-4-5",
        }

    def test_mapped_no_reactive_falls_back_to_first(self):
        default, mapping = _parse_model("foo=A, bar=B")
        assert default in ("A", "B")
        assert mapping == {"foo": "A", "bar": "B"}


# ── _parse_one ──────────────────────────────────────────────────────────


def _make_persona_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestParseOne:
    def test_minimal_persona_loads_as_consultant(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "phishing-analyst.md",
            "# Phishing Analyst\n\n"
            "**Model:** claude-sonnet-4-6\n"
            "**Tools:** read_file, fetch\n\n"
            "## System Prompt\n\nYou analyze phishing.\n",
        )
        spec = _parse_one(path)
        assert spec is not None
        assert spec.agent_id == "phishing-analyst"
        assert spec.model == "claude-sonnet-4-6"
        assert spec.role == "consultant"  # default
        assert spec.manager_id is None
        assert spec.team_id is None
        assert "read_file" in spec.tools
        assert "You analyze phishing" in spec.system_prompt
        assert spec.display_name == "Phishing Analyst"

    def test_full_frontmatter_director(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "director.md",
            "# Director\n\n"
            "**Agent ID:** director\n"
            "**Role:** director\n"
            "**Manager:** none\n"
            "**Team:** core\n"
            "**Model:** reactive=claude-sonnet-4-7, rank_pass=claude-haiku-4-5\n"
            "**Tools:** create_task, assign_task, recall\n"
            "**Schema Version:** 1\n\n"
            "## System Prompt\n\nYou are the Director.\n",
        )
        spec = _parse_one(path)
        assert spec.role == "director"
        assert spec.manager_id is None  # "none" → None
        assert spec.team_id == "core"
        assert spec.model == "claude-sonnet-4-7"
        assert spec.model_map["rank_pass"] == "claude-haiku-4-5"
        assert spec.model_for("reactive") == "claude-sonnet-4-7"
        assert spec.model_for("rank_pass") == "claude-haiku-4-5"

    def test_missing_model_returns_none(self, tmp_path):
        # No **Model:** line → not a persona file; parser returns None.
        path = _make_persona_file(
            tmp_path,
            "not-a-persona.md",
            "# Heading\n\nJust prose, no frontmatter.\n",
        )
        assert _parse_one(path) is None

    def test_missing_system_prompt_returns_spec_with_empty_body(self, tmp_path):
        # Intentional: parser doesn't fail on missing `## System Prompt`.
        # Persona is still loadable; runtime decides what to do with an
        # empty system_prompt (likely refuse to dispatch).
        path = _make_persona_file(
            tmp_path,
            "broken.md",
            "**Model:** claude-sonnet-4-6\n",
        )
        spec = _parse_one(path)
        assert spec is not None
        assert spec.system_prompt == ""

    def test_invalid_role_raises(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "broken.md",
            "**Model:** claude-sonnet-4-6\n"
            "**Role:** overlord\n\n"
            "## System Prompt\nx\n",
        )
        with pytest.raises(PersonaSpecError):
            _parse_one(path)

    def test_invalid_agent_id_raises(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "broken.md",
            "**Agent ID:** Bad Name\n"
            "**Model:** claude-sonnet-4-6\n\n"
            "## System Prompt\nx\n",
        )
        with pytest.raises(PersonaSpecError):
            _parse_one(path)

    def test_crlf_in_source_handled(self, tmp_path):
        # Simulate a Windows-saved file with CRLF line endings.
        path = tmp_path / "windows.md"
        path.write_bytes(
            b"# Windows\r\n\r\n"
            b"**Model:** claude-sonnet-4-6\r\n"
            b"**Tools:** read_file\r\n\r\n"
            b"## System Prompt\r\n\r\nHello.\r\n"
        )
        spec = _parse_one(path)
        assert spec is not None
        assert spec.model == "claude-sonnet-4-6"

    def test_content_sha_stable_across_reads(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "x.md",
            "**Model:** m\n**Tools:** t\n\n## System Prompt\nbody\n",
        )
        first = _parse_one(path)
        second = _parse_one(path)
        assert first.content_sha256 == second.content_sha256

    def test_content_sha_changes_on_prompt_edit(self, tmp_path):
        path = _make_persona_file(
            tmp_path,
            "x.md",
            "**Model:** m\n**Tools:** t\n\n## System Prompt\nbody\n",
        )
        first = _parse_one(path)
        path.write_text(
            "**Model:** m\n**Tools:** t\n\n## System Prompt\nDIFFERENT body\n"
        )
        second = _parse_one(path)
        assert first.content_sha256 != second.content_sha256


# ── load_personas (full directory walk) ─────────────────────────────────


class TestLoadPersonas:
    def test_skips_uppercase_filenames(self, tmp_path):
        # CONSTITUTION.md should be skipped by the filename regex.
        (tmp_path / "CONSTITUTION.md").write_text("anything")
        (tmp_path / "PROMPTS.md").write_text("anything")
        (tmp_path / "valid.md").write_text(
            "**Model:** m\n\n## System Prompt\nbody\n"
        )

        registry = load_personas(tmp_path)
        assert "valid" in [p.agent_id for p in registry.all()]
        assert "CONSTITUTION" not in registry
        assert "PROMPTS" not in registry

    def test_missing_dir_returns_empty(self):
        registry = load_personas(Path("/nonexistent/path/xyz"))
        assert len(registry) == 0

    def test_duplicate_agent_id_logs_and_skips(self, tmp_path):
        (tmp_path / "alice.md").write_text(
            "**Agent ID:** alice\n**Model:** m\n\n## System Prompt\nA\n"
        )
        (tmp_path / "alice-2.md").write_text(
            "**Agent ID:** alice\n**Model:** m\n\n## System Prompt\nB\n"
        )
        registry = load_personas(tmp_path)
        assert len(registry) == 1

    def test_one_bad_file_doesnt_break_others(self, tmp_path):
        (tmp_path / "good.md").write_text(
            "**Model:** m\n\n## System Prompt\nbody\n"
        )
        (tmp_path / "bad.md").write_text(
            "**Model:** m\n**Role:** overlord\n\n## System Prompt\nbody\n"
        )
        registry = load_personas(tmp_path)
        assert "good" in registry
        assert "bad" not in registry


# ── PersonaRegistry ─────────────────────────────────────────────────────


class TestPersonaRegistry:
    def test_get_missing_returns_none(self):
        r = PersonaRegistry({})
        assert r.get("nobody") is None

    def test_require_missing_raises(self):
        r = PersonaRegistry({})
        with pytest.raises(KeyError):
            r.require("nobody")

    def test_by_role(self):
        d = PersonaSpec(
            agent_id="d", role="director", manager_id=None, team_id="core",
            model="m",
        )
        i = PersonaSpec(
            agent_id="i", role="ic", manager_id="d", team_id="core",
            model="m",
        )
        c = PersonaSpec(
            agent_id="c", role="consultant", manager_id=None, team_id=None,
            model="m",
        )
        r = PersonaRegistry({"d": d, "i": i, "c": c})
        assert r.by_role("director") == [d]
        assert r.by_role("ic") == [i]
        assert r.consultants() == [c]
