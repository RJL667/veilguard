"""
Backfill ``topic_dicts`` and ``entity_dicts`` on archive.lance rows
that have populated string lists but empty typed-dict columns.

Background: until commit 05454e9 (2026-04-24) the Vertex NLP adapter
was emitting topics in the wrong shape, and the persistence filter at
``archive.py:1411-1413`` dropped them on write. Result: 7.3% of
archive rows have ``topic_dicts``, vs. 73% with the plain ``topics``
list. ``entity_dicts`` has a similar smaller gap (58.8% vs 65.8% on
``entities``).

Two modes:

  --synthesize  (default if --execute, no LLM calls)
    For each gap row, build typed dicts from the existing string lists:
        topic_dicts  = [{"name": t, "type": "topic",  "score": 0.8 - 0.05*i}
                        for i, t in enumerate(topics)]
        entity_dicts = [{"name": e, "type": "OTHER",  "score": 0.5}
                        for e in entities]
    Same format the local Gemma adapter produces (nlp_adapter.py:629).
    Lossy on entity TYPES — they all get "OTHER" — but populates the
    column so consumers can rely on it being non-empty.

  --reextract
    Call the Vertex AI Gemini adapter on each row's ``text`` field to
    re-derive REAL types (PERSON/ORG/LOC for entities, category-vs-topic
    distinction). Slow (~1-2s/row) and costs Gemini tokens. Use only if
    you genuinely need the type information and not just non-empty
    columns. Requires GOOGLE_CLOUD_PROJECT env + ADC creds (same setup
    TCMM uses).

Safety:
  - Dry-run by default. Pass --execute to actually write.
  - Idempotent: skips rows whose typed dict is already populated.
  - Batches updates 50 rows at a time → bounded fragment growth (instead
    of one fragment per row).
  - Read with TCMM either stopped or running. Direct merge_insert on
    (user_id, namespace, aid) is MVCC-safe; worst case TCMM's in-memory
    archive cache shows stale data until it reloads (next access /
    restart). Recommend stopping TCMM for the --reextract long run.

    sudo systemctl stop veilguard-tcmm
    python3 scripts/tcmm_backfill_typed_dicts.py --execute --synthesize
    sudo systemctl start veilguard-tcmm
"""

import argparse
import os
import sys
import time
from pathlib import Path

import lance
import pyarrow as pa


DB_DIR = Path(os.environ.get(
    "LANCE_DB_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
))
ARCHIVE_PATH = DB_DIR / "archive.lance"

BATCH_SIZE = int(os.environ.get("BACKFILL_BATCH", "50"))
RATE_LIMIT_RPS = float(os.environ.get("BACKFILL_RATE_RPS", "5"))


def synthesize_topic_dicts(topics: list) -> list:
    """Build typed dicts from a string list. Mirrors nlp_adapter.py:629."""
    if not topics:
        return []
    return [
        {"name": str(t), "type": "topic",
         "score": round(max(0.2, 0.8 - 0.05 * i), 3)}
        for i, t in enumerate(topics) if t
    ]


def synthesize_entity_dicts(entities: list) -> list:
    """Build typed dicts from a string list.

    Lossy on type — we don't know if these were PERSON/ORG/LOC originally,
    so they all get "OTHER". The recall path filters by ``name`` not
    type, so this is a non-blocker for current consumers.
    """
    if not entities:
        return []
    return [
        {"name": str(e), "type": "OTHER", "score": 0.5}
        for e in entities if e
    ]


_AI_STUDIO_PROMPT = """You are an information extractor. Given a block of text, identify:
1. ENTITIES: named entities (people, organizations, locations, products, dates, numbers)
2. TOPICS: 3-7 short topical phrases describing what the block is about

Return ONLY valid JSON with this exact shape, no prose, no markdown fences:
{"entities":[{"name":"...","type":"PERSON|ORG|LOC|PRODUCT|DATE|NUMBER|OTHER"}],"topics":["..."]}

Text:
\"\"\"
{text}
\"\"\""""


def reextract_via_ai_studio(text: str, model) -> tuple[list, list]:
    """Call Google AI Studio Gemini with the configured model on one
    block's text. Returns (topic_dicts, entity_dicts) ready for
    merge_insert.

    NOT going through TCMM's vertex_nlp_adapter on purpose — the user
    asked us not to push TCMM code changes, and the adapter is set up
    for Vertex (ADC auth), not AI Studio (API key). This standalone
    call uses google-generativeai SDK directly with the API key from
    the VM's .env (GOOGLE_API_KEY).
    """
    if not text or not text.strip():
        return [], []
    # Trim to ~2000 chars (matches what the production adapter does at
    # vertex_nlp_adapter.py:452). Long blocks blow the prompt budget
    # and don't yield better extraction in practice.
    snippet = text[:2000]
    prompt = _AI_STUDIO_PROMPT.replace("{text}", snippet)
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        raw = resp.text.strip() if hasattr(resp, "text") else ""
    except Exception as e:
        print(f"  ! AI Studio call failed: {e}", file=sys.stderr)
        return [], []

    if not raw:
        return [], []

    import json as _json
    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError:
        # Sometimes the model wraps JSON in fences despite the
        # response_mime_type hint. Strip and retry once.
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            parsed = _json.loads(cleaned)
        except _json.JSONDecodeError as e:
            print(f"  ! AI Studio returned non-JSON: {raw[:200]} ({e})",
                  file=sys.stderr)
            return [], []

    raw_topics = parsed.get("topics") or []
    raw_entities = parsed.get("entities") or []

    # Convert to the same {name, type, score} shape TCMM uses. Score
    # mirrors the local Gemma adapter's gentle decline (nlp_adapter.py:629).
    topic_dicts = [
        {"name": str(t)[:120], "type": "topic",
         "score": round(max(0.2, 0.8 - 0.05 * i), 3)}
        for i, t in enumerate(raw_topics) if t
    ]

    # Entities arrive as {name, type} from the prompt — give them a
    # constant 0.7 score (they don't have a per-item score from the
    # model, but score is required by the schema). Filter to the
    # canonical type set so weird model outputs don't poison the column.
    valid_types = {"PERSON", "ORG", "LOC", "PRODUCT", "DATE", "NUMBER", "OTHER"}
    entity_dicts = []
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()[:120]
        etype = str(e.get("type", "OTHER")).strip().upper()
        if etype not in valid_types:
            etype = "OTHER"
        if name:
            entity_dicts.append({"name": name, "type": etype, "score": 0.7})

    return topic_dicts, entity_dicts


def _is_synthetic(dicts: list, expected_type: str) -> bool:
    """Heuristic: detect dicts produced by --synthesize.

    Synthesize uses fixed placeholder types: ``"topic"`` for topic_dicts
    and ``"OTHER"`` for entity_dicts. If every dict in the list has the
    expected placeholder type, it's almost certainly synthetic. Real
    Gemini extraction returns mixed types (PERSON / ORG / LOC / DATE /
    NUMBER / PRODUCT / OTHER for entities; "category" mixed with
    "topic" for topic_dicts).
    """
    if not dicts:
        return False
    return all(
        isinstance(d, dict) and d.get("type") == expected_type
        for d in dicts
    )


def find_gap_rows(ds, upgrade_synthetic: bool = False) -> list[dict]:
    """Pull (aid, user_id, namespace, topics, entities, topic_dicts,
    entity_dicts, text) for every archive row, then filter to those
    that need work.

    Default: target rows where the typed dict is empty but the string
    list is populated.

    With ``upgrade_synthetic=True``: also target rows whose typed dicts
    appear to have been filled by --synthesize (placeholder types only).
    These rows would otherwise be skipped because their dicts are
    "non-empty"; --upgrade-synthetic makes the script revisit them so
    --reextract can replace placeholders with real types from Gemini.
    """
    cols = ["aid", "user_id", "namespace",
            "topics", "entities",
            "topic_dicts", "entity_dicts",
            "text"]
    arr = ds.to_table(columns=cols).to_pylist()

    gaps = []
    for r in arr:
        topics = r.get("topics") or []
        entities = r.get("entities") or []
        topic_dicts = r.get("topic_dicts") or []
        entity_dicts = r.get("entity_dicts") or []

        # Always-true gap conditions
        needs_topic = bool(topics) and not topic_dicts
        needs_entity = bool(entities) and not entity_dicts

        # Optional: synthetic upgrade conditions
        if upgrade_synthetic:
            if topic_dicts and _is_synthetic(topic_dicts, "topic"):
                needs_topic = True
            if entity_dicts and _is_synthetic(entity_dicts, "OTHER"):
                needs_entity = True

        if needs_topic or needs_entity:
            gaps.append({
                "aid": r["aid"],
                "user_id": r.get("user_id", ""),
                "namespace": r.get("namespace", ""),
                "topics": topics,
                "entities": entities,
                "topic_dicts_present": bool(topic_dicts) and not (
                    upgrade_synthetic and _is_synthetic(topic_dicts, "topic")
                ),
                "entity_dicts_present": bool(entity_dicts) and not (
                    upgrade_synthetic and _is_synthetic(entity_dicts, "OTHER")
                ),
                "text": r.get("text", ""),
            })
    return gaps


_DICT_STRUCT = pa.struct([
    pa.field("name", pa.utf8()),
    pa.field("type", pa.utf8()),
    pa.field("score", pa.float64()),
])


def _apply_one_column(ds, updates: list[dict], col_name: str,
                       source_key: str) -> int:
    """Update ONE typed-dict column on a batch of rows.

    Builds a source Arrow table with ONLY the merge keys + the target
    column. merge_insert with when_matched_update_all() then has nothing
    else to copy from the source, so other columns on matched rows are
    preserved.

    This is the corrected pattern. The previous version put BOTH
    topic_dicts + entity_dicts into the source and defaulted missing
    ones to [], which silently nulled the un-targeted column on every
    matched row — that's how entity_dicts dropped from 58.8% → 8.9%
    on the first run.
    """
    rows = [
        {
            "aid": u["aid"],
            "user_id": u["user_id"],
            "namespace": u["namespace"],
            col_name: u[source_key],
        }
        for u in updates if source_key in u
    ]
    if not rows:
        return 0

    schema = pa.schema([
        pa.field("aid", pa.int64()),
        pa.field("user_id", pa.utf8()),
        pa.field("namespace", pa.utf8()),
        pa.field(col_name, pa.list_(_DICT_STRUCT)),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)

    builder = ds.merge_insert(["user_id", "namespace", "aid"])
    builder.when_matched_update_all().execute(table)
    return len(rows)


def apply_batch(ds, updates: list[dict]) -> tuple[int, int]:
    """Apply N updates by issuing one merge_insert per column.

    Returns (n_topic_updates, n_entity_updates). Two passes is fine
    here — each pass is one transaction, so worst case we add 2
    fragments per batch (vs 50 with per-row writes).
    """
    if not updates:
        return 0, 0
    n_t = _apply_one_column(ds, updates, "topic_dicts", "new_topic_dicts")
    n_e = _apply_one_column(ds, updates, "entity_dicts", "new_entity_dicts")
    return n_t, n_e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Actually write changes. Default is dry-run.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--synthesize", action="store_true",
                      help="Build typed dicts from existing string lists. "
                           "Cheap, instant, lossy on entity TYPES.")
    mode.add_argument("--reextract", action="store_true",
                      help="Call Google AI Studio Gemini to re-derive "
                           "types. Slow, costs API tokens. Requires "
                           "GOOGLE_API_KEY env var (AI Studio key, AIza...).")
    ap.add_argument("--model", default="gemini-2.0-flash",
                    help="Gemini model name (default: gemini-2.0-flash). "
                         "Use gemini-2.5-flash for higher quality at ~3x cost.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows processed (0 = all). Useful for testing.")
    ap.add_argument("--upgrade-synthetic", action="store_true",
                    help="Also target rows whose typed dicts were filled "
                         "by a prior --synthesize pass (entity types all "
                         "'OTHER', topic types all 'topic'). Only meaningful "
                         "with --reextract — re-runs extraction to replace "
                         "placeholders with real PERSON/ORG/LOC/etc types.")
    args = ap.parse_args()

    # Default mode if neither specified
    if not args.synthesize and not args.reextract:
        args.synthesize = True

    if not ARCHIVE_PATH.is_dir():
        print(f"ERROR: {ARCHIVE_PATH} not found", file=sys.stderr)
        return 1

    ds = lance.dataset(str(ARCHIVE_PATH))
    total = ds.count_rows()
    print(f"=== archive.lance: {total:,} rows total ===")

    print("\n=== finding gap rows ===")
    if args.upgrade_synthetic and not args.reextract:
        print("  WARNING: --upgrade-synthetic without --reextract is a no-op "
              "(synthesize would just regenerate the same placeholders)")
    gaps = find_gap_rows(ds, upgrade_synthetic=args.upgrade_synthetic)
    label = "missing or synthetic" if args.upgrade_synthetic else "missing"
    print(f"  {len(gaps):,} rows have at least one {label} typed dict")

    n_topic_gaps = sum(1 for g in gaps if not g["topic_dicts_present"])
    n_entity_gaps = sum(1 for g in gaps if not g["entity_dicts_present"])
    print(f"    topic_dicts gap : {n_topic_gaps:,}")
    print(f"    entity_dicts gap: {n_entity_gaps:,}")

    if args.limit > 0 and len(gaps) > args.limit:
        gaps = gaps[: args.limit]
        print(f"  (limited to {args.limit} rows for this run)")

    if not gaps:
        print("\nNothing to do. ✓")
        return 0

    mode_label = "synthesize" if args.synthesize else "reextract via Vertex Gemini"
    print(f"\n=== mode: {mode_label} ===")
    if args.reextract:
        # Estimate cost: ~$0.0001/row at current Gemini Flash pricing.
        est = len(gaps) * 0.0001
        est_min = len(gaps) / RATE_LIMIT_RPS / 60
        print(f"  estimated cost  : ~${est:.2f} ({len(gaps)} rows × ~$0.0001)")
        print(f"  estimated time  : ~{est_min:.1f} min "
              f"(rate-limited to {RATE_LIMIT_RPS} rps)")

    if not args.execute:
        print("\n--- DRY RUN — no writes. Pass --execute to apply. ---")
        # Show a sample of what would change
        for g in gaps[:5]:
            print(f"\n  aid={g['aid']} user={g['user_id'][:12]}... "
                  f"ns={g['namespace'][:24]}...")
            if not g["topic_dicts_present"]:
                if args.synthesize:
                    new = synthesize_topic_dicts(g["topics"])
                    print(f"    topics ({len(g['topics'])}) → topic_dicts ({len(new)})")
                    for t in new[:3]:
                        print(f"      {t}")
                else:
                    print(f"    topics ({len(g['topics'])}) → would call Gemini on text[:80]"
                          f"={g['text'][:80]!r}")
            if not g["entity_dicts_present"]:
                if args.synthesize:
                    new = synthesize_entity_dicts(g["entities"])
                    print(f"    entities ({len(g['entities'])}) → entity_dicts ({len(new)})")
                    for e in new[:3]:
                        print(f"      {e}")
                else:
                    print(f"    entities ({len(g['entities'])}) → would call Gemini")
        if len(gaps) > 5:
            print(f"\n  ... and {len(gaps) - 5} more.")
        return 0

    # Execute mode. The reextract path uses Google AI Studio directly
    # (NOT TCMM's vertex_nlp_adapter, which would require Vertex ADC).
    # Standalone google-generativeai SDK call with the API key from
    # the VM's .env (GOOGLE_API_KEY).
    model = None
    if args.reextract:
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            print("ERROR: GOOGLE_API_KEY env var required for --reextract.\n"
                  "Set it in the systemd EnvironmentFile or:\n"
                  "  GOOGLE_API_KEY=AIza... python3 backfill_typed_dicts.py ...",
                  file=sys.stderr)
            return 1
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(args.model)
            print(f"  loaded AI Studio model: {args.model}")
        except Exception as e:
            print(f"ERROR: could not init AI Studio Gemini: {e}",
                  file=sys.stderr)
            return 1

    print(f"\n=== executing — {len(gaps)} rows in batches of {BATCH_SIZE} ===")
    t_start = time.time()
    n_done = 0
    n_topic_filled = 0
    n_entity_filled = 0

    batch: list[dict] = []
    last_call = 0.0
    interval = 1.0 / RATE_LIMIT_RPS if args.reextract else 0.0

    for g in gaps:
        update = {
            "aid": g["aid"],
            "user_id": g["user_id"],
            "namespace": g["namespace"],
        }

        if args.synthesize:
            if not g["topic_dicts_present"] and g["topics"]:
                update["new_topic_dicts"] = synthesize_topic_dicts(g["topics"])
                n_topic_filled += 1
            if not g["entity_dicts_present"] and g["entities"]:
                update["new_entity_dicts"] = synthesize_entity_dicts(g["entities"])
                n_entity_filled += 1
        else:
            # Rate-limit Gemini calls
            now = time.time()
            wait = (last_call + interval) - now
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            tdicts, edicts = reextract_via_ai_studio(g["text"], model)
            if not g["topic_dicts_present"] and tdicts:
                update["new_topic_dicts"] = tdicts
                n_topic_filled += 1
            if not g["entity_dicts_present"] and edicts:
                update["new_entity_dicts"] = edicts
                n_entity_filled += 1

        # Skip rows where extraction returned nothing — no point rewriting
        if "new_topic_dicts" not in update and "new_entity_dicts" not in update:
            continue

        batch.append(update)
        n_done += 1

        if len(batch) >= BATCH_SIZE:
            nt, ne = apply_batch(ds, batch)
            elapsed = time.time() - t_start
            print(f"  applied batch: {n_done}/{len(gaps)} rows "
                  f"(topic+={nt}, entity+={ne}, {elapsed:.1f}s elapsed)",
                  flush=True)
            batch = []

    if batch:
        nt, ne = apply_batch(ds, batch)
        print(f"  applied final batch: topic+={nt}, entity+={ne}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== done in {elapsed:.1f}s ===")
    print(f"  rows updated      : {n_done}")
    print(f"  topic_dicts filled: {n_topic_filled}")
    print(f"  entity_dicts filled: {n_entity_filled}")
    print(f"\nRecommend running maintenance after this to compact:")
    print(f"  sudo systemctl start veilguard-tcmm-maintenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
