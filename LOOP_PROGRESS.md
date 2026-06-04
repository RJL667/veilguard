# Overnight bug-fix loop — progress log

## Iteration 1 — Bug #1: entity_dicts / topic_dicts 0% in archive
**CONCLUSION: not a current code bug — SKIP (low impact + historical).**

Traced the full path, all layers correct:
- adapter `ai_studio_nlp_adapter.py` `_parse` (line 629) + `process_batch_gemma` (1340) emit `entity_dicts`/`topic_dicts` alongside flat lists (result dict line 1490-1498).
- `core/workers.py:502/547` reads them from the NLP result, `:596/598` puts them in the overlay `data`.
- `core/archive.py:1561` (`attach_semantic_overlay`) writes them (`if raw_ed:` guard).
- `core/providers/lance.py:1851-1852` serializes the `list<struct<name,type,score>>` columns; schema has them (244/256).

Root cause: the 915 existing records were enriched BEFORE this dict-emission code existed (in-code comment line 706: *"topic_dicts column was always empty in Lance"* = historical). Flat `entities`/`topics` are ~70% (they always worked); the typed dicts are the newer addition.

Impact: LOW. Dream engine + recall fall back to flat lists (`core/dream/arc_builder.py:370` "Fallback: plain entity strings"). Nothing is broken.

Fix options (NOT done — both deferred): (a) backfill = re-enrich 915 rows = 915 AIStudio calls (expensive/risky overnight); (b) nothing — new records will populate them.
**VERIFY LATER:** after the user next chats, query the newest archive record's `entity_dicts` — if populated, confirmed historical-only and no action needed.

## Next: Bug #2 — ~27% of archive records missing session_id (ingest gap).
## Then: Bug #3 — test-debris cleanup (DESTRUCTIVE — be very conservative; only clearly-mine conv-* test convs, verify before any delete).

## Iteration 2 — Bug #2: 27% of archive records missing session_id
**CONCLUSION: not a bug — expected behavior.** The 240 missing-session records are
almost all dream-cycle synthesized nodes (extracted_by=agent:dream-cycle,
namespace=agent_dream-cycle_observ, 238) + 2 critic-promoted team-knowledge nodes.
Those are DERIVED memory, not chat turns — they correctly have no chat session_id.
The 666 real chat turns all HAVE session_id. No fix needed. (Cross-session recall
grouping handles these via namespace, not session_id, so they downplay correctly.)

## Iteration 3 — Bug #3: test-debris cleanup
**CONCLUSION: nothing to clean (safely).** 0 rows match conv-*-0601* (my chat tests
never reached the searchable archive). The archive = dream nodes (238, keep) +
multi-agent IC sub-task-* blocks (the rest, from earlier platform testing). The
sub-task-* blocks are the ones that surface as recall "debris", but they are NOT
clearly mine to delete, and deleting from the live Lance archive while TCMM holds it
open in-memory risks on-disk/in-memory divergence. SKIP per "if unsure / risky, skip".
NOTE for user: if you want them gone, stop TCMM first, then delete by namespace.

## Iteration 4 — Bug #4 + loop status
Reviewed for other concrete safe bugs. The candidates all need a product decision or
are risky overnight: topic_dicts backfill (915 AIStudio calls, may 429-storm);
excluding agent namespaces (sub-task-*) from user chat recall (recall-scoping change —
the downplay grouping already mitigates this); .wslconfig memory cap (needs
wsl --shutdown, bounces the whole stack). All deferred per constraints.

## LOOP STOPPED — backlog exhausted of safe, clear overnight fixes.
Net: #1 #2 #3 all turned out to be non-bugs / expected behavior (system is healthy).
The real fixes from this session were already landed live during the session
(persona double-write, marker-collision, MCP-suffix dispatch, _first_user_text
list-content, recall grouping + downplay, admin 3GB leak + request_detail leak).
Nothing left that's safe to change unattended. Awaiting user.
