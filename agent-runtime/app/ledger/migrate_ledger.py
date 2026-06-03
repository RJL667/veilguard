"""Copy the agentic ledger tables from a LanceDB ledger dir into Postgres.

Straight per-table copy — ledger rows carry their own ids (task_id, proposal_id,
etc.), so there's no aid remap. Only tables registered in TABLE_SCHEMAS are
copied; unregistered legacy tables (e.g. org_memory) are skipped, matching the
PgLedgerStore surface.
"""
import logging

logger = logging.getLogger(__name__)


def migrate_ledger(lance_db_path: str, dsn: str) -> dict:
    """Copy every registered ledger table from the Lance ledger dir to Postgres.
    Returns {table: row_count}."""
    import lancedb
    from .schemas import TABLE_SCHEMAS
    from .pg_store import PgLedgerStore

    src = lancedb.connect(str(lance_db_path))
    dst = PgLedgerStore(dsn)
    existing = set(src.table_names())
    counts = {}
    for name in TABLE_SCHEMAS:
        if name not in existing:
            continue
        try:
            rows = src.open_table(name).to_arrow().to_pylist()
        except Exception as e:
            logger.warning("[migrate_ledger] read %s failed: %s", name, e)
            continue
        if rows:
            dst.table(name).add(rows)
        counts[name] = len(rows)
        logger.info("[migrate_ledger] %s -> postgres: %d rows", name, len(rows))
    return counts
