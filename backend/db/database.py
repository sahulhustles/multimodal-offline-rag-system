"""SQLite database engine and session management via SQLModel.

The database file is stored at the path configured by DATABASE_URL
(default: sqlite:///data/app.db).
"""

from pathlib import Path

from sqlalchemy import text, inspect
from sqlmodel import SQLModel, Session, create_engine

from backend.config import settings
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Ensure the database directory exists before creating the engine.
# sqlite:///data/app.db  →  data/app.db  (relative path, triple-slash)
_db_path = settings.database_url.replace("sqlite:///", "")
Path(_db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},
)


def _run_migrations() -> None:
    """Apply safe schema migrations for new columns on existing tables.

    Each migration checks whether the column already exists before
    attempting an ALTER TABLE, making this idempotent.
    """
    inspector = inspect(engine)

    # --- Phase 3: Add indexing columns to source_documents ---
    if "source_documents" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("source_documents")}
        with engine.begin() as conn:
            if "indexing_status" not in existing_cols:
                conn.execute(text(
                    "ALTER TABLE source_documents "
                    "ADD COLUMN indexing_status VARCHAR DEFAULT 'not_started'"
                ))
                logger.info("Migration: added indexing_status to source_documents")
            if "indexing_error_message" not in existing_cols:
                conn.execute(text(
                    "ALTER TABLE source_documents "
                    "ADD COLUMN indexing_error_message TEXT"
                ))
                logger.info("Migration: added indexing_error_message to source_documents")

    # --- Phase 3 bugfix: Add image occurrence columns to indexed_records ---
    if "indexed_records" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("indexed_records")}
        with engine.begin() as conn:
            if "image_index" not in existing_cols:
                conn.execute(text(
                    "ALTER TABLE indexed_records "
                    "ADD COLUMN image_index INTEGER"
                ))
                logger.info("Migration: added image_index to indexed_records")
            if "image_sha256" not in existing_cols:
                conn.execute(text(
                    "ALTER TABLE indexed_records "
                    "ADD COLUMN image_sha256 VARCHAR"
                ))
                logger.info("Migration: added image_sha256 to indexed_records")


def init_db() -> None:
    """Create all tables defined by SQLModel metadata.

    Safe to call multiple times — SQLModel/SQLAlchemy will skip tables
    that already exist.
    """
    # Import models so their tables are registered with SQLModel.metadata
    import backend.db.models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _run_migrations()
    logger.info("Database tables initialised at %s", settings.database_url)


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    return Session(engine)

