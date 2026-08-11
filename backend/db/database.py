"""SQLite database engine and session management via SQLModel.

The database file is stored at the path configured by DATABASE_URL
(default: sqlite:///data/app.db).
"""

from pathlib import Path

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


def init_db() -> None:
    """Create all tables defined by SQLModel metadata.

    Safe to call multiple times — SQLModel/SQLAlchemy will skip tables
    that already exist.
    """
    # Import models so their tables are registered with SQLModel.metadata
    import backend.db.models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    logger.info("Database tables initialised at %s", settings.database_url)


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    return Session(engine)
