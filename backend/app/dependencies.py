"""FastAPI dependencies shared by application routers."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy session for the lifetime of a request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
