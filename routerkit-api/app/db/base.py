from collections.abc import Generator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# The actual schema lives in the Postgres schema "routerkit" (see
# postgres/init/001_schema.sql), not in "public".
metadata = MetaData(schema="routerkit")


class Base(DeclarativeBase):
    metadata = metadata


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
