from collections.abc import Generator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# El schema real vive en el schema de Postgres "routerkit" (ver
# postgres/init/001_schema.sql), no en "public".
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
