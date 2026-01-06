"""
Database connection management using SQLAlchemy.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Create SQLAlchemy engine with connection pooling."""
        try:
            self.engine = create_engine(
                settings.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_pre_ping=True,
                echo=settings.DEBUG,
            )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

            self._add_event_listeners()

            logger.info(
                f"Database engine initialized: "
                f"{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def _add_event_listeners(self) -> None:
        """Add SQLAlchemy event listeners for monitoring."""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("Database connection established")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope for database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rollback due to error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        """Create all tables defined in SQLAlchemy models."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    def execute_sql_file(self, sql_file_path: str) -> None:
        """
        Execute a SQL file as a single script.
        Required for PostgreSQL functions & triggers.
        """
        try:
            with open(sql_file_path, "r", encoding="utf-8") as f:
                sql = f.read()

            with self.engine.begin() as connection:
                connection.execute(text(sql))

            logger.info(f"SQL file executed successfully: {sql_file_path}")

        except Exception as e:
            logger.error(f"Failed to execute SQL file: {e}")
            raise

    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# -------------------------------------------------------------------
# Global database manager
# -------------------------------------------------------------------

db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get a database session.
    """
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    """
    Initialize database using raw SQL schema.
    Run once during setup.
    """
    try:
        # Execute SQL schema file
        db_manager.execute_sql_file("app/db/schemas/create_tables.sql")
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
