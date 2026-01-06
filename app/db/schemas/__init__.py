"""
Database schemas and SQL scripts.
"""

from pathlib import Path

SCHEMA_DIR = Path(__file__).parent
CREATE_TABLES_SQL = SCHEMA_DIR / "create_tables.sql"

__all__ = ['SCHEMA_DIR', 'CREATE_TABLES_SQL']