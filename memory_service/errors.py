"""Errors shared by the PostgreSQL and SQLite storage backends."""


class RequestConflictError(Exception):
    """A request ID was reused with different content."""
