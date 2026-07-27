"""Typed application errors.

Every failure the user can trigger has a class here so the API can map it to a
status code and a human-readable message instead of leaking a traceback.
"""

from __future__ import annotations


class AnalystError(Exception):
    """Base class for all expected (non-bug) failures."""

    status_code: int = 400
    code: str = "analyst_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class IngestionError(AnalystError):
    code = "ingestion_error"


class FileTooLargeError(IngestionError):
    code = "file_too_large"
    status_code = 413


class UnsupportedFileError(IngestionError):
    code = "unsupported_file"
    status_code = 415


class EmptyFileError(IngestionError):
    code = "empty_file"


class SchemaError(AnalystError):
    code = "schema_error"


class UnsafeQueryError(AnalystError):
    """Raised by the SQL guard. Never contains executable SQL in `message`."""

    code = "unsafe_query"
    status_code = 422


class QueryExecutionError(AnalystError):
    code = "query_execution_error"


class QueryTimeoutError(QueryExecutionError):
    code = "query_timeout"
    status_code = 504


class LLMNotConfiguredError(AnalystError):
    code = "llm_not_configured"
    status_code = 503


class LLMError(AnalystError):
    code = "llm_error"
    status_code = 502


class LLMRateLimitError(LLMError):
    code = "llm_rate_limited"
    status_code = 429


class SessionNotFoundError(AnalystError):
    code = "session_not_found"
    status_code = 404


class DatasetNotFoundError(AnalystError):
    code = "dataset_not_found"
    status_code = 404


class AgentError(AnalystError):
    code = "agent_error"


class ToolError(AnalystError):
    """A tool failed in a way the agent is allowed to see and retry from."""

    code = "tool_error"
