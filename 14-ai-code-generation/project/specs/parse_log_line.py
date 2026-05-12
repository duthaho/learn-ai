import re


def parse_log_line(line: str) -> dict | None:
    """Parse a log line of the form '<ISO_TIMESTAMP> <LEVEL> <MESSAGE>'.

    Returns a dict with keys 'timestamp', 'level', 'message' if the line
    is well-formed; returns None if it cannot be parsed.

    LEVEL is one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    TIMESTAMP is an ISO 8601 datetime (e.g., 2026-01-15T12:34:56).
    """
    ...


def test_well_formed():
    result = parse_log_line("2026-01-15T12:34:56 INFO server started")
    assert result == {
        "timestamp": "2026-01-15T12:34:56",
        "level": "INFO",
        "message": "server started",
    }


def test_error_level():
    result = parse_log_line("2026-01-15T12:34:56 ERROR connection refused")
    assert result is not None
    assert result["level"] == "ERROR"


def test_malformed_returns_none():
    assert parse_log_line("not a log line") is None
    assert parse_log_line("") is None


def test_extra_whitespace_in_message():
    # Multiple spaces inside the message should be preserved.
    result = parse_log_line("2026-01-15T12:34:56 WARNING  multiple   spaces")
    assert result is not None
    assert result["message"] == " multiple   spaces" or result["message"] == "multiple   spaces"


def test_unknown_level_returns_none():
    # FOO is not a valid level.
    assert parse_log_line("2026-01-15T12:34:56 FOO something") is None
