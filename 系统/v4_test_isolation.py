"""Explicit pytest plugin for the audited synthetic fourth-version suites.

Use ``-p v4_test_isolation``. Product behavior is not changed. Synthetic
engineering checks must not use packaged historical statistics as sample data.
"""
import pytest


@pytest.fixture(autouse=True)
def exclude_historical_report_sources(monkeypatch):
    from reporting import evidence_package
    monkeypatch.setattr(evidence_package, "_read", lambda _path: {})
    monkeypatch.setattr(evidence_package, "_known_issues", lambda: {
        "available": False, "note": "Synthetic isolation: historical documents not loaded"})
