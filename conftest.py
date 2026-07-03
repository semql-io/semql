"""Workspace-wide pytest configuration.

Registers the Hypothesis settings profiles (property-testing.md §3.1) so
individual property tests no longer carry per-test ``@settings`` boilerplate.
Select a profile with the ``HYPOTHESIS_PROFILE`` env var:

- ``dev`` (default) — fast inner loop, 50 examples.
- ``ci`` — 200 examples, derandomised so PR runs are reproducible, and the
  failing-example blob is printed for copy-paste repro.
- ``nightly`` — 5000 examples, stays random to keep exploring.

Also assigns the W5/§10.5 test-tier markers by path in one place (see
``pytest_collection_modifyitems``) so the tier policy is centralised and
greppable rather than scattered as per-file ``pytestmark`` lines. An
unmarked test is a ``unit`` test by convention.
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
    print_blob=True,
)
settings.register_profile("nightly", max_examples=5_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# --- W5/§10.5 test tiers -----------------------------------------------------
# Every test in the engine package drives a real (in-process) engine/adapter
# oracle, so the whole package is ``integration``. Security lives in dedicated
# files across packages — enumerate them here rather than sprinkling
# ``pytestmark`` so the policy is one greppable list.
# DuckDB execution-oracle tests outside the engine package.
_INTEGRATION_FILES = frozenset(
    {
        "test_symmetric_aggregation.py",
        "test_federate_where_segments.py",
    }
)
_SECURITY_FILES = frozenset(
    {
        "test_auth.py",
        "test_auth_attrs.py",
        "test_scope.py",
        "test_tenancy.py",
        "test_sql_injection.py",
        "test_security_sql.py",
        "test_core_security_regressions.py",
        "test_mcp_security_regressions.py",
        "test_prompt_security_regressions.py",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply tier markers by path (see module docstring)."""
    for item in items:
        parts = item.path.parts
        if "semql-engine" in parts or item.path.name in _INTEGRATION_FILES:
            item.add_marker(pytest.mark.integration)
        if item.path.name in _SECURITY_FILES:
            item.add_marker(pytest.mark.security)
