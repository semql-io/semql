"""Workspace-wide pytest configuration.

Registers the Hypothesis settings profiles (property-testing.md §3.1) so
individual property tests no longer carry per-test ``@settings`` boilerplate.
Select a profile with the ``HYPOTHESIS_PROFILE`` env var:

- ``dev`` (default) — fast inner loop, 50 examples.
- ``ci`` — 200 examples, derandomised so PR runs are reproducible, and the
  failing-example blob is printed for copy-paste repro.
- ``nightly`` — 5000 examples, stays random to keep exploring.
"""

from __future__ import annotations

import os

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
