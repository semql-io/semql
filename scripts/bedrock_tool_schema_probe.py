#!/usr/bin/env -S uv run --with boto3
"""Probe how Amazon Bedrock's Converse API validates tool ``inputSchema``\\ s.

Bedrock validates every tool schema at request time, *above* the model: the
top level must carry ``type: "object"``, so a bare root ``$ref`` (what Pydantic
emits for a recursive root model) is rejected on every model family with
``inputSchema.json.type must be one of the following: object``. Internal
``$ref`` / ``$defs`` (including recursive cycles) and ``prefixItems`` tuples are
accepted fine. ``semql_prompt.bedrock.flatten_root_ref`` exists to fix the root.

This script keeps that knowledge executable instead of tribal. Two modes:

* ``constraints`` — submit a fixed battery of schema shapes (root ``$ref``,
  nested ``$ref``, ``prefixItems``, recursion) and check Bedrock's behavior
  against documented expectations. Re-run when AWS ships new models or changes
  the validator; a mismatch means the platform moved and our assumptions
  (and ``flatten_root_ref``) need revisiting.
* ``repo`` — take the *actual* schema this library ships
  (``SemanticQuery.model_json_schema()``), run it through ``flatten_root_ref``,
  and confirm Bedrock accepts it across models. With ``--raw`` it submits the
  un-flattened schema too, to demonstrate the rejection the post-processor cures.

Needs AWS credentials with ``bedrock:InvokeModel`` and the models enabled in the
target region. boto3 is pulled in ephemerally via ``--with`` (no manifest entry),
and ``uv run`` overlays it on the project env so ``repo`` mode can still import
the workspace packages. Run from the repo root::

    uv run --with boto3 scripts/bedrock_tool_schema_probe.py
    uv run --with boto3 scripts/bedrock_tool_schema_probe.py repo --raw
    uv run --with boto3 scripts/bedrock_tool_schema_probe.py --region us-east-1 \
        --models us.amazon.nova-lite-v1:0 us.meta.llama4-scout-17b-instruct-v1:0

The shebang already embeds ``--with boto3``, so ``./scripts/bedrock_tool_schema_probe.py``
works directly too.

Exit code is non-zero if any result diverges from expectation (a real schema
rejection of a schema we expect to pass, or a constraint behaving differently
than documented), so it doubles as a regression check for ``flatten_root_ref``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

TOOL_NAME = "probe_tool"

# Representative cross-family default set. Override with --models. These are
# us. cross-region inference-profile ids; swap the prefix for another geo.
DEFAULT_MODELS: tuple[str, ...] = (
    "us.amazon.nova-lite-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
)

# ---------------------------------------------------------------------------
# Constraint battery. ``schema_ok`` is the documented expectation: True means
# Bedrock should accept the schema (generation may still fail model-side);
# False means Bedrock should reject it at validation time.
# ---------------------------------------------------------------------------
CONSTRAINT_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "root_ref",
        "schema_ok": False,  # the canonical failure flatten_root_ref fixes
        "prompt": "call the tool",
        "schema": {
            "$ref": "#/$defs/Q",
            "$defs": {
                "Q": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
            },
        },
    },
    {
        "name": "nested_ref",
        "schema_ok": True,
        "prompt": "call the tool with inner.a = hi",
        "schema": {
            "type": "object",
            "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "string"}}}},
            "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            "required": ["inner"],
        },
    },
    {
        "name": "prefixItems_tuple",
        "schema_ok": True,
        "prompt": "Query starting 2026-01-01 grouped by month.",
        "schema": {
            "type": "object",
            "properties": {
                "time_window": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "string", "description": "ISO start date"},
                        {"type": "string", "enum": ["day", "week", "month"]},
                    ],
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["time_window"],
        },
    },
    {
        "name": "internal_recursion",
        "schema_ok": True,  # validates everywhere; generation may fail (Nova)
        "prompt": "Find rows where status = open AND priority = high.",
        "schema": {
            "type": "object",
            "$defs": {
                "Filter": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "op": {"type": "string"},
                        "value": {"type": "string"},
                        "and_": {"type": "array", "items": {"$ref": "#/$defs/Filter"}},
                    },
                }
            },
            "properties": {"where": {"$ref": "#/$defs/Filter"}},
            "required": ["where"],
        },
    },
)


def _classify(
    client: Any,  # noqa: ANN401 — boto3 bedrock-runtime client is untyped (no stubs)
    model: str,
    schema: Mapping[str, Any],
    prompt: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Submit one (model, schema) pair; return ``(bucket, detail)``.

    Buckets: ``accepted`` (schema valid, tool emitted), ``no_tool`` (schema
    valid, no toolUse block — reasoning-channel / auto mode), ``gen_fail``
    (schema valid, model produced an invalid sequence), ``schema_reject``
    (Bedrock rejected the inputSchema), ``unsupported`` (model lacks forced
    toolChoice — retried with auto), ``access`` (model not enabled / denied),
    ``error`` (anything else). Only ``schema_reject`` counts as a schema verdict
    of "invalid".
    """
    from botocore.exceptions import ClientError

    def _invoke(force_tool: bool) -> dict[str, Any]:
        tool_config: dict[str, Any] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": TOOL_NAME,
                        "description": "probe tool",
                        "inputSchema": {"json": dict(schema)},
                    }
                }
            ]
        }
        if force_tool:
            tool_config["toolChoice"] = {"tool": {"name": TOOL_NAME}}
        return client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )

    try:
        resp = _invoke(force_tool=True)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        msg = exc.response["Error"]["Message"]
        low = msg.lower()
        if code == "ValidationException" and "toolchoice" in low:
            try:
                resp = _invoke(force_tool=False)
            except ClientError as exc2:
                return _classify_error(exc2)
            return _summarize(resp)
        if code == "ValidationException" and ("inputschema" in low or "json schema" in low):
            return "schema_reject", msg
        return _classify_error(exc)
    return _summarize(resp)


def _summarize(resp: dict[str, Any]) -> tuple[str, str]:
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return "accepted", json.dumps(block["toolUse"]["input"])
    return "no_tool", "schema valid; no toolUse block emitted"


def _classify_error(exc: Any) -> tuple[str, str]:  # noqa: ANN401 — botocore ClientError is untyped
    code = exc.response["Error"]["Code"]
    msg = exc.response["Error"]["Message"]
    if code == "ModelErrorException":
        return "gen_fail", "schema valid; model produced an invalid sequence"
    if code in ("AccessDeniedException", "ResourceNotFoundException"):
        return "access", f"{code}: {msg}"
    return "error", f"{code}: {msg}"


# Buckets where Bedrock accepted the schema (the verdict we usually care about).
_SCHEMA_OK_BUCKETS = frozenset({"accepted", "no_tool", "gen_fail"})


def _run_cases(
    client: Any,  # noqa: ANN401 — boto3 bedrock-runtime client is untyped (no stubs)
    models: tuple[str, ...],
    cases: list[dict[str, Any]],
    max_tokens: int,
) -> int:
    """Run every case against every model; print a matrix; return failure count."""
    failures = 0
    for case in cases:
        expected = case.get("schema_ok")
        exp_str = "" if expected is None else f"  (expect schema {'OK' if expected else 'REJECT'})"
        print(f"\n### {case['name']}{exp_str}")
        for model in models:
            bucket, detail = _classify(client, model, case["schema"], case["prompt"], max_tokens)
            flag = " "
            if expected is not None and bucket != "access":
                got_ok = bucket in _SCHEMA_OK_BUCKETS
                if got_ok != expected:
                    flag = "✗"
                    failures += 1
                else:
                    flag = "✓"
            print(f"  {flag} {model:<48} {bucket:<13} {detail[:90]}")
    return failures


def _repo_cases(*, raw: bool, include_boolexpr: bool) -> list[dict[str, Any]]:
    """Build cases from the schemas this library actually ships."""
    from semql import flatten_root_ref
    from semql.spec import BoolExpr, SemanticQuery

    cases: list[dict[str, Any]] = [
        {
            "name": "SemanticQuery (flattened)",
            "schema_ok": True,
            "prompt": "List the regions.",
            "schema": flatten_root_ref(SemanticQuery.model_json_schema()),
        }
    ]
    if include_boolexpr:
        cases.append(
            {
                "name": "BoolExpr (flattened)",
                "schema_ok": True,
                "prompt": "status = open AND priority = high",
                "schema": flatten_root_ref(BoolExpr.model_json_schema()),
            }
        )
    if raw:
        # The un-flattened recursive root — expected to be rejected, proving
        # what flatten_root_ref cures.
        cases.append(
            {
                "name": "BoolExpr (raw root $ref)",
                "schema_ok": False,
                "prompt": "status = open AND priority = high",
                "schema": BoolExpr.model_json_schema(),
            }
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["constraints", "repo", "both"],
        default="constraints",
        help="Which battery to run (default: constraints).",
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model / inference-profile ids to probe (default: a cross-family set).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="maxTokens per call (default: 128)."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="repo mode: also submit the un-flattened recursive schema to show the rejection.",
    )
    parser.add_argument(
        "--include-boolexpr",
        action="store_true",
        help="repo mode: also probe the flattened recursive BoolExpr schema.",
    )
    args = parser.parse_args(argv)

    try:
        import boto3
    except ImportError:
        print(
            "boto3 is required. Run via: uv run --with boto3 scripts/bedrock_tool_schema_probe.py",
            file=sys.stderr,
        )
        return 2

    client = boto3.client("bedrock-runtime", region_name=args.region)
    models = tuple(args.models)

    cases: list[dict[str, Any]] = []
    if args.mode in ("constraints", "both"):
        cases.extend(CONSTRAINT_CASES)
    if args.mode in ("repo", "both"):
        cases.extend(_repo_cases(raw=args.raw, include_boolexpr=args.include_boolexpr))

    print(f"Region: {args.region}   Models: {len(models)}   Cases: {len(cases)}")
    print("Legend: ✓ matches expectation  ✗ diverges  (blank = no expectation / access-blocked)")
    failures = _run_cases(client, models, cases, args.max_tokens)

    if failures:
        print(f"\n{failures} result(s) diverged from expectation.")
        return 1
    print("\nAll results matched expectation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
