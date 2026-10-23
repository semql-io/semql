"""Dimension-value resolution helpers.

A :class:`semql.model.Lookup` declares the finite set of valid values
for a string dimension. ``Lookup`` lives in the model; this module
turns lookups into something callers can *use* at request time:

- :func:`materialize` — fire any ``loader`` against a
  :class:`ResolutionContext` and return the canonical
  ``(values, labels?)`` tuple, or ``None`` when the lookup is dynamic
  and no ``ctx`` was provided.
- :func:`resolve` — turn a free-text query ("paid east", "europe")
  into a list of canonical dimension values via exact / substring /
  fuzzy matching.

This module is the I/O surface of the lookup system. The compiler
never touches it; ``semql_prompt.planner_prompt(...)`` and any user-supplied
``resolve_<dim>`` tool do.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from types import MappingProxyType
from typing import Any, Literal

from semql.analysis import EnrichmentProvenance, SemanticAnalysis
from semql.catalog import Catalog
from semql.errors import ContractError
from semql.model import Lookup, MultiFieldEnricher, ResolutionContext
from semql.safe import is_safe_sql_identifier
from semql.spec import Filter, FilterOp, SemanticQuery

# ---------------------------------------------------------------------------
# Materialization — turn a Lookup into a concrete (values, labels) tuple
# ---------------------------------------------------------------------------


def materialize(
    lookup: Lookup, ctx: ResolutionContext | None
) -> tuple[list[str], dict[str, str] | None] | None:
    """Materialize ``lookup`` to ``(values, labels?)`` for the given ``ctx``.

    Static lookups (``values=`` declared) return their inlined tuple
    regardless of ``ctx``. Dynamic lookups (``loader=`` declared) fire
    the loader against ``ctx`` — this is the I/O boundary. Returns
    ``None`` for dynamic lookups when ``ctx`` is ``None``, so callers
    can route to a "values resolved at runtime" fallback instead of
    inventing a stale answer."""
    if lookup.values is not None:
        return list(lookup.values), dict(lookup.labels) if lookup.labels else None
    if lookup.loader is None:
        return None  # validator forbids this; defensive
    if ctx is None:
        return None
    result = lookup.loader(ctx)
    if isinstance(result, dict):
        return list(result.keys()), dict(result)
    return list(result), dict(lookup.labels) if lookup.labels else None


# ---------------------------------------------------------------------------
# Resolution — free-text query → canonical dimension values
# ---------------------------------------------------------------------------


def _match_values(
    values: list[str], labels: dict[str, str] | None, query: str, max_candidates: int
) -> tuple[tuple[str, ...], Literal["exact", "substring", "fuzzy"] | None, int]:
    needle = query.strip().casefold()
    if not needle:
        return (), None, 0
    label_for = labels or {}
    exact = tuple(
        value
        for value in values
        if value.casefold() == needle or label_for.get(value, "").casefold() == needle
    )
    if exact:
        return exact[:max_candidates], "exact", len(exact)
    substring = tuple(
        value
        for value in values
        if needle in value.casefold() or needle in label_for.get(value, "").casefold()
    )
    if substring:
        return substring[:max_candidates], "substring", len(substring)
    scored: list[tuple[float, str]] = []
    for value in values:
        score = SequenceMatcher(None, needle, value.casefold()).ratio()
        if value in label_for:
            score = max(score, SequenceMatcher(None, needle, label_for[value].casefold()).ratio())
        if score >= 0.5:
            scored.append((score, value))
    scored.sort(key=lambda item: -item[0])
    return tuple(value for _, value in scored[:max_candidates]), "fuzzy", len(scored)


def resolve(
    catalog: Catalog,
    dimension: str,
    query: str,
    *,
    ctx: ResolutionContext | None = None,
    max_candidates: int = 5,
) -> list[str]:
    """Return bounded candidates, not evidence of unique authoritative resolution.

    Use :func:`resolve_outcome` for status, authority, and completeness. A
    single returned candidate may be fuzzy or drawn from a truncated list.
    """
    if dimension not in catalog.lookups:
        raise KeyError(
            f"No Lookup registered for {dimension!r}. Registered: {sorted(catalog.lookups)}."
        )
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    values_labels = materialize(catalog.lookups[dimension], ctx)
    if values_labels is None:
        return []
    values, labels = values_labels
    matches, _match_kind, _total = _match_values(values, labels, query, max_candidates)
    return list(matches)


# ---------------------------------------------------------------------------
# Resolution stage — free-text filter phrases -> canonical keys, typed
# ---------------------------------------------------------------------------

# Operators whose values are membership keys worth canonicalizing. ``contains``
# / ``gt`` / ``lt`` carry patterns or bounds, not values to resolve; ``is_null``
# / ``not_null`` carry none.
_RESOLVABLE_OPS: frozenset[FilterOp] = frozenset({"eq", "neq", "in", "not_in"})


@dataclass(frozen=True)
class ResolutionOutcome:
    """One bounded lookup result with explicit authority and completeness."""

    dimension: str
    query: str
    status: Literal["resolved", "ambiguous", "not_found", "unavailable"]
    values: tuple[str, ...] = ()
    labels: Mapping[str, str] | None = None
    match_kind: Literal["exact", "substring", "fuzzy"] | None = None
    authoritative: bool = False
    complete: bool = True
    truncated: bool = False
    total_candidates: int = 0

    def __post_init__(self) -> None:
        if self.labels is not None:
            object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def value(self) -> str:
        if not self.resolved:
            raise ValueError(
                f"Resolution for {self.dimension!r} is {self.status!r}; inspect values."
            )
        return self.values[0]


def resolve_outcome(
    catalog: Catalog,
    dimension: str,
    query: str,
    *,
    ctx: ResolutionContext | None = None,
    max_candidates: int = 5,
) -> ResolutionOutcome:
    """Resolve once, preserving certainty before candidate truncation."""
    if dimension not in catalog.lookups:
        raise KeyError(
            f"No Lookup registered for {dimension!r}. Registered: {sorted(catalog.lookups)}."
        )
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    try:
        materialized = materialize(catalog.lookups[dimension], ctx)
    except Exception:
        return ResolutionOutcome(dimension, query, "unavailable", complete=False)
    if materialized is None:
        return ResolutionOutcome(dimension, query, "unavailable", complete=False)
    values, all_labels = materialized
    matches, match_kind, total = _match_values(values, all_labels, query, max_candidates)
    labels = (
        {value: all_labels[value] for value in matches if value in all_labels} or None
        if all_labels
        else None
    )
    truncated = total > len(matches)
    complete = not truncated
    if total == 0:
        return ResolutionOutcome(dimension, query, "not_found")
    authoritative = match_kind != "fuzzy" and total == 1 and not truncated
    status: Literal["resolved", "ambiguous", "not_found", "unavailable"] = (
        "resolved" if authoritative else "ambiguous"
    )
    return ResolutionOutcome(
        dimension,
        query,
        status,
        matches,
        labels,
        match_kind,
        authoritative,
        complete,
        truncated,
        total,
    )


@dataclass(frozen=True)
class QueryResolution:
    """A query whose lookup-backed filter values have been canonicalized where
    possible, plus every :class:`ResolutionOutcome` attempted.

    ``query`` has resolved values spliced in; any filter with an ambiguous /
    unresolved value is left exactly as the caller wrote it (see ``blocked``).
    Feed ``query`` to :func:`semql.autoplan.autoplan` only when ``ok``."""

    query: SemanticQuery
    outcomes: tuple[ResolutionOutcome, ...] = ()

    @property
    def blocked(self) -> tuple[ResolutionOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.resolved)

    @property
    def ok(self) -> bool:
        return not self.blocked


def resolve_query_filters(
    q: SemanticQuery,
    catalog: Catalog,
    *,
    ctx: ResolutionContext | None = None,
    max_candidates: int = 5,
) -> QueryResolution:
    """Canonicalize every free-text membership filter on a Lookup-backed
    dimension (the sans-io compiler / Auto-Planner then sees canonical values).

    A filter resolves *atomically*: only when every one of its values resolves
    to a single canonical key is it spliced; if any value is ambiguous or
    unresolved the filter is left verbatim and the blocking outcomes are
    reported via ``blocked``. Non-string values, non-membership ops, and
    dimensions without a registered ``Lookup`` pass through untouched. The
    boolean ``q.where`` tree is out of scope (kept as-is)."""
    outcomes: list[ResolutionOutcome] = []
    new_filters: list[Filter] = []
    for f in q.filters:
        if f.dimension not in catalog.lookups or f.op not in _RESOLVABLE_OPS:
            new_filters.append(f)
            continue
        resolved_values: list[str | int | float | bool] = []
        blocked = False
        for v in f.values:
            if not isinstance(v, str):
                resolved_values.append(v)
                continue
            outcome = resolve_outcome(
                catalog, f.dimension, v, ctx=ctx, max_candidates=max_candidates
            )
            outcomes.append(outcome)
            if outcome.resolved:
                resolved_values.append(outcome.value)
            else:
                blocked = True
        new_filters.append(f if blocked else f.model_copy(update={"values": resolved_values}))
    return QueryResolution(
        query=q.model_copy(update={"filters": new_filters}), outcomes=tuple(outcomes)
    )


def enrich_result(
    rows: list[dict[str, object]],
    dim_name: str,
    lookup: Lookup,
    ctx: ResolutionContext,
) -> list[dict[str, object]]:
    """Attach reference fields to each row via the lookup's enricher.

    Reads ``lookup.enricher`` only — the post-query slot. The vocabulary
    (``values`` / ``loader``) is never touched here, so a lookup's
    prompt-time vocabulary and its result-time enrichment stay decoupled:
    an enricher-only lookup attaches labels without ever surfacing ids to
    the planner. Two enricher shapes, checked in this order:

    - :class:`~semql.model.MultiFieldEnricher` (``enrich_fields``) attaches
      *several* columns per id — one ``<dim_name>__<field>`` column per
      resolved field (e.g. ``region_id__name``, ``region_id__manager``).
      An id absent from the mapping adds no columns for that row; a field
      absent for a present id is simply omitted.
    - :class:`~semql.model.LookupEnricher` (``enrich``) attaches a single
      ``<dim_name>__label`` column. Missing ids echo the raw id as the
      label.

    A lookup with no ``enricher`` leaves rows unchanged (the model validator
    guarantees any ``enricher`` satisfies one of the two protocols, so there
    is no "implements neither" case to handle here). Rows whose dimension
    value is ``None`` are always skipped. When the enricher implements both
    protocols the multi-field path wins. Mutates the row dicts in place — the
    enriched columns are a guaranteed attachment, not an alternative view.
    """
    enricher = lookup.enricher
    if enricher is None:
        return rows
    ids = list({str(r[dim_name]) for r in rows if r.get(dim_name) is not None})
    if not ids:
        return rows

    if isinstance(enricher, MultiFieldEnricher):
        field_map = enricher.enrich_fields(ids, ctx)
        for row in rows:
            raw = row.get(dim_name)
            if raw is None:
                continue
            fields = field_map.get(str(raw))
            if not fields:
                continue
            for field_name, value in fields.items():
                row[f"{dim_name}__{field_name}"] = value
        return rows

    # The only remaining union member is LookupEnricher (single-label path).
    mapping = enricher.enrich(ids, ctx)
    label_col = f"{dim_name}__label"
    for row in rows:
        raw = row.get(dim_name)
        if raw is None:
            continue
        raw_str = str(raw)
        row[label_col] = mapping.get(raw_str, raw_str)
    return rows


@dataclass(frozen=True)
class EnrichedResult:
    """Rows and their original semantic contract plus lookup provenance."""

    rows: list[dict[str, object]]
    analysis: SemanticAnalysis


def enrich_all(
    rows: list[dict[str, object]],
    catalog: Catalog,
    ctx: ResolutionContext,
    *,
    analysis: SemanticAnalysis,
) -> EnrichedResult:
    """Attach lookup fields using resolved semantic outputs, including aliases.

    Original keys, rows, grain, nodes, and output references are preserved.
    Attached fields carry lookup provenance rather than becoming new metrics.
    An unavailable analysis cannot establish lookup identity and is rejected.
    Masked outputs never invoke an enricher.
    """
    if not analysis.outputs and rows:
        raise ContractError(
            "Enrichment requires semantic output mappings.",
            reason="analysis_unavailable",
            operation="enrichment",
        )
    nodes = {node.node_id: node for node in analysis.nodes}
    enriched = [dict(row) for row in rows]
    provenance = list(analysis.enrichments)
    for output in analysis.outputs:
        node = nodes[output.node_id]
        if node.masked or node.catalog_ref not in catalog.lookups:
            continue
        lookup = catalog.lookups[node.catalog_ref]
        if lookup.enricher is None:
            continue
        column = output.sql_alias
        if any(column not in row for row in rows):
            raise ContractError(
                "An enrichment input does not match its output contract.",
                reason="output_correspondence",
                names=(column,),
                operation="enrichment",
            )
        attachments = [{column: row[column]} for row in rows]
        enrich_result(attachments, column, lookup, ctx)
        attached: set[str] = set()
        for row, attachment in zip(enriched, attachments, strict=True):
            for name, value in attachment.items():
                if name == column:
                    continue
                if name in row:
                    raise ContractError(
                        "An enrichment field collides with an existing output.",
                        reason="duplicate_output",
                        names=(name,),
                        operation="enrichment",
                    )
                row[name] = value
                attached.add(name)
        provenance.extend(
            EnrichmentProvenance(
                output_id=output.output_id,
                sql_alias=name,
                lookup_ref=lookup.dimension,
                field=name[len(column) + 2 :],
            )
            for name in sorted(attached)
        )
    return EnrichedResult(
        rows=enriched,
        analysis=analysis.model_copy(update={"enrichments": tuple(provenance)}),
    )


# ---------------------------------------------------------------------------
# Declarative SQL enricher — the common "SELECT … FROM <reference table>" case
# ---------------------------------------------------------------------------


class _SafeFormat(dict):  # type: ignore[type-arg]
    """``str.format_map`` helper that leaves unknown ``{placeholders}`` intact
    so a ``table`` template only substitutes the keys ``ctx.context`` carries."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class _SqlEnricher:
    """A :class:`~semql.model.MultiFieldEnricher` that reads its fields from a
    reference table via a caller-supplied ``execute``. Post-query only — it
    is not a vocabulary loader and never feeds the prompt. Built by
    :func:`sql_enricher`; see it for the contract."""

    table: str
    key: str
    fields: tuple[str, ...]
    execute: Callable[[str, list[Any]], Sequence[Mapping[str, Any]]]
    paramstyle: Literal["qmark", "format"] = "qmark"
    key_cast: str | None = None

    def _table_for(self, ctx: ResolutionContext | None) -> str:
        if ctx is not None and "{" in self.table:
            # ``table`` is catalog-author-controlled, but the values that fill
            # its ``{placeholders}`` come from ``ctx.context`` — which a host
            # may populate from request data. They land in an identifier
            # position (a schema / table name) with no bind-parameter form, so
            # validate each substituted value as a safe SQL identifier before
            # splicing, exactly as the compiler's ``security_sql`` path does
            # (SEMQL-LOOKUP-ENRICHER-IDENT). Refuse rather than emit raw.
            for key, value in ctx.context.items():
                if "{" + key + "}" in self.table and not is_safe_sql_identifier(value):
                    raise ValueError(
                        f"Lookup enricher table placeholder {{{key}}} resolved "
                        f"to {value!r}, which is not a safe SQL identifier; "
                        "refusing to splice it into the FROM clause."
                    )
            return self.table.format_map(_SafeFormat(ctx.context))
        return self.table

    def enrich_fields(self, ids: list[str], ctx: ResolutionContext) -> dict[str, dict[str, str]]:
        if not ids:
            return {}
        placeholder = "?" if self.paramstyle == "qmark" else "%s"
        in_clause = ", ".join([placeholder] * len(ids))
        cols = ", ".join((self.key, *self.fields))
        # Only the WHERE-side comparison needs the cast: a uuid (or other
        # non-text) key column compared against bound string parameters
        # fails at the DB level (Postgres does not implicitly cast a bound
        # parameter to uuid). The SELECT projection returns the raw column
        # value, which is fine — it already goes through ``str(...)`` in
        # Python when building ``out`` below.
        where_key = f"CAST({self.key} AS {self.key_cast})" if self.key_cast else self.key
        sql = (  # noqa: S608 — identifiers are catalog-author-controlled; ids bind as params
            f"SELECT {cols} FROM {self._table_for(ctx)} WHERE {where_key} IN ({in_clause})"
        )
        out: dict[str, dict[str, str]] = {}
        for r in self.execute(sql, list(ids)):
            out[str(r[self.key])] = {f: "" if r.get(f) is None else str(r[f]) for f in self.fields}
        return out


def sql_enricher(
    *,
    table: str,
    key: str,
    fields: Sequence[str],
    execute: Callable[[str, list[Any]], Sequence[Mapping[str, Any]]],
    paramstyle: Literal["qmark", "format"] = "qmark",
    key_cast: str | None = None,
) -> _SqlEnricher:
    """Build a post-query multi-field enricher for the common "reference
    table" case without hand-writing a class.

    The enricher attaches reference columns to result rows *after* the query
    runs. It is **not** a vocabulary loader — assign it to ``enricher=``, and
    the reference table's ids never reach the planner prompt::

        Lookup(
            dimension="orders.region_id",
            enricher=sql_enricher(
                table="{schema}.regions", key="id",
                fields=["name", "manager", "currency"],
                execute=db.execute,        # (sql, params) -> list[dict]
            ),
        )

    - ``table`` / ``key`` / ``fields`` name the reference table, its join
      column, and the columns to attach (as ``<dim>__<field>``). They are
      catalog-author-controlled identifiers, interpolated into SQL; the
      dimension *values* always bind as parameters, never as literals.
    - ``table`` may carry ``{placeholders}`` (e.g. ``{schema}``) filled from
      ``ctx.context`` per request (multi-tenant); unknown keys stay literal.
    - ``execute(sql, params)`` is your DB driver returning a list of row
      mappings — the only I/O, kept at the edge so the catalog stays sans-io.
    - ``paramstyle`` picks the placeholder: ``"qmark"`` (``?``, the
      default — sqlite/duckdb) or ``"format"`` (``%s`` — psycopg/mysql).
    - ``key_cast`` casts the key column (via ANSI ``CAST(col AS key_cast)``)
      on the WHERE-side comparison only, when the reference table's primary
      key is typed as something other than text. Ids reaching ``enrich_fields``
      are always Python ``str`` (coerced by :func:`enrich_result`), and some
      databases — Postgres in particular — refuse to implicitly cast a bound
      text *parameter* to a non-text column type such as ``uuid`` in a
      comparison (only literal constants get that implicit cast). A ``uuid``
      PK reference table therefore needs ``key_cast="text"``::

          Lookup(
              dimension="orders.customer_id",
              enricher=sql_enricher(
                  table="customers", key="id", fields=["name"],
                  execute=db.execute, key_cast="text",
              ),
          )

      which emits ``WHERE CAST(id AS text) IN (%s, %s)`` instead of the
      unsafe ``WHERE id IN (%s, %s)``. Left unset (the default), the SQL is
      unchanged from before this parameter existed.
    """
    return _SqlEnricher(
        table=table,
        key=key,
        fields=tuple(fields),
        execute=execute,
        paramstyle=paramstyle,
        key_cast=key_cast,
    )


__all__ = [
    "EnrichedResult",
    "QueryResolution",
    "ResolutionOutcome",
    "enrich_all",
    "enrich_result",
    "materialize",
    "resolve",
    "resolve_outcome",
    "resolve_query_filters",
    "sql_enricher",
]
