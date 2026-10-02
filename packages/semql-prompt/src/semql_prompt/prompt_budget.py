"""Prompt-budget enforcement.

``PromptBudget`` trims catalog prompt sections and reports whether the
remaining text fits. The built-in chars/4 estimate is deliberately labeled as a
heuristic; callers may supply a model-specific ``count_tokens`` callback.
Trimming removes only known optional catalog sections. Required instructions,
protected cubes, and other remaining text survive even when the result cannot
fit.

Trim order: domain relations, glossary, field descriptions, then unprotected
cube blocks. Callers must check ``BudgetResult.fits`` before sending.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from semql.model import Dialect

# Rough heuristic. Matches OpenAI's "1 token ~ 4 chars of English
# text" rule of thumb. Good enough for guardrail use; do not rely on
# this for model-specific BPE.
_CHARS_PER_TOKEN = 4

# Trim-pass patterns. Compiled once at import rather than per ``apply()`` call.
_FIELD_DESCRIPTION_RE = re.compile(
    r"^(  - `[^`\n]+`[^\n]*?) — [^\n]*$",
    re.MULTILINE,
)
_DIALECTS = "|".join(re.escape(dialect.value) for dialect in Dialect)
_CUBE_HEADER_RE = re.compile(rf"\n### ([\w-]+) \(({_DIALECTS})\)")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ``ceil(len(text) / 4)``.

    Empty string yields 0 tokens (we don't want the budget to fire
    on the empty string)."""
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _catalog_span(text: str) -> tuple[int, int] | None:
    """Return semantic-catalog bounds inside a full prompt."""
    start = text.find("## SEMANTIC CATALOG")
    if start == -1:
        return None
    end = text.find("\n## ", start + len("## SEMANTIC CATALOG"))
    return start, len(text) if end == -1 else end


def _drop_domain_subsection(text: str, marker: str) -> tuple[str, bool]:
    """Drop one recognized subsection from ``## DOMAIN CONTEXT`` only."""
    domain_start = text.find("## DOMAIN CONTEXT")
    if domain_start == -1:
        return text, False
    domain_end = text.find("\n## ", domain_start + len("## DOMAIN CONTEXT"))
    if domain_end == -1:
        domain_end = len(text)
    start = text.find(marker, domain_start, domain_end)
    if start == -1:
        return text, False
    if marker == "**Glossary:**":
        relations = text.find("**Relations:**", start + len(marker), domain_end)
        end = relations if relations != -1 else domain_end
    else:
        end = domain_end
    return text[:start] + text[end:], True


def _drop_field_descriptions(text: str) -> tuple[str, int]:
    """Remove field prose inside the catalog while preserving metadata."""
    span = _catalog_span(text)
    if span is None:
        return text, 0
    start, end = span
    catalog = text[start:end]
    trimmed, count = _FIELD_DESCRIPTION_RE.subn(r"\1", catalog)
    return text[:start] + trimmed + text[end:], count


def _drop_lowest_priority_cube(
    text: str, protected_cubes: frozenset[str]
) -> tuple[str, str | None]:
    """Drop the last unprotected cube within the semantic catalog."""
    span = _catalog_span(text)
    if span is None:
        return text, None
    catalog_start, catalog_end = span
    catalog = text[catalog_start:catalog_end]
    matches = list(_CUBE_HEADER_RE.finditer(catalog))
    for match in reversed(matches):
        name = match.group(1)
        if name in protected_cubes:
            continue
        end = catalog.find("\n### ", match.end())
        if end == -1:
            end = len(catalog)
        trimmed = catalog[: match.start()] + catalog[end:]
        return text[:catalog_start] + trimmed + text[catalog_end:], name
    return text, None


class PromptBudget(BaseModel):
    """Trim optional catalog prompt content and report whether it fits."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(ge=0, description="Maximum count accepted by the active counter.")

    def apply(
        self,
        text: str,
        *,
        count_tokens: Callable[[str], int] | None = None,
        protected_cubes: frozenset[str] = frozenset(),
    ) -> BudgetResult:
        """Trim optional prompt content, preserving required text and cubes.

        ``count_tokens`` may supply a model-specific counter. Without it, the
        built-in chars/4 heuristic is used. ``protected_cubes`` names catalog
        blocks that must survive. A result can remain over budget when no
        further catalog content can be safely removed.
        """
        counter = _estimate_tokens if count_tokens is None else count_tokens

        def count(value: str) -> int:
            result = counter(value)
            if result < 0:
                raise ValueError("count_tokens must return a non-negative integer")
            return result

        if count(text) <= self.max_tokens:
            return self._result(text, [], count, count_tokens is not None)

        dropped: list[str] = []
        current = text
        for marker, label in (
            ("**Relations:**", "relations"),
            ("**Glossary:**", "glossary"),
        ):
            current, removed = _drop_domain_subsection(current, marker)
            if removed:
                dropped.append(label)
                if count(current) <= self.max_tokens:
                    return self._result(current, dropped, count, count_tokens is not None)

        current, n = _drop_field_descriptions(current)
        if n > 0:
            dropped.append(f"descriptions({n})")
            if count(current) <= self.max_tokens:
                return self._result(current, dropped, count, count_tokens is not None)

        while count(current) > self.max_tokens:
            new_text, name = _drop_lowest_priority_cube(current, protected_cubes)
            if name is None:
                break
            dropped.append(f"cube:{name}")
            current = new_text

        return self._result(current, dropped, count, count_tokens is not None)

    def _result(
        self,
        text: str,
        dropped: list[str],
        count: Callable[[str], int],
        custom_counter: bool,
    ) -> BudgetResult:
        token_count = count(text)
        return BudgetResult(
            text=text,
            token_count=token_count,
            count_source="callback" if custom_counter else "heuristic",
            fits=token_count <= self.max_tokens,
            was_truncated=len(dropped) > 0,
            dropped=tuple(dropped),
        )


class BudgetResult(BaseModel):
    """Output of :meth:`PromptBudget.apply` with explicit fit status.

    ``fits`` may be false when protected or non-catalog content alone is too
    large to trim safely. ``count_source`` identifies whether ``token_count``
    came from the built-in heuristic or a caller-supplied counter.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    token_count: int = Field(ge=0, description="Count from the labeled active counter.")
    count_source: Literal["heuristic", "callback"]
    fits: bool
    was_truncated: bool
    dropped: tuple[str, ...] = Field(default_factory=tuple)


def apply_budget(
    text: str,
    max_tokens: int,
    *,
    count_tokens: Callable[[str], int] | None = None,
    protected_cubes: frozenset[str] = frozenset(),
) -> BudgetResult:
    """Apply a one-shot prompt budget with optional protected cubes."""
    return PromptBudget(max_tokens=max_tokens).apply(
        text,
        count_tokens=count_tokens,
        protected_cubes=protected_cubes,
    )


__all__ = [
    "BudgetResult",
    "PromptBudget",
    "apply_budget",
    "estimate_tokens",
]


# Re-export for callers that want the heuristic directly.
estimate_tokens = _estimate_tokens
