"""Prompt-budget enforcement.

``PromptBudget`` trims catalog prompt sections and reports whether the
remaining text fits. The built-in chars/4 estimate is deliberately labeled as a
heuristic; callers may supply a model-specific ``count_tokens`` callback.
Trimming removes only known optional catalog sections. Required instructions
and other remaining text are preserved even when the result cannot fit.

Trim order: relations, glossary, descriptions, then cube blocks.
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

# Trim-pass patterns. Compiled once at import rather than per
# ``apply()`` call — the trim path can run several passes over a large
# prompt, and re-compiling these on each pass was pure waste.
_DESCRIPTION_LINE_RE = re.compile(r"^[ \t]*description:[ \t].*$")
_DIALECTS = "|".join(re.escape(dialect.value) for dialect in Dialect)
_CUBE_HEADER_RE = re.compile(rf"\n### ([\w-]+) \(({_DIALECTS})\)")
_CUBE_LINE_RE = re.compile(rf"^### [\w-]+ \(({_DIALECTS})\)")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ``ceil(len(text) / 4)``.

    Empty string yields 0 tokens (we don't want the budget to fire
    on the empty string)."""
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def _slice_section(text: str, start_marker: str, end_marker: str | None) -> tuple[str, str]:
    """Find a markdown section by its header marker; return
    ``(kept, dropped)`` where ``kept`` is text without the section
    and ``dropped`` is the section text (or empty if not found).

    The end marker defaults to the next ``## `` header. If neither
    is present, the section is taken to extend to the end of the
    document.
    """
    start = text.find(start_marker)
    if start == -1:
        return text, ""
    if end_marker is None:
        # Default: next section header.
        end = text.find("\n## ", start + len(start_marker))
        if end == -1:
            end = len(text)
    else:
        end = text.find(end_marker, start + len(start_marker))
        if end == -1:
            end = len(text)
    kept = text[:start] + text[end:]
    dropped = text[start:end]
    return kept, dropped


def _drop_descriptions(text: str) -> tuple[str, int]:
    """Strip optional description lines inside rendered cube blocks only."""
    lines = text.splitlines(keepends=True)
    in_cube = False
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if line.startswith(("## ", "### ")):
            in_cube = False
        if _CUBE_LINE_RE.match(line.rstrip("\r\n")):
            in_cube = True
        if in_cube and _DESCRIPTION_LINE_RE.match(line.rstrip("\r\n")):
            dropped += 1
            continue
        kept.append(line)
    return "".join(kept), dropped


def _drop_lowest_priority_cube(text: str) -> tuple[str, str | None]:
    """Drop the lowest-priority cube block.

    The rendered prompt's cube blocks use ``### <name> (<dialect>)`` headers.
    Other Markdown headings are not treated as removable catalog content.
    The priority metadata is not rendered, so the last listed cube is the
    last-written = least-cached = best candidate for pruning. Returns
    ``(trimmed, dropped_cube_name)``.

    This is intentionally crude: a smarter implementation would
    plumb the priority order through. For the budget guardrail
    use-case, "drop the last cube" is correct enough — it's the
    one the prompt builder appended last, so the rest of the
    ordering is preserved.
    """
    matches = list(_CUBE_HEADER_RE.finditer(text))
    if not matches:
        return text, None
    # Take the last cube block.
    last_match = matches[-1]
    name = last_match.group(1)
    end = text.find("\n### ", last_match.end())
    if end == -1:
        end = text.find("\n## ", last_match.end())
    if end == -1:
        end = len(text)
    return text[: last_match.start()] + text[end:], name


class PromptBudget(BaseModel):
    """Trim optional catalog prompt content and report whether it fits."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(ge=0, description="Maximum count accepted by the active counter.")

    def apply(
        self,
        text: str,
        *,
        count_tokens: Callable[[str], int] | None = None,
    ) -> BudgetResult:
        """Trim optional prompt content, preserving all remaining required text.

        ``count_tokens`` may supply a model-specific counter. Without it, the
        built-in chars/4 heuristic is used. A result can remain over budget
        when no further catalog content can be safely removed.
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
        if "## Cross-cube relations" in current or "## Relations" in current:
            current, removed = _slice_section(current, "## Cross-cube relations", None)
            if not removed and "## Relations" in current:
                current, removed = _slice_section(current, "## Relations", None)
            if removed:
                dropped.append("relations")
                if count(current) <= self.max_tokens:
                    return self._result(current, dropped, count, count_tokens is not None)

        if "## Glossary" in current:
            current, removed = _slice_section(current, "## Glossary", None)
            if removed:
                dropped.append("glossary")
                if count(current) <= self.max_tokens:
                    return self._result(current, dropped, count, count_tokens is not None)

        current, n = _drop_descriptions(current)
        if n > 0:
            dropped.append(f"descriptions({n})")
            if count(current) <= self.max_tokens:
                return self._result(current, dropped, count, count_tokens is not None)

        while count(current) > self.max_tokens:
            new_text, name = _drop_lowest_priority_cube(current)
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
    """Output of :meth:`PromptBudget.apply` with explicit fit status."""

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
) -> BudgetResult:
    """Convenience wrapper for :meth:`PromptBudget.apply`."""
    return PromptBudget(max_tokens=max_tokens).apply(text, count_tokens=count_tokens)


__all__ = [
    "BudgetResult",
    "PromptBudget",
    "apply_budget",
    "estimate_tokens",
]


# Re-export for callers that want the heuristic directly.
estimate_tokens = _estimate_tokens
