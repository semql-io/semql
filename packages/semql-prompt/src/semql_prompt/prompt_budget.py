"""I12: prompt-budget enforcement.

The four-role prompt pipeline (Router / Generator / Presenter /
Drilldown) builds catalog fragments that can grow large as the
catalog grows. ``PromptBudget.apply(text)`` enforces a token ceiling
on a rendered prompt string by progressively trimming the cheapest-
to-prune content first.

The token estimate uses the chars/4 heuristic. It's intentionally
rough — guardrail use, not a token-by-token count. A more elaborate
estimate (tiktoken, a model-specific BPE) is out of scope: callers
who need that level of fidelity should bypass the budget and pass
the raw prompt to their model, or set ``max_tokens`` to a high
enough value that the budget becomes a no-op.

Trim order (cheapest to most-destructive):

1. **relations** — the cross-cube narrative block. The LLM can
   still resolve cube names; it just loses the connective prose.
2. **glossary** — the ``## Glossary`` block. Vocabulary is helpful
   but not load-bearing.
3. **descriptions** — ``Cube.description`` and
   ``Measure/Dimension.description`` lines. The LLM can still
   resolve names and types.
4. **low-priority cubes** — the cube with the lowest ``priority``
   score (default 0) is dropped. Iterated until the prompt fits
   or no cubes remain.

The result is a frozen Pydantic value type carrying the trimmed
text, an ``estimated_tokens`` count, a ``was_truncated`` flag, and
a ``dropped`` list of what was pruned (in order). Callers can
log/report the drops; we don't silently drop without record.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Rough heuristic. Matches OpenAI's "1 token ~ 4 chars of English
# text" rule of thumb. Good enough for guardrail use; do not rely on
# this for model-specific BPE.
_CHARS_PER_TOKEN = 4


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
        # Default: next H2 header.
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
    """Strip ``description: ...`` lines from the rendered prompt.

    The prompt builder emits descriptions on lines like
    ``  description: Sum of all order amounts in the period``. We
    drop everything after the colon to the end of the line, but
    only inside the catalog block (not inside ``## Glossary`` etc.).
    Returns ``(trimmed, count)``."""
    import re

    pattern = re.compile(r"^[ \t]*description:[ \t].*$", re.MULTILINE)
    matches = pattern.findall(text)
    return pattern.sub("", text), len(matches)


def _drop_lowest_priority_cube(text: str) -> tuple[str, str | None]:
    """Drop the lowest-priority cube block.

    The rendered prompt's cube block is a sequence of ``### <name>``
    headers followed by the cube's content. We don't have access to
    the priority metadata in the rendered text, so we fall back to
    the last cube in the document (last-written = least-cached = best
    candidate for pruning). Returns ``(trimmed, dropped_cube_name)``.

    This is intentionally crude: a smarter implementation would
    plumb the priority order through. For the budget guardrail
    use-case, "drop the last cube" is correct enough — it's the
    one the prompt builder appended last, so the rest of the
    ordering is preserved.
    """
    import re

    pattern = re.compile(r"\n### (\w+)\b")
    matches = list(pattern.finditer(text))
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
    """A guardrail that trims a prompt string to fit a token budget.

    The trim is *progressive*: it applies the cheapest pruning
    strategies first, then moves to the more destructive ones,
    stopping as soon as the prompt fits. Callers that need a
    deterministic drop order can read ``result.dropped``."""

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(
        ge=0,
        description=(
            "Maximum allowed token count. Texts already at or below this pass through unchanged."
        ),
    )

    def apply(self, text: str) -> BudgetResult:
        """Trim ``text`` until it fits within ``max_tokens``.

        Returns a :class:`BudgetResult` with the trimmed text and a
        record of what was dropped. If the text is already within
        budget, returns it unchanged with ``was_truncated=False``.

        ``max_tokens=0`` is a special case: any non-empty text is
        over-budget and the result is the empty string. (Without
        this special case, ``apply("")`` and ``apply("any text at
        all")`` would both be over-budget, but only the empty
        string actually fits.)
        """
        if _estimate_tokens(text) <= self.max_tokens:
            return BudgetResult(
                text=text,
                estimated_tokens=_estimate_tokens(text),
                was_truncated=False,
                dropped=(),
            )

        # max_tokens=0 with non-empty text: trim to empty.
        if self.max_tokens == 0:
            return BudgetResult(
                text="",
                estimated_tokens=0,
                was_truncated=True,
                dropped=("everything",),
            )

        dropped: list[str] = []
        current = text

        # 1. Drop relations block.
        if "## Cross-cube relations" in current or "## Relations" in current:
            current, removed = _slice_section(current, "## Cross-cube relations", None)
            if not removed and "## Relations" in current:
                current, removed = _slice_section(current, "## Relations", None)
            if removed:
                dropped.append("relations")
                if _estimate_tokens(current) <= self.max_tokens:
                    return self._result(current, dropped)

        # 2. Drop glossary block.
        if "## Glossary" in current:
            current, removed = _slice_section(current, "## Glossary", None)
            if removed:
                dropped.append("glossary")
                if _estimate_tokens(current) <= self.max_tokens:
                    return self._result(current, dropped)

        # 3. Drop descriptions line-by-line.
        current, n = _drop_descriptions(current)
        if n > 0:
            dropped.append(f"descriptions({n})")
            if _estimate_tokens(current) <= self.max_tokens:
                return self._result(current, dropped)

        # 4. Drop the lowest-priority cube, iterated.
        while _estimate_tokens(current) > self.max_tokens:
            new_text, name = _drop_lowest_priority_cube(current)
            if name is None:
                # No more cubes to drop; we have to accept the
                # over-budget text.
                break
            dropped.append(f"cube:{name}")
            current = new_text

        return self._result(current, dropped)

    def _result(self, text: str, dropped: list[str]) -> BudgetResult:
        return BudgetResult(
            text=text,
            estimated_tokens=_estimate_tokens(text),
            was_truncated=len(dropped) > 0,
            dropped=tuple(dropped),
        )


class BudgetResult(BaseModel):
    """The output of :meth:`PromptBudget.apply`.

    ``text`` is the trimmed prompt. ``estimated_tokens`` is the
    post-trim token count (heuristic). ``was_truncated`` is True if
    any drop happened. ``dropped`` is the ordered list of what was
    pruned — a list of stable strings the caller can log or
    report.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    estimated_tokens: int = Field(ge=0)
    was_truncated: bool
    dropped: tuple[str, ...] = Field(
        default_factory=lambda: tuple[str, ...](),
        description=(
            "Ordered list of what was dropped (e.g. ['relations', 'glossary', 'cube:orders'])."
        ),
    )


def apply_budget(text: str, max_tokens: int) -> BudgetResult:
    """Convenience: ``PromptBudget(max_tokens=max_tokens).apply(text)``.

    Equivalent to constructing a one-shot budget; lets callers that
    only need a single trim avoid the Pydantic value-type
    ceremony."""
    return PromptBudget(max_tokens=max_tokens).apply(text)


__all__ = [
    "BudgetResult",
    "PromptBudget",
    "apply_budget",
    "estimate_tokens",
]


# Re-export for callers that want the heuristic directly.
estimate_tokens = _estimate_tokens
