# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# ``graphviz`` ships no type stubs; pyright reports every method
# return as Unknown. The functions wrap it tightly enough that local
# inference covers the actual contract.
"""PNG/SVG rendering for catalog ER diagrams.

Shells out to Graphviz via the optional ``graphviz`` Python bindings.
Install with ``pip install "semql-erd[image]"`` and a system ``dot``
binary on PATH. The pure-Python ``render_dot`` path stays usable
without these.
"""

from __future__ import annotations

from pathlib import Path

from semql import Catalog

from semql_erd.dot import RankDir, render_dot


def render_image(
    catalog: Catalog,
    path: str | Path,
    *,
    format: str = "png",
    only_exposed: bool = True,
    rankdir: RankDir = "LR",
    title: str | None = None,
) -> Path:
    """Render the catalog as a PNG / SVG / PDF image at ``path``.

    Calls the system ``dot`` binary via the ``graphviz`` Python
    bindings. Raises ``ImportError`` if the optional ``image`` extra
    isn't installed, and ``graphviz.ExecutableNotFound`` (re-raised
    from the bindings) if the ``dot`` binary isn't on PATH.
    """
    try:
        # graphviz has no type stubs; mypy is handled by a tool.mypy override.
        from graphviz import Source  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "render_image requires the ``image`` extra. Install with "
            "``pip install 'semql-erd[image]'``."
        ) from exc

    dot_source = render_dot(catalog, only_exposed=only_exposed, rankdir=rankdir, title=title)
    target = Path(path)
    # ``graphviz.Source`` writes ``<filename>.<format>``; we want the
    # exact path the caller asked for, so strip the suffix and pass it
    # as ``filename`` with ``format=`` matching the requested type.
    filename = target.with_suffix("")
    src = Source(dot_source, format=format)
    rendered = src.render(filename=str(filename), cleanup=True)
    return Path(rendered)


__all__ = ["render_image"]
