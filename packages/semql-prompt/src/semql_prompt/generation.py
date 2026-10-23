"""Aligned output model, projected schema, loss report, and prompt text."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from semql.catalog import Catalog
from semql.hooks import CubePromptHook
from semql.introspect import PolicyFn
from semql.model import AuthContext, Cube, GlossaryEntry, Lookup, ResolutionContext, View
from semql.retrieve import Retriever
from semql.schema import (
    SchemaProfile,
    SchemaProjection,
    SemanticFeature,
    project_schema,
)
from semql.spec import SavedQuery

from semql_prompt.continuation import ContinuationState
from semql_prompt.outcomes import PlannerResponse
from semql_prompt.prompt import build_query_generator_prompt_fragment


@dataclass(frozen=True)
class GenerationContract:
    """The exact schema and guidance pair to use for one generation request."""

    output_model: type[BaseModel]
    projection: SchemaProjection
    text: str
    disabled_features: frozenset[str]

    @property
    def schema(self) -> dict[str, Any]:
        return self.projection.schema

    @property
    def report(self) -> SchemaProjection:
        return self.projection

    def validate_payload(self, payload: object) -> BaseModel:
        """Apply the canonical model and profile feature restrictions."""
        return self.projection.validate_payload(payload)


def build_generation_contract(
    catalog: dict[str, Cube],
    *,
    profile: SchemaProfile = SchemaProfile.TOOL,
    disabled_features: Sequence[SemanticFeature | str] = (),
    output_model: type[BaseModel] = PlannerResponse,
    scope_to: list[str] | None = None,
    only_exposed: bool = True,
    include_introspection: bool = False,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    lookups: dict[str, Lookup] | None = None,
    ctx: ResolutionContext | None = None,
    glossary: list[GlossaryEntry] | None = None,
    relations: str = "",
    user_query: str | None = None,
    retriever: Retriever | None = None,
    top_k: int = 10,
    retrieval_threshold: int = 50,
    saved_queries: Sequence[SavedQuery] | None = None,
    cube_prompt_hooks: list[CubePromptHook] | None = None,
    instructions: str | None = None,
    continuation: ContinuationState | None = None,
    continuation_catalog: Catalog | None = None,
    current_authorized_cubes: Sequence[str] | None = None,
    current_visibility_revision: str | None = None,
) -> GenerationContract:
    """Build guidance and schema together from the same profile inputs."""
    projection = project_schema(
        output_model,
        profile,
        disabled_features=disabled_features,
    )
    text = build_query_generator_prompt_fragment(
        catalog,
        profile=profile,
        disabled_features=disabled_features,
        output_model=output_model,
        scope_to=scope_to,
        only_exposed=only_exposed,
        include_introspection=include_introspection,
        views=views,
        viewer=viewer,
        policy=policy,
        lookups=lookups,
        ctx=ctx,
        glossary=glossary,
        relations=relations,
        user_query=user_query,
        retriever=retriever,
        top_k=top_k,
        retrieval_threshold=retrieval_threshold,
        saved_queries=saved_queries,
        cube_prompt_hooks=cube_prompt_hooks,
        instructions=instructions,
        continuation=continuation,
        continuation_catalog=continuation_catalog,
        current_authorized_cubes=current_authorized_cubes,
        current_visibility_revision=current_visibility_revision,
    )
    return GenerationContract(
        output_model=output_model,
        projection=projection,
        text=text,
        disabled_features=frozenset(
            SemanticFeature(feature).value for feature in disabled_features
        ),
    )


__all__ = ["GenerationContract", "build_generation_contract"]
