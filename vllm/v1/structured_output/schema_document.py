# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-merge prototype of a shared JSON Schema document for design review."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TypeAlias
from urllib.parse import unquote


class IncompleteSchemaAnalysis(Exception):
    """A schema feature needs resolution beyond this prototype's coverage."""


@dataclass
class BoolSchema:
    value: bool


@dataclass
class GeneralSchema:
    """One schema node with typed behavior over shared keyword data."""

    values: dict[str, Any]
    subschemas: dict[str, Any]
    order: tuple[str, ...]

    def has_keyword(self, keyword: str) -> bool:
        return keyword in self.values or keyword in self.subschemas

    @cached_property
    def _explicit_types(self) -> frozenset[str] | None:
        schema_type = self.values.get("type")
        if schema_type is None:
            return None
        if isinstance(schema_type, str):
            return frozenset({schema_type})
        if isinstance(schema_type, list) and all(
            isinstance(value, str) for value in schema_type
        ):
            return frozenset(schema_type)
        raise IncompleteSchemaAnalysis("invalid type keyword")

    def explicit_types(self) -> frozenset[str] | None:
        return self._explicit_types

    @property
    def may_be_numeric(self) -> bool:
        types = self._explicit_types
        return types is None or "integer" in types or "number" in types

    @property
    def may_be_string(self) -> bool:
        types = self._explicit_types
        return types is None or "string" in types

    @property
    def may_be_array(self) -> bool:
        types = self._explicit_types
        return types is None or "array" in types

    @property
    def may_be_object(self) -> bool:
        types = self._explicit_types
        return types is None or "object" in types

    @property
    def has_multiple_of(self) -> bool:
        return "multipleOf" in self.values

    @property
    def has_format(self) -> bool:
        return "format" in self.values

    @property
    def string_format(self) -> Any:
        return self.values.get("format")

    @property
    def has_pattern_or_format_with_length_bounds(self) -> bool:
        return ("pattern" in self.values or "format" in self.values) and (
            "minLength" in self.values or "maxLength" in self.values
        )

    @property
    def has_unique_items(self) -> bool:
        return "uniqueItems" in self.values

    @property
    def contains(self) -> SchemaNode | None:
        return self.subschemas.get("contains")

    @property
    def has_min_contains(self) -> bool:
        return "minContains" in self.values

    @property
    def has_max_contains(self) -> bool:
        return "maxContains" in self.values

    @property
    def properties(self) -> dict[str, SchemaNode] | None:
        return self.subschemas.get("properties")

    @property
    def pattern_properties(self) -> dict[str, SchemaNode] | None:
        return self.subschemas.get("patternProperties")

    @property
    def property_names(self) -> SchemaNode | None:
        return self.subschemas.get("propertyNames")

    @property
    def additional_properties(self) -> SchemaNode | None:
        return self.subschemas.get("additionalProperties")

    @property
    def unevaluated_properties(self) -> SchemaNode | None:
        return self.subschemas.get("unevaluatedProperties")

    def add_subschema(self, keyword: str, child: SchemaNode) -> None:
        if self.has_keyword(keyword):
            raise ValueError(f"{keyword} already exists")
        self.subschemas[keyword] = child
        self.order += (keyword,)


SchemaNode: TypeAlias = BoolSchema | GeneralSchema

_SCHEMA_MAP = frozenset(
    {"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
)
_SCHEMA_ARRAY = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_SINGLE = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "items",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)
_DEFINITION_CONTAINERS = frozenset({"$defs", "definitions"})


def _parse_node(raw: Any) -> SchemaNode:
    if isinstance(raw, bool):
        return BoolSchema(raw)
    if not isinstance(raw, dict):
        raise ValueError("a JSON Schema must be an object or boolean")
    if any(not isinstance(keyword, str) for keyword in raw):
        raise ValueError("JSON Schema keywords must be strings")

    values: dict[str, Any] = {}
    subschemas: dict[str, Any] = {}
    for keyword, value in raw.items():
        if keyword in _SCHEMA_MAP:
            if not isinstance(value, dict):
                raise ValueError(f"{keyword} must be a schema map")
            subschemas[keyword] = {
                name: _parse_node(child) for name, child in value.items()
            }
        elif keyword in _SCHEMA_ARRAY or (
            keyword == "items" and isinstance(value, list)
        ):
            if not isinstance(value, list):
                raise ValueError(f"{keyword} must be a schema list")
            subschemas[keyword] = [_parse_node(child) for child in value]
        elif keyword in _SCHEMA_SINGLE:
            subschemas[keyword] = _parse_node(value)
        else:
            values[keyword] = deepcopy(value)
    return GeneralSchema(values, subschemas, tuple(raw))


def _export_node(node: SchemaNode) -> Any:
    if isinstance(node, BoolSchema):
        return node.value
    exported: dict[str, Any] = {}
    for keyword in node.order:
        if keyword in node.values:
            exported[keyword] = deepcopy(node.values[keyword])
        else:
            children = node.subschemas[keyword]
            if isinstance(children, dict):
                exported[keyword] = {
                    name: _export_node(child) for name, child in children.items()
                }
            elif isinstance(children, list):
                exported[keyword] = [_export_node(child) for child in children]
            else:
                exported[keyword] = _export_node(children)
    return exported


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


@dataclass
class SchemaDocument:
    root: SchemaNode
    dialect: str = "2020-12"
    nodes: dict[str, SchemaNode] = field(default_factory=dict)
    application_edges: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, dialect: str = "2020-12") -> SchemaDocument:
        if dialect != "2020-12":
            raise IncompleteSchemaAnalysis(f"unsupported dialect: {dialect}")
        document = cls(_parse_node(raw), dialect)
        document._index()
        return document

    def _index(self) -> None:
        self.nodes = {}
        self.application_edges = {}

        def visit(pointer: str, node: SchemaNode) -> None:
            self.nodes[pointer] = node
            self.application_edges[pointer] = []
            if isinstance(node, BoolSchema):
                return
            for keyword, children in node.subschemas.items():
                prefix = f"{pointer}/{_pointer_part(keyword)}"
                child_nodes: Iterable[tuple[str, SchemaNode]]
                if isinstance(children, dict):
                    child_nodes = (
                        (f"{prefix}/{_pointer_part(name)}", child)
                        for name, child in children.items()
                    )
                elif isinstance(children, list):
                    child_nodes = (
                        (f"{prefix}/{index}", child)
                        for index, child in enumerate(children)
                    )
                else:
                    child_nodes = ((prefix, children),)
                for child_pointer, child in child_nodes:
                    visit(child_pointer, child)
                    if keyword not in _DEFINITION_CONTAINERS:
                        self.application_edges[pointer].append(child_pointer)

        visit("", self.root)

    def export(self) -> Any:
        return _export_node(self.root)

    def walk_all_schema_nodes(self) -> Iterator[tuple[str, SchemaNode]]:
        yield from self.nodes.items()

    def walk_potential_constraints(self) -> Iterator[tuple[str, SchemaNode]]:
        pending = [""]
        visited: set[str] = set()
        while pending:
            pointer = pending.pop()
            if pointer in visited:
                continue
            visited.add(pointer)
            node = self.nodes[pointer]
            yield pointer, node
            pending.extend(reversed(self.application_edges[pointer]))
            if isinstance(node, GeneralSchema):
                if any(
                    keyword in node.values
                    for keyword in ("$id", "$anchor", "$dynamicAnchor", "dependencies")
                ):
                    raise IncompleteSchemaAnalysis(
                        "resource scope, anchors, or legacy dependencies unresolved"
                    )
                if "$dynamicRef" in node.values or "$recursiveRef" in node.values:
                    raise IncompleteSchemaAnalysis("dynamic references are unresolved")
                ref = node.values.get("$ref")
                if ref is not None:
                    if not isinstance(ref, str) or not ref.startswith("#"):
                        raise IncompleteSchemaAnalysis(
                            "only local references are resolved"
                        )
                    target = unquote(ref[1:])
                    if target not in self.nodes:
                        raise IncompleteSchemaAnalysis(f"unresolved reference: {ref}")
                    pending.append(target)

    def transform_schema_nodes(
        self, transform: Callable[[GeneralSchema], None]
    ) -> SchemaDocument:
        prepared = deepcopy(self)
        for node in list(prepared.nodes.values()):
            if isinstance(node, GeneralSchema):
                transform(node)
                node.__dict__.pop("_explicit_types", None)
        prepared._index()
        return prepared
