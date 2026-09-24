# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-merge prototype of a shared JSON Schema document for design review."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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


def _copy_value(value: Any) -> Any:
    if type(value) in (str, int, float, bool, type(None)):
        return value
    return deepcopy(value)


def _parse_node(raw: Any) -> SchemaNode:
    if isinstance(raw, bool):
        return BoolSchema(raw)
    if not isinstance(raw, dict):
        raise ValueError("a JSON Schema must be an object or boolean")

    values: dict[str, Any] = {}
    subschemas: dict[str, Any] = {}
    for keyword, value in raw.items():
        if not isinstance(keyword, str):
            raise ValueError("JSON Schema keywords must be strings")
        if keyword in _SCHEMA_MAP:
            if not isinstance(value, dict):
                raise ValueError(f"{keyword} must be a schema map")
            parsed_children: dict[str, SchemaNode] = {}
            for name, child in value.items():
                if not isinstance(name, str):
                    raise ValueError(f"{keyword} names must be strings")
                parsed_children[name] = _parse_node(child)
            subschemas[keyword] = parsed_children
        elif keyword in _SCHEMA_ARRAY or (
            keyword == "items" and isinstance(value, list)
        ):
            if not isinstance(value, list):
                raise ValueError(f"{keyword} must be a schema list")
            subschemas[keyword] = [_parse_node(child) for child in value]
        elif keyword in _SCHEMA_SINGLE:
            subschemas[keyword] = _parse_node(value)
        else:
            values[keyword] = _copy_value(value)
    return GeneralSchema(values, subschemas, tuple(raw))


def _export_node(node: SchemaNode) -> Any:
    if isinstance(node, BoolSchema):
        return node.value
    exported: dict[str, Any] = {}
    for keyword in node.order:
        if keyword in node.values:
            exported[keyword] = _copy_value(node.values[keyword])
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


def _schema_children(
    node: GeneralSchema, *, include_definitions: bool
) -> Iterator[tuple[str, str | int | None, SchemaNode]]:
    for keyword, children in node.subschemas.items():
        if not include_definitions and keyword in _DEFINITION_CONTAINERS:
            continue
        if isinstance(children, dict):
            for name, child in children.items():
                yield keyword, name, child
        elif isinstance(children, list):
            for index, child in enumerate(children):
                yield keyword, index, child
        else:
            yield keyword, None, children


def _child_pointer(pointer: str, keyword: str, name: str | int | None) -> str:
    child_pointer = f"{pointer}/{_pointer_part(keyword)}"
    if name is not None:
        child_pointer += f"/{_pointer_part(name) if isinstance(name, str) else name}"
    return child_pointer


def _push_children(
    pending: list[SchemaNode], node: GeneralSchema, *, include_definitions: bool
) -> None:
    for keyword in reversed(node.subschemas):
        if not include_definitions and keyword in _DEFINITION_CONTAINERS:
            continue
        children = node.subschemas[keyword]
        if isinstance(children, dict):
            pending.extend(reversed(children.values()))
        elif isinstance(children, list):
            pending.extend(reversed(children))
        else:
            pending.append(children)


@dataclass
class SchemaDocument:
    root: SchemaNode
    dialect: str = "2020-12"
    _node_index: dict[str, SchemaNode] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def parse(cls, raw: Any, dialect: str = "2020-12") -> SchemaDocument:
        if dialect != "2020-12":
            raise IncompleteSchemaAnalysis(f"unsupported dialect: {dialect}")
        return cls(_parse_node(raw), dialect)

    @property
    def nodes(self) -> dict[str, SchemaNode]:
        if self._node_index is None:
            index: dict[str, SchemaNode] = {}

            def visit(pointer: str, node: SchemaNode) -> None:
                index[pointer] = node
                if isinstance(node, BoolSchema):
                    return
                for keyword, children in node.subschemas.items():
                    prefix = f"{pointer}/{_pointer_part(keyword)}"
                    if isinstance(children, dict):
                        for name, child in children.items():
                            visit(f"{prefix}/{_pointer_part(name)}", child)
                    elif isinstance(children, list):
                        for child_index, child in enumerate(children):
                            visit(f"{prefix}/{child_index}", child)
                    else:
                        visit(prefix, children)

            visit("", self.root)
            self._node_index = index
        return self._node_index

    def _local_reference(self, node: GeneralSchema) -> str | None:
        values = node.values
        if (
            "$id" in values
            or "$anchor" in values
            or "$dynamicAnchor" in values
            or "dependencies" in values
        ):
            raise IncompleteSchemaAnalysis(
                "resource scope, anchors, or legacy dependencies unresolved"
            )
        if "$dynamicRef" in values or "$recursiveRef" in values:
            raise IncompleteSchemaAnalysis("dynamic references are unresolved")
        ref = values.get("$ref")
        if ref is None:
            return None
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise IncompleteSchemaAnalysis("only local references are resolved")
        target = unquote(ref[1:])
        if target not in self.nodes:
            raise IncompleteSchemaAnalysis(f"unresolved reference: {ref}")
        return target

    def export(self) -> Any:
        return _export_node(self.root)

    def walk_all_schema_nodes(self) -> Iterator[tuple[str, SchemaNode]]:
        yield from self.nodes.items()

    def walk_potential_constraint_nodes(self) -> Iterator[SchemaNode]:
        """Visit applicable nodes without building pointers unless a ref needs them."""
        pending = [self.root]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            yield node
            if isinstance(node, GeneralSchema):
                _push_children(pending, node, include_definitions=False)
                target = self._local_reference(node)
                if target is not None:
                    pending.append(self.nodes[target])

    def walk_potential_constraints(self) -> Iterator[tuple[str, SchemaNode]]:
        pending = [("", self.root)]
        visited: set[str] = set()
        while pending:
            pointer, node = pending.pop()
            if pointer in visited:
                continue
            visited.add(pointer)
            yield pointer, node
            if isinstance(node, GeneralSchema):
                children = [
                    (_child_pointer(pointer, keyword, name), child)
                    for keyword, name, child in _schema_children(
                        node, include_definitions=False
                    )
                ]
                pending.extend(reversed(children))
                target = self._local_reference(node)
                if target is not None:
                    pending.append((target, self.nodes[target]))

    def transform_schema_nodes(
        self, transform: Callable[[GeneralSchema], None]
    ) -> SchemaDocument:
        prepared = SchemaDocument(deepcopy(self.root), self.dialect)
        pending = [prepared.root]
        nodes: list[GeneralSchema] = []
        while pending:
            node = pending.pop()
            if isinstance(node, GeneralSchema):
                nodes.append(node)
                _push_children(pending, node, include_definitions=True)
        for node in nodes:
            transform(node)
            node.__dict__.pop("_explicit_types", None)
        return prepared
