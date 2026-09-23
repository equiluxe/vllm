# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest

from vllm.v1.structured_output.backend_guidance import (
    has_guidance_unsupported_json_features,
    process_for_additional_properties,
)
from vllm.v1.structured_output.backend_xgrammar import (
    has_xgrammar_unsupported_json_features,
)
from vllm.v1.structured_output.schema_document import (
    GeneralSchema,
    IncompleteSchemaAnalysis,
    SchemaDocument,
)

pytestmark = pytest.mark.cpu_test


def test_document_preserves_schema_positions_and_literal_data():
    schema = {
        "properties": {"patternProperties": {"type": "string"}},
        "const": {"properties": {"status": "ok"}},
        "x-annotation": {"patternProperties": {"type": "integer"}},
        "$defs": {"Unused": False},
    }
    original = deepcopy(schema)

    document = SchemaDocument.parse(schema)

    assert document.export() == original
    assert schema == original
    assert {pointer for pointer, _ in document.walk_all_schema_nodes()} == {
        "",
        "/properties/patternProperties",
        "/$defs/Unused",
    }
    assert {pointer for pointer, _ in document.walk_potential_constraints()} == {
        "",
        "/properties/patternProperties",
    }


def test_guidance_check_uses_schema_keywords_and_reachable_references():
    data_only = {
        "properties": {"patternProperties": {"type": "string"}},
        "const": {"patternProperties": {"^x$": True}},
        "$defs": {"Unused": {"patternProperties": {"^x$": True}}},
    }
    assert not has_guidance_unsupported_json_features(data_only)
    assert has_guidance_unsupported_json_features(
        {**data_only, "$ref": "#/$defs/Unused"}
    )
    assert has_guidance_unsupported_json_features(
        {"properties": {"code": {"patternProperties": {"^x$": True}}}}
    )


def test_guidance_rewrite_closes_only_schema_objects():
    schema = {
        "properties": {"name": {"type": "string"}},
        "const": {"properties": {"status": "ok"}},
        "enum": [{"properties": {"status": "ok"}}],
        "$defs": {"Child": {"properties": {"id": {"type": "integer"}}}},
    }
    original = deepcopy(schema)

    prepared = process_for_additional_properties(schema)

    assert schema == original
    assert prepared["additionalProperties"] is False
    assert prepared["const"] == original["const"]
    assert prepared["enum"] == original["enum"]
    assert prepared["$defs"]["Child"]["additionalProperties"] is False


def test_type_behaviors_share_the_node_without_inserting_a_type():
    schema = {"pattern": "^[a-z]+$", "maxLength": 3}
    document = SchemaDocument.parse(schema)
    root = document.root

    assert isinstance(root, GeneralSchema)
    assert root.explicit_types() is None
    assert root.may_be_string
    assert root.has_pattern_or_format_with_length_bounds
    assert document.export() == schema
    integer_root = SchemaDocument.parse({"type": "integer"}).root
    assert isinstance(integer_root, GeneralSchema)
    assert integer_root.may_be_numeric
    assert not integer_root.may_be_string


def test_type_behavior_cache_is_refreshed_after_transformation():
    document = SchemaDocument.parse({"type": "string"})
    root = document.root
    assert isinstance(root, GeneralSchema)
    assert root.may_be_string

    def replace_type(node: GeneralSchema) -> None:
        node.values["type"] = "integer"

    prepared = document.transform_schema_nodes(replace_type)
    prepared_root = prepared.root
    assert isinstance(prepared_root, GeneralSchema)
    assert prepared_root.may_be_numeric
    assert not prepared_root.may_be_string
    assert document.export() == {"type": "string"}


@pytest.mark.parametrize(
    "schema",
    [
        {"multipleOf": 2},
        {"type": ["null", "number"], "multipleOf": 2},
        {"uniqueItems": True},
        {"format": "made-up-format"},
        {"pattern": "^a$", "maxLength": 1},
        {"propertyNames": {"pattern": "^a$", "maxLength": 1}},
    ],
)
def test_xgrammar_check_uses_node_behaviors_for_applicable_constraints(schema):
    assert has_xgrammar_unsupported_json_features(schema)


def test_xgrammar_check_ignores_literal_data_and_unused_definitions():
    unsupported = {"type": "string", "pattern": "^a$", "maxLength": 1}
    schema = {
        "type": "object",
        "const": unsupported,
        "$defs": {"Unused": unsupported},
    }

    assert not has_xgrammar_unsupported_json_features(schema)
    assert has_xgrammar_unsupported_json_features({**schema, "$ref": "#/$defs/Unused"})


def test_unresolved_reference_is_reported_as_incomplete():
    document = SchemaDocument.parse({"$ref": "https://example.com/schema"})

    with pytest.raises(IncompleteSchemaAnalysis):
        list(document.walk_potential_constraints())


def test_escaped_recursive_local_reference_visits_each_node_once():
    schema = {
        "$defs": {"a/b": {"$ref": "#/$defs/a~1b"}},
        "$ref": "#/$defs/a~1b",
    }

    document = SchemaDocument.parse(schema)

    assert [pointer for pointer, _ in document.walk_potential_constraints()] == [
        "",
        "/$defs/a~1b",
    ]
