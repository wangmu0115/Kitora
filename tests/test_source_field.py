import json
from typing import Annotated

import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, ValidationError

from kitora.pandas import dataframe_to_models
from kitora.pydantic import SourceField


@pytest.mark.parametrize("transform", [None, int, lambda value: int(value)])
def test_source_field_generates_serializable_valid_json_schema(transform):
    class Row(BaseModel):
        value: int = SourceField(source="input", required=True, transform=transform)

    schema = Row.model_json_schema()

    json.dumps(schema)
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["value"]
    assert schema["properties"]["value"]["type"] == "integer"


def test_custom_schema_extra_is_preserved_without_mutation():
    extras = {"examples": [1], "description": "A value"}

    class Row(BaseModel):
        value: int = SourceField(source="input", transform=int, json_schema_extra=extras)

    schema = Row.model_json_schema()

    Draft202012Validator.check_schema(schema)
    assert extras == {"examples": [1], "description": "A value"}
    assert schema["properties"]["value"]["examples"] == [1]
    assert schema["properties"]["value"]["description"] == "A value"


def test_callable_schema_extra_is_supported():
    def customize(schema):
        schema["examples"] = [1]

    class Row(BaseModel):
        value: int = SourceField(source="input", json_schema_extra=customize)

    schema = Row.model_json_schema()

    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["value"]["examples"] == [1]


def test_pydantic_constraints_defaults_and_aliases_are_preserved():
    class Row(BaseModel):
        model_config = ConfigDict(validate_by_name=True)
        value: int = SourceField(source="input", alias="external", gt=0, default=1)

    assert Row().value == 1
    with pytest.raises(ValidationError) as error:
        Row.model_validate({"external": 0})
    assert error.value.errors()[0]["type"] == "greater_than"
    result = dataframe_to_models(pd.DataFrame({"input": [2]}), Row)
    assert result[0].model_dump(by_alias=True) == {"external": 2}


def test_source_field_works_inside_annotated_and_is_inherited():
    class Parent(BaseModel):
        value: Annotated[int, SourceField(source="input", transform=int)]

    class Child(Parent):
        label: str

    result = dataframe_to_models(pd.DataFrame({"input": ["12"], "label": ["ok"]}), Child)

    assert result[0].model_dump() == {"value": 12, "label": "ok"}
    Draft202012Validator.check_schema(Child.model_json_schema())
