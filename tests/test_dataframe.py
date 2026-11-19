from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, ValidationError

from kitora.pandas import dataframe_to_models
from kitora.pydantic import SourceField


def test_complete_rows_produce_one_model_each():
    class Row(BaseModel):
        id: int
        name: str

    records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    result = dataframe_to_models(pd.DataFrame(records), Row)

    assert [row.model_dump() for row in result] == records


def test_fields_with_defaults_do_not_produce_partial_models():
    class Row(BaseModel):
        id: int = 0
        name: str = "default"

    result = dataframe_to_models(pd.DataFrame([{"id": 1, "name": "Alice"}]), Row)

    assert [row.model_dump() for row in result] == [{"id": 1, "name": "Alice"}]


def test_large_integer_is_preserved_beside_float_column():
    class Row(BaseModel):
        id: int

    identifier = 2**53 + 1
    frame = pd.DataFrame({"id": [identifier], "score": [0.5]})

    assert dataframe_to_models(frame, Row)[0].id == identifier


@pytest.mark.parametrize(
    "field_kwargs",
    [
        {"alias": "id"},
        {"validation_alias": "id"},
        {"validation_alias": AliasPath("user", "id")},
        {"validation_alias": AliasChoices("id", "ID")},
    ],
    ids=["alias", "validation-alias", "alias-path", "alias-choices"],
)
def test_model_can_enable_validation_by_field_name(field_kwargs):
    class Row(BaseModel):
        model_config = ConfigDict(validate_by_name=True)
        user_id: int = Field(**field_kwargs)

    result = dataframe_to_models(pd.DataFrame({"user_id": [1]}), Row)

    assert result[0].user_id == 1


@pytest.mark.parametrize("use_alias,expected_left", [(False, 1), (True, 2)])
def test_model_controls_alias_precedence(use_alias, expected_left):
    class Row(BaseModel):
        model_config = ConfigDict(validate_by_name=True, validate_by_alias=use_alias)
        left: int = Field(alias="right")
        right: int

    result = dataframe_to_models(pd.DataFrame({"left": [1], "right": [2]}), Row)

    assert result[0].model_dump() == {"left": expected_left, "right": 2}


@pytest.mark.parametrize("source", ["extra_fields", "dataframe"])
def test_nested_alias_is_validated_with_child_config(source):
    class User(BaseModel):
        user_id: int = Field(alias="userId")

    class Row(BaseModel):
        model_config = ConfigDict(validate_by_name=True, validate_by_alias=False)
        owner: User = Field(alias="ownerInfo")

    value = {"userId": 7}
    if source == "extra_fields":
        frame, extras = pd.DataFrame(index=[0]), {"owner": value}
    else:
        frame, extras = pd.DataFrame({"owner": [value]}), None

    result = dataframe_to_models(frame, Row, extra_fields=extras)

    assert result[0].owner.user_id == 7


def test_nested_alias_only_model_rejects_field_name_in_extra_fields():
    class User(BaseModel):
        user_id: int = Field(alias="userId")

    class Row(BaseModel):
        owner: User

    with pytest.raises(ValidationError) as error:
        dataframe_to_models(pd.DataFrame(index=[0]), Row, extra_fields={"owner": {"user_id": 7}})

    assert error.value.errors()[0]["loc"] == ("owner", "userId")
    assert error.value.errors()[0]["type"] == "missing"


def test_missing_fields_use_defaults_and_independent_factories():
    class Row(BaseModel):
        status: str | None = "active"
        count: int = 7
        tags: list[str] = SourceField(default_factory=list)

    result = dataframe_to_models(pd.DataFrame({"other": [1, 2]}), Row, strict=False)

    assert [row.model_dump() for row in result] == [
        {"status": "active", "count": 7, "tags": []},
        {"status": "active", "count": 7, "tags": []},
    ]
    assert result[0].tags is not result[1].tags


@pytest.mark.parametrize("missing", [None, pd.NA, float("nan"), pd.NaT])
def test_explicit_scalar_null_does_not_use_default_or_transform(missing):
    transform = Mock(return_value="transformed")

    class Row(BaseModel):
        status: str | None = SourceField(default="active", transform=transform)

    frame = pd.DataFrame({"status": pd.Series([missing], dtype=object)})

    assert dataframe_to_models(frame, Row)[0].status is None
    transform.assert_not_called()


def test_missing_required_model_field_is_reported_by_pydantic():
    class Row(BaseModel):
        value: int

    with pytest.raises(ValidationError) as error:
        dataframe_to_models(pd.DataFrame({"other": [1]}), Row, strict=False)

    assert error.value.errors()[0]["type"] == "missing"
    assert error.value.errors()[0]["loc"] == ("value",)


@pytest.mark.parametrize("strict,required", [(True, False), (False, True)])
def test_missing_source_still_raises_when_required(strict, required):
    class Row(BaseModel):
        value: int = SourceField(source="input", required=required, default=1)

    with pytest.raises(ValueError, match="Missing source column for value"):
        dataframe_to_models(pd.DataFrame({"other": [1]}), Row, strict=strict)


def test_list_with_null_item_is_not_replaced_by_none():
    class Row(BaseModel):
        values: list[int | None]

    result = dataframe_to_models(pd.DataFrame({"values": [[None]]}), Row)

    assert result[0].values == [None]


@pytest.mark.parametrize(
    "value",
    [[], [None], (None,), {"key": None}, np.array([None]), np.array([1, 2])],
    ids=["empty-list", "list", "tuple", "dict", "one-item-array", "array"],
)
def test_container_cells_are_preserved(value):
    class Row(BaseModel):
        value: Any

    frame = pd.DataFrame({"value": pd.Series([value], dtype=object)})

    assert dataframe_to_models(frame, Row)[0].value is value


@pytest.mark.parametrize(
    "columns,extra_fields,expected",
    [
        ({"Input value": ["1"], "backup": ["2"], "value": ["3"]}, {"value": "4"}, 1),
        ({"backup": ["2"], "value": ["3"]}, {"value": "4"}, 2),
        ({"value": ["3"]}, {"value": "4"}, 3),
        ({"other": [0]}, {"value": "4"}, 4),
    ],
    ids=["first-source", "second-source", "field-name", "extra-fields"],
)
def test_source_priority_and_transform(columns, extra_fields, expected):
    class Row(BaseModel):
        value: int = SourceField(source=("Input value", "backup"), transform=int, strict=True)

    result = dataframe_to_models(pd.DataFrame(columns), Row, extra_fields=extra_fields)

    assert result[0].value == expected


def test_transform_failure_reports_field_and_value_and_chains_original_exception():
    class Row(BaseModel):
        value: int = SourceField(source="input", transform=int)

    with pytest.raises(ValueError, match="Transform failed for value") as error:
        dataframe_to_models(pd.DataFrame({"input": ["invalid"]}), Row)

    assert "'invalid'" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


def test_empty_dataframe_returns_empty_list():
    class Row(BaseModel):
        value: int

    assert dataframe_to_models(pd.DataFrame(), Row) == []


def test_rows_without_columns_can_use_extra_fields_and_defaults():
    class Row(BaseModel):
        value: int
        status: str = "active"

    frame = pd.DataFrame(index=[10, 10])
    result = dataframe_to_models(frame, Row, extra_fields={"value": 1}, strict=False)

    assert [row.model_dump() for row in result] == [
        {"value": 1, "status": "active"},
        {"value": 1, "status": "active"},
    ]


def test_duplicate_columns_are_rejected_instead_of_overwritten():
    class Row(BaseModel):
        value: int

    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])

    with pytest.raises(ValueError, match="column names must be unique"):
        dataframe_to_models(frame, Row)
