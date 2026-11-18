from collections.abc import Callable
from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo


def dataframe_to_models[T: BaseModel](
    df: pd.DataFrame,
    model_cls: type[T],
    *,
    extra_fields: dict[str, Any] | None = None,
    strict: bool = True,
) -> list[T]:
    """将 pandas DataFrame 的每一行映射为一个 Pydantic 模型实例，并返回一个 Pydantic 模型列表。

    目标模型字段可以使用 `SourceField` 声明来源列、必填策略和转换函数，未使用 `SourceField` 的字段会默认尝试读取同名列。

    字段解析规则：
        - 如果字段通过 `SourceField(source=...)` 声明来源列，会优先从这些列名中读取值。
        - 如果未声明 `source`，会使用模型字段名作为来源列名。
        - 如果 DataFrame 中没有匹配列，但 `extra_fields` 中存在同名字段，则使用 `extra_fields` 的值。
        - 如果字段声明了 `SourceField(transform=...)`，会在读取并归一化空值后调用该转换函数。
        - `None` 和 pandas 空值会被视为 `None`。

    Args:
        df: 来源 pandas DataFrame。
        model_cls: 目标 Pydantic 模型类，字段可使用 `SourceField` 补充来源映射元信息。
        extra_fields: 可选的额外字段值，用于补充 DataFrame 中不存在的字段。
        strict: 是否在映射阶段对缺失来源列提前抛出 `ValueError`。

    Returns:
        转换并通过 `model_cls.model_validate()` 校验后的模型列表。
    """

    rows: list[T] = []
    for _, record in df.iterrows():
        payload: dict[str, Any] = {}
        for field_name, field_info in model_cls.model_fields.items():
            value = _parse_field_value(
                record,
                field_name,
                field_info,
                extra_fields or {},
                strict=strict,
            )
            payload[field_name] = value
            rows.append(model_cls.model_validate(payload))
    return rows


def _parse_field_value(
    record: pd.Series,
    field_name: str,
    field_info: FieldInfo,
    extra_fields: dict[str, Any],
    *,
    strict: bool = True,
) -> Any:
    field_extra = (
        field_info.json_schema_extra
        if isinstance(field_info.json_schema_extra, dict)
        else {}
    )

    source_candidates = _normalize_value_sources(field_extra.get("source"), field_name)
    required = bool(field_extra.get("required", False))
    transform = field_extra.get("transform")

    found = False
    value = None
    used_source = None
    for source_candidate in source_candidates:
        if source_candidate in record:
            found = True
            value = record.get(source_candidate)
            used_source = source_candidate
            break
    if not found and field_name in extra_fields:
        found = True
        value = extra_fields.get(field_name)
        used_source = f"extra_fields.{field_name}"

    if not found and (strict or required):
        raise ValueError(
            f"Missing source colume for {field_name}, candidates is {source_candidates}"
        )

    value = _normalize_pandas_value(value)
    if value is None:
        return None
    if transform is not None and isinstance(transform, Callable):
        try:
            value = transform(value)
        except Exception as e:
            raise ValueError(
                f"Transform failed for {field_name}, source is {used_source}, value is {value!r}"
            ) from e
    return value


def _normalize_value_sources(source: Any, field_name: str) -> list[str]:
    if source is None:
        return [field_name]

    if isinstance(source, str):
        return [source, field_name]

    return [*source, field_name]


def _normalize_pandas_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value
