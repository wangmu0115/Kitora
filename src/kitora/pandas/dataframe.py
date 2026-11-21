from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from kitora.pydantic.augmented_fields import _SourceMetadata

_MISSING = object()


def dataframe_to_models[T: BaseModel](
    df: pd.DataFrame,
    model_cls: type[T],
    *,
    extra_fields: dict[str, Any] | None = None,
    strict: bool = True,
) -> list[T]:
    """将 pandas DataFrame 的每一行映射为一个 Pydantic 模型实例。

    - 对模型的每个字段，按 `SourceField(source=...)` 声明的顺序查找候选列，再查找字段同名的列。
    取第一个存在的列，无论值是否为空。当候选列和同名列都不存在时，才按模型字段名从 `extra_fields` 取值。

    - 如果取到的值为 `None` 或 pandas 标量缺失值，则将字段值设为 `None` 且跳过 `transform` 转换。
    如果配置了 `transform`，则先调用该函数转换取到的值，再交给 Pydantic 校验。

    - 每行数据以模型**字段名**为键传给 `model_cls.model_validate()`，顶层模型**必须**接受按字段名输入的数据。
    如果字段声明了校验别名，可在模型上设置 `ConfigDict(validate_by_name=True)`。嵌套字典的键由对应子模型按自身配置解析。

    Args:
        df: 来源 pandas DataFrame，列名必须唯一。
        model_cls: 具有命名字段的 Pydantic 模型类。顶层 `RootModel` 不在支持范围内。
            每个字段至多声明一个 `SourceField`；如有多个候选来源，应在同一个 `SourceField` 中通过 `source` 元组指定。
        extra_fields: 以模型字段名为键的补充值，所有行共用。仅在找不到来源列时使用。
        strict: 是否要求每个字段都有来源，此参数仅控制来源检查。为 True 时，缺少来源即报错。
            为 False 时，只有声明了 `SourceField(required=True)` 的字段要求必须有来源；
            其他缺少来源的字段会从输入中省略，由 Pydantic 应用默认值或报告必填字段缺失。

    Returns:
        按 DataFrame 原始行序排列的 `model_cls` 实例列表，每行对应一个实例。没有数据行时返回空列表。

    Raises:
        ValueError: DataFrame 列名重复、字段缺少要求的来源，或 `transform` 执行失败。
        pydantic.ValidationError: 映射后的行数据未通过模型校验。
    """

    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")

    results: list[T] = []
    for row in df.itertuples(index=False, name=None):
        record = dict(zip(df.columns, row, strict=True))
        payload: dict[str, Any] = {}
        for field_name, field_info in model_cls.model_fields.items():
            value = _parse_field_value(
                record,
                field_name,
                field_info,
                extra_fields if extra_fields is not None else {},
                strict=strict,
            )
            if value is not _MISSING:
                payload[field_name] = value
        results.append(model_cls.model_validate(payload))
    return results


def _parse_field_value(
    record: dict[str, Any],
    field_name: str,
    field_info: FieldInfo,
    extra_fields: dict[str, Any],
    *,
    strict: bool,
) -> Any:
    metadata = next(
        (item for item in field_info.metadata if isinstance(item, _SourceMetadata)),
        _SourceMetadata(),
    )

    source_candidates = _build_source_candidates(metadata.source, field_name)

    found = False
    value = None
    for source_candidate in source_candidates:
        if source_candidate in record:
            found = True
            value = record[source_candidate]
            break
    if not found and field_name in extra_fields:
        found = True
        value = extra_fields[field_name]

    if not found:
        if strict or metadata.required:
            raise ValueError(f"Missing source column for {field_name}, candidates are {source_candidates}")
        return _MISSING

    value = _normalize_pandas_value(value)
    if value is None:
        return None
    if callable(metadata.transform):
        try:
            value = metadata.transform(value)
        except Exception as e:
            raise ValueError(f"Transform failed for {field_name}, value is {value!r}") from e
    return value


def _build_source_candidates(source: tuple[str, ...] | None, field_name: str) -> list[str]:
    if source is None:
        return [field_name]

    return [*source, field_name]


def _normalize_pandas_value(value: Any) -> Any:
    if value is None:
        return None

    if not pd.api.types.is_scalar(value):
        # 只对标量检查是否为 pandas None 值
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value
