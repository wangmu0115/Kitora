from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, cast

from pydantic import Field
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class _SourceMetadata:
    source: tuple[str, ...] | None = None
    required: bool = False
    transform: Callable[[Any], Any] | None = None


def SourceField(
    *,
    source: str | tuple[str, ...] | None = None,
    required: bool = False,
    transform: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """配置 Pydantic 模型字段，在 `pydantic.Field` 的基础上可额外配置来源映射和数据转换规则。

    - 字段的默认值、别名、校验约束和描述等配置由 `pydantic.Field` 处理。
    - `source`、`required` 和 `transform` 作为附加元数据，供映射器读取，例如 `kitora.pandas.dataframe_to_models`。

    ***每个模型字段应只声明一个 SourceField。如有多个候选来源，请通过 source 元组按优先级指定。***

    Args:
        source: 外部数据中的来源名称。可传入单个名称，或按查找优先级排列的名称元组。None 表示未指定来源名称。
        required: 映射时是否必须找到数据来源。字段本身是否必填由 Pydantic 的字段声明决定。
        transform: 对来源值执行转换的函数，接收一个值并返回转换结果。None 表示未指定转换函数。
        **kwargs: 传给 `pydantic.Field` 的关键字参数，例如 `default`、`alias`、`description` 等等。

    Returns:
        包含 Pydantic 字段配置和来源映射元数据的 `FieldInfo` 对象。
    """
    metadata = _SourceMetadata(
        source=_normalize_source(source),
        required=required,
        transform=transform,
    )
    return FieldInfo.from_annotated_attribute(
        cast(type[Any], Annotated[Any, metadata]),
        Field(**kwargs),
    )


def _normalize_source(source: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
    if source is None:
        return None
    elif isinstance(source, str):
        return (source,)
    else:
        return source
