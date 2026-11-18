from collections.abc import Callable
from typing import Any

from pydantic import Field


def SourceField(
    *,
    source: str | list[str] | tuple[str, ...] | None = None,
    required: bool = False,
    transform: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """创建带有外部来源映射元信息的 Pydantic 字段。

    `SourceField` 描述外部数据结构的列如何映射到 Pydantic 模型的字段，例如 pandas DataFrame、数据库查询结果等。
    """
    _json_schema_extra = kwargs.pop("json_schema_extra", {}) or {}
    _json_schema_extra.update(
        {
            "source": source,
            "required": required,
            "transform": transform,
        }
    )
    return Field(json_schema_extra=_json_schema_extra, **kwargs)
