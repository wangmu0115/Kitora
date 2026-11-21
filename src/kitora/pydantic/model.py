from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

type SourceModelName = str
type SourceFieldName = str
type FieldRule = str | tuple[SourceModelName, SourceFieldName] | Callable[[dict[str, BaseModel]], Any]
type SingleFieldRule[M: BaseModel] = str | Callable[[M], Any]


def model_to_model[S: BaseModel, T: BaseModel](
    source: S,
    target_cls: type[T],
    *,
    field_rules: dict[str, SingleFieldRule[S]] | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """将单个源 Pydantic 模型的字段映射为目标 Pydantic 模型实例。

    1. 每个目标字段按 `overrides`、`field_rules`、源模型导出数据的同名字段顺序取值。取到 `None` 时也传给目标模型校验。
    2. 若未指定覆盖值或规则，且没有同名字段，则省略该字段，由目标模型应用默认值或报告必填字段缺失。

    Args:
        source: 普通 Pydantic 模型实例，必须具有命名字段，且 `model_dump()` 必须返回以字符串为键的字段字典。
        target_cls: 具有命名字段的目标 `BaseModel` 类。顶层校验输入使用 Python 字段名作为键。
            如果字段声明了与字段名不同的 `alias` 或 `validation_alias`，目标模型必须同时接受字段名，例如设置
            `model_config = ConfigDict(validate_by_name=True)`。
        field_rules: 以目标字段名为键的取值规则。
            - 字符串表示从源模型的字段字典中读取指定键。
            - 可调用对象接收原始 `source` 实例并返回目标字段值，其抛出的异常会向上传播。
            - 不传或者传入空字典时按同名字段匹配。
        overrides: 以目标字段名为键的显式覆盖值。键存在时直接使用对应值，跳过该字段的取值规则。

    Returns:
        通过校验的 `target_cls` 实例。

    Raises:
        ValueError: 执行的字符串规则引用了源模型导出数据中不存在的键。
        TypeError: 执行的字段规则类型不受支持。
        pydantic.ValidationError: 映射后的字段数据未通过目标模型校验。
    """
    adapted_mapping: dict[str, FieldRule] | None = None
    if field_rules is not None:
        adapted_mapping = {}
        for field, rule in field_rules.items():
            if callable(rule):
                adapted_mapping[field] = lambda _sources, apply_rule=rule: apply_rule(source)
                continue
            adapted_mapping[field] = rule

    return models_to_model(
        {"__source__": source},
        target_cls,
        field_rules=adapted_mapping,
        overrides=overrides,
    )


def models_to_model[T: BaseModel](
    sources: dict[str, BaseModel],
    target_cls: type[T],
    *,
    field_rules: dict[str, FieldRule] | None = None,
    overrides: dict[str, Any] | None = None,
) -> T:
    """将多个具名源 Pydantic 模型的字段值合并为一个目标 Pydantic 模型实例。

    1. 每个目标字段按 `overrides`、`field_rules`、合并数据中的同名字段这一顺序取值。取到 `None` 时也传给目标模型校验。
    2. 若未指定覆盖值或规则，且没有同名字段，则省略该字段，由目标模型应用默认值或报告必填字段缺失。

    Args:
        sources: 从来源名称到普通 `BaseModel` 实例的非空字典。
            每个源模型必须具有命名字段，且 `model_dump()` 必须返回以字符串为键的字段字典。
            多个来源含有同名键时，按 `sources` 的插入顺序取最后一个值，包括该来源导出的默认值和 `None`。
        target_cls: 具有命名字段的目标 `BaseModel` 类。顶层校验输入使用 Python 字段名作为键。
            如果字段声明了与字段名不同的 `alias` 或 `validation_alias`，目标模型必须同时接受字段名，例如设置
            `model_config = ConfigDict(validate_by_name=True)`。
        field_rules: 以目标字段名为键的取值规则。
            - 字符串表示从合并数据中读取指定键。
            - 二元组 `(source_name, key)` 表示从指定来源的字段字典中读取该键。
            - 可调用对象接收原始 `sources` 字典并返回目标字段值，其抛出的异常会向上传播。
        overrides: 以目标字段名为键的显式覆盖值。键存在时直接使用对应值，跳过该字段的取值规则。

    Returns:
        通过校验的 `target_cls` 实例。

    Raises:
        ValueError: `sources` 为空，或执行的字段规则引用了不存在的来源或键。
        TypeError: 执行的字段规则类型不受支持。
        pydantic.ValidationError: 合并后的字段数据未通过目标模型校验。
    """
    if not sources:
        raise ValueError("At least one source model is required.")
    overrides = overrides or {}
    field_rules = field_rules or {}

    source_payloads = {source_name: source_model.model_dump() for source_name, source_model in sources.items()}
    merged_payload: dict[str, Any] = {}
    for source_payload in source_payloads.values():
        merged_payload.update(source_payload)

    target_payload: dict[str, Any] = {}
    for target_field_name in target_cls.model_fields:
        if target_field_name in overrides:
            target_payload[target_field_name] = overrides[target_field_name]
            continue

        field_rule = field_rules.get(target_field_name)
        if field_rule is not None:
            if callable(field_rule):
                target_payload[target_field_name] = field_rule(sources)
                continue

            if isinstance(field_rule, tuple):
                source_model_name, source_field_name = field_rule
                if source_model_name not in sources:
                    raise ValueError(f"Unknown source model name: {source_model_name}.")
                source_playload = source_payloads[source_model_name]
                if source_field_name not in source_playload:
                    raise ValueError(f"Missing field {source_field_name} from source {source_model_name}.")
                target_payload[target_field_name] = source_playload[source_field_name]
                continue

            if isinstance(field_rule, str):
                if field_rule not in merged_payload:
                    raise ValueError(f"Missing {field_rule} for {target_field_name}.")
                target_payload[target_field_name] = merged_payload[field_rule]
                continue

            raise TypeError(f"Unsupported field rule {type(field_rule).__name__} for {target_field_name}.")

        if target_field_name in merged_payload:
            target_payload[target_field_name] = merged_payload[target_field_name]
            continue

    return target_cls.model_validate(target_payload)
