from typing import TYPE_CHECKING

from kitora.import_utils import import_attr

if TYPE_CHECKING:
    from kitora.pandas.dataframe import dataframe_to_models

__all__ = ("dataframe_to_models",)

_dynamic_imports = {
    "dataframe_to_models": "dataframe",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
