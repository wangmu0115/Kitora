from typing import TYPE_CHECKING

from kitora.import_utils import import_attr

if TYPE_CHECKING:
    from kitora.pydantic.augmented_fields import SourceField
    from kitora.pydantic.model import model_to_model, models_to_model

__all__ = (
    "SourceField",
    "model_to_model",
    "models_to_model",
)

_dynamic_imports = {
    "SourceField": "augmented_fields",
    "model_to_model": "model",
    "models_to_model": "model",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
