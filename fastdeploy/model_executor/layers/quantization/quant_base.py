# 1. 基础
# 1.1 抽象类
# 1.2 任何类型，Optional[int, None]选择类型
from abc import ABC, abstractmethod
from typing import Any, Optional


# 1. 量化方法类
# 1.1 是所有对权重进行FP8, INT4量化的基类
class QuantMethodBase(ABC):
    # 1.1 给Transformer第5层，创建量化权重
    @abstractmethod
    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        raise NotImplementedError

    # 1.2 使用Transformer第5层，的量化权重，参与计算
    @abstractmethod
    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    # 1.3 使用Transformer第5层，加载好的权重，进行一些后处理比如矩阵转置一下
    def process_loaded_weights(self, layer, weights):
        return


class QuantConfigBase(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "QuantConfigBase":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's " "quantization config.")

    @abstractmethod
    def get_quant_method(self, layer, prefix) -> Optional[QuantMethodBase]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError
