# 1. 抽象类
# 1.1 抽象类
# 1.2 nn基类
# 1.3 FD配置，Load配置，Model配置
from abc import ABC, abstractmethod
from paddle import nn
from fastdeploy.config import FDConfig, LoadConfig, ModelConfig


# 1. Base Model Loader基类
class BaseModelLoader(ABC):
    # 1.1 初始化，设置Load配置
    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    # 1.2 下载模型，子类需要实现
    @abstractmethod
    def download_model(self, load_config: ModelConfig) -> None:
        raise NotImplementedError

    # 1.3 加载参数，子类需要实现
    @abstractmethod
    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        raise NotImplementedError
