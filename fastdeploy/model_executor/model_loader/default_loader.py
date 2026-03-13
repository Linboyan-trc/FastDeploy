import contextlib
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (load_composite_checkpoint, measure_time)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.platforms import current_platform


# 1. 默认使用的Default
class DefaultModelLoader(BaseModelLoader):
    # 1.1 初始化
    # 1.1.1 打印日志，使用Default Model Loader
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("Load the model and weights using DefaultModelLoader")

    # 1.2 下载模型
    # 1.2.1 实现父类下载模型
    def download_model(self, model_config: ModelConfig) -> None:
        pass

    # 1.3 加载参数
    # 1.3.1 实现父类加载参数
    def load_model(self, fd_config: FDConfig) -> nn.Layer:
        # 1.3.1 获取结构
        architectures = fd_config.model_config.architectures[0]

        # 1.3.2 打印日志，开始加载参数
        logger.info(f"Starting to load model {architectures}")

        # 1.3.3 加载参数
        # 1.3.3.1 动态加载参数
        if fd_config.load_config.dynamic_load_weight:
            import fastdeploy.rl
            if fd_config.speculative_config.model_type != "mtp": architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MoeForCausalLM")
            else: architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MTPForCausalLM")
            architectures = architectures + "RL"
            context = paddle.LazyGuard()
        # 1.3.3.2 正常加载参数
        else:
            context = contextlib.nullcontext()

        # 1.3.4 看不懂
        with context:
            model_cls = ModelRegistry.get_class(architectures)
            model = model_cls(fd_config)
        model.eval()

        # TODO(gongshaotian): Now, only support safetensor
        # 1.3.5 返回加载好参数的模型
        if fd_config.load_config.dynamic_load_weight:
            return model
        else:
            # 1.3.5 静态加载参数，传入model, fd_config, architectures
            self.load_weights(model, fd_config, architectures)
            return model
    
    # 1.3.2 静态加载参数
    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, architectures: str) -> None:
        # 1.1 不知道干嘛的
        model_class = ModelRegistry.get_pretrain_cls(architectures)

        # 1.2 从磁盘读到CPU内存
        state_dict = load_composite_checkpoint(fd_config.model_config.model, model_class, fd_config, return_numpy=True)

        # 1.3 从CPU内存填进GPU显存中的模型参数
        model.set_state_dict(state_dict)
        self.clean_memory_fragments(state_dict)

    def clean_memory_fragments(self, state_dict: dict) -> None:
        if current_platform.is_cuda() or current_platform.is_maca():
            if state_dict:
                for k, v in state_dict.items():
                    if isinstance(v, paddle.Tensor):
                        v.value().get_tensor()._clear()
            paddle.device.empty_cache()
            paddle.device.synchronize()



