# 1. Paddle库
# 1.1 Paddle库
# 1.2 nn基类
# 1.3 保证不进入多余分支，做类型合规检查
# 1.4 日志
import paddle
from paddle import nn
from typing_extensions import assert_never
from paddleformers.utils.log import logger

# 1. FastDeploy
# 1.1 FD配置, Load配置, Model配置
# 1.2 model_executor下的加载参数工具
# 1.3 model_executor下的工具，包括对加载好的参数进行后处理
from fastdeploy.config import FDConfig, LoadConfig, ModelConfig
from fastdeploy.model_executor.load_weight_utils import (get_model_path, get_weight_iterator, is_weight_cache_enabled, load_weights_from_cache, measure_time, save_model)
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.model_executor.utils import (need_memory_reconstruction, process_final_after_loading, reconstruct_memory)
from fastdeploy.platforms import current_platform


# 1. 默认使用的Default V1
class DefaultModelLoaderV1(BaseModelLoader):
    # 1.1 初始化
    # 1.1.1 打印日志，使用Default Model Loader
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("Load the model and weights using DefaultModelLoader V1")

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
        context = paddle.LazyGuard()
        if fd_config.load_config.dynamic_load_weight:
            import fastdeploy.rl  # noqa
            if fd_config.speculative_config.model_type != "mtp": architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MoeForCausalLM")
            else: architectures = architectures.replace("Ernie5ForCausalLM", "Ernie5MTPForCausalLM")
            architectures = architectures + "RL"

        # 1.3.4 看不懂
        enable_cache, _, weight_cache_context = is_weight_cache_enabled(fd_config)
        fd_config.model_config.enable_cache = enable_cache
        with weight_cache_context:
            with context:
                model_cls = ModelRegistry.get_class(architectures)
                convert_type = fd_config.model_config.convert_type
                if convert_type == "none": pass
                elif convert_type == "embed": model_cls = as_embedding_model(model_cls)
                else: assert_never(convert_type)
                model = model_cls(fd_config)
                if fd_config.load_config.dynamic_load_weight or fd_config.model_config.enable_cache:
                    process_final_after_loading(model, fd_config)
        model.eval()

        # RL model not need set_state_dict
        # 1.3.5 返回加载好参数的模型
        if fd_config.load_config.dynamic_load_weight:
            return model
        else:
            # 1.3.5 静态加载参数，传入model, fd_config, enable_cache
            self.load_weights(model, fd_config, enable_cache)
            if need_memory_reconstruction(fd_config):
                reconstruct_memory(model)
            return model

    # 1.3.2 静态加载参数
    @save_model()
    @measure_time()
    def load_weights(self, model, fd_config: FDConfig, enable_cache: bool = False) -> None:
        # 1.1 从模型参数下，获取参数迭代器
        model_path = get_model_path(fd_config)
        weights_iterator = get_weight_iterator(model_path)
        
        # 1.2 从缓存中读取
        if enable_cache:
            load_weights_from_cache(model, weights_iterator)
        
        # 1.3 从GPU显存中读取
        else:
            model.load_weights(weights_iterator)

        # 1.4 有一些参数读取进Model之后，还需要多一些后处理
        process_final_after_loading(model, fd_config)

        # 1.5 清理显存
        self.clean_memory_fragments()

    def clean_memory_fragments(self) -> None:
        if current_platform.is_cuda() or current_platform.is_maca():
            paddle.device.empty_cache()
            paddle.device.synchronize()
