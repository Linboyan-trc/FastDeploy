# 1. 配置
# 1.1 Load配置
# 1.2 Load选择, default, default_v1, dummy
# 1.3 Loader类
# 1.3.1 Base Model Loader是父类
# 1.3.2 Default, Default V1, Dummy是子类
from fastdeploy.config import LoadConfig, LoadChoices
from fastdeploy.model_executor.model_loader.base_loader import BaseModelLoader
from fastdeploy.model_executor.model_loader.default_loader import DefaultModelLoader
from fastdeploy.model_executor.model_loader.default_loader_v1 import DefaultModelLoaderV1
from fastdeploy.model_executor.model_loader.dummy_loader import DummyModelLoader


# 2. 选择三个子类
# 2.1 Default, Default V1, Dummy是子类
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    # 2.1 根据Load配置中的选择，返回Default, Default V1, Dummy中的一个
    # 2.1.1 其中Load配置中的选择若空缺，默认返回Default
    if load_config.load_choices == LoadChoices.DEFAULT_V1:
        return DefaultModelLoaderV1(load_config)
    if load_config.load_choices == LoadChoices.DUMMY:
        return DummyModelLoader(load_config)
    return DefaultModelLoader(load_config)


# 3. 当import fastdeploy.model_executor.model_loader的时候，设置导出内容
# 3.1 只会导出get_model_loader()这个方法，而不导出Default, Default V1, Dummy三个子类
__all__ = ["get_model_loader"]

