"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.utils import get_logger
from fastdeploy.worker.output import ModelRunnerOutput

logger = get_logger("model_runner_base", "model_runner_base.log")


# 1. dataclass作用
# 1.1 快速将一个类变成具有__init__(), __repr__(), __eq__()方法的类，成为一种专用于表示数据的类
# 1.2 有了@dataclass装饰器之后只需要声明需要的成员变量及其类型，就可以自动具有__init__(), __repr__(), __eq__()方法
# 1.3 这个类是分布式执行时的输入状态信息
@dataclass
class DistributedStatus:
    # 1.1 当前是否处于decode阶段
    # 1.2 MoE切成多少chunk
    only_decode: bool = True
    moe_num_chunk: int = 1


# 2. dataclass作用
# 2.1 分布式执行后返回给调度层的信息
@dataclass
class DistributedOut:
    # 2.1 是否只执行了 decode
    # 2.2 实际使用了多少 MoE chunk
    if_only_decode: bool = True
    max_moe_num_chunk: Optional[int] = None


# 3. 执行单卡推理的Worker
class ModelRunnerBase(ABC):
    """
    Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
    ModelRunner interface abstracts the model execution logic that
    contain input preparation, token generation, and tokenprocessing.
    """

    # 3.1 初始化，获取模型配置、加载配置、设备配置
    def __init__(self, fd_config: FDConfig, device: str) -> None:
        """
        Initialize FDConfig
        """
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        self.load_config = fd_config.load_config
        self.device_config = fd_config.device_config
        self.speculative_config = fd_config.speculative_config
        self.parallel_config = fd_config.parallel_config
        self.graph_opt_config = fd_config.graph_opt_config
        self.quant_config = fd_config.quant_config
        self.cache_config = fd_config.cache_config
        self.scheduler_config = fd_config.scheduler_config
        # ... config

        self.device = device

    # 3.2 子类需要实现加载模型、获取模型、执行推理、分析推理
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model from local path or remote(will download) path
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """
        Get current model
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
    ) -> ModelRunnerOutput:
        """
        Execute model with and get output
        """
        raise NotImplementedError

    @abstractmethod
    def profile_run(self) -> None:
        """
        Execute a forward pass with dummy inputs to profile the memory usage of the model."
        """
        raise NotImplementedError

    def vision_encoder_compile(self):
        """
        Compile the vision encoder if applicable
        """
        logger.info(f"No vision encoder compilation for base {self.__class__.__name__}")
