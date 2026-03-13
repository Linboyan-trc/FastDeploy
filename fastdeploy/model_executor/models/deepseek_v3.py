# 1. Python依赖
# 1.1 future: 延迟引用解析
# 1.2 math: 数学
# 1.3 re: 正则表达式
# 1.4 partial: partial(add, 5)，给函数固定一些参数
from __future__ import annotations
import math
import re
from functools import partial

# 2. paddle依赖
# 2.1 paddle: paddle库
# 2.2 paddle.nn: 神经网络层基础库
# 2.3 paddleformers.transformers.PretrainedModel: 预训练模型基类，带有下载模型，加载权重的功能
# 2.4 paddleformers.utils.log.logger: 日志系统
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel
from paddleformers.utils.log import logger

# 3. fastdeploy引用
# 3.1 fastdeploy.config.FDConfig: 自定义配置类
# 3.2 fastdeploy.distributed.communication.tensor_model_parallel_all_reduce: 张量并行之后，有的计算步骤需要进行GPU通信进行Reduce
from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce

# 3.3 推理元信息，包括推理阶段，词表大小，上下文长度seq_len
# 3.4 图优化装饰器
# 3.5 模型类别、语言模型基类、模型注册器
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
from fastdeploy.model_executor.models.model_base import (ModelCategory, ModelForCasualLM, ModelRegistry)

# 3.9 词表，token ids -> embedding
# 3.13 归一化算子
# 3.10 投影计算，列分片、行分片、完全复制一样的、融合列分片、融合复制分片、KV Cache优化
# 3.14 RoPE计算算子
# 3.8 注意力计算算子，对Q, K, V进行注意力计算，得到P
# 3.7 FFN层融合算子，将gate_proj投影之后的激活值 与 up_proj投影逐元素点乘法作为一个操作
# 3.12 MoE计算的融合算子
# 3.11 logits计算的算子
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.linear import (ColumnParallelLinear, KVBatchLinear, MergedColumnParallelLinear, MergedReplicatedLinear, ReplicatedLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead

# 3.15 硬件平台
# 3.16 硬件平台导入算子
# 3.16.1 如果是NVIDIA GPU或者Mac Apple Silicon，就引入一个批量生成position_ids、attention_mask的算子
from fastdeploy.platforms import current_platform
if current_platform.is_cuda() or current_platform.is_maca():
    from fastdeploy.model_executor.ops.gpu import get_position_ids_and_mask_encoder_batch


# 1. MLP计算
# 1.1 就是基础的FFN、shared experts计算
# 1.2 gate_proj把7168放大到18432，然后silu激活，然后乘法；然后down_proj还原回7168
# 1.3 up_proj把7168放大到18432
# 1.3.其中hidden_sizes = 7168，intermediate_size = 18432，hidden_act是silu
# 2.1 这里对于gate、up、silu激活、逐元素乘法，使用融合列分片
# 2.2 这里对于down还原，使用行分片
# 2.3 SiluAndMul怎么像纯激活函数
class DeepSeekV3MLP(nn.Layer):
    # 1.1 初始化FFN需要用到的三个参数、一个激活操作
    def __init__(
        self,
        fd_config: FDConfig,
        intermediate_size: int,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.up_gate_proj = MergedColumnParallelLinear( fd_config=fd_config, prefix=f"{prefix}.up_gate_proj",   input_size=fd_config.model_config.hidden_size,  output_size=intermediate_size * 2,              with_bias=False, activation=fd_config.model_config.hidden_act)
        self.down_proj = RowParallelLinear(             fd_config=fd_config, prefix=f"{prefix}.down_proj",      input_size=intermediate_size,                   output_size=fd_config.model_config.hidden_size, with_bias=False, reduce_results=reduce_results)
        self.act_fn = SiluAndMul(fd_config=fd_config, bias=None, act_method=fd_config.model_config.hidden_act)

    # 1.2 加载参数
    # 1.2.1 up_gate_proj是融合列分片，是我们自定义的类，也继承了nn.Layer；其中又写了gate、up，所以这里直接up_gate_proj可以从字典里找到两个参数
    # 1.2.2 down_proj是行分片，也是我们自定义的泪，继承了nn.Layer，从字典里找参数
    # 1.3 这里state_dict就是所有参数名和张量组成的字典，此时所有参数都已经加载进内存，然后从所有参数里找key符合的，然后赋值给self.up_gate_proj，self.down_proj
    def load_state_dict(self, state_dict):
        self.up_gate_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    # 1.3 FFN计算
    # 1.3.1 x.shape = [100, 7168]
    # 1.3.2 原始计算流程为：x先进行gate投影、激活，up投影，逐元素点乘，得到shape = [100, 18432]
    # 1.3.2 然后shape = [100, 18432]再进行down投影得到[100, 7168]
    # 1.3.3 这里up_gate_proj()先仅仅计算gate投影、up投影，然后act_fn()完成gate激活、与up投影的逐元素点乘，最后down_proj()完成down投影
    def forward(self, x, forward_meta=None):
        gate_up_out = self.up_gate_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

# 2. MoE计算
# 2.1 x维度是7168,，gate、up投影放大到2048，然后down还原到7168
# 2.2 hidden_sizes = 7168，moe_intermediate_size = 2048，hidden_act是silu
class DeepSeekV3MoE(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        # 2.1 多卡并行
        super().__init__()
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.attn_tp_size = fd_config.parallel_config.tensor_parallel_size
        if self.ep_size > 1:
            self.tp_size = 1

        # 2.2 专家权重归一化，在每个token选出其topk = 8个专家后
        self.norm_topk_prob = fd_config.model_config.norm_topk_prob

        # 2.3 MoE运算需要的参数有两套：gate + gate_e_score_correction_bias
        # 2.3 MoE运算需要的参数有两套：experts的gate、up，experts的gate、up、down
        weight_key_map = {
            "gate_correction_bias_key": f"{prefix}.gate.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
        }

        # 2.4 专家打分算子,x.shape = [100, 7168] -> score.shape = [100, 256]
        # 2.4.1 hidden_size = 7168， n_routed_experts = 256
        self.gate = ReplicatedLinear(fd_config=fd_config, prefix=f"{prefix}.gate", input_size=fd_config.model_config.hidden_size, output_size=fd_config.model_config.n_routed_experts, with_bias=False, skip_quant=True, weight_dtype="float32")

        # 2.5 专家打分之后的修正参数，gate_e_score_correction_bias = [256]
        if fd_config.model_config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = self.create_parameter(shape=[1, fd_config.model_config.n_routed_experts], dtype="float32", default_initializer=paddle.nn.initializer.Constant(0))
        else:
            self.gate.e_score_correction_bias = None

        # 2.6 专家MoE算子
        self.experts = FusedMoE(
            # 2.6.1 打分修正
            gate_correction_bias=self.gate.e_score_correction_bias,

            # 2.6.2 权重归一化
            renormalize=self.norm_topk_prob,

            # 2.6.3 7168 -> 2048 -> 7168
            # 2.6.3.1 moe_intermediate_size
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,

            # 2.6.4 专家分组，和选前几组
            # 2.6.4 专家数量，每个token选择的专家数量，不使用辅助Loss
            # 2.6.4 MoE输出放大
            n_group=fd_config.model_config.n_group,
            topk_group=fd_config.model_config.topk_group,
            num_experts=fd_config.model_config.n_routed_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method=fd_config.model_config.topk_method,
            routed_scaling_factor=fd_config.model_config.routed_scaling_factor,

            # 2.6.5 配置
            fd_config=fd_config, 
            layer_idx=layer_id,
            reduce_results=False, 
            weight_key_map=weight_key_map,
        )

        # 2.7 有一个共享专家算子
        # 2.7.1 一个共享专家的中间也是2048
        # 2.7.2 共享专家算子是普通FFN算子
        self.num_shared_experts = fd_config.model_config.n_shared_experts
        shared_experts_intermediate_size = self.num_shared_experts * fd_config.model_config.moe_intermediate_size
        self.shared_experts = DeepSeekV3MLP(fd_config=fd_config, intermediate_size=shared_experts_intermediate_size, prefix=f"{prefix}.shared_experts", reduce_results=False,)

    # 2. 加载参数
    def load_state_dict(self, state_dict):
        # 2.1 打分参数
        # 2.2 打分修正参数
        # 2.3 共享专家参数
        # 2.3 MoE专家参数
        self.gate.load_state_dict(state_dict)
        if self.experts.gate_correction_bias is not None:
            gate_correction_bias_tensor = state_dict.pop(self.experts.gate_correction_bias_key)
            if self.experts.gate_correction_bias.shape != gate_correction_bias_tensor.shape:
                gate_correction_bias_tensor = gate_correction_bias_tensor.reshape(self.experts.gate_correction_bias.shape)
            self.experts.gate_correction_bias.set_value(gate_correction_bias_tensor)
        
        self.shared_experts.load_state_dict(state_dict)
        self.experts.load_state_dict(state_dict)

    # 3. MoE计算
    def forward(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta):
        # 3.1 打分 + 完成MoE计算
        # 3.1.1 hidden_states + self.gate打分
        # 3.1.2 然后再MoE计算得到输出
        moe_out = self.experts(hidden_states, self.gate, forward_meta)

        # 3.2 共享专家计算
        shared_experts_out = self.shared_experts(hidden_states)
        if self.attn_tp_size > 1 and self.ep_size > 1:
            shared_experts_out = tensor_model_parallel_all_reduce(shared_experts_out)

        # 3.3 共享专家结果 + MoE专家结果作为输出
        moe_out = moe_out + shared_experts_out
        if self.tp_size > 1:
            moe_out = tensor_model_parallel_all_reduce(moe_out)
        return moe_out


# 3. Attention计算
class DeepseekV3MLAAttention(nn.Layer):
    # 3.1 获取参数
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str = "") -> None:
        # 3.1 x维度是7168，hidden_size = 7168，不过做完投影Q到了24576，K到了128*192 = 24576
        # 3.1 128头注意力，num_attention_heads = 128
        super().__init__()
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.hidden_size = fd_config.model_config.hidden_size
        self.num_attention_heads = fd_config.model_config.num_attention_heads
        self.num_attention_heads_tp = self.num_attention_heads // self.tp_size

        # 3.2 旋转位置编码
        # 3.2.1 Q.shape = [100, 128, 194]，194拆成128 + 64，注意力计算的时候还是192
        # 3.2.1 qk_nope_head_dim=128，qk_rope_head_dim=64，qk_head_dim=128+64=192
        # 3.2.2 V每个头是128维
        # 3.2.3 Q,k,V投影缩小维度，把x.shape = [100, 7168]缩小成1536，512(来自576拆去64)
        self.qk_nope_head_dim = fd_config.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = fd_config.model_config.v_head_dim
        self.q_lora_rank = fd_config.model_config.q_lora_rank
        self.kv_lora_rank = fd_config.model_config.kv_lora_rank

        # 3.3 超参数
        # 3.3.1 一个是归一化防止根均方为0
        # 3.3.1 一个是旋转位置编码theta = 10000 rad
        # 3.3.1 一个是注意力计算除以sqrt(d)
        self.attn_softmax_scale = self.qk_head_dim**-0.5
        self.rope_theta = fd_config.model_config.rope_theta
        self.rms_norm_eps = fd_config.model_config.rms_norm_eps

        assert self.q_lora_rank is not None, "self.q_lora_rank is None, Please Check your config."
        # NOTE: (changwenbin) qkv_a_proj horizontal fusion

        # 3.4 q_a_proj, kv_a_proj一起算，q缩到1536维，kv缩到576维
        # 3.4.1 hidden_size = 7168, q_lora_rank = 1536, kv_lora_rank + qk_rope_head_dim = 576
        self.qkv_a_proj_with_mqa = MergedReplicatedLinear(fd_config=fd_config, prefix=f"{prefix}.qkv_a_proj_with_mqa", input_size=self.hidden_size, output_sizes=[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],with_bias=False,)

        # 3.5 q_a_layernormd，对1536归一化
        # 3.5.1 q_lora_rank = 1536，rms_norm_eps防止除零
        self.q_a_layernorm = RMSNorm(fd_config, hidden_size=self.q_lora_rank, eps=self.rms_norm_eps, prefix=f"{prefix}.q_a_layernorm")

        # 3.6 q_b_proj，将1536放大到24576
        # 3.6.1 q_lora_rank = 1536，num_attention_heads*qk_head_dim=128*192=24576
        self.q_b_proj = ColumnParallelLinear(fd_config=fd_config, prefix=f"{prefix}.q_b_proj", input_size=self.q_lora_rank, output_size=self.num_attention_heads * self.qk_head_dim, with_bias=False)

        # 3.5 kv_a_layernorm，对512归一化
        # 3.5.1 kv_lora_rank = 512，rms_norm_eps防止除零
        self.kv_a_layernorm = RMSNorm(fd_config, hidden_size=self.kv_lora_rank, eps=self.rms_norm_eps, prefix=f"{prefix}.kv_a_layernorm")

        # 3.6 kv_b_proj，将512放大到32768
        # 3.6.1 后续拆成K、V，各自16384
        # 3.6.1 kv_lora_rank = 512，num_attention_heads*(qk_nope_head_dim + v_head_dim) = 128 * (128 + 128) = 16384
        self.kv_b_proj = ColumnParallelLinear(fd_config=fd_config, prefix=f"{prefix}.kv_b_proj", input_size=self.kv_lora_rank, output_size=self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim), with_bias=False)

        # 3.7 attn_o_proj，对注意力计算结果投影，从128*128=16384降低到7168维
        # 3.7.1 num_attention_heads*v_head_dim = 128*128 = 16384，hidden_size = 7168
        self.o_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.o_proj", input_size=self.num_attention_heads * self.v_head_dim, output_size=self.hidden_size, with_bias=False, layer_id=layer_id)

        # 3.8 Batch相关
        self.kv_b_proj_bmm = KVBatchLinear(
            fd_config=fd_config,
            kv_b_proj=self.kv_b_proj,
            prefix=f"{prefix}.kv_b_proj",
            kv_lora_rank=self.kv_lora_rank,
            num_attention_heads=self.num_attention_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            v_head_dim=self.v_head_dim,
        )

        # 3.9 RoPE扩展
        self.rope_scaling = fd_config.model_config.rope_scaling
        if self.rope_scaling:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", False)
            scaling_factor = self.rope_scaling["factor"]
            mscale = self.yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        rope_scaling_kwargs = {
            key: self.rope_scaling[key]
            for key in [
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.rope_scaling
        }
        self.rope_scaling_factor = self.rope_scaling["factor"]
        self.rope_scaling_original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]
        
        # 3.10 RoPE算子
        self.rotary_emb = DeepseekScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.rope_scaling_original_max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
            **rope_scaling_kwargs,
        )

        # 3.11 注意力计算
        self.mla_attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

        # 3.12 前缀
        self.prefix = prefix

    # 3.2 RoPE扩展
    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        """ """
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # 3.3 Attention计算
    # 3.3.1 Prefill需要x，mask
    # 3.3.1 Decode 需要x，position ids
    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        # 3.1 计算q_a_proj，kv_a_proj
        # NOTE: (changwenbin) Bring out the public calculation in PD MIX to avoid repeated calculation.
        # NOTE: (changwenbin) qkv_a_proj horizontal fusion
        fmha_out = None
        qkv_a_out = self.qkv_a_proj_with_mqa(hidden_states)

        # 3.2 结果裂成Q = 1536，K = 512，K_rope = 64
        # 3.2.1 Q记为query
        # 3.2.2 K记为压缩kv，compressed_kv
        # 3.2.3 K_rope记为key_pe，就是key + positional embedding，key位置编码
        # 3.2.3 就是q是query，kv压缩，k是key位置编码是pe
        query, compressed_kv, key_pe = qkv_a_out.split([self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)

        # 3.3 q归一化，q投影放大，q分裂成[100, 128, 192]
        # 3.3.1 然后拆出q_nope.shape = [100, 128, 128]，    q_rope = [100, 128, 64]准备位置编码
        # 3.3.1 然后拆出query_nope.shape = [100, 128, 128]，query_pe = [100, 128, 64]准备位置编码
        query = self.q_a_layernorm(query)[0]
        query = self.q_b_proj(query)
        query.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
        query_nope, query_pe = query.split([self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        # 3.4 key_pe.shape = [100, 64]，要广播到[100, 128, 64]
        # 3.4 之后q_rope和k_rope一起进行RoPE，也就是query_pe，key_pe进行RoPE
        key_pe.reshape_([-1, 1, self.qk_rope_head_dim])
        query_pe, key_pe = self.rotary_emb(position_ids, query_pe, key_pe)

        # 3.5 kv_a_layernorm归一化
        compressed_kv = self.kv_a_layernorm(compressed_kv)[0]

        # 3.6 判断Prefill、Decode
        need_do_prefill = forward_meta.max_len_tensor_cpu[1] > 0
        need_do_decode = forward_meta.max_len_tensor_cpu[2] > 0

        # max_enc_len_this_time
        # 3.7 Prefill阶段
        if need_do_prefill:  
            # 3.7.1 kv_b_proj放大到32768维，然后拆成多头，然后拆出k、v
            key_value = self.kv_b_proj(compressed_kv)
            key_value.reshape_([-1,self.num_attention_heads_tp,self.qk_nope_head_dim + self.v_head_dim,])
            key_nope, value = key_value.split([self.qk_nope_head_dim, self.v_head_dim], axis=-1)

            # 3.7.2 query和key拼上已经旋转位置编码好的query_pe, key_pe
            query[..., self.qk_nope_head_dim :] = query_pe
            key = paddle.empty_like(query)
            key[..., : self.qk_nope_head_dim] = key_nope
            key[..., self.qk_nope_head_dim :] = key_pe
            value = paddle.nn.functional.pad(value, [0, self.qk_head_dim - self.v_head_dim], value=0)

            # 3.7.3 注意力计算
            fmha_out_prefill = self.mla_attn(
                q=query,
                k=key,
                v=value,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmha_out_prefill.reshape_([-1, self.num_attention_heads_tp, self.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, : self.v_head_dim]
            fmha_out_prefill.reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            fmha_out_prefill = fmha_out_prefill * mask_encoder_batch.cast(fmha_out_prefill.dtype)
            fmha_out = fmha_out_prefill

        # max_dec_len_this_time
        if need_do_decode:  
            q_nope_out = self.kv_b_proj_bmm(query_nope.transpose([1, 0, 2]), proj_type="k").transpose([1, 0, 2])

            q_input = paddle.concat([q_nope_out, query_pe], axis=-1)
            q_input.reshape_(
                [
                    -1,
                    self.num_attention_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )

            fmha_out_decode = self.mla_attn(
                q=q_input,
                k=None,
                v=None,
                qkv=None,
                compressed_kv=compressed_kv,
                k_pe=key_pe,
                forward_meta=forward_meta,
            )

            fmha_out_decode = fmha_out_decode.reshape_([-1, self.num_attention_heads_tp, self.kv_lora_rank]).transpose(
                [1, 0, 2]
            )

            fmha_out_decode = (
                self.kv_b_proj_bmm(fmha_out_decode, proj_type="v")
                .transpose([1, 0, 2])
                .reshape_([-1, self.num_attention_heads_tp * self.v_head_dim])
            )

            if need_do_prefill:
                fmha_out += fmha_out_decode
            else:
                fmha_out = fmha_out_decode

        # 3.8 注意力输出为16384维，还原到7168维
        output = self.o_proj(fmha_out)
        return output

    # 3.4 加载参数
    def load_state_dict(self, state_dict):
        # 3.4.1 把__init__()中的参数都加载一下
        self.q_a_layernorm.load_state_dict(state_dict)
        self.qkv_a_proj_with_mqa.load_state_dict(state_dict)
        self.kv_a_layernorm.load_state_dict(state_dict)
        self.q_b_proj.load_state_dict(state_dict)
        self.kv_b_proj_bmm.load_state_dict(state_dict)
        self.kv_b_proj.load_state_dict(state_dict)
        # NOTE(Ryan):Make sure kv_b_proj_bmm loaded before kv_b_proj,
        # The same weight key will be poped after kv_b_proj.
        self.o_proj.load_state_dict(state_dict)
        self.mla_attn.load_state_dict(state_dict)


# 1. Transformer
class DeepSeekV3DecoderLayer(nn.Layer):
    # 1.1 Transformer id
    # 1.2 Q、K、V投影 + Attention计算
    # 1.3 MLP计算
    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        # 1.1 id
        super().__init__()
        layer_id = int(prefix.split(sep=".")[-1])

        # 1.2 输入归一化
        self.input_layernorm = RMSNorm(fd_config, hidden_size=fd_config.model_config.hidden_size, eps=fd_config.model_config.rms_norm_eps, prefix=f"{prefix}.input_layernorm", layer_id=layer_id)

        # 1.3 Q、K、V投影 + Attention计算
        self.self_attn = DeepseekV3MLAAttention(fd_config=fd_config,layer_id=layer_id,prefix=f"{prefix}.self_attn",)

        # 1.4 注意力计算输出归一化
        self.post_attention_layernorm = RMSNorm(fd_config, hidden_size=fd_config.model_config.hidden_size, eps=fd_config.model_config.rms_norm_eps, prefix=f"{prefix}.post_attention_layernorm", layer_id=layer_id)

        # 1.5 MLP计算
        if fd_config.model_config.n_routed_experts is not None and layer_id >= fd_config.model_config.first_k_dense_replace:
            self.mlp = DeepSeekV3MoE(fd_config=fd_config, layer_id=layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = DeepSeekV3MLP(fd_config=fd_config, intermediate_size=fd_config.model_config.intermediate_size, prefix=f"{prefix}.mlp")
        
    # 1.2 加载参数
    def load_state_dict(self, state_dict):
        self.input_layernorm.load_state_dict(state_dict)
        self.self_attn.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)

    # 1.3 推理
    # 1.3 返回输出，残差
    # 1.3.1 这里的残差是Transformer第0层的残差，Transformer第1层的残差，Transformer第2层的残差，...，Transformer第61层的残差
    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        if hidden_states.shape[0] > 0:
            hidden_states, residual = self.input_layernorm(hidden_states, residual_input=residual, forward_meta=forward_meta)
            hidden_states = self.self_attn(forward_meta, hidden_states, position_ids, mask_encoder_batch)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
        hidden_states = self.mlp(hidden_states, forward_meta)
        return hidden_states, residual


# 1. 输入层 + Transformer + 输出层归一化
@support_graph_optimization
class DeepSeekV3Model(nn.Layer):
    # 1. 初始化
    # 1.1 有输出处理embed_tokens
    # 1.2 有61层Transfomer
    def __init__(self, fd_config: FDConfig = None,):
        # 1.1 Transformer有61层，模型名字
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "deepseek_v3"

        # 1.2 输入层 + Transformer层 + 输出层
        self.embed_tokens = VocabParallelEmbedding(fd_config,num_embeddings=fd_config.model_config.vocab_size,embedding_dim=fd_config.model_config.hidden_size,params_dtype=paddle.get_default_dtype(),prefix="deepseek_v3.embed_tokens")
        self.layers = nn.LayerList([DeepSeekV3DecoderLayer(fd_config,prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}") for i in range(self.num_layers)])
        self.norm = RMSNorm(fd_config,hidden_size=fd_config.model_config.hidden_size,eps=fd_config.model_config.rms_norm_eps,prefix="deepseek_v3.norm",)

    # 1.2 加载参数
    def load_state_dict(self, state_dict):
        self.embed_tokens.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            logger.info(f"Start load layer {i}")
            self.layers[i].load_state_dict(state_dict)

    # 1.3 推理
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        position_ids: paddle.Tensor,
        mask_encoder_batch: paddle.Tensor,
    ):
        # 1.1 输出token ids到embedding
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        # 1.2 Transformer第0, 1, 2, ..., 61层
        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.layers[i](forward_meta,hidden_states,residual,position_ids,mask_encoder_batch)
        
        # 1.3 输出层归一化
        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]
        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


# 1. 模型，词表投影，最后一行作为logits
# 1.1 模型: 输入层 + Transformer + 输出层归一化
# 1.2 对其config: "architectures", "model_type"
@ModelRegistry.register_model_class(
    architecture="DeepseekV3ForCausalLM",
    module_name="deepseek_v3",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class DeepseekV3ForCausalLM(ModelForCasualLM):
    # 1. 初始化
    def __init__(self, fd_config: FDConfig):
        # 1.1 配置
        super().__init__(fd_config)

        # 1.1 Prefill:  mask.shape          = [batch, 1]
        # 1.2 Decode:   position ids.shape  = [batch]
        self.mask_encoder_batch_buffer  = paddle.empty([fd_config.scheduler_config.max_num_batched_tokens, 1],  dtype=paddle.int32)
        self.position_ids_buffer        = paddle.empty([fd_config.scheduler_config.max_num_batched_tokens   ],  dtype=paddle.int32)
        
        # 1.3 Model
        # 1.4 词表投影；作为logits
        self.model      = DeepSeekV3Model(fd_config)
        self.lm_head    = ParallelLMHead(fd_config,     embedding_dim=fd_config.model_config.hidden_size,   num_embeddings=fd_config.model_config.vocab_size,   prefix="lm_head")

        # 1.5 ori_vocab_size就是padding前的词表大小，就是vocab_size = 129380
        # 1.5.1 只是在实际推理中可能为了对齐之后可能加了padding使得推理中的vocab_size略大于129380
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

    # 2. Default Loader加载权重方法，直接从CPU内存填进GPU显存上的模型参数
    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.
        """
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    # 3. Default Loader V1加载权重方法，使用weights_iterator从填进模型参数
    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        # 3.1 导入model_executor的工具
        from fastdeploy.model_executor.utils import (default_weight_loader, process_weights_after_loading)

        # 3.2 模型参数和.safetensors中参数映射
        # 3.2.1 第一列param_name是指在本代码中参数的key的名称
        # 3.2.2 第二列shard_name是指在.safetensors中参数的key的名称
        # 3.2.3 第三列是指该参数会和其它参数合并作为一个参数，然后该参数位于合并后的参数的前半部分，或者后半部分
        stacked_params_mapping = [
            # 3.2.1 输入层
            # 3.2.1 模型参数中为，embed_tokens.embeddings
            # 3.2.1 文件safetensors中为，model.embed_tokens.weight
            ("embed_tokens.embeddings",         "embed_tokens",                 None),

            # 3.2.2 Transformer第一次残差连接
            # 没有input_layernorm
            ("qkv_a_proj_with_mqa",             "q_a_proj",                     "q_a"),
            ("qkv_a_proj_with_mqa",             "kv_a_proj_with_mqa",           "kv_a"),
            # 没有q_a_layernorm
            # 没有q_b_proj
            # 没有kv_a_layernorm
            # 没有kv_b_proj
            # 没有o_proj

            # 3.2.3 Transformer第二次残差连接FFN
            # 没有post_attention_layernorm
            ("up_gate_proj",                    "gate_proj",                    "gate"),
            ("up_gate_proj",                    "up_proj",                      "up"),
            # 没有down_proj

            # 3.2.3 Transformer第二次残差连接MoE
            # 没有post_attention_layernorm
            # 没有gate打分
            ("experts.gate_correction_bias",    "gate.e_score_correction_bias", None),
            # 没有共享专家
            # 没有专家自己的gate, up, down

            # 3.2.4 输出层
            # 没有norm归一化
            ("lm_head.linear",                  "lm_head",                      None), 
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj", ckpt_up_proj_name="up_proj", ckpt_down_proj_name="down_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",  param_down_proj_name="experts.down_proj_"
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        for loaded_weight_name, loaded_weight in weights_iterator:
            logger.debug(f"Loading weight: {loaded_weight_name}")
            loaded_weight_name = loaded_weight_name.replace("deepseek_v3", "model")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                if "mlp.experts." in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)

                if model_param_name not in params_dict:
                    continue

                param = params_dict[model_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in loaded_weight_name:
                        continue
                    model_param_name = loaded_weight_name.replace(weight_name, param_name)
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    model_param_name = loaded_weight_name
                    if model_param_name not in params_dict:
                        continue
                    param = params_dict[model_param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                    weight_loader(param, loaded_weight)

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            if "kv_b_proj" in model_sublayer_name:
                kv_model_sublayer_name = model_sublayer_name.replace("kv_b_proj", "kv_b_proj_bmm")
                process_weights_after_loading_fn(kv_model_sublayer_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    # 4. 词表投影；作为logits
    def compute_logits(self, hidden_states: paddle.Tensor):
        """ """
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def pre_process(self, forward_meta):
        """ """
        seq_lens_encoder = forward_meta.seq_lens_encoder
        seq_lens_decoder = forward_meta.seq_lens_decoder
        seq_lens_this_time = forward_meta.seq_lens_this_time

        current_total_tokens = forward_meta.ids_remove_padding.shape[0]
        position_ids = self.position_ids_buffer[:current_total_tokens]
        mask_encoder_batch = self.mask_encoder_batch_buffer[:current_total_tokens]

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            position_ids,
            mask_encoder_batch,
        )
        return position_ids, mask_encoder_batch

    def empty_input_forward(self, forward_meta):
        """
        empty_input_forward
        """
        fake_hidden_states = paddle.empty(
            shape=[1, self.fd_config.model_config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )
        for i in range(
            self.fd_config.model_config.first_k_dense_replace,
            self.fd_config.model_config.num_hidden_layers,
        ):
            self.model.layers[i].mlp.experts(fake_hidden_states, self.model.layers[i].mlp.gate, forward_meta)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """ """
        position_ids, mask_encoder_batch = self.pre_process(forward_meta)
        hidden_states = self.model(
            ids_remove_padding=ids_remove_padding,
            forward_meta=forward_meta,
            position_ids=position_ids,
            mask_encoder_batch=mask_encoder_batch,
        )
        return hidden_states

    def clear_grpah_opt_backend(self):
        """Clear graph optimization backend, the captured cuda graph will be cleaned"""
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)

    @classmethod
    def name(cls):
        return "DeepseekV3ForCausalLM"

####################################################################################################
# 1. 不知道有什么用
class DeepSeekV3PretrainedModel(PretrainedModel):
    config_class = FDConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def arch_name(self):
        return "DeepseekV3ForCausalLM"

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):

        logger.info("DeepseekV3 inference model _get_tensor_parallel_mappings")

        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}

            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                "embed_tokens.weight": partial(fn, is_column=False),
                "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
            }

            # Self Attention Layer which are need TP.
            base_actions["layers.0.self_attn.q_b_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.kv_b_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.q_b_proj.weight_scale_inv"] = partial(fn, is_column=True)
            base_actions["layers.0.self_attn.kv_b_proj.weight_scale_inv"] = partial(fn, is_column=True)

            # MLP Layer
            base_actions["layers.0.mlp.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.down_proj.weight"] = partial(fn, is_column=False)

            # Moe Layer
            for expert_idx in range(config.n_routed_experts):
                base_actions[f"layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = partial(fn, is_column=True)
                base_actions[f"layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = partial(fn, is_column=False)

            # Shared Expert Layer
            base_actions["layers.0.mlp.shared_experts.up_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.gate_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.0.mlp.shared_experts.down_proj.weight"] = partial(fn, is_column=False)

            # MTP parts
            base_actions["layers.61.embed_tokens.weight"] = partial(fn, is_column=False)
            base_actions["layers.61.eh_proj.weight"] = partial(fn, is_column=True)
            base_actions["layers.61.shared_head.head.weight"] = partial(fn, is_column=True)

            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings
