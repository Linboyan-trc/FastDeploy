// 1. 需要ATTENTION_HOPPER_KERNEL_TRAITS_CUH_宏进入该代码
#ifndef ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
#define ATTENTION_HOPPER_KERNEL_TRAITS_CUH_

// 2. 头文件
#include <type_traits>
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

// 3. 名字空间
namespace mla_attn {
using namespace cute;

// 4. shared memory 内存布局描述
// 4.1 假设Q.shape = [2, 100, 128, 192]
// 4.1 假设K.shape = [2, 100, 128, 192]
// 4.1 假设V.shape = [2, 100, 128, 128]
// 4.2 DTypeQ           = fp16
// 4.2 DTypeKV          = fp16
// 4.2 DTypeQKAccum     = float
// 4.2 DTypeOut         = fp16
// 4.2 IdType           = int32_t
// 4.2 BLOCK_SHAPE_KV   = 64，每次加载64个token的K，V
// 4.2 SmemLayoutQ      = Layout<Shape<64, 192>>，一个Q的tile.shape = [64, 192]
// 4.2 SmemLayoutK      = Layout<Shape<64, 192>>，一个K的tile.shape = [64, 192]
// 4.2 SmemLayoutP      = Layout<Shape<64, 64>>，一个P的tile.shape = [64, 64]，这个形状来自Q.tile * K.tile^T = [64, 64]
// 4.2 SmemLayoutRow    = Layout<Shape<64>>，sofamax用
// 4.2 SmemLayoutO      = Layout<Shape<64, 128>>，一个O的tile.shape = [64, 128]，这个形状来自P.tile * V.tile = [64, 128]
// 4.3 这里Q.shape = [2, 100, 128, 192]，那么Q.tile读的是第一个batch的前64个token的第一个head的192个元素，也就是[1, 64, 1, 192]
// 4.3 这里Q.shape = [2, 100, 128, 192]，那么Q.tile读的是第一个batch的前64个token的第一个head的192个元素，也就是[1, 64, 2, 192]
// ....................................................................................................................
// 4.3 这里Q.shape = [2, 100, 128, 192]，那么Q.tile读的是第一个batch的前64个token的第一个head的192个元素，也就是[1, 64, 128, 192]
// 4.2 也就是对于第一个batch的前64个token，需要128个Q.tile去读，因为有128个头，每个头192个元素
// 4.2 也就是对于第一个batch的后36个token，需要128个Q.tile去读，因为有128个头，每个头192个元素，加padding
// 4.2 因此对于一共两个batch，需要128*4 = 512个Q.tile才能读完Q[2, 100, 128, 192]
template <
    typename MainloopPipeline,
    typename MainloopPipelineQ,
    
    class DTypeQ,
    class DTypeKV,
    class DTypeQKAccum,
    class DTypeOut,
    class IdType,
    
    int BLOCK_SHAPE_KV,
    class SmemLayoutQ,
    class SmemLayoutK,
    class SmemLayoutP,
    class SmemLayoutRow,
    class SmemLayoutO
>
struct alignas(16) SharedStorageQKVO {
    // 4.3 smem_q[64 * 192]，也就是一个Q.tile是第一个batch前64个token的第一个头，然后依次内容替换成第二个头，...，第一百二十八个头
    // 4.3 smem_p[64 * 64]
    // 4.3 smem_scale[64]
    // 4.3 smem_kv[64 * 192]
    // 4.3 smem_o[64 * 128]
    alignas(16) cute::array_aligned<DTypeQ,       cute::cosize_v<SmemLayoutQ>>    smem_q;
    alignas(16) cute::array_aligned<DTypeQ,       cute::cosize_v<SmemLayoutP>>    smem_p;
    alignas(16) cute::array_aligned<DTypeQKAccum, cute::cosize_v<SmemLayoutRow>>  smem_scale;
    union {
        alignas(16) cute::array_aligned<DTypeKV,    cute::cosize_v<SmemLayoutK>> smem_kv;
        alignas(16) cute::array_aligned<DTypeOut,   cute::cosize_v<SmemLayoutO>> smem_o;
    };

    // 4.4 双缓存，计算Q.tile[0]的时候，加载Q.tile[1]在另一块shared memory上
    struct {
        alignas(16) typename MainloopPipelineQ::SharedStorage pipeline_q;
        alignas(16) typename MainloopPipeline::SharedStorage pipeline_kv;
    };
};

// 5. attention计算配置设定
// 5.1 假设Q = [2, 100, 128, 192]
// 5.1 假设K = [2, 100, 128, 192]
// 5.1 假设V = [2, 100, 128, 128]
// 5.2 USE_TMA_LOAD_KV_ = true，使用张量内存加速器
// 5.2 HEAD_DIM_QK      = 192，Q,K的head_dim是192
// 5.2 HEAD_DIM_VO      = 128，V,O的head_dim是128
// 5.2 GROUP_SIZE_      = 1，一个tile处理1个 head
// 5.2 BLOCK_SHAPE_Q_   = 64，64个token
// 5.2 BLOCK_SHAPE_KV_  = 64，64个token
// 5.2 NUM_STAGES_      = 2，双缓冲
// 5.3 DTypeQ_      = fp16
// 5.3 DTypeKV_     = fp16
// 5.3 DTypeO_      = fp16
// 5.3 IdType_      = int32_t
// 5.3 NV_TYPE_     = uint32_t，没什么用，只是为了对齐接口保持参数数量统一
template <
    bool USE_TMA_LOAD_KV_,
    
    int HEAD_DIM_QK_,
    int HEAD_DIM_VO_,
    int GROUP_SIZE_,
    int BLOCK_SHAPE_Q_,
    int BLOCK_SHAPE_KV_,
    int NUM_STAGES_,
    
    typename DTypeQ_,
    typename DTypeKV_,
    typename DTypeO_,
    typename IdType_,
    typename NV_TYPE_
>
struct AttentionKernelTraits {
    // 5.1 DTypeQ       = fp16
    // 5.1 DTypeKV      = fp16
    // 5.1 DTypeO       = fp16
    // 5.1 IdType       = int32_t
    // 5.1 DTypeQKAccum = float
    // 5.1 DTypePVAccum = float
    // 5.1 NV_TYPE      = uint32_t
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;
    using DTypeQKAccum = float;
    using DTypePVAccum = float;
    using NV_TYPE = NV_TYPE_;

    // 5.2 USE_TMA_LOAD_KV  = true
    // 5.2 GROUP_SIZE       = 1
    // 5.2 BLOCK_SHAPE_Q    = 64，必须对齐64
    static constexpr bool USE_TMA_LOAD_KV = USE_TMA_LOAD_KV_;
    static constexpr int GROUP_SIZE = GROUP_SIZE_;
    static constexpr int BLOCK_SHAPE_Q = BLOCK_SHAPE_Q_;
    static_assert(BLOCK_SHAPE_Q % 64 == 0, "BLOCK_SHAPE_Q must be a multiple of 64");

    // 5.2 BLOCK_SHAPE_KV  = 64
    // 5.2 HEAD_DIM_QK     = 192，必须对齐32
    // 5.2 HEAD_DIM_VO     = 128，必须对齐32
    // 5.2 NUM_PER_STAGE   = 64 * 192，就是一次stage要处理的元素总数量，就是一个Q.tile里的元素数量
    static constexpr int BLOCK_SHAPE_KV = BLOCK_SHAPE_KV_;
    static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
    static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
    static constexpr int NUM_PER_STAGE = BLOCK_SHAPE_KV * HEAD_DIM_QK;
    static_assert(HEAD_DIM_QK % 32 == 0, "HEAD_DIM_QK must be a multiple of 32");
    static_assert(HEAD_DIM_VO % 32 == 0, "HEAD_DIM_VO must be a multiple of 32");
    
    // 5.3 一个blcok里有12个warp
    // 5.3 一个blcok里有384个thread
    // 5.3 384个thread中，128个用于从HMB到shared memory搬数据，叫做Producer
    // 5.3 384个thread中，256个用于利用shared memory中的数据做GMMA，矩阵乘加，叫做Comsumer
    static constexpr int NUM_WARPS = 12;
    static constexpr int NUM_THREADS = 384;
    static constexpr int NUM_PRODUCER_THREADS = 128;

    // 5.4 TileShape_QKD = [64, 64, 192]
    // 5.4 TileShape_PDV = [64, 128, 64]
    // 5.4.1 using就是typedef, 用TileShape_QKD代替Shape<..., ..., ...>
    using TileShape_QKD = Shape<Int<BLOCK_SHAPE_Q>, Int<BLOCK_SHAPE_KV>, Int<HEAD_DIM_QK>>;
    using TileShape_PDV = Shape<Int<BLOCK_SHAPE_Q>, Int<HEAD_DIM_VO>, Int<BLOCK_SHAPE_KV>>;

    // 5.5 NUM_STAGES = 2
    static constexpr int NUM_STAGES = NUM_STAGES_;

    // 5.6 AtomLayoutQKD = [1, 1, 1]
    // 5.6 AtomLayoutQKD = [1, 2, 1]
    using AtomLayoutQKD = Layout<Shape<Int<BLOCK_SHAPE_Q / 64>, _1, _1>>;
    using AtomLayoutPV = Layout<Shape<Int<BLOCK_SHAPE_Q / 64>, _2, _1>>;
    using TiledMmaQK = decltype(
        cute::make_tiled_mma(
            cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(),
            AtomLayoutQKD{}
        )
    );
    using TiledMmaPV = decltype(
        cute::make_tiled_mma(
            cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/DTypePVAccum, TileShape_PDV, GMMA::Major::K, GMMA::Major::MN>(), AtomLayoutPV{}
        )
    );
    using TiledMmaPVSS = decltype(
        cute::make_tiled_mma(
            cute::GMMA::ss_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/DTypePVAccum, TileShape_PDV, GMMA::Major::K, GMMA::Major::MN>(), AtomLayoutPV{}
        )
    );

    static constexpr int NUM_MMA_THREADS = size(TiledMmaPV{});
    using SmemLayoutAtomQ = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})), decltype(cute::get<2>(TileShape_QKD{}))>()
    );
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_QKD{})));
    using SmemLayoutAtomK = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})), decltype(cute::get<2>(TileShape_QKD{}))>()
    );
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));
    using SmemLayoutVt = decltype(
        composition( 
            SmemLayoutK{},
            make_ordered_layout(
                make_shape(get<2>(TileShape_QKD{}),
                get<1>(TileShape_QKD{}),
                Int<NUM_STAGES>{}),
                Step<_2, _1, _3>{}
            )
        )
    );
    using SmemLayoutAtomV = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})), decltype(cute::get<1>(TileShape_PDV{}))>()
    );
    using SmemLayoutV = decltype(
        tile_to_shape(
            SmemLayoutAtomV{},
            make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<1>{}))
    );

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutVtOneStage = decltype(composition(
        SmemLayoutV{},
        make_ordered_layout(
            make_shape(
                get<1>(TileShape_PDV{}), get<2>(TileShape_PDV{}), Int<1>{}),
            Step<_2, _1, _3>{})));

    using SmemLayoutAtomO =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                GMMA::Major::K,
                DTypeO,
                decltype(cute::get<0>(TileShape_PDV{})),
                decltype(cute::get<1>(TileShape_PDV{}))>());
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));

    using SmemCopyAtom = Copy_Atom<cute::SM90_U32x4_STSM_N, DTypeQ>;

    static constexpr bool IS_CTA_32 = (BLOCK_SHAPE_KV == 32);
    using SmemLayoutRowOneStage = Layout<Shape<_2, Int<128>>, Stride<_1, _2>>;
    using SmemLayoutRowTwoStage = Layout<Shape<_2, Int<128>, _2>, Stride<_1, _2, _256>>;
    using SmemLayoutRow = std::conditional_t<IS_CTA_32, SmemLayoutRowTwoStage, SmemLayoutRowOneStage>;

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                GMMA::Major::K,
                DTypeQ,
                decltype(cute::get<0>(TileShape_QKD{})),
                decltype(cute::get<1>(TileShape_QKD{}))>());
    using SmemLayoutPSSOneStage = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_QKD{})));
    using SmemLayoutPSSTwoStage = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        make_shape(Int<BLOCK_SHAPE_Q>{}, Int<BLOCK_SHAPE_KV>{}, Int<2>{})));
    using SmemLayoutP = std::conditional_t<IS_CTA_32, SmemLayoutPSSTwoStage, SmemLayoutPSSOneStage>;

    using MainloopPipelineQ = typename cutlass::PipelineAsync<1>;
    using PipelineStateQ = typename cutlass::PipelineState<1>;
    using MainloopPipeline = std::conditional_t<USE_TMA_LOAD_KV,
                            typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                            typename cutlass::PipelineAsync<NUM_STAGES>>;
    using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

    using SharedStorage = SharedStorageQKVO<
        MainloopPipeline,
        MainloopPipelineQ,
        DTypeQ,
        DTypeKV,
        DTypeQKAccum,
        DTypeO,
        IdType,
        BLOCK_SHAPE_KV,
        SmemLayoutQ,
        SmemLayoutK,
        SmemLayoutP,
        SmemLayoutRow,
        SmemLayoutO
    >;
};

}  // namespace mla_attn

#endif
