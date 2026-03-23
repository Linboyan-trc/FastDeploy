#pragma once
#include "paddle/extension.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/allocator.h"
#include "append_attn/utils.cuh"

template <typename T>
void BatchMLAWithPagedKVCacheKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& q,  // [token_num, q_head_num, head_dim]
    const paddle::Tensor&
        latent_cache,  // [max_block_num, q_head_num, block_size, head_dim]
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>&
        cache_k_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_scale,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_k_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        cache_v_zp,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        shift_bias,  // [num_kv_heads, head_dim]
    const paddle::optional<paddle::Tensor>&
        smooth_weight,  // [num_kv_heads, head_dim]
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids_per_batch,
    const paddle::Tensor& num_blocks_x_device,
    const std::string& cache_quant_type_str,
    const paddle::Tensor& decoder_chunk_size_device,
    const int max_seq_len,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float in_scale,
    const int draft_token_num,
    const bool causal,
    cudaStream_t& stream,
    paddle::Tensor* out);
