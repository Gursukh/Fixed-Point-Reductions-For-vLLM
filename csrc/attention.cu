// Unified prefill + decode attention with paged KV.
//
// One CTA per (request, query_token_index, head). For each query
// token the CTA does two passes over the key range:
//   Pass 1: per-key qk dot product (fxp accumulator, integer sum
//           across threads); track row_max in fp32 (max is
//           order-invariant in fp32).
//   Pass 2: recompute qk, derive attention weight = exp2(qk -
//           row_max), quantise to fxp, and accumulate softmax
//           denominator + weighted V into fxp running tiles.
// At the end, divide weighted V by denominator (one fp32 div per
// output element) and store.
//
// Determinism notes:
//   * The qk dot is an integer sum of per-element products. Threads
//     reduce via __shfl_xor_sync; integer add is associative.
//   * The pv dot is also integer. attention_weights is in registers
//     (no per-row scale possible, see Migrate.md D7) so the dot
//     stays scalar.
//   * Two separate passes (max, then weighted sum) avoid the
//     re-association implied by online-softmax.
//   * softmax_scale is pre-multiplied by RCP_LN2 so qk lives in
//     log2 domain; the running max and exp2() match.

#include "fixed_point.cuh"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cmath>

namespace fxpr {

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 256;
constexpr float kRcpLn2 = 1.4426950408889634f;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

template <typename FxpInt>
__device__ FxpInt block_reduce_sum(FxpInt v, FxpInt* shmem) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  v = warp_reduce_sum<FxpInt>(v);
  if (lane == 0) shmem[warp] = v;
  __syncthreads();

  if (warp == 0) {
    FxpInt x = lane < num_warps ? shmem[lane] : FxpInt(0);
    x = warp_reduce_sum<FxpInt>(x);
    if (lane == 0) shmem[0] = x;
  }
  __syncthreads();
  return shmem[0];
}

__device__ float block_reduce_max(float v, float* shmem) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  v = warp_reduce_max(v);
  if (lane == 0) shmem[warp] = v;
  __syncthreads();

  if (warp == 0) {
    float x = lane < num_warps ? shmem[lane] : -INFINITY;
    x = warp_reduce_max(x);
    if (lane == 0) shmem[0] = x;
  }
  __syncthreads();
  return shmem[0];
}

// kv_cache layout: (num_blocks, 2, block_size, num_kv_heads, head_dim).
// Resolve a logical key position to a pointer to its head-dim row.
template <typename T>
__device__ __forceinline__ const T* paged_kv_row(
    const T* kv_base,
    const int* block_table_row,
    int key_pos,
    int page_size,
    int kv_head,
    int64_t stride_block,
    int64_t stride_slot,
    int64_t stride_head) {
  const int logical_block = key_pos / page_size;
  const int slot = key_pos % page_size;
  const int physical_block = block_table_row[logical_block];
  return kv_base
      + physical_block * stride_block
      + slot * stride_slot
      + kv_head * stride_head;
}

template <typename FxpInt, typename IOFloat>
__global__ void unified_attention_kernel(
    const IOFloat* __restrict__ Q,                  // (T, H, D)
    const IOFloat* __restrict__ K_cache,            // (B, page_size, H_kv, D)
    const IOFloat* __restrict__ V_cache,            // (B, page_size, H_kv, D)
    float*         __restrict__ O,                  // (T, H, D), fp32
    const int*   __restrict__ query_start_loc,      // (R+1,)
    const int*   __restrict__ seq_lens,             // (R,)
    const int*   __restrict__ block_table,          // (R, max_logical_blocks)
    const float* __restrict__ alibi_slopes,         // (H,) or nullptr; pre * RCP_LN2
    int   num_heads,
    int   num_kv_heads,
    int   head_dim,
    int   page_size,
    int64_t stride_q_token,
    int64_t stride_q_head,
    int64_t stride_k_block,
    int64_t stride_k_slot,
    int64_t stride_k_head,
    int64_t stride_v_block,
    int64_t stride_v_slot,
    int64_t stride_v_head,
    int64_t stride_o_token,
    int64_t stride_o_head,
    int64_t stride_block_table_row,
    float softmax_scale_log2,    // base scale, already * RCP_LN2
    float logit_softcap,         // 0 -> off
    int   window_size,           // 0 -> off
    bool  is_causal,
    int   frac_bits) {
  const int request_index = blockIdx.x;
  const int q_in_request = blockIdx.y;
  const int head_index = blockIdx.z;
  const int kv_head_index = head_index / (num_heads / num_kv_heads);

  const int q_start = query_start_loc[request_index];
  const int q_end = query_start_loc[request_index + 1];
  const int q_len = q_end - q_start;
  if (q_in_request >= q_len) return;

  const int seq_len = seq_lens[request_index];
  const int context_len = seq_len - q_len;
  const int q_token_idx = q_start + q_in_request;
  const int q_abs_pos = context_len + q_in_request;

  const int key_end =
      is_causal ? (q_abs_pos + 1) : seq_len;
  const int key_start =
      window_size > 0 ? max(0, q_abs_pos - window_size + 1) : 0;

  // --- Per-thread layout ---
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // ALiBi slope (already * RCP_LN2 by the host).
  const float alibi_slope =
      alibi_slopes != nullptr ? alibi_slopes[head_index] : 0.0f;

  const IOFloat* q_row = Q + q_token_idx * stride_q_token + head_index * stride_q_head;
  const int*     bt_row = block_table + request_index * stride_block_table_row;

  // Shared buffers (sized for max threads). Final two scalars are the
  // single-row reductions.
  __shared__ FxpInt shm_int[kMaxThreads / kWarpSize];
  __shared__ float  shm_flt[kMaxThreads / kWarpSize];
  __shared__ float  s_row_max;

  // ============================================================
  // Pass 1: scan keys, compute qk in fxp, find row_max in fp32.
  // ============================================================
  float partial_row_max = -INFINITY;
  for (int kp = key_start; kp < key_end; ++kp) {
    const IOFloat* k_row = paged_kv_row<IOFloat>(
        K_cache, bt_row, kp, page_size, kv_head_index,
        stride_k_block, stride_k_slot, stride_k_head);

    // qk dot in fxp: each thread accumulates products for its
    // strided slice of head_dim. Q / K are loaded in their native
    // dtype and widened to fp32 element-wise (single op, deterministic).
    FxpInt partial = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float qd = static_cast<float>(q_row[d]);
      const float kd = static_cast<float>(k_row[d]);
      const float prod = __fmul_rn(qd, kd);
      partial += float_to_fixed<FxpInt>(prod, frac_bits);
    }
    partial = block_reduce_sum<FxpInt>(partial, shm_int);

    // Convert qk to fp32, apply scale (log2 domain), softcap, alibi.
    if (tid == 0) {
      float qk = fixed_to_float<FxpInt>(partial, frac_bits);
      qk = qk * softmax_scale_log2;
      if (logit_softcap > 0.0f) {
        const float softcap_log2 = logit_softcap * kRcpLn2;
        qk = softcap_log2 * tanhf(qk / softcap_log2);
      }
      qk += alibi_slope * (static_cast<float>(kp) - static_cast<float>(q_abs_pos));
      // No mask check here: key_start/key_end already account for
      // causal + window. Anything outside that range is skipped.
      partial_row_max = fmaxf(partial_row_max, qk);
      shm_flt[0] = partial_row_max;  // staging for next loop iter
    }
    __syncthreads();
    // Broadcast running max to all threads via shm_flt[0] for
    // subsequent iters (only thread 0 updates it).
    partial_row_max = shm_flt[0];
  }

  if (tid == 0) {
    s_row_max = partial_row_max;
  }
  __syncthreads();
  const float row_max = s_row_max;

  // ============================================================
  // Pass 2: recompute qk, derive weights, accumulate fxp denominator
  //         and per-d weighted-V.
  // ============================================================
  // Per-thread output accumulator: each thread owns a strided slice
  // of head_dim. Use FxpInt (same dtype as qk).
  // Allocate a small array; head_dim / nthreads. With head_dim<=128
  // and nthreads=128, each thread holds at most 1 slot (head_dim=128)
  // or up to 4 if head_dim=128 and nthreads=32.
  // Keep it dynamic via a small fixed array.
  constexpr int kMaxPerThread = 8;
  FxpInt out_acc[kMaxPerThread];
  #pragma unroll
  for (int i = 0; i < kMaxPerThread; ++i) out_acc[i] = 0;

  FxpInt denom_partial = 0;

  for (int kp = key_start; kp < key_end; ++kp) {
    const IOFloat* k_row = paged_kv_row<IOFloat>(
        K_cache, bt_row, kp, page_size, kv_head_index,
        stride_k_block, stride_k_slot, stride_k_head);
    const IOFloat* v_row = paged_kv_row<IOFloat>(
        V_cache, bt_row, kp, page_size, kv_head_index,
        stride_v_block, stride_v_slot, stride_v_head);

    // Recompute qk_fxp identically to pass 1 (deterministic).
    FxpInt qk_fxp_partial = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float qd = static_cast<float>(q_row[d]);
      const float kd = static_cast<float>(k_row[d]);
      const float prod = __fmul_rn(qd, kd);
      qk_fxp_partial += float_to_fixed<FxpInt>(prod, frac_bits);
    }
    qk_fxp_partial = block_reduce_sum<FxpInt>(qk_fxp_partial, shm_int);

    // Compute attention_weight = exp2(qk - row_max) in thread 0,
    // broadcast via shared memory.
    if (tid == 0) {
      float qk = fixed_to_float<FxpInt>(qk_fxp_partial, frac_bits);
      qk = qk * softmax_scale_log2;
      if (logit_softcap > 0.0f) {
        const float softcap_log2 = logit_softcap * kRcpLn2;
        qk = softcap_log2 * tanhf(qk / softcap_log2);
      }
      qk += alibi_slope * (static_cast<float>(kp) - static_cast<float>(q_abs_pos));
      const float w = exp2f(qk - row_max);
      shm_flt[0] = w;
    }
    __syncthreads();
    const float weight = shm_flt[0];
    const FxpInt weight_fxp = float_to_fixed<FxpInt>(weight, frac_bits);

    // Denominator: each thread sees the same weight; only thread 0
    // accumulates to avoid double-counting.
    if (tid == 0) denom_partial += weight_fxp;

    // Output accumulator: each thread covers its slice of head_dim.
    int local_idx = 0;
    for (int d = tid; d < head_dim; d += nthreads) {
      const float vd = static_cast<float>(v_row[d]);
      const float prod = __fmul_rn(weight, vd);
      out_acc[local_idx] += float_to_fixed<FxpInt>(prod, frac_bits);
      ++local_idx;
    }
  }

  // Convert denominator to fp32 (only thread 0 holds it).
  __shared__ float s_denom;
  if (tid == 0) {
    const float d = fixed_to_float<FxpInt>(denom_partial, frac_bits);
    s_denom = fmaxf(d, 1.0e-6f);
  }
  __syncthreads();
  const float denom = s_denom;

  // Write output: each thread writes its slice.
  float* o_row = O + q_token_idx * stride_o_token + head_index * stride_o_head;
  int local_idx = 0;
  for (int d = tid; d < head_dim; d += nthreads) {
    const float num = fixed_to_float<FxpInt>(out_acc[local_idx], frac_bits);
    o_row[d] = num / denom;
    ++local_idx;
  }
}

template <typename FxpInt, typename IOFloat>
void launch_attention_typed(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& o,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& block_table,
    const at::Tensor* alibi_slopes,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    float softmax_scale_log2,
    float logit_softcap,
    int window_size,
    bool is_causal,
    int frac_bits,
    int max_query_len,
    int num_requests) {
  if (num_requests == 0) return;

  int threads = 32;
  while (threads < head_dim && threads < kMaxThreads) threads <<= 1;
  if (threads > kMaxThreads) threads = kMaxThreads;

  dim3 grid(num_requests, max_query_len, num_heads);

  auto stream = at::cuda::getCurrentCUDAStream();
  unified_attention_kernel<FxpInt, IOFloat><<<grid, threads, 0, stream>>>(
      q.data_ptr<IOFloat>(),
      k_cache.data_ptr<IOFloat>(),
      v_cache.data_ptr<IOFloat>(),
      o.data_ptr<float>(),
      query_start_loc.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      block_table.data_ptr<int>(),
      alibi_slopes ? alibi_slopes->data_ptr<float>() : nullptr,
      num_heads, num_kv_heads, head_dim, page_size,
      q.stride(0), q.stride(1),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
      o.stride(0), o.stride(1),
      block_table.stride(0),
      softmax_scale_log2,
      logit_softcap,
      window_size,
      is_causal,
      frac_bits);
}

template <typename FxpInt>
void launch_attention(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    at::Tensor& o,
    const at::Tensor& query_start_loc,
    const at::Tensor& seq_lens,
    const at::Tensor& block_table,
    const at::Tensor* alibi_slopes,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    float softmax_scale_log2,
    float logit_softcap,
    int window_size,
    bool is_causal,
    int frac_bits,
    int max_query_len,
    int num_requests) {
  switch (q.scalar_type()) {
    case at::kFloat:
      launch_attention_typed<FxpInt, float>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          frac_bits, max_query_len, num_requests);
      break;
    case at::kHalf:
      launch_attention_typed<FxpInt, at::Half>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          frac_bits, max_query_len, num_requests);
      break;
    case at::kBFloat16:
      launch_attention_typed<FxpInt, at::BFloat16>(
          q, k_cache, v_cache, o, query_start_loc, seq_lens, block_table,
          alibi_slopes, num_heads, num_kv_heads, head_dim, page_size,
          softmax_scale_log2, logit_softcap, window_size, is_causal,
          frac_bits, max_query_len, num_requests);
      break;
    default:
      TORCH_CHECK(false, "unified_attention_fxp: unsupported q dtype ",
                  q.scalar_type());
  }
}

}  // namespace

void unified_attention_fxp_op(
    torch::Tensor q,
    torch::Tensor kv_cache,
    torch::Tensor o,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    int64_t max_query_len,
    c10::optional<torch::Tensor> alibi_slopes,
    bool is_causal,
    c10::optional<double> softmax_scale,
    int64_t frac_bits,
    int64_t fxp_int_bits,
    double logit_softcap,
    int64_t window_size) {
  TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda() && o.is_cuda(),
              "unified_attention_fxp: all tensors must be CUDA");
  TORCH_CHECK(kv_cache.dim() == 5 && kv_cache.size(1) == 2,
              "kv_cache must be (num_blocks, 2, page_size, num_kv_heads, head_dim)");
  TORCH_CHECK(
      q.scalar_type() == at::kFloat || q.scalar_type() == at::kHalf ||
          q.scalar_type() == at::kBFloat16,
      "q dtype must be float32 / float16 / bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == q.scalar_type(),
              "kv_cache dtype must match q dtype");
  TORCH_CHECK(o.scalar_type() == at::kFloat, "o must be float32");
  TORCH_CHECK(fxp_int_bits == 16 || fxp_int_bits == 32 || fxp_int_bits == 64,
              "fxp_int_bits must be 16/32/64");

  const c10::cuda::CUDAGuard device_guard(q.device());

  // Split kv into K, V views. No .contiguous() — the kernel uses
  // explicit strides, and copying the full cache would double HBM use.
  auto k_cache = kv_cache.select(1, 0);
  auto v_cache = kv_cache.select(1, 1);

  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int num_kv_heads = k_cache.size(2);
  const int page_size = k_cache.size(1);
  const int num_requests = seq_lens.size(0);

  TORCH_CHECK(num_heads % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");

  const float ss = softmax_scale.has_value()
      ? static_cast<float>(softmax_scale.value())
      : 1.0f / sqrtf(static_cast<float>(head_dim));
  const float ss_log2 = ss * 1.4426950408889634f;

  const at::Tensor* alibi_ptr = nullptr;
  at::Tensor alibi_holder;
  if (alibi_slopes.has_value() && alibi_slopes->numel() > 0) {
    alibi_holder = alibi_slopes->contiguous();
    TORCH_CHECK(alibi_holder.scalar_type() == at::kFloat,
                "alibi_slopes must be float32");
    alibi_ptr = &alibi_holder;
  }

  switch (fxp_int_bits) {
    case 16:
      launch_attention<int16_t>(q, k_cache, v_cache, o,
                                query_start_loc, seq_lens, block_table,
                                alibi_ptr,
                                num_heads, num_kv_heads, head_dim, page_size,
                                ss_log2,
                                static_cast<float>(logit_softcap),
                                static_cast<int>(window_size),
                                is_causal,
                                static_cast<int>(frac_bits),
                                static_cast<int>(max_query_len),
                                num_requests);
      break;
    case 32:
      launch_attention<int32_t>(q, k_cache, v_cache, o,
                                query_start_loc, seq_lens, block_table,
                                alibi_ptr,
                                num_heads, num_kv_heads, head_dim, page_size,
                                ss_log2,
                                static_cast<float>(logit_softcap),
                                static_cast<int>(window_size),
                                is_causal,
                                static_cast<int>(frac_bits),
                                static_cast<int>(max_query_len),
                                num_requests);
      break;
    case 64:
      launch_attention<int64_t>(q, k_cache, v_cache, o,
                                query_start_loc, seq_lens, block_table,
                                alibi_ptr,
                                num_heads, num_kv_heads, head_dim, page_size,
                                ss_log2,
                                static_cast<float>(logit_softcap),
                                static_cast<int>(window_size),
                                is_causal,
                                static_cast<int>(frac_bits),
                                static_cast<int>(max_query_len),
                                num_requests);
      break;
  }
}

}  // namespace fxpr
