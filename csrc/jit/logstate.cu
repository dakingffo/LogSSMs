#include <concepts>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <tvm/ffi/container/tensor.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>

namespace logssms {

struct LogStateKernelParams {
  const void *key_log;
  const void *value_log;
  const float *decay_log;
  void *recurrent_state;

  const std::int32_t *__restrict__ base_slots;
  const std::int32_t *__restrict__ task_log_offsets;
  const std::int32_t *__restrict__ log_slots;
  const std::int32_t *__restrict__ log_begins;
  const std::int32_t *__restrict__ log_ends;

  std::int64_t key_slot_stride;
  std::int64_t key_layer_stride;
  std::int64_t key_token_stride;
  std::int64_t key_head_stride;
  std::int64_t value_slot_stride;
  std::int64_t value_layer_stride;
  std::int64_t value_token_stride;
  std::int64_t value_head_stride;
  std::int64_t decay_slot_stride;
  std::int64_t decay_layer_stride;
  std::int64_t decay_token_stride;
  std::int64_t decay_head_stride;
  std::int64_t state_slot_stride;
  std::int64_t state_layer_stride;
  std::int64_t state_head_stride;
  std::int64_t num_layers;
  std::int64_t num_heads;
  std::int64_t num_tasks;
};

struct LogStateStoreParams {
  void *__restrict__             key_log;
  void *__restrict__             value_log;
  float *__restrict__            decay_log;
  const std::int32_t *__restrict__ slots;
  const std::int32_t *__restrict__ token_offsets;
  const void *__restrict__       keys;
  const void *__restrict__       values;
  const float *__restrict__      decays;

  std::int64_t key_slot_stride;
  std::int64_t key_layer_stride;
  std::int64_t key_token_stride;
  std::int64_t key_head_stride;
  std::int64_t value_slot_stride;
  std::int64_t value_layer_stride;
  std::int64_t value_token_stride;
  std::int64_t value_head_stride;
  std::int64_t decay_slot_stride;
  std::int64_t decay_layer_stride;
  std::int64_t decay_token_stride;
  std::int64_t decay_head_stride;
  std::int64_t key_row_stride;
  std::int64_t key_source_head_stride;
  std::int64_t value_row_stride;
  std::int64_t value_source_head_stride;
  std::int64_t decay_row_stride;
  std::int64_t decay_source_head_stride;
  std::int64_t layer;
  std::int64_t num_heads;
  std::int64_t num_tokens;
};

struct LogStateConstructionParams : LogStateKernelParams {
  float *__restrict__ affine_gamma;
  float *__restrict__ affine_bias;
  const std::int32_t *__restrict__ output_offsets;
  const std::int32_t *__restrict__ construction_output_slots;
  std::int64_t num_segments;
  std::int64_t num_outputs;
};

template <typename T>
LOGSSMS_DEVICE float to_float(T value) {
  return static_cast<float>(value);
}

template <>
LOGSSMS_DEVICE float to_float(__half value) {
  return __half2float(value);
}

template <>
LOGSSMS_DEVICE float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
LOGSSMS_DEVICE T from_float(float value) {
  return static_cast<T>(value);
}

template <>
LOGSSMS_DEVICE __half from_float(float value) {
  return __float2half_rn(value);
}

template <>
LOGSSMS_DEVICE __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE const T *key_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t token
) {
  return static_cast<const T *>(params.key_log)
       + slot  * params.key_slot_stride
       + layer * params.key_layer_stride
       + token * params.key_token_stride
       + head  * params.key_head_stride;
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE const T *value_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t token
) {
  return static_cast<const T *>(params.value_log)
       + slot  * params.value_slot_stride
       + layer * params.value_layer_stride
       + token * params.value_token_stride
       + head  * params.value_head_stride;
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE T *state_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head
) {
  return static_cast<T *>(params.recurrent_state)
       + slot  * params.state_slot_stride
       + layer * params.state_layer_stride
       + head  * params.state_head_stride;
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
__global__ __launch_bounds__(256)
void build_page_affine_kernel(
  const __grid_constant__ LogStateConstructionParams params
) {
  using namespace nvcuda;
  constexpr auto kTileK = 32;
  constexpr auto kTileV = 64;
  constexpr auto kChunk = 16;
  constexpr auto kTilesK = div_ceil(kHeadKDim, kTileK);
  constexpr auto kTilesV = div_ceil(kHeadVDim, kTileV);
  constexpr auto kMacroTiles = kTilesK * kTilesV;

  auto linear = blockIdx.x;
  const auto macro_tile = linear % kMacroTiles;
  linear /= kMacroTiles;
  const auto head = linear % params.num_heads;
  linear /= params.num_heads;
  const auto layer = linear % params.num_layers;
  const auto segment = linear / params.num_layers;
  if (segment >= params.num_segments) {
    return;
  }

  const auto tile_k = macro_tile / kTilesV;
  const auto tile_v = macro_tile % kTilesV;
  const auto k_begin = tile_k * kTileK;
  const auto v_begin = tile_v * kTileV;
  const auto slot = params.log_slots[segment];
  const auto begin = params.log_begins[segment];
  const auto end = params.log_ends[segment];
  const auto length = end - begin;

  __shared__ T shared_key[kChunk * kTileK];
  __shared__ T shared_value[kChunk * kTileV];
  __shared__ float suffix[64];
  __shared__ float gamma;

  if (threadIdx.x == 0) {
    float product = 1.0f;
    for (auto token = length - 1; token >= 0; --token) {
      suffix[token] = product;
      product *= params.decay_log[
          slot  * params.decay_slot_stride
        + layer * params.decay_layer_stride
        + (begin + token) * params.decay_token_stride
        + head  * params.decay_head_stride
      ];
    }
    gamma = product;
    if (macro_tile == 0) {
      params.affine_gamma[
        (segment * params.num_layers + layer) * params.num_heads + head
      ] = product;
    }
  }
  __syncthreads();

  const auto warp = threadIdx.x / 32;
  const auto warp_k = warp / 4;
  const auto warp_v = warp % 4;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (auto chunk = 0; chunk < length; chunk += kChunk) {
    for (auto index = threadIdx.x; index < kChunk * kTileK; index += 256) {
      const auto token = index / kTileK;
      const auto dim = k_begin + index % kTileK;
      const auto *key = key_pointer<T, kHeadKDim, kHeadVDim>(
        params, slot, layer, head, begin + chunk + token
      );
      shared_key[index] = dim < kHeadKDim ? key[dim] : from_float<T>(0.0f);
    }
    for (auto index = threadIdx.x; index < kChunk * kTileV; index += 256) {
      const auto token = index / kTileV;
      const auto dim = v_begin + index % kTileV;
      const auto *value = value_pointer<T, kHeadKDim, kHeadVDim>(
        params, slot, layer, head, begin + chunk + token
      );
      const auto weighted = dim < kHeadVDim
        ? to_float(value[dim]) * suffix[chunk + token]
        : 0.0f;
      shared_value[index] = from_float<T>(weighted);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> key_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> value_fragment;
    wmma::load_matrix_sync(
      key_fragment, shared_key + warp_k * 16, kTileK
    );
    wmma::load_matrix_sync(
      value_fragment, shared_value + warp_v * 16, kTileV
    );
    wmma::mma_sync(accumulator, key_fragment, value_fragment, accumulator);
    __syncthreads();
  }

  const auto output = params.affine_bias
    + (((segment * params.num_layers + layer) * params.num_heads + head)
       * kHeadKDim + k_begin + warp_k * 16) * kHeadVDim
    + v_begin + warp_v * 16;
  if (k_begin + warp_k * 16 < kHeadKDim &&
      v_begin + warp_v * 16 < kHeadVDim) {
    wmma::store_matrix_sync(output, accumulator, kHeadVDim, wmma::mem_row_major);
  }
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
__global__ __launch_bounds__(256)
void scan_page_affine_kernel(
  const __grid_constant__ LogStateConstructionParams params
) {
  constexpr auto kTileK = 32;
  constexpr auto kTileV = 64;
  constexpr auto kTileSize = kTileK * kTileV;
  constexpr auto kTilesK = div_ceil(kHeadKDim, kTileK);
  constexpr auto kTilesV = div_ceil(kHeadVDim, kTileV);
  constexpr auto kMacroTiles = kTilesK * kTilesV;

  auto linear = blockIdx.x;
  const auto macro_tile = linear % kMacroTiles;
  linear /= kMacroTiles;
  const auto head = linear % params.num_heads;
  linear /= params.num_heads;
  const auto layer = linear % params.num_layers;
  const auto task = linear / params.num_layers;
  if (task >= params.num_tasks) {
    return;
  }

  const auto k_begin = (macro_tile / kTilesV) * kTileK;
  const auto v_begin = (macro_tile % kTilesV) * kTileV;
  float state[div_ceil(kTileSize, 256)];
  const auto base_slot = params.base_slots[task];
  const auto *base = base_slot >= 0
    ? state_pointer<T, kHeadKDim, kHeadVDim>(params, base_slot, layer, head)
    : nullptr;
#pragma unroll
  for (auto item = 0; item < div_ceil(kTileSize, 256); ++item) {
    const auto local = threadIdx.x + item * 256;
    const auto k = k_begin + local / kTileV;
    const auto v = v_begin + local % kTileV;
    state[item] = base_slot >= 0 && k < kHeadKDim && v < kHeadVDim
      ? to_float(base[k * kHeadVDim + v]) : 0.0f;
  }

  for (auto segment = params.task_log_offsets[task];
       segment < params.task_log_offsets[task + 1]; ++segment) {
    const auto gamma = params.affine_gamma[
      (segment * params.num_layers + layer) * params.num_heads + head
    ];
    const auto *bias = params.affine_bias
      + ((segment * params.num_layers + layer) * params.num_heads + head)
        * kHeadKDim * kHeadVDim;
#pragma unroll
    for (auto item = 0; item < div_ceil(kTileSize, 256); ++item) {
      const auto local = threadIdx.x + item * 256;
      const auto k = k_begin + local / kTileV;
      const auto v = v_begin + local % kTileV;
      if (k < kHeadKDim && v < kHeadVDim) {
        state[item] = gamma * state[item] + bias[k * kHeadVDim + v];
      }
    }
    for (auto output = params.output_offsets[segment];
         output < params.output_offsets[segment + 1]; ++output) {
      auto *destination = state_pointer<T, kHeadKDim, kHeadVDim>(
        params, params.construction_output_slots[output], layer, head
      );
#pragma unroll
      for (auto item = 0; item < div_ceil(kTileSize, 256); ++item) {
        const auto local = threadIdx.x + item * 256;
        const auto k = k_begin + local / kTileV;
        const auto v = v_begin + local % kTileV;
        if (k < kHeadKDim && v < kHeadVDim) {
          destination[k * kHeadVDim + v] = from_float<T>(state[item]);
        }
      }
    }
  }
}

template <
  typename T,
  std::size_t kHeadKDim,
  std::size_t kHeadVDim,
  std::size_t kNumThreads
>
__global__ __launch_bounds__(kNumThreads)
void store_log_state_kernel(
  const __grid_constant__ LogStateStoreParams params
) {
  const auto row = blockIdx.x;
  const auto token = row / params.num_heads;
  const auto head = row % params.num_heads;
  if (token >= params.num_tokens) {
    return;
  }

  const auto slot = params.slots[token];
  const auto token_offset = params.token_offsets[token];
  auto *key_output = static_cast<T *>(params.key_log)
      + slot * params.key_slot_stride
      + params.layer * params.key_layer_stride
      + token_offset * params.key_token_stride
      + head * params.key_head_stride;
  auto *value_output = static_cast<T *>(params.value_log)
      + slot * params.value_slot_stride
      + params.layer * params.value_layer_stride
      + token_offset * params.value_token_stride
      + head * params.value_head_stride;
  const auto *key_input = static_cast<const T *>(params.keys)
      + token * params.key_row_stride
      + head * params.key_source_head_stride;
  const auto *value_input = static_cast<const T *>(params.values)
      + token * params.value_row_stride
      + head * params.value_source_head_stride;

  for (auto index = threadIdx.x; index < kHeadKDim; index += kNumThreads) {
    key_output[index] = key_input[index];
  }
  for (auto index = threadIdx.x; index < kHeadVDim; index += kNumThreads) {
    value_output[index] = value_input[index];
  }
  if (threadIdx.x == 0) {
    params.decay_log[
        slot * params.decay_slot_stride
      + params.layer * params.decay_layer_stride
      + token_offset * params.decay_token_stride
      + head * params.decay_head_stride
    ] = params.decays[
        token * params.decay_row_stride
      + head * params.decay_source_head_stride
    ];
  }
}

template <
  std::size_t page_size,
  std::size_t head_k_dim,
  std::size_t head_v_dim,
  std::size_t num_threads = 256
>
struct LogStateKernel {
private:
  struct Views {
    LogStateKernelParams params;
    DLDataType           dtype;
    DLDevice             device;
  };

  static Views verify_views(
    const tvm::ffi::TensorView key_log,
    const tvm::ffi::TensorView value_log,
    const tvm::ffi::TensorView decay_log,
    const tvm::ffi::TensorView recurrent_state
  ) {
    using namespace host;

    auto num_slots  = SymbolicSize{"num_slots"};
    auto num_layers = SymbolicSize{"num_layers"};
    auto num_heads  = SymbolicSize{"num_heads"};
    auto dtype      = SymbolicDType{};
    auto device     = SymbolicDevice{};

    TensorMatcher({num_slots, num_layers, page_size, num_heads, head_k_dim})
        .with_strides({-1, -1, -1, -1, 1})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(key_log);
    TensorMatcher({num_slots, num_layers, page_size, num_heads, head_v_dim})
        .with_strides({-1, -1, -1, -1, 1})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(value_log);
    TensorMatcher({num_slots, num_layers, page_size, num_heads})
        .with_strides({-1, -1, -1, -1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(decay_log);
    TensorMatcher({num_slots, num_layers, num_heads, head_k_dim, head_v_dim})
        .with_strides({-1, -1, -1, head_v_dim, 1})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(recurrent_state);

    const auto value = dtype.unwrap();
    runtime_assert(
      value.lanes == 1 && (
        (value.code == kDLBfloat && value.bits == 16) ||
        (value.code == kDLFloat && (value.bits == 16 || value.bits == 32))
      ),
      "LogState kernel supports float16, bfloat16 and float32, got ", value
    );
    runtime_assert(
      key_log.data_ptr() == recurrent_state.data_ptr(),
      "The key log and recurrent state must share a head-local overlay"
    );
    runtime_assert(
      key_log.stride(0) == recurrent_state.stride(0) &&
      key_log.stride(1) == recurrent_state.stride(1) &&
      key_log.stride(3) == recurrent_state.stride(2),
      "The LogState head-local overlay has incompatible strides"
    );
    runtime_assert(
      value_log.stride(0) == key_log.stride(0) &&
      value_log.stride(1) == key_log.stride(1) &&
      value_log.stride(3) == key_log.stride(3),
      "The key and residual-value logs must use the same head-local layout"
    );
    runtime_assert(
      recurrent_state.stride(2) >= head_k_dim * head_v_dim,
      "Adjacent LogState heads overlap"
    );

    return Views{
      .params = LogStateKernelParams{
        .key_log             = key_log.data_ptr(),
        .value_log           = value_log.data_ptr(),
        .decay_log           = static_cast<const float *>(decay_log.data_ptr()),
        .recurrent_state      = recurrent_state.data_ptr(),
        .key_slot_stride      = key_log.stride(0),
        .key_layer_stride     = key_log.stride(1),
        .key_token_stride     = key_log.stride(2),
        .key_head_stride      = key_log.stride(3),
        .value_slot_stride    = value_log.stride(0),
        .value_layer_stride   = value_log.stride(1),
        .value_token_stride   = value_log.stride(2),
        .value_head_stride    = value_log.stride(3),
        .decay_slot_stride    = decay_log.stride(0),
        .decay_layer_stride   = decay_log.stride(1),
        .decay_token_stride   = decay_log.stride(2),
        .decay_head_stride    = decay_log.stride(3),
        .state_slot_stride    = recurrent_state.stride(0),
        .state_layer_stride   = recurrent_state.stride(1),
        .state_head_stride    = recurrent_state.stride(2),
        .num_layers           = num_layers.unwrap(),
        .num_heads            = num_heads.unwrap(),
      },
      .dtype  = value,
      .device = device.unwrap(),
    };
  }

  static void verify_int32_vector(
    const tvm::ffi::TensorView tensor,
    DLDevice device
  ) {
    using namespace host;
    TensorMatcher({-1})
        .with_dtype<std::int32_t>()
        .with_device(device)
        .verify(tensor);
  }

public:
  static void store(
    const tvm::ffi::TensorView key_log,
    const tvm::ffi::TensorView value_log,
    const tvm::ffi::TensorView decay_log,
    const tvm::ffi::TensorView recurrent_state,
    const tvm::ffi::TensorView slots,
    const tvm::ffi::TensorView token_offsets,
    const tvm::ffi::TensorView keys,
    const tvm::ffi::TensorView values,
    const tvm::ffi::TensorView decays,
    std::int64_t layer
  ) {
    using namespace host;
    auto views = verify_views(
      key_log, value_log, decay_log, recurrent_state
    );
    verify_int32_vector(slots, views.device);
    verify_int32_vector(token_offsets, views.device);

    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_heads  = SymbolicSize{"num_heads"};
    auto dtype      = SymbolicDType{};
    TensorMatcher({num_tokens, num_heads, head_k_dim})
        .with_strides({-1, -1, 1})
        .with_dtype(dtype)
        .with_device(views.device)
        .verify(keys);
    TensorMatcher({num_tokens, num_heads, head_v_dim})
        .with_strides({-1, -1, 1})
        .with_dtype(dtype)
        .with_device(views.device)
        .verify(values);
    TensorMatcher({num_tokens, num_heads})
        .with_strides({-1, 1})
        .with_dtype<float>()
        .with_device(views.device)
        .verify(decays);
    runtime_assert(dtype.unwrap() == views.dtype);
    runtime_assert(slots.size(0) == num_tokens.unwrap());
    runtime_assert(token_offsets.size(0) == num_tokens.unwrap());
    runtime_assert(layer >= 0 && layer < views.params.num_layers);
    runtime_assert(num_heads.unwrap() == views.params.num_heads);

    const auto params = LogStateStoreParams{
      .key_log                 = key_log.data_ptr(),
      .value_log               = value_log.data_ptr(),
      .decay_log               = static_cast<float *>(decay_log.data_ptr()),
      .slots                   = static_cast<const std::int32_t *>(slots.data_ptr()),
      .token_offsets           = static_cast<const std::int32_t *>(token_offsets.data_ptr()),
      .keys                    = keys.data_ptr(),
      .values                  = values.data_ptr(),
      .decays                  = static_cast<const float *>(decays.data_ptr()),
      .key_slot_stride         = key_log.stride(0),
      .key_layer_stride        = key_log.stride(1),
      .key_token_stride        = key_log.stride(2),
      .key_head_stride         = key_log.stride(3),
      .value_slot_stride       = value_log.stride(0),
      .value_layer_stride      = value_log.stride(1),
      .value_token_stride      = value_log.stride(2),
      .value_head_stride       = value_log.stride(3),
      .decay_slot_stride       = decay_log.stride(0),
      .decay_layer_stride      = decay_log.stride(1),
      .decay_token_stride      = decay_log.stride(2),
      .decay_head_stride       = decay_log.stride(3),
      .key_row_stride          = keys.stride(0),
      .key_source_head_stride  = keys.stride(1),
      .value_row_stride        = values.stride(0),
      .value_source_head_stride= values.stride(1),
      .decay_row_stride        = decays.stride(0),
      .decay_source_head_stride= decays.stride(1),
      .layer                   = layer,
      .num_heads               = num_heads.unwrap(),
      .num_tokens              = num_tokens.unwrap(),
    };
    if (params.num_tokens == 0) {
      return;
    }
    const auto blocks = params.num_tokens * params.num_heads;
    if (views.dtype.code == kDLBfloat) {
      host::LaunchKernel(blocks, num_threads, views.device)(
        store_log_state_kernel<__nv_bfloat16, head_k_dim,
                               head_v_dim, num_threads>, params
      );
    }
    else if (views.dtype.bits == 16) {
      host::LaunchKernel(blocks, num_threads, views.device)(
        store_log_state_kernel<__half, head_k_dim,
                               head_v_dim, num_threads>, params
      );
    }
    else {
      host::LaunchKernel(blocks, num_threads, views.device)(
        store_log_state_kernel<float, head_k_dim,
                               head_v_dim, num_threads>, params
      );
    }
  }

  static void construct(
    const tvm::ffi::TensorView key_log,
    const tvm::ffi::TensorView value_log,
    const tvm::ffi::TensorView decay_log,
    const tvm::ffi::TensorView recurrent_state,
    const tvm::ffi::TensorView base_slots,
    const tvm::ffi::TensorView task_segment_offsets,
    const tvm::ffi::TensorView log_slots,
    const tvm::ffi::TensorView log_begins,
    const tvm::ffi::TensorView log_ends,
    const tvm::ffi::TensorView output_offsets,
    const tvm::ffi::TensorView output_slots,
    const tvm::ffi::TensorView affine_gamma,
    const tvm::ffi::TensorView affine_bias
  ) {
    using namespace host;
    auto views = verify_views(key_log, value_log, decay_log, recurrent_state);
    for (const auto tensor : {
           base_slots, task_segment_offsets, log_slots, log_begins,
           log_ends, output_offsets, output_slots
         }) {
      verify_int32_vector(tensor, views.device);
    }
    const auto num_tasks = base_slots.size(0);
    const auto num_segments = log_slots.size(0);
    runtime_assert(task_segment_offsets.size(0) == num_tasks + 1);
    runtime_assert(log_begins.size(0) == num_segments);
    runtime_assert(log_ends.size(0) == num_segments);
    runtime_assert(output_offsets.size(0) == num_segments + 1);
    runtime_assert(head_k_dim % 16 == 0 && head_v_dim % 16 == 0);
    runtime_assert(page_size <= 64);
    TensorMatcher({num_segments, views.params.num_layers, views.params.num_heads})
        .with_strides({-1, -1, 1})
        .with_dtype<float>()
        .with_device(views.device)
        .verify(affine_gamma);
    TensorMatcher({num_segments, views.params.num_layers, views.params.num_heads,
                   head_k_dim, head_v_dim})
        .with_strides({-1, -1, -1, head_v_dim, 1})
        .with_dtype<float>()
        .with_device(views.device)
        .verify(affine_bias);

    LogStateConstructionParams params{};
    static_cast<LogStateKernelParams&>(params) = views.params;
    params.base_slots = static_cast<const std::int32_t *>(base_slots.data_ptr());
    params.task_log_offsets = static_cast<const std::int32_t *>(
      task_segment_offsets.data_ptr()
    );
    params.log_slots = static_cast<const std::int32_t *>(log_slots.data_ptr());
    params.log_begins = static_cast<const std::int32_t *>(log_begins.data_ptr());
    params.log_ends = static_cast<const std::int32_t *>(log_ends.data_ptr());
    params.output_offsets = static_cast<const std::int32_t *>(
      output_offsets.data_ptr()
    );
    params.construction_output_slots = static_cast<const std::int32_t *>(
      output_slots.data_ptr()
    );
    params.affine_gamma = static_cast<float *>(affine_gamma.data_ptr());
    params.affine_bias = static_cast<float *>(affine_bias.data_ptr());
    params.num_tasks = num_tasks;
    params.num_segments = num_segments;
    params.num_outputs = output_slots.size(0);
    if (num_tasks == 0 || num_segments == 0) {
      return;
    }

    constexpr auto macro_tiles = div_ceil(head_k_dim, std::size_t{32})
                               * div_ceil(head_v_dim, std::size_t{64});
    const auto affine_blocks = num_segments * params.num_layers
                             * params.num_heads * macro_tiles;
    const auto scan_blocks = num_tasks * params.num_layers
                           * params.num_heads * macro_tiles;
    if (views.dtype.code == kDLBfloat) {
      host::LaunchKernel(affine_blocks, 256, views.device)(
        build_page_affine_kernel<__nv_bfloat16, head_k_dim, head_v_dim>, params
      );
      host::LaunchKernel(scan_blocks, 256, views.device)(
        scan_page_affine_kernel<__nv_bfloat16, head_k_dim, head_v_dim>, params
      );
    }
    else if (views.dtype.bits == 16) {
      host::LaunchKernel(affine_blocks, 256, views.device)(
        build_page_affine_kernel<__half, head_k_dim, head_v_dim>, params
      );
      host::LaunchKernel(scan_blocks, 256, views.device)(
        scan_page_affine_kernel<__half, head_k_dim, head_v_dim>, params
      );
    }
    else {
      runtime_assert(false, "Tensor-Core LogState construction requires a 16-bit dtype");
    }
  }
};

} // namespace logssms
