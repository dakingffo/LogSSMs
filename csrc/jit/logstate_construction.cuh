#pragma once

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
  std::uint8_t *frame_storage;
  const std::int32_t *payload_frames;
  const std::int32_t *decay_frames;
  std::int64_t frame_bytes;
  std::int64_t payload_frames_per_slot;
  std::int64_t head_payload_bytes;
  std::int64_t decay_record_bytes;
  std::int64_t page_size;

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

struct LogStateConstructionParams : LogStateKernelParams {
  const std::int32_t *__restrict__ output_offsets;
  const std::int32_t *__restrict__ construction_output_slots;
};

struct LogStateRestoreParams : LogStateKernelParams {
  const std::int32_t *__restrict__ output_slots;
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
  if (params.frame_storage != nullptr) {
    const auto offset = (layer * params.num_heads + head) * params.head_payload_bytes;
    const auto frame = params.payload_frames[
      slot * params.payload_frames_per_slot + offset / params.frame_bytes
    ];
    const auto *head_ptr = params.frame_storage
      + frame * params.frame_bytes + offset % params.frame_bytes;
    return reinterpret_cast<const T *>(head_ptr) + token * kHeadKDim;
  }
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
  if (params.frame_storage != nullptr) {
    const auto offset = (layer * params.num_heads + head) * params.head_payload_bytes;
    const auto frame = params.payload_frames[
      slot * params.payload_frames_per_slot + offset / params.frame_bytes
    ];
    const auto *head_ptr = params.frame_storage
      + frame * params.frame_bytes + offset % params.frame_bytes;
    return reinterpret_cast<const T *>(head_ptr)
      + params.page_size * kHeadKDim + token * kHeadVDim;
  }
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
  if (params.frame_storage != nullptr) {
    const auto offset = (layer * params.num_heads + head) * params.head_payload_bytes;
    const auto frame = params.payload_frames[
      slot * params.payload_frames_per_slot + offset / params.frame_bytes
    ];
    return reinterpret_cast<T *>(
      params.frame_storage + frame * params.frame_bytes + offset % params.frame_bytes
    );
  }
  return static_cast<T *>(params.recurrent_state)
       + slot  * params.state_slot_stride
       + layer * params.state_layer_stride
       + head  * params.state_head_stride;
}

LOGSSMS_DEVICE const float *decay_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t token,
  std::int64_t page_size
) {
  if (params.frame_storage != nullptr) {
    const auto frame = params.decay_frames[slot * 2];
    const auto index = params.decay_frames[slot * 2 + 1];
    return reinterpret_cast<const float *>(
      params.frame_storage + frame * params.frame_bytes
      + index * params.decay_record_bytes
    ) + (layer * page_size + token) * params.num_heads + head;
  }
  return params.decay_log
    + slot * params.decay_slot_stride
    + layer * params.decay_layer_stride
    + token * params.decay_token_stride
    + head * params.decay_head_stride;
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE void prefetch_log_chunk(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t begin,
  std::int32_t chunk,
  std::int32_t length,
  T *shared_key,
  T *shared_value
) {
  constexpr auto kChunk = 16;
  constexpr auto kBytes = 16;
  constexpr auto kKeyBlocks = kChunk * kHeadKDim * sizeof(T) / kBytes;
  constexpr auto kValueBlocks = kChunk * kHeadVDim * sizeof(T) / kBytes;
#pragma unroll
  for (auto step = 0; step < div_ceil(kKeyBlocks, 256); ++step) {
    const auto index = step * 256 + threadIdx.x;
    if (index >= kKeyBlocks) {
      continue;
    }
    const auto token = index / (kHeadKDim * sizeof(T) / kBytes);
    const auto valid = chunk + token < length;
    const auto *source = reinterpret_cast<const std::uint8_t *>(
      key_pointer<T, kHeadKDim, kHeadVDim>(
        params, slot, layer, head, begin + chunk + (valid ? token : 0)
      )
    ) + (index % (kHeadKDim * sizeof(T) / kBytes)) * kBytes;
    const auto shared = static_cast<std::uint32_t>(__cvta_generic_to_shared(
      reinterpret_cast<std::uint8_t *>(shared_key) + index * kBytes
    ));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared), "l"(source), "r"(valid ? 16 : 0));
  }
#pragma unroll
  for (auto step = 0; step < div_ceil(kValueBlocks, 256); ++step) {
    const auto index = step * 256 + threadIdx.x;
    if (index >= kValueBlocks) {
      continue;
    }
    const auto token = index / (kHeadVDim * sizeof(T) / kBytes);
    const auto valid = chunk + token < length;
    const auto *source = reinterpret_cast<const std::uint8_t *>(
      value_pointer<T, kHeadKDim, kHeadVDim>(
        params, slot, layer, head, begin + chunk + (valid ? token : 0)
      )
    ) + (index % (kHeadVDim * sizeof(T) / kBytes)) * kBytes;
    const auto shared = static_cast<std::uint32_t>(__cvta_generic_to_shared(
      reinterpret_cast<std::uint8_t *>(shared_value) + index * kBytes
    ));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared), "l"(source), "r"(valid ? 16 : 0));
  }
  asm volatile("cp.async.commit_group;");
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
__global__ __launch_bounds__(256)
void construct_log_state_kernel(
  const __grid_constant__ LogStateConstructionParams params
) {
  using namespace nvcuda;
  constexpr auto kChunk = 16;
  constexpr auto kTile = 16;
  constexpr auto kWarps = 8;
  constexpr auto kTilesV = div_ceil(kHeadVDim, kTile);
  constexpr auto kTiles = div_ceil(kHeadKDim, kTile) * kTilesV;
  constexpr auto kTileSteps = div_ceil(kTiles, kWarps);
  using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  auto linear = blockIdx.x;
  const auto head = linear % params.num_heads;
  linear /= params.num_heads;
  const auto layer = linear % params.num_layers;
  const auto task = linear / params.num_layers;
  if (task >= params.num_tasks) {
    return;
  }

  const auto warp = threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;
  __shared__ __align__(16) T shared_key[2][kChunk * kHeadKDim];
  __shared__ __align__(16) T shared_value[2][kChunk * kHeadVDim];
  __shared__ float shared_tile[kWarps * kTile * kTile];
  __shared__ float suffix[64];
  __shared__ float gamma;

  Accumulator state[kTileSteps];
  const auto base_slot = params.base_slots[task];
  const auto *base = base_slot >= 0
    ? state_pointer<T, kHeadKDim, kHeadVDim>(params, base_slot, layer, head)
    : nullptr;
#pragma unroll
  for (auto step = 0; step < kTileSteps; ++step) {
    const auto tile = step * kWarps + warp;
    if (tile >= kTiles) {
      continue;
    }
    if (base == nullptr) {
      wmma::fill_fragment(state[step], 0.0f);
    } else {
      const auto k_begin = (tile / kTilesV) * kTile;
      const auto v_begin = (tile % kTilesV) * kTile;
      for (auto index = lane; index < kTile * kTile; index += 32) {
        const auto k = k_begin + index / kTile;
        const auto v = v_begin + index % kTile;
        shared_tile[warp * kTile * kTile + index] =
          k < kHeadKDim && v < kHeadVDim
            ? to_float(base[k * kHeadVDim + v]) : 0.0f;
      }
      __syncwarp();
      wmma::load_matrix_sync(
        state[step], shared_tile + warp * kTile * kTile,
        kTile, wmma::mem_row_major
      );
      __syncwarp();
    }
  }

  for (auto segment = params.task_log_offsets[task];
       segment < params.task_log_offsets[task + 1]; ++segment) {
    const auto slot = params.log_slots[segment];
    const auto begin = params.log_begins[segment];
    const auto end = params.log_ends[segment];
    const auto length = end - begin;
    if (threadIdx.x == 0) {
      float log_sum = 0.0f;
      for (auto token = length - 1; token >= 0; --token) {
        suffix[token] = expf(log_sum);
        log_sum += *decay_pointer(
          params, slot, layer, head, begin + token, params.page_size
        );
      }
      gamma = expf(log_sum);
    }
    __syncthreads();

#pragma unroll
    for (auto step = 0; step < kTileSteps; ++step) {
      if (step * kWarps + warp >= kTiles) {
        continue;
      }
#pragma unroll
      for (auto item = 0; item < Accumulator::num_elements; ++item) {
        state[step].x[item] *= gamma;
      }
    }

    if (length > 0) {
      prefetch_log_chunk<T, kHeadKDim, kHeadVDim>(
        params, slot, layer, head, begin, 0, length,
        shared_key[0], shared_value[0]
      );
    }
    for (auto chunk = 0; chunk < length; chunk += kChunk) {
      const auto buffer = (chunk / kChunk) % 2;
      asm volatile("cp.async.wait_group 0;");
      __syncthreads();
      if (chunk + kChunk < length) {
        prefetch_log_chunk<T, kHeadKDim, kHeadVDim>(
          params, slot, layer, head, begin, chunk + kChunk, length,
          shared_key[1 - buffer], shared_value[1 - buffer]
        );
      }
      for (auto index = threadIdx.x; index < kChunk * kHeadVDim;
           index += blockDim.x) {
        const auto token = index / kHeadVDim;
        shared_value[buffer][index] = from_float<T>(
          to_float(shared_value[buffer][index]) * suffix[chunk + token]
        );
      }
      __syncthreads();

#pragma unroll
      for (auto step = 0; step < kTileSteps; ++step) {
        const auto tile = step * kWarps + warp;
        if (tile >= kTiles) {
          continue;
        }
        const auto k_begin = (tile / kTilesV) * kTile;
        const auto v_begin = (tile % kTilesV) * kTile;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> key_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> value_fragment;
        wmma::load_matrix_sync(
          key_fragment, shared_key[buffer] + k_begin, kHeadKDim
        );
        wmma::load_matrix_sync(
          value_fragment, shared_value[buffer] + v_begin, kHeadVDim
        );
        wmma::mma_sync(
          state[step], key_fragment, value_fragment, state[step]
        );
      }
      __syncthreads();
    }

    // One CTA owns a complete layer/head. All its Log reads have finished
    // before any warp is allowed to overwrite the corresponding payload.
    for (auto output = params.output_offsets[segment];
         output < params.output_offsets[segment + 1]; ++output) {
      auto *destination = state_pointer<T, kHeadKDim, kHeadVDim>(
        params, params.construction_output_slots[output], layer, head
      );
#pragma unroll
      for (auto step = 0; step < kTileSteps; ++step) {
        const auto tile = step * kWarps + warp;
        if (tile >= kTiles) {
          continue;
        }
        const auto k_begin = (tile / kTilesV) * kTile;
        const auto v_begin = (tile % kTilesV) * kTile;
        wmma::store_matrix_sync(
          shared_tile + warp * kTile * kTile,
          state[step], kTile, wmma::mem_row_major
        );
        __syncwarp();
        for (auto index = lane; index < kTile * kTile; index += 32) {
          const auto k = k_begin + index / kTile;
          const auto v = v_begin + index % kTile;
          if (k < kHeadKDim && v < kHeadVDim) {
            destination[k * kHeadVDim + v] = from_float<T>(
              shared_tile[warp * kTile * kTile + index]
            );
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
__global__ __launch_bounds__(256)
void restore_log_state_kernel(
  const __grid_constant__ LogStateRestoreParams params
) {
  using namespace nvcuda;
  constexpr auto kTileK = 32;
  constexpr auto kTileV = 64;
  constexpr auto kTileSize = kTileK * kTileV;
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
  const auto task = linear / params.num_layers;
  if (task >= params.num_tasks) {
    return;
  }

  const auto tile_k = macro_tile / kTilesV;
  const auto tile_v = macro_tile % kTilesV;
  const auto k_begin = tile_k * kTileK;
  const auto v_begin = tile_v * kTileV;
  const auto warp = threadIdx.x / 32;
  const auto warp_k = warp / 4;
  const auto warp_v = warp % 4;

  __shared__ T shared_key[kChunk * kTileK];
  __shared__ T shared_value[kChunk * kTileV];
  __shared__ float suffix[64];
  __shared__ float gamma;
  __shared__ __align__(32) float bias[kTileSize];

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
    const auto slot = params.log_slots[segment];
    const auto begin = params.log_begins[segment];
    const auto end = params.log_ends[segment];
    const auto length = end - begin;
    if (threadIdx.x == 0) {
      float log_sum = 0.0f;
      for (auto token = length - 1; token >= 0; --token) {
        suffix[token] = expf(log_sum);
        log_sum += *decay_pointer(
          params, slot, layer, head, begin + token, params.page_size
        );
      }
      gamma = expf(log_sum);
    }
    __syncthreads();

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
      wmma::load_matrix_sync(key_fragment, shared_key + warp_k * 16, kTileK);
      wmma::load_matrix_sync(value_fragment, shared_value + warp_v * 16, kTileV);
      wmma::mma_sync(accumulator, key_fragment, value_fragment, accumulator);
      __syncthreads();
    }
    wmma::store_matrix_sync(
      bias + (warp_k * 16) * kTileV + warp_v * 16,
      accumulator, kTileV, wmma::mem_row_major
    );
    __syncthreads();
#pragma unroll
    for (auto item = 0; item < div_ceil(kTileSize, 256); ++item) {
      const auto local = threadIdx.x + item * 256;
      const auto k = k_begin + local / kTileV;
      const auto v = v_begin + local % kTileV;
      if (k < kHeadKDim && v < kHeadVDim) {
        state[item] = gamma * state[item] + bias[local];
      }
    }
    __syncthreads();
  }

  auto *destination = state_pointer<T, kHeadKDim, kHeadVDim>(
    params, params.output_slots[task], layer, head
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

} // namespace logssms
