#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <tvm/ffi/container/tensor.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>

namespace logssms {

struct LogStateKernelParams {
  std::uint8_t *frame_storage;
  const std::int32_t *payload_frames;
  const std::int32_t *decay_frames;
  std::int64_t frame_bytes;
  std::int64_t payload_frames_per_slot;
  std::int64_t head_payload_bytes;
  std::int64_t decay_record_bytes;
  std::int64_t page_size;
  std::int64_t granularity;

  const std::int32_t *__restrict__ base_slots;
  const std::int32_t *__restrict__ task_log_offsets;
  const std::int32_t *__restrict__ log_slots;
  const std::int32_t *__restrict__ log_begins;
  const std::int32_t *__restrict__ log_ends;

  std::int64_t num_layers;
  std::int64_t num_heads;
  std::int64_t num_key_heads;
  std::int64_t num_tasks;
};

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE T *payload_pointer(
  const LogStateKernelParams& params, std::int32_t slot,
  std::int64_t layer, std::int64_t head, std::int64_t element
) {
  const auto bytes = (layer * params.num_heads + head) * params.head_payload_bytes;
  const auto frame = params.payload_frames[
    slot * params.payload_frames_per_slot + bytes / params.frame_bytes
  ];
  return reinterpret_cast<T *>(
    params.frame_storage + frame * params.frame_bytes + bytes % params.frame_bytes
  ) + element;
}

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
  std::int32_t token,
  std::int64_t column = 0
) {
  const auto group = head / (params.num_heads / params.num_key_heads);
  const auto layer_size = params.num_key_heads == params.num_heads
    ? params.num_heads * params.head_payload_bytes / sizeof(T) / (params.page_size / params.granularity)
    : params.granularity * (params.num_key_heads * kHeadKDim + params.num_heads * kHeadVDim);
  const auto offset = (token / params.granularity * params.num_layers + layer) * layer_size
    + (group * params.granularity + token % params.granularity) * kHeadKDim + column;
  const auto bytes = offset * sizeof(T);
  const auto frame = params.payload_frames[
    slot * params.payload_frames_per_slot + bytes / params.frame_bytes
  ];
  return reinterpret_cast<const T *>(params.frame_storage + frame * params.frame_bytes + bytes % params.frame_bytes);
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE const T *value_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t token,
  std::int64_t column = 0
) {
  const auto key_size = params.granularity * params.num_key_heads * kHeadKDim;
  const auto layer_size = params.num_key_heads == params.num_heads
    ? params.num_heads * params.head_payload_bytes / sizeof(T) / (params.page_size / params.granularity)
    : key_size + params.granularity * params.num_heads * kHeadVDim;
  const auto offset = (token / params.granularity * params.num_layers + layer) * layer_size
    + key_size + (head * params.granularity + token % params.granularity) * kHeadVDim + column;
  const auto bytes = offset * sizeof(T);
  const auto frame = params.payload_frames[
    slot * params.payload_frames_per_slot + bytes / params.frame_bytes
  ];
  return reinterpret_cast<const T *>(params.frame_storage + frame * params.frame_bytes + bytes % params.frame_bytes);
}

template <typename T>
struct LogChunk {
  const T *key;
  const T *value;
  bool contiguous;
};

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE LogChunk<T> resolve_log_chunk(
  const LogStateKernelParams& params, std::int32_t slot,
  std::int64_t layer, std::int64_t head, std::int32_t begin, std::int32_t end
) {
  const auto group = head / (params.num_heads / params.num_key_heads);
  const auto key_size = params.granularity * params.num_key_heads * kHeadKDim;
  const auto layer_size = params.num_key_heads == params.num_heads
    ? params.num_heads * params.head_payload_bytes / sizeof(T) / (params.page_size / params.granularity)
    : key_size + params.granularity * params.num_heads * kHeadVDim;
  const auto base = (begin / params.granularity * params.num_layers + layer) * layer_size;
  const auto key_bytes = (base + (group * params.granularity + begin % params.granularity) * kHeadKDim) * sizeof(T);
  const auto value_bytes = (base + key_size + (head * params.granularity + begin % params.granularity) * kHeadVDim) * sizeof(T);
  const auto key_frame = params.payload_frames[
    slot * params.payload_frames_per_slot + key_bytes / params.frame_bytes
  ];
  const auto value_frame = params.payload_frames[
    slot * params.payload_frames_per_slot + value_bytes / params.frame_bytes
  ];
  const auto contiguous = begin / params.granularity == (end - 1) / params.granularity
    && key_bytes % params.frame_bytes + (end - begin) * kHeadKDim * sizeof(T) <= params.frame_bytes
    && value_bytes % params.frame_bytes + (end - begin) * kHeadVDim * sizeof(T) <= params.frame_bytes;
  return {
    reinterpret_cast<const T *>(
      params.frame_storage + key_frame * params.frame_bytes + key_bytes % params.frame_bytes
    ),
    reinterpret_cast<const T *>(
      params.frame_storage + value_frame * params.frame_bytes + value_bytes % params.frame_bytes
    ),
    contiguous,
  };
}

template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
LOGSSMS_DEVICE T *state_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int64_t element = 0
) {
  return payload_pointer<T, kHeadKDim, kHeadVDim>(params, slot, layer, head, element);
}

LOGSSMS_DEVICE const float *decay_pointer(
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int64_t layer,
  std::int64_t head,
  std::int32_t token,
  std::int64_t page_size
) {
  const auto frame = params.decay_frames[slot * 2];
  const auto index = params.decay_frames[slot * 2 + 1];
  return reinterpret_cast<const float *>(
    params.frame_storage + frame * params.frame_bytes
    + index * params.decay_record_bytes
  ) + (layer * page_size + token) * params.num_heads + head;
}


template <typename T, int kHeadKDim, int kHeadVDim, int kTileK, int kTileV, int kChunk>
__device__ void prefetch_log_chunk(
  const LogStateKernelParams& params, int slot, int layer, int head,
  int begin, int chunk, int length, int k_begin, int v_begin, T *keys, T *values
) {
  constexpr int kVectorElements = 16 / sizeof(T);
  const auto pointers = resolve_log_chunk<T, kHeadKDim, kHeadVDim>(
    params, slot, layer, head, begin + chunk, begin + min(length, chunk + kChunk)
  );
  for (int index = threadIdx.x; index < kChunk * kTileK / kVectorElements; index += blockDim.x) {
    const int token = index / (kTileK / kVectorElements), column = k_begin + index % (kTileK / kVectorElements) * kVectorElements;
    const bool valid = chunk + token < length && column + kVectorElements <= kHeadKDim;
    const auto *source = pointers.contiguous
      ? pointers.key + (valid ? token : 0) * kHeadKDim + (column < kHeadKDim ? column : 0)
      : key_pointer<T, kHeadKDim, kHeadVDim>(params, slot, layer, head,
          begin + chunk + (valid ? token : 0), column < kHeadKDim ? column : 0);
    const auto destination = static_cast<unsigned>(__cvta_generic_to_shared(keys + index * kVectorElements));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(destination), "l"(source), "r"(valid ? 16 : 0) : "memory");
  }
  for (int index = threadIdx.x; index < kChunk * kTileV / kVectorElements; index += blockDim.x) {
    const int token = index / (kTileV / kVectorElements), column = v_begin + index % (kTileV / kVectorElements) * kVectorElements;
    const bool valid = chunk + token < length && column + kVectorElements <= kHeadVDim;
    const auto *source = pointers.contiguous
      ? pointers.value + (valid ? token : 0) * kHeadVDim + (column < kHeadVDim ? column : 0)
      : value_pointer<T, kHeadKDim, kHeadVDim>(params, slot, layer, head,
          begin + chunk + (valid ? token : 0), column < kHeadVDim ? column : 0);
    const auto destination = static_cast<unsigned>(__cvta_generic_to_shared(values + index * kVectorElements));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
      :: "r"(destination), "l"(source), "r"(valid ? 16 : 0) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <
  typename T, int kHeadKDim, int kHeadVDim, int kTileK, int kTileV,
  int kWarps, int kTileSteps, typename Params
>
__device__ void write_state_tiles(
  const Params& params, int slot, int layer, int head, int k_begin, int v_begin,
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> (&state)[kTileSteps],
  float *tiles
) {
  constexpr int tiles_v = kTileV / 16, tile_count = kTileK / 16 * tiles_v;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  auto *destination = state_pointer<T, kHeadKDim, kHeadVDim>(params, slot, layer, head);
#pragma unroll
  for (int step = 0; step < kTileSteps; ++step) {
    const int tile = step * kWarps + warp;
    if (tile >= tile_count) {
      continue;
    }
    const int tile_k = k_begin + tile / tiles_v * 16;
    const int tile_v = v_begin + tile % tiles_v * 16;
    nvcuda::wmma::store_matrix_sync(tiles + warp * 256, state[step],
      16, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < 256; index += 32) {
      const int k = tile_k + index / 16, v = tile_v + index % 16;
      if (k < kHeadKDim && v < kHeadVDim) {
        destination[k * kHeadVDim + v] = from_float<T>(tiles[warp * 256 + index]);
      }
    }
    __syncwarp();
  }
}

template <bool kConstruct, typename T, int kHeadKDim, int kHeadVDim>
__global__ __launch_bounds__((kConstruct ? 8 : 4) * 32)
void log_state_kernel(
  const __grid_constant__ std::conditional_t<kConstruct,
    LogStateConstructionParams, LogStateRestoreParams> params
) {
  using namespace nvcuda;
  constexpr int kWarps = kConstruct ? 8 : 4;
  constexpr int kTileK = kConstruct ? kHeadKDim : 64;
  constexpr int kTileV = kConstruct ? kHeadVDim : 64;
  constexpr auto kPrefetchBytes = 2 * 32 * (kHeadKDim + kHeadVDim) * sizeof(T)
                               + 65 * sizeof(float);
  constexpr int kChunk = kConstruct && kPrefetchBytes <= 48 * 1024 ? 32 : 16;
  constexpr int kBuffers = 2;
  constexpr int tiles_v = kTileV / 16, tile_count = kTileK / 16 * tiles_v;
  constexpr int steps = (tile_count + kWarps - 1) / kWarps;
  constexpr int macro_v = (kHeadVDim + kTileV - 1) / kTileV;
  constexpr int macros = ((kHeadKDim + kTileK - 1) / kTileK) * macro_v;
  static_assert(kTileK % 16 == 0 && kTileV % 16 == 0 && kChunk % 16 == 0);
  int linear = blockIdx.x;
  const int macro = linear % macros;
  linear /= macros;
  const int head = linear % params.num_heads;
  linear /= params.num_heads;
  const int layer = linear % params.num_layers, task = linear / params.num_layers;
  if (task >= params.num_tasks) {
    return;
  }
  const int k_begin = macro / macro_v * kTileK, v_begin = macro % macro_v * kTileV;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  union Scratch {
    struct {
      T key[kBuffers][kChunk * kTileK];
      T value[kBuffers][kChunk * kTileV];
    } log;
    float tiles[kWarps * 256];
  };
  __shared__ __align__(32) Scratch scratch;
  __shared__ float suffix[64];
  __shared__ float gamma;
  using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  Accumulator state[steps];
  const int base_slot = params.base_slots[task];
  const auto *base = base_slot >= 0
    ? state_pointer<T, kHeadKDim, kHeadVDim>(params, base_slot, layer, head)
    : nullptr;
#pragma unroll
  for (int step = 0; step < steps; ++step) {
    const int tile = step * kWarps + warp;
    if (tile >= tile_count) {
      continue;
    }
    if (base == nullptr) {
      wmma::fill_fragment(state[step], 0.f);
    } else {
      for (int index = lane; index < 256; index += 32) {
        const int k = k_begin + tile / tiles_v * 16 + index / 16;
        const int v = v_begin + tile % tiles_v * 16 + index % 16;
        scratch.tiles[warp * 256 + index] = k < kHeadKDim && v < kHeadVDim
          ? to_float(base[k * kHeadVDim + v]) : 0.f;
      }
      __syncwarp();
      wmma::load_matrix_sync(state[step], scratch.tiles + warp * 256,
        16, wmma::mem_row_major);
      __syncwarp();
    }
  }
  __syncthreads();
  for (int segment = params.task_log_offsets[task]; segment < params.task_log_offsets[task + 1]; ++segment) {
    const int slot = params.log_slots[segment], begin = params.log_begins[segment];
    const int length = params.log_ends[segment] - begin;
    if (threadIdx.x == 0) {
      float sum = 0.f;
      const auto *decays = length > 0
        ? decay_pointer(params, slot, layer, head, begin, params.page_size)
        : nullptr;
      for (int token = length - 1; token >= 0; --token) {
        suffix[token] = expf(sum);
        sum += decays[token * params.num_heads];
      }
      gamma = expf(sum);
    }
    __syncthreads();
#pragma unroll
    for (int step = 0; step < steps; ++step) {
      if (step * kWarps + warp < tile_count) {
#pragma unroll
        for (int index = 0; index < Accumulator::num_elements; ++index) {
          state[step].x[index] *= gamma;
        }
      }
    }
    if (length > 0) {
      prefetch_log_chunk<T, kHeadKDim, kHeadVDim, kTileK, kTileV, kChunk>(
        params, slot, layer, head, begin, 0, length, k_begin, v_begin,
        scratch.log.key[0], scratch.log.value[0]
      );
    }
    for (int chunk = 0; chunk < length; chunk += kChunk) {
      const int buffer = chunk / kChunk % kBuffers;
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      if (chunk + kChunk < length) {
        prefetch_log_chunk<T, kHeadKDim, kHeadVDim, kTileK, kTileV, kChunk>(
          params, slot, layer, head, begin, chunk + kChunk, length, k_begin, v_begin,
          scratch.log.key[1 - buffer], scratch.log.value[1 - buffer]
        );
      }
      for (int index = threadIdx.x; index < kChunk * kTileV; index += blockDim.x) {
        const int token = chunk + index / kTileV;
        scratch.log.value[buffer][index] = from_float<T>(token < length
          ? to_float(scratch.log.value[buffer][index]) * suffix[token] : 0.f);
      }
      __syncthreads();
#pragma unroll
      for (int sub = 0; sub < kChunk; sub += 16) {
#pragma unroll
        for (int step = 0; step < steps; ++step) {
          const int tile = step * kWarps + warp;
          if (tile >= tile_count) {
            continue;
          }
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> key_fragment;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> value_fragment;
          wmma::load_matrix_sync(key_fragment, scratch.log.key[buffer] + sub * kTileK + tile / tiles_v * 16, kTileK);
          wmma::load_matrix_sync(value_fragment, scratch.log.value[buffer] + sub * kTileV + tile % tiles_v * 16, kTileV);
          wmma::mma_sync(state[step], key_fragment, value_fragment, state[step]);
        }
      }
      __syncthreads();
    }
    if constexpr (kConstruct) {
      for (int output = params.output_offsets[segment]; output < params.output_offsets[segment + 1]; ++output) {
        write_state_tiles<T, kHeadKDim, kHeadVDim, kTileK, kTileV, kWarps, steps>(params,
          params.construction_output_slots[output], layer, head, k_begin, v_begin, state, scratch.tiles);
        __syncthreads();
      }
    }
    __syncthreads();
  }
  if constexpr (!kConstruct) {
    write_state_tiles<T, kHeadKDim, kHeadVDim, kTileK, kTileV, kWarps, steps>(params,
      params.output_slots[task], layer, head, k_begin, v_begin, state, scratch.tiles);
  }
}


template <typename T, std::size_t kHeadKDim, std::size_t kHeadVDim>
__global__ void copy_log_state_kernel(const __grid_constant__ LogStateRestoreParams params) {
  const auto head = blockIdx.x % params.num_heads;
  const auto layer = blockIdx.x / params.num_heads % params.num_layers;
  const auto task = blockIdx.x / params.num_heads / params.num_layers;
  auto *source = state_pointer<T, kHeadKDim, kHeadVDim>(params, params.base_slots[task], layer, head);
  auto *target = state_pointer<T, kHeadKDim, kHeadVDim>(params, params.output_slots[task], layer, head);
  for (auto index = threadIdx.x; index < kHeadKDim * kHeadVDim; index += blockDim.x) {
    target[index] = source[index];
  }
}

}
