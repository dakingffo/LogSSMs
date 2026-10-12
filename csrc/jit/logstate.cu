#include <cstddef>
#include <cstdint>

#include <tvm/ffi/container/tensor.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>

namespace logssms {

struct LogStoreParams {
  std::uint8_t *__restrict__ storage;
  const std::int32_t *__restrict__ payload_frames;
  const std::int32_t *__restrict__ decay_frames;
  const std::int32_t *__restrict__ slots;
  const std::int32_t *__restrict__ token_offsets;
  const std::uint8_t *__restrict__ keys;
  const std::uint8_t *__restrict__ residual_values;
  const float *__restrict__ decays;
  std::int64_t frame_bytes;
  std::int64_t payload_frames_per_slot;
  std::int64_t head_payload_bytes;
  std::int64_t decay_record_bytes;
  std::int64_t num_heads;
  std::int64_t num_key_heads;
  std::int64_t input_key_heads;
  std::int64_t num_tokens;
  std::int64_t layer;
  std::int64_t num_layers;
  std::int64_t key_bytes;
  std::int64_t value_bytes;
  std::int64_t key_row_bytes;
  std::int64_t value_row_bytes;
  std::int64_t decay_row;
};

template <std::size_t kPageSize, std::size_t kHeadKDim, std::size_t kHeadVDim,
          std::size_t kGranularity>
__global__ void log_store_kernel(
  const __grid_constant__ LogStoreParams params
) {
  const auto row = blockIdx.x;
  const auto token = row / params.num_heads;
  const auto head = row % params.num_heads;
  if (token >= params.num_tokens) {
    return;
  }
  const auto slot = params.slots[token];
  const auto position = params.token_offsets[token];
  const auto head_granule = params.head_payload_bytes / (kPageSize / kGranularity);
  const auto key_head = head / (params.num_heads / params.num_key_heads);
  const auto layer_bytes = params.num_key_heads == params.num_heads
    ? params.num_heads * head_granule
    : kGranularity * (params.num_key_heads * params.key_bytes + params.num_heads * params.value_bytes);
  const auto layer_offset = (position / kGranularity * params.num_layers + params.layer) * layer_bytes;
  const auto key_offset = layer_offset
    + (position % kGranularity * params.num_key_heads + key_head) * params.key_bytes;
  const auto value_offset = layer_offset + kGranularity * params.num_key_heads * params.key_bytes
    + (position % kGranularity * params.num_heads + head) * params.value_bytes;
  const auto *key = params.keys
      + token * params.key_row_bytes
      + (params.input_key_heads == params.num_heads ? head : key_head) * params.key_bytes;
  const auto *value = params.residual_values
      + token * params.value_row_bytes + head * params.value_bytes;
  for (auto index = threadIdx.x; head % (params.num_heads / params.num_key_heads) == 0
       && index < params.key_bytes; index += blockDim.x) {
    const auto offset = key_offset + index;
    const auto frame = params.payload_frames[
      slot * params.payload_frames_per_slot + offset / params.frame_bytes
    ];
    params.storage[frame * params.frame_bytes + offset % params.frame_bytes] = key[index];
  }
  for (auto index = threadIdx.x; index < params.value_bytes; index += blockDim.x) {
    const auto offset = value_offset + index;
    const auto frame = params.payload_frames[
      slot * params.payload_frames_per_slot + offset / params.frame_bytes
    ];
    params.storage[frame * params.frame_bytes + offset % params.frame_bytes] = value[index];
  }
  if (threadIdx.x == 0) {
    const auto decay_frame = params.decay_frames[slot * 2];
    const auto decay_index = params.decay_frames[slot * 2 + 1];
    auto *decay = reinterpret_cast<float *>(
      params.storage + decay_frame * params.frame_bytes
      + decay_index * params.decay_record_bytes
    );
    decay[(params.layer * kPageSize + position) * params.num_heads + head]
      = params.decays[token * params.decay_row + head];
  }
}

template <std::size_t page_size, std::size_t head_k_dim, std::size_t head_v_dim,
          std::size_t granularity>
struct LogStateKernel {
  static_assert(granularity > 0 && page_size % granularity == 0);
  static void store(
    const tvm::ffi::TensorView storage,
    const tvm::ffi::TensorView payload_frames,
    const tvm::ffi::TensorView decay_frames,
    const tvm::ffi::TensorView slots,
    const tvm::ffi::TensorView token_offsets,
    const tvm::ffi::TensorView keys,
    const tvm::ffi::TensorView residual_values,
    const tvm::ffi::TensorView decays,
    std::int64_t layer,
    std::int64_t head_payload_bytes,
    std::int64_t decay_record_bytes,
    std::int64_t num_layers,
    std::int64_t num_key_heads
  ) {
    using namespace host;
    auto num_frames = SymbolicSize{"num_frames"};
    auto frame_bytes = SymbolicSize{"frame_bytes"};
    auto num_slots = SymbolicSize{"num_slots"};
    auto num_payload_frames = SymbolicSize{"num_payload_frames"};
    auto num_tokens = SymbolicSize{"num_tokens"};
    auto num_heads = SymbolicSize{"num_heads"};
    auto input_key_heads = SymbolicSize{"input_key_heads"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    TensorMatcher({num_frames, frame_bytes})
        .with_strides({-1, 1})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>(device)
        .verify(storage);
    TensorMatcher({num_slots, num_payload_frames})
        .with_strides({-1, 1})
        .with_dtype<std::int32_t>()
        .with_device(device)
        .verify(payload_frames);
    TensorMatcher({num_slots, 2})
        .with_strides({-1, 1})
        .with_dtype<std::int32_t>()
        .with_device(device)
        .verify(decay_frames);
    TensorMatcher({num_tokens})
        .with_dtype<std::int32_t>()
        .with_device(device)
        .verify(slots)
        .verify(token_offsets);
    TensorMatcher({num_tokens, input_key_heads, head_k_dim})
        .with_strides({-1, head_k_dim, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(keys);
    TensorMatcher({num_tokens, num_heads, head_v_dim})
        .with_strides({-1, head_v_dim, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(residual_values);
    TensorMatcher({num_tokens, num_heads})
        .with_strides({-1, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(decays);

    const auto itemsize = dtype_bytes(dtype.unwrap());
    const auto key_bytes = head_k_dim * itemsize;
    const auto value_bytes = head_v_dim * itemsize;
    runtime_assert(num_layers > 0 && layer >= 0 && layer < num_layers);
    runtime_assert(num_key_heads > 0 && num_heads.unwrap() % num_key_heads == 0);
    runtime_assert(input_key_heads.unwrap() == num_key_heads || input_key_heads.unwrap() == num_heads.unwrap());
    runtime_assert(num_layers * num_heads.unwrap() * head_payload_bytes
      <= num_payload_frames.unwrap() * frame_bytes.unwrap());
    runtime_assert(head_payload_bytes >= page_size * (key_bytes + value_bytes));
    runtime_assert(head_payload_bytes <= frame_bytes.unwrap());
    runtime_assert(frame_bytes.unwrap() % head_payload_bytes == 0);
    runtime_assert(decay_record_bytes > 0 && decay_record_bytes <= frame_bytes.unwrap());
    runtime_assert(frame_bytes.unwrap() % itemsize == 0);
    if (num_tokens.unwrap() == 0) {
      return;
    }

    const auto params = LogStoreParams{
      .storage = static_cast<std::uint8_t *>(storage.data_ptr()),
      .payload_frames = static_cast<const std::int32_t *>(payload_frames.data_ptr()),
      .decay_frames = static_cast<const std::int32_t *>(decay_frames.data_ptr()),
      .slots = static_cast<const std::int32_t *>(slots.data_ptr()),
      .token_offsets = static_cast<const std::int32_t *>(token_offsets.data_ptr()),
      .keys = static_cast<const std::uint8_t *>(keys.data_ptr()),
      .residual_values = static_cast<const std::uint8_t *>(residual_values.data_ptr()),
      .decays = static_cast<const float *>(decays.data_ptr()),
      .frame_bytes = frame_bytes.unwrap(),
      .payload_frames_per_slot = num_payload_frames.unwrap(),
      .head_payload_bytes = head_payload_bytes,
      .decay_record_bytes = decay_record_bytes,
      .num_heads = num_heads.unwrap(),
      .num_key_heads = num_key_heads,
      .input_key_heads = input_key_heads.unwrap(),
      .num_tokens = num_tokens.unwrap(),
      .layer = layer,
      .num_layers = num_layers,
      .key_bytes = static_cast<std::int64_t>(key_bytes),
      .value_bytes = static_cast<std::int64_t>(value_bytes),
      .key_row_bytes = keys.stride(0) * itemsize,
      .value_row_bytes = residual_values.stride(0) * itemsize,
      .decay_row = decays.stride(0),
    };
    LaunchKernel(params.num_tokens * params.num_heads, 128, device.unwrap())(
      log_store_kernel<page_size, head_k_dim, head_v_dim, granularity>, params
    );
  }
};

}
