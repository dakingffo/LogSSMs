#include <concepts>
#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tvm/ffi/container/tensor.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>

namespace logssms {

struct LogStateKernelParams {
  const void *key_log;
  const void *value_log;
  const float *decay_log;
  void *recurrent_state;

  const std::int32_t *__restrict__ output_slots;
  const std::int32_t *__restrict__ base_slots;
  const std::int32_t *__restrict__ task_log_offsets;
  const std::int32_t *__restrict__ log_slots;
  const std::int32_t *__restrict__ log_begins;
  const std::int32_t *__restrict__ log_ends;
  const std::int32_t *__restrict__ checkpoint_flags;

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

template <
  typename T,
  std::size_t kHeadKDim,
  std::size_t kHeadVDim,
  std::size_t kNumThreads
>
LOGSSMS_DEVICE void load_initial_state(
  float (&state)[div_ceil(kHeadKDim * kHeadVDim, kNumThreads)],
  const LogStateKernelParams& params,
  std::int32_t base_slot,
  std::int64_t layer,
  std::int64_t head
) {
  constexpr auto kStateSize = kHeadKDim * kHeadVDim;
  constexpr auto kItems = div_ceil(kStateSize, kNumThreads);
  const auto *source = state_pointer<T, kHeadKDim, kHeadVDim>(
    params, base_slot, layer, head
  );

#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item) {
    const auto index = threadIdx.x + item * kNumThreads;
    state[item] = base_slot >= 0 && index < kStateSize
                    ? to_float(source[index])
                    : 0.0f;
  }
}

template <
  typename T,
  std::size_t kHeadKDim,
  std::size_t kHeadVDim,
  std::size_t kNumThreads
>
LOGSSMS_DEVICE void consume_log(
  float (&state)[div_ceil(kHeadKDim * kHeadVDim, kNumThreads)],
  const LogStateKernelParams& params,
  std::int32_t slot,
  std::int32_t begin,
  std::int32_t end,
  std::int64_t layer,
  std::int64_t head,
  float *shared_key,
  float *shared_value,
  float *shared_decay
) {
  constexpr auto kStateSize = kHeadKDim * kHeadVDim;
  constexpr auto kItems = div_ceil(kStateSize, kNumThreads);

  for (auto token = begin; token < end; ++token) {
    const auto *key = key_pointer<T, kHeadKDim, kHeadVDim>(
      params, slot, layer, head, token
    );
    const auto *value = value_pointer<T, kHeadKDim, kHeadVDim>(
      params, slot, layer, head, token
    );
    if (threadIdx.x < kHeadKDim) {
      shared_key[threadIdx.x] = to_float(key[threadIdx.x]);
    }
    if (threadIdx.x < kHeadVDim) {
      shared_value[threadIdx.x] = to_float(value[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
      shared_decay[0] = params.decay_log[
          slot  * params.decay_slot_stride
        + layer * params.decay_layer_stride
        + token * params.decay_token_stride
        + head  * params.decay_head_stride
      ];
    }
    __syncthreads();

#pragma unroll
    for (std::size_t item = 0; item < kItems; ++item) {
      const auto index = threadIdx.x + item * kNumThreads;
      if (index < kStateSize) {
        const auto k = index / kHeadVDim;
        const auto v = index % kHeadVDim;
        state[item] = state[item] * shared_decay[0]
                    + shared_key[k] * shared_value[v];
      }
    }
    __syncthreads();
  }
}

template <
  typename T,
  std::size_t kHeadKDim,
  std::size_t kHeadVDim,
  std::size_t kNumThreads
>
LOGSSMS_DEVICE void store_recurrent_state(
  const float (&state)[div_ceil(kHeadKDim * kHeadVDim, kNumThreads)],
  const LogStateKernelParams& params,
  std::int32_t output_slot,
  std::int64_t layer,
  std::int64_t head
) {
  constexpr auto kStateSize = kHeadKDim * kHeadVDim;
  constexpr auto kItems = div_ceil(kStateSize, kNumThreads);
  auto *output = state_pointer<T, kHeadKDim, kHeadVDim>(
    params, output_slot, layer, head
  );

#pragma unroll
  for (std::size_t item = 0; item < kItems; ++item) {
    const auto index = threadIdx.x + item * kNumThreads;
    if (index < kStateSize) {
      output[index] = from_float<T>(state[item]);
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
void restore_log_state_kernel(
  const __grid_constant__ LogStateKernelParams params
) {
  const auto task = blockIdx.x / (params.num_layers * params.num_heads);
  const auto layer_head = blockIdx.x % (params.num_layers * params.num_heads);
  const auto layer = layer_head / params.num_heads;
  const auto head = layer_head % params.num_heads;
  if (task >= params.num_tasks) {
    return;
  }

  constexpr auto kItems = div_ceil(kHeadKDim * kHeadVDim, kNumThreads);
  float state[kItems];
  extern __shared__ float shared[];
  auto *shared_key = shared;
  auto *shared_value = shared_key + kHeadKDim;
  auto *shared_decay = shared_value + kHeadVDim;

  load_initial_state<T, kHeadKDim, kHeadVDim, kNumThreads>(
    state, params, params.base_slots[task], layer, head
  );
  for (auto index = params.task_log_offsets[task];
       index < params.task_log_offsets[task + 1]; ++index) {
    consume_log<T, kHeadKDim, kHeadVDim, kNumThreads>(
      state, params, params.log_slots[index], params.log_begins[index],
      params.log_ends[index], layer, head, shared_key, shared_value,
      shared_decay
    );
  }
  store_recurrent_state<T, kHeadKDim, kHeadVDim, kNumThreads>(
    state, params, params.output_slots[task], layer, head
  );
}

template <
  typename T,
  std::size_t kHeadKDim,
  std::size_t kHeadVDim,
  std::size_t kNumThreads
>
__global__ __launch_bounds__(kNumThreads)
void materialize_log_state_kernel(
  const __grid_constant__ LogStateKernelParams params
) {
  const auto chain = blockIdx.x / (params.num_layers * params.num_heads);
  const auto layer_head = blockIdx.x % (params.num_layers * params.num_heads);
  const auto layer = layer_head / params.num_heads;
  const auto head = layer_head % params.num_heads;
  if (chain >= params.num_tasks) {
    return;
  }

  constexpr auto kItems = div_ceil(kHeadKDim * kHeadVDim, kNumThreads);
  float state[kItems];
  extern __shared__ float shared[];
  auto *shared_key = shared;
  auto *shared_value = shared_key + kHeadKDim;
  auto *shared_decay = shared_value + kHeadVDim;

  load_initial_state<T, kHeadKDim, kHeadVDim, kNumThreads>(
    state, params, params.base_slots[chain], layer, head
  );
  for (auto index = params.task_log_offsets[chain];
       index < params.task_log_offsets[chain + 1]; ++index) {
    const auto slot = params.log_slots[index];
    consume_log<T, kHeadKDim, kHeadVDim, kNumThreads>(
      state, params, slot, params.log_begins[index], params.log_ends[index],
      layer, head, shared_key, shared_value, shared_decay
    );
    if (params.checkpoint_flags[index]) {
      store_recurrent_state<T, kHeadKDim, kHeadVDim, kNumThreads>(
        state, params, slot, layer, head
      );
    }
    __syncthreads();
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

  template <auto Kernel>
  static void launch(
    const LogStateKernelParams& params,
    DLDevice device
  ) {
    if (params.num_tasks == 0) {
      return;
    }
    const auto blocks = params.num_tasks * params.num_layers * params.num_heads;
    const auto shared_mem = (head_k_dim + head_v_dim + 1) * sizeof(float);
    host::LaunchKernel(blocks, num_threads, device, shared_mem)(Kernel, params);
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

  static void restore(
    const tvm::ffi::TensorView key_log,
    const tvm::ffi::TensorView value_log,
    const tvm::ffi::TensorView decay_log,
    const tvm::ffi::TensorView recurrent_state,
    const tvm::ffi::TensorView output_slots,
    const tvm::ffi::TensorView base_slots,
    const tvm::ffi::TensorView task_log_offsets,
    const tvm::ffi::TensorView log_slots,
    const tvm::ffi::TensorView log_begins,
    const tvm::ffi::TensorView log_ends
  ) {
    using namespace host;
    auto views = verify_views(key_log, value_log, decay_log, recurrent_state);
    for (const auto tensor : {
           output_slots, base_slots, task_log_offsets, log_slots,
           log_begins, log_ends
         }) {
      verify_int32_vector(tensor, views.device);
    }
    const auto num_tasks = output_slots.size(0);
    runtime_assert(base_slots.size(0) == num_tasks);
    runtime_assert(task_log_offsets.size(0) == num_tasks + 1);
    runtime_assert(log_slots.size(0) == log_begins.size(0));
    runtime_assert(log_slots.size(0) == log_ends.size(0));

    auto& params = views.params;
    params.output_slots =
        static_cast<const std::int32_t *>(output_slots.data_ptr());
    params.base_slots =
        static_cast<const std::int32_t *>(base_slots.data_ptr());
    params.task_log_offsets =
        static_cast<const std::int32_t *>(task_log_offsets.data_ptr());
    params.log_slots =
        static_cast<const std::int32_t *>(log_slots.data_ptr());
    params.log_begins =
        static_cast<const std::int32_t *>(log_begins.data_ptr());
    params.log_ends =
        static_cast<const std::int32_t *>(log_ends.data_ptr());
    params.num_tasks = num_tasks;

    if (views.dtype.code == kDLBfloat) {
      launch<restore_log_state_kernel<__nv_bfloat16, head_k_dim,
                                      head_v_dim, num_threads>>(
        params, views.device
      );
    }
    else if (views.dtype.bits == 16) {
      launch<restore_log_state_kernel<__half, head_k_dim,
                                      head_v_dim, num_threads>>(
        params, views.device
      );
    }
    else {
      launch<restore_log_state_kernel<float, head_k_dim,
                                      head_v_dim, num_threads>>(
        params, views.device
      );
    }
  }

  static void materialize(
    const tvm::ffi::TensorView key_log,
    const tvm::ffi::TensorView value_log,
    const tvm::ffi::TensorView decay_log,
    const tvm::ffi::TensorView recurrent_state,
    const tvm::ffi::TensorView base_slots,
    const tvm::ffi::TensorView chain_log_offsets,
    const tvm::ffi::TensorView log_slots,
    const tvm::ffi::TensorView log_begins,
    const tvm::ffi::TensorView log_ends,
    const tvm::ffi::TensorView checkpoint_flags
  ) {
    using namespace host;
    auto views = verify_views(key_log, value_log, decay_log, recurrent_state);
    for (const auto tensor : {
           base_slots, chain_log_offsets, log_slots, log_begins,
           log_ends, checkpoint_flags
         }) {
      verify_int32_vector(tensor, views.device);
    }
    const auto num_chains = base_slots.size(0);
    runtime_assert(chain_log_offsets.size(0) == num_chains + 1);
    runtime_assert(log_slots.size(0) == log_begins.size(0));
    runtime_assert(log_slots.size(0) == log_ends.size(0));
    runtime_assert(log_slots.size(0) == checkpoint_flags.size(0));

    auto& params = views.params;
    params.base_slots =
        static_cast<const std::int32_t *>(base_slots.data_ptr());
    params.task_log_offsets =
        static_cast<const std::int32_t *>(chain_log_offsets.data_ptr());
    params.log_slots =
        static_cast<const std::int32_t *>(log_slots.data_ptr());
    params.log_begins =
        static_cast<const std::int32_t *>(log_begins.data_ptr());
    params.log_ends =
        static_cast<const std::int32_t *>(log_ends.data_ptr());
    params.checkpoint_flags =
        static_cast<const std::int32_t *>(checkpoint_flags.data_ptr());
    params.num_tasks = num_chains;

    if (views.dtype.code == kDLBfloat) {
      launch<materialize_log_state_kernel<__nv_bfloat16, head_k_dim,
                                          head_v_dim, num_threads>>(
        params, views.device
      );
    }
    else if (views.dtype.bits == 16) {
      launch<materialize_log_state_kernel<__half, head_k_dim,
                                          head_v_dim, num_threads>>(
        params, views.device
      );
    }
    else {
      launch<materialize_log_state_kernel<float, head_k_dim,
                                          head_v_dim, num_threads>>(
        params, views.device
      );
    }
  }
};

} // namespace logssms
