#include "logstate_construction.cuh"

namespace logssms {

template <std::size_t page_size, std::size_t head_k_dim, std::size_t head_v_dim,
          std::size_t granularity>
struct LogStateConstructionKernel {
  static_assert(granularity > 0 && page_size % granularity == 0);
private:
  struct Views {
    LogStateKernelParams params;
    DLDevice device;
  };

  static Views verify_views(
    const tvm::ffi::TensorView storage,
    const tvm::ffi::TensorView payload_frames,
    const tvm::ffi::TensorView decay_frames,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t head_payload_bytes,
    std::int64_t decay_record_bytes,
    std::int64_t num_key_heads
  ) {
    using namespace host;
    auto num_frames = SymbolicSize{"num_frames"};
    auto frame_bytes = SymbolicSize{"frame_bytes"};
    auto num_slots = SymbolicSize{"num_slots"};
    auto num_payload_frames = SymbolicSize{"num_payload_frames"};
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
    runtime_assert(num_layers > 0 && num_heads > 0);
    runtime_assert(head_k_dim % 16 == 0 && head_v_dim % 16 == 0);
    runtime_assert(page_size <= 64);
    runtime_assert(frame_bytes.unwrap() % head_payload_bytes == 0);
    runtime_assert(head_payload_bytes >= page_size * (head_k_dim + head_v_dim) * 2);
    runtime_assert(head_payload_bytes >= head_k_dim * head_v_dim * 2);
    runtime_assert(num_layers * num_heads * head_payload_bytes
      <= num_payload_frames.unwrap() * frame_bytes.unwrap());
    runtime_assert(decay_record_bytes == num_layers * num_heads * page_size * 4);
    runtime_assert(decay_record_bytes <= frame_bytes.unwrap());
    return Views{
      .params = LogStateKernelParams{
        .frame_storage = static_cast<std::uint8_t *>(storage.data_ptr()),
        .payload_frames = static_cast<const std::int32_t *>(payload_frames.data_ptr()),
        .decay_frames = static_cast<const std::int32_t *>(decay_frames.data_ptr()),
        .frame_bytes = frame_bytes.unwrap(),
        .payload_frames_per_slot = num_payload_frames.unwrap(),
        .head_payload_bytes = head_payload_bytes,
        .decay_record_bytes = decay_record_bytes,
        .page_size = page_size,
        .granularity = granularity,
        .num_layers = num_layers,
        .num_heads = num_heads,
        .num_key_heads = num_key_heads,
      },
      .device = device.unwrap(),
    };
  }

  static void verify_vector(
    const tvm::ffi::TensorView tensor, DLDevice device
  ) {
    host::TensorMatcher({-1})
        .with_dtype<std::int32_t>()
        .with_device(device)
        .verify(tensor);
  }

public:
  static void restore(
    const tvm::ffi::TensorView storage,
    const tvm::ffi::TensorView payload_frames,
    const tvm::ffi::TensorView decay_frames,
    const tvm::ffi::TensorView base_slots,
    const tvm::ffi::TensorView task_segment_offsets,
    const tvm::ffi::TensorView log_slots,
    const tvm::ffi::TensorView log_begins,
    const tvm::ffi::TensorView log_ends,
    const tvm::ffi::TensorView output_slots,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t head_payload_bytes,
    std::int64_t decay_record_bytes,
    std::int64_t num_key_heads,
    bool is_bfloat16
  ) {
    using namespace host;
    auto views = verify_views(
      storage, payload_frames, decay_frames,
      num_layers, num_heads, head_payload_bytes, decay_record_bytes, num_key_heads
    );
    for (const auto tensor : {
           base_slots, task_segment_offsets, log_slots, log_begins,
           log_ends, output_slots
         }) {
      verify_vector(tensor, views.device);
    }
    const auto num_tasks = base_slots.size(0);
    const auto num_segments = log_slots.size(0);
    runtime_assert(task_segment_offsets.size(0) == num_tasks + 1);
    runtime_assert(log_begins.size(0) == num_segments);
    runtime_assert(log_ends.size(0) == num_segments);
    runtime_assert(output_slots.size(0) == num_tasks);
    if (num_tasks == 0 || num_segments == 0) {
      return;
    }
    LogStateRestoreParams params{};
    static_cast<LogStateKernelParams&>(params) = views.params;
    params.base_slots = static_cast<const std::int32_t *>(base_slots.data_ptr());
    params.task_log_offsets = static_cast<const std::int32_t *>(task_segment_offsets.data_ptr());
    params.log_slots = static_cast<const std::int32_t *>(log_slots.data_ptr());
    params.log_begins = static_cast<const std::int32_t *>(log_begins.data_ptr());
    params.log_ends = static_cast<const std::int32_t *>(log_ends.data_ptr());
    params.output_slots = static_cast<const std::int32_t *>(output_slots.data_ptr());
    params.num_tasks = num_tasks;
    const auto macro_tiles = div_ceil(head_k_dim, std::size_t{32})
                           * div_ceil(head_v_dim, std::size_t{64});
    const auto blocks = num_tasks * num_layers * num_heads * macro_tiles;
    if (is_bfloat16) {
      LaunchKernel(blocks, 256, views.device)(
        restore_log_state_kernel<__nv_bfloat16, head_k_dim, head_v_dim>, params
      );
    } else {
      LaunchKernel(blocks, 256, views.device)(
        restore_log_state_kernel<__half, head_k_dim, head_v_dim>, params
      );
    }
  }

  static void construct(
    const tvm::ffi::TensorView storage,
    const tvm::ffi::TensorView payload_frames,
    const tvm::ffi::TensorView decay_frames,
    const tvm::ffi::TensorView base_slots,
    const tvm::ffi::TensorView task_segment_offsets,
    const tvm::ffi::TensorView log_slots,
    const tvm::ffi::TensorView log_begins,
    const tvm::ffi::TensorView log_ends,
    const tvm::ffi::TensorView output_offsets,
    const tvm::ffi::TensorView output_slots,
    std::int64_t num_layers,
    std::int64_t num_heads,
    std::int64_t head_payload_bytes,
    std::int64_t decay_record_bytes,
    std::int64_t num_key_heads,
    bool is_bfloat16
  ) {
    using namespace host;
    auto views = verify_views(
      storage, payload_frames, decay_frames,
      num_layers, num_heads, head_payload_bytes, decay_record_bytes, num_key_heads
    );
    for (const auto tensor : {
           base_slots, task_segment_offsets, log_slots, log_begins,
           log_ends, output_offsets, output_slots
         }) {
      verify_vector(tensor, views.device);
    }
    const auto num_tasks = base_slots.size(0);
    const auto num_segments = log_slots.size(0);
    runtime_assert(task_segment_offsets.size(0) == num_tasks + 1);
    runtime_assert(log_begins.size(0) == num_segments);
    runtime_assert(log_ends.size(0) == num_segments);
    runtime_assert(output_offsets.size(0) == num_segments + 1);
    if (num_tasks == 0 || num_segments == 0) {
      return;
    }
    LogStateConstructionParams params{};
    static_cast<LogStateKernelParams&>(params) = views.params;
    params.base_slots = static_cast<const std::int32_t *>(base_slots.data_ptr());
    params.task_log_offsets = static_cast<const std::int32_t *>(task_segment_offsets.data_ptr());
    params.log_slots = static_cast<const std::int32_t *>(log_slots.data_ptr());
    params.log_begins = static_cast<const std::int32_t *>(log_begins.data_ptr());
    params.log_ends = static_cast<const std::int32_t *>(log_ends.data_ptr());
    params.output_offsets = static_cast<const std::int32_t *>(output_offsets.data_ptr());
    params.construction_output_slots = static_cast<const std::int32_t *>(output_slots.data_ptr());
    params.num_tasks = num_tasks;
    const auto blocks = num_tasks * num_layers * num_heads;
    if (is_bfloat16) {
      LaunchKernel(blocks, 256, views.device)(
        construct_log_state_kernel<__nv_bfloat16, head_k_dim, head_v_dim>,
        params
      );
    } else {
      LaunchKernel(blocks, 256, views.device)(
        construct_log_state_kernel<__half, head_k_dim, head_v_dim>,
        params
      );
    }
  }

  static void copy(
    const tvm::ffi::TensorView storage,
    const tvm::ffi::TensorView payload_frames,
    const tvm::ffi::TensorView decay_frames,
    const tvm::ffi::TensorView sources,
    const tvm::ffi::TensorView destinations,
    std::int64_t num_layers, std::int64_t num_heads,
    std::int64_t head_payload_bytes, std::int64_t decay_record_bytes,
    std::int64_t num_key_heads, bool is_bfloat16
  ) {
    auto views = verify_views(storage, payload_frames, decay_frames,
      num_layers, num_heads, head_payload_bytes, decay_record_bytes, num_key_heads);
    verify_vector(sources, views.device);
    verify_vector(destinations, views.device);
    host::runtime_assert(sources.size(0) == destinations.size(0));
    if (sources.size(0) == 0) {
      return;
    }
    LogStateRestoreParams params{};
    static_cast<LogStateKernelParams&>(params) = views.params;
    params.base_slots = static_cast<const std::int32_t *>(sources.data_ptr());
    params.output_slots = static_cast<const std::int32_t *>(destinations.data_ptr());
    params.num_tasks = sources.size(0);
    const auto blocks = params.num_tasks * num_layers * num_heads;
    if (is_bfloat16) {
      host::LaunchKernel(blocks, 256, views.device)(
        copy_log_state_kernel<__nv_bfloat16, head_k_dim, head_v_dim>, params);
    } else {
      host::LaunchKernel(blocks, 256, views.device)(
        copy_log_state_kernel<__half, head_k_dim, head_v_dim>, params);
    }
  }
};

}
