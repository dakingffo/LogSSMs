#include <concepts>
#include <cstddef>
#include <cstdint>

#include <tvm/ffi/container/tensor.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>
#include <logssms/warp.hpp>

namespace logssms {

struct KVStoreKernelParams {
  void *__restrict__       k_cache;
  void *__restrict__       v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t              cache_stride;
  std::size_t              k_stride;
  std::size_t              v_stride;
  std::size_t              length;
};

template <
  std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
  std::size_t kElementSize, std::integral IndexType
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy)
void store_kv_kernel(
  const __grid_constant__ KVStoreKernelParams params
) {
  using namespace device;

  constexpr auto kWarpsPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[k_cache, v_cache, indices_raw, k, v, cache_stride,
               k_stride, v_stride, length] = params;
  const auto warp_id =
      threadIdx.x / kWarpThreads + blockIdx.x * kWarpsPerBlock;

  PDL::wait<kUsePDL>();
  if (warp_id < length) {
    const auto position = static_cast<const IndexType *>(indices_raw)[warp_id];
    const auto destination_k =
        pointer::offset(k_cache, position * cache_stride);
    const auto source_k = pointer::offset(k, warp_id * k_stride);
    warp::copy<kElementSize>(destination_k, source_k);

    const auto destination_v =
        pointer::offset(v_cache, position * cache_stride);
    const auto source_v = pointer::offset(v, warp_id * v_stride);
    warp::copy<kElementSize>(destination_v, source_v);
  }
  PDL::launch<kUsePDL>();
}

template <
  std::size_t element_size,
  std::size_t num_threads     = 128,
  std::size_t max_concurrency = 1,
  bool        use_pdl         = false
>
struct KVStoreKernel {
  static void run(
    const tvm::ffi::TensorView k_cache,
    const tvm::ffi::TensorView v_cache,
    const tvm::ffi::TensorView indices,
    const tvm::ffi::TensorView k,
    const tvm::ffi::TensorView v
  ) {
    using namespace host;

    auto row_size      = SymbolicSize{"D"};
    auto length        = SymbolicSize{"L"};
    auto cache_stride  = SymbolicSize{"cache_stride"};
    auto k_stride      = SymbolicSize{"k_stride"};
    auto v_stride      = SymbolicSize{"v_stride"};
    auto indices_dtype = SymbolicDType{};
    auto dtype         = SymbolicDType{};
    auto device        = SymbolicDevice{};

    TensorMatcher({-1, row_size})
        .with_strides({cache_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({length, row_size})
        .with_strides({k_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(k);
    TensorMatcher({length, row_size})
        .with_strides({v_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(v);
    TensorMatcher({length})
        .with_device<kDLCUDA>(device)
        .with_dtype<std::int32_t, std::int64_t>(indices_dtype)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype.unwrap());
    runtime_assert(
      element_size == dtype_size * row_size.unwrap(),
      "StoreKernel element_size mismatch"
    );

    if (length.unwrap() == 0) {
      return;
    }

    const auto params = KVStoreKernelParams{
      .k_cache      = k_cache.data_ptr(),
      .v_cache      = v_cache.data_ptr(),
      .indices      = indices.data_ptr(),
      .k            = k.data_ptr(),
      .v            = v.data_ptr(),
      .cache_stride = static_cast<std::size_t>(cache_stride.unwrap()) * dtype_size,
      .k_stride     = static_cast<std::size_t>(k_stride.unwrap()) * dtype_size,
      .v_stride     = static_cast<std::size_t>(v_stride.unwrap()) * dtype_size,
      .length       = static_cast<std::size_t>(length.unwrap()),
    };

    constexpr auto kWarpsPerBlock =
        num_threads / logssms::device::kWarpThreads;
    static_assert(num_threads % logssms::device::kWarpThreads == 0);
    const auto num_blocks = div_ceil(params.length, kWarpsPerBlock);
    const auto use_int32  = indices_dtype.unwrap().bits == 32;
    const auto kernel =
        use_int32 ? store_kv_kernel<num_threads, max_concurrency, use_pdl,
                                    element_size, std::int32_t>
                  : store_kv_kernel<num_threads, max_concurrency, use_pdl,
                                    element_size, std::int64_t>;
    LaunchKernel(num_blocks, num_threads, device.unwrap())
        .with_attr(use_pdl)(kernel, params);
  }
};

struct StateStoreKernelParams {
  void *__restrict__       pool;
  const void *__restrict__ indices;
  const void *__restrict__ states;
  std::size_t              pool_stride;
  std::size_t              state_stride;
  std::size_t              length;
};

template <
  std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
  std::size_t kElementSize, std::integral IndexType
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy)
void store_state_kernel(
  const __grid_constant__ StateStoreKernelParams params
) {
  using namespace device;

  constexpr auto kWarpsPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[pool, indices_raw, states, pool_stride, state_stride, length] =
      params;
  const auto warp_id =
      threadIdx.x / kWarpThreads + blockIdx.x * kWarpsPerBlock;

  PDL::wait<kUsePDL>();
  if (warp_id < length) {
    const auto position = static_cast<const IndexType *>(indices_raw)[warp_id];
    const auto destination = pointer::offset(pool, position * pool_stride);
    const auto source = pointer::offset(states, warp_id * state_stride);
    warp::copy<kElementSize>(destination, source);
  }
  PDL::launch<kUsePDL>();
}

template <
  std::size_t element_size,
  std::size_t num_threads     = 128,
  std::size_t max_concurrency = 1,
  bool        use_pdl         = false
>
struct StateStoreKernel {
  static void run(
    const tvm::ffi::TensorView pool,
    const tvm::ffi::TensorView indices,
    const tvm::ffi::TensorView states
  ) {
    using namespace host;

    auto row_size      = SymbolicSize{"D"};
    auto length        = SymbolicSize{"L"};
    auto pool_stride   = SymbolicSize{"pool_stride"};
    auto state_stride  = SymbolicSize{"state_stride"};
    auto indices_dtype = SymbolicDType{};
    auto dtype         = SymbolicDType{};
    auto device        = SymbolicDevice{};

    TensorMatcher({-1, row_size})
        .with_strides({pool_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(pool);
    TensorMatcher({length, row_size})
        .with_strides({state_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(dtype)
        .verify(states);
    TensorMatcher({length})
        .with_device<kDLCUDA>(device)
        .with_dtype<std::int32_t, std::int64_t>(indices_dtype)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype.unwrap());
    runtime_assert(
      element_size == dtype_size * row_size.unwrap(),
      "StateStoreKernel element_size mismatch"
    );
    if (length.unwrap() == 0) {
      return;
    }

    const auto params = StateStoreKernelParams{
      .pool         = pool.data_ptr(),
      .indices      = indices.data_ptr(),
      .states       = states.data_ptr(),
      .pool_stride  = static_cast<std::size_t>(pool_stride.unwrap()) * dtype_size,
      .state_stride = static_cast<std::size_t>(state_stride.unwrap()) * dtype_size,
      .length       = static_cast<std::size_t>(length.unwrap()),
    };

    constexpr auto kWarpsPerBlock =
        num_threads / logssms::device::kWarpThreads;
    static_assert(num_threads % logssms::device::kWarpThreads == 0);
    const auto num_blocks = div_ceil(params.length, kWarpsPerBlock);
    const auto use_int32  = indices_dtype.unwrap().bits == 32;
    const auto kernel =
        use_int32 ? store_state_kernel<num_threads, max_concurrency, use_pdl,
                                       element_size, std::int32_t>
                  : store_state_kernel<num_threads, max_concurrency, use_pdl,
                                       element_size, std::int64_t>;
    LaunchKernel(num_blocks, num_threads, device.unwrap())
        .with_attr(use_pdl)(kernel, params);
  }
};

}
