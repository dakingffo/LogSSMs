#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <logssms/tensor.hpp>
#include <logssms/utils.hpp>
#include <logssms/warp.hpp>

namespace logssms {

struct IndexKernelParams {
  void *__restrict__       output;
  const void *__restrict__ weight;
  const void *__restrict__ indices;
  std::size_t              num_warps;
};

struct MaskedIndexKernelParams {
  IndexKernelParams params;
  std::size_t       start;
  std::size_t       length;
};

template <
  std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
  std::size_t kElementSize, std::size_t kNumSplits, std::integral IndexType
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy)
void index_kernel(
  const __grid_constant__ IndexKernelParams params
) {
  using namespace device;

  constexpr auto kSizePerWarp = kElementSize / kNumSplits;
  constexpr auto kWarpsPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[output, weight, indices_raw, num_warps] = params;
  const auto indices = static_cast<const IndexType *>(indices_raw);
  const auto warp_id =
      threadIdx.x / kWarpThreads + blockIdx.x * kWarpsPerBlock;

  PDL::wait<kUsePDL>();
  if (warp_id < num_warps) {
    const auto position = indices[warp_id / kNumSplits];
    const auto destination =
        pointer::offset(output, warp_id * kSizePerWarp);
    const auto source = pointer::offset(
      weight,
      position * kElementSize,
      (warp_id % kNumSplits) * kSizePerWarp
    );
    warp::copy<kSizePerWarp>(destination, source);
  }
  PDL::launch<kUsePDL>();
}

template <
  std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
  std::size_t kElementSize, std::size_t kNumSplits, std::integral IndexType
>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy)
void masked_index_kernel(
  const __grid_constant__ MaskedIndexKernelParams masked_params
) {
  using namespace device;

  constexpr auto kSizePerWarp = kElementSize / kNumSplits;
  constexpr auto kWarpsPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = masked_params;
  const auto &[output, weight, indices_raw, num_warps] = params;
  const auto indices = static_cast<const IndexType *>(indices_raw);
  const auto warp_id =
      threadIdx.x / kWarpThreads + blockIdx.x * kWarpsPerBlock;

  PDL::wait<kUsePDL>();
  if (warp_id < num_warps) {
    const auto position = indices[warp_id / kNumSplits] - start;
    const auto destination =
        pointer::offset(output, warp_id * kSizePerWarp);
    if (position < length) {
      const auto source = pointer::offset(
        weight,
        position * kElementSize,
        (warp_id % kNumSplits) * kSizePerWarp
      );
      warp::copy<kSizePerWarp>(destination, source);
    }
    else {
      warp::reset<kSizePerWarp>(destination);
    }
  }
  PDL::launch<kUsePDL>();
}

template <
  std::size_t element_size,          // depends on data type and embedding dim
  std::size_t num_splits      = 1,   // how many warps handles one element
  std::size_t num_threads     = 128, // number of threads per block
  std::size_t max_concurrency = 1,   // max blocks per SM
  bool        use_pdl         = false
>
struct IndexKernel {
  static void run(
    const tvm::ffi::TensorView weights, // [V / tp_size, D]
    const tvm::ffi::TensorView indices, // [L]
    const tvm::ffi::TensorView output,  // [L, D]
    tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_options
  ) {
    using namespace host;

    auto embedding_size = SymbolicSize{"D"};
    auto num_indices     = SymbolicSize{"L"};
    auto device          = SymbolicDevice{};
    auto weights_dtype   = SymbolicDType{};
    auto indices_dtype   = SymbolicDType{};

    TensorMatcher({-1, embedding_size})
        .with_dtype(weights_dtype)
        .with_device<kDLCUDA>(device)
        .verify(weights);
    TensorMatcher({num_indices, embedding_size})
        .with_dtype(weights_dtype)
        .with_device<kDLCUDA>(device)
        .verify(output);
    TensorMatcher({num_indices})
        .with_dtype<std::int32_t, std::int64_t>(indices_dtype)
        .with_device<kDLCUDA>(device)
        .verify(indices);

    const auto entry_size =
        dtype_bytes(weights_dtype.unwrap()) * embedding_size.unwrap();
    runtime_assert(
      entry_size == element_size,
      "IndexKernel element_size mismatch: expected ", entry_size,
      " but got ", element_size
    );

    if (num_indices.unwrap() == 0) {
      return;
    }

    constexpr auto kWarpsPerBlock =
        num_threads / logssms::device::kWarpThreads;
    const auto total_warps = num_splits * num_indices.unwrap();
    const auto num_blocks  = div_ceil(total_warps, kWarpsPerBlock);
    const auto params = IndexKernelParams{
      .output    = output.data_ptr(),
      .weight    = weights.data_ptr(),
      .indices   = indices.data_ptr(),
      .num_warps = static_cast<std::size_t>(total_warps),
    };

    const auto use_int32 = indices_dtype.unwrap().bits == 32;
    if (mask_options.has_value()) {
      const auto [start, length] = mask_options.value();
      const auto masked_params = MaskedIndexKernelParams{
        .params = params,
        .start  = static_cast<std::size_t>(start),
        .length = static_cast<std::size_t>(length),
      };
      const auto kernel =
          use_int32 ? masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, std::int32_t>
                    : masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, std::int64_t>;
      LaunchKernel(num_blocks, num_threads, device.unwrap())
          .with_attr(use_pdl)(kernel, masked_params);
    }
    else {
      const auto kernel =
          use_int32 ? index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, std::int32_t>
                    : index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, std::int64_t>;
      LaunchKernel(num_blocks, num_threads, device.unwrap())
          .with_attr(use_pdl)(kernel, params);
    }
  }
};

} // namespace logssms
