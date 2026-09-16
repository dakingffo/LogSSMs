#pragma once

#include <cstddef>
#include <sys/cdefs.h>

#include <logssms/utils.hpp>

namespace logssms::device {

inline constexpr auto kWarpThreads = 32u;

namespace detail {

template <std::size_t kBytes>
inline constexpr auto get_mem_package() {
  if constexpr (kBytes == 16) {
    return uint4{};
  }
  if constexpr (kBytes == 8) {
    return uint2{};
  }
  if constexpr (kBytes == 4) {
    return uint1{};
  }
  static_assert(
    kBytes == 16 || kBytes == 8 || kBytes == 4,
    "Unsupported memory package size"
  );
}

template <std::size_t kBytes>
using mem_package_t = decltype(get_mem_package<kBytes>());

inline constexpr std::size_t resolve_unit_size(std::size_t x) {
  if (x % (16 * kWarpThreads) == 0) {
    return 16;
  }
  if (x % (8 * kWarpThreads) == 0) {
    return 8;
  }
  if (x % (4 * kWarpThreads) == 0) {
    return 4;
  }
  return 0; // trigger static assert in _get_mem_package
}

} // namespace detail

namespace warp {

template <
  std::size_t kBytes,
  std::size_t kUnitBytes = detail::resolve_unit_size(kBytes)
>
LOGSSMS_DEVICE void copy(void* __restrict__ dst, const void* __restrict__ src) {
  using Package = detail::mem_package_t<kUnitBytes>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(
    kBytes % kBytesPerLoop == 0,
    "kBytes must be multiple of 128 bytes"
  );

  const auto dst_packed = static_cast<Package*>(dst);
  const auto src_packed = static_cast<const Package*>(src);
  const auto lane_id = threadIdx.x % kWarpThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kWarpThreads + lane_id;
    dst_packed[j] = src_packed[j];
  }
}

template <
  std::size_t kBytes,
  std::size_t kUnitBytes = detail::resolve_unit_size(kBytes)
>
LOGSSMS_DEVICE void reset(void* __restrict__ dst) {
  using Package = detail::mem_package_t<kUnitBytes>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(
    kBytes % kBytesPerLoop == 0,
    "warp_copy: kBytes must be multiple of 128 bytes"
  );

  const auto dst_ = static_cast<Package*>(dst);
  const auto lane_id = threadIdx.x % kWarpThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    dst_[i * kWarpThreads + lane_id] = Package{};
  }
}

} // namespace warp

} // namespace logssms::device
