#pragma once


#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")

#define __cpp_consteval 201811L
#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif
#define consteval constexpr
#include <source_location>
#undef consteval

#pragma pop_macro("_NODISCARD")
#pragma pop_macro("__cpp_consteval")
#else
#include <source_location>
#endif

#include <concepts>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <dlpack/dlpack.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#define LOGSSMS_DEVICE __forceinline__ __device__
#define LOGSSMS_HOST_DEVICE __forceinline__ __host__ __device__
#else
#define LOGSSMS_HOST_DEVICE inline
#endif

namespace logssms {

template <std::integral T, std::integral U>
LOGSSMS_HOST_DEVICE constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

LOGSSMS_HOST_DEVICE std::size_t dtype_bytes(DLDataType dtype) {
  return static_cast<std::size_t>(dtype.bits) * dtype.lanes / 8;
}

namespace pointer {

template <std::integral... Ints>
LOGSSMS_HOST_DEVICE void* offset(void* ptr, Ints... offsets) {
  return static_cast<char*>(ptr) + (... + offsets);
}

template <std::integral... Ints>
LOGSSMS_HOST_DEVICE const void* offset(const void* ptr, Ints... offsets) {
  return static_cast<const char*>(ptr) + (... + offsets);
}

}

namespace host {

class PanicError : public std::runtime_error {
public:
  explicit PanicError(std::string message)
      : std::runtime_error(message), message_(std::move(message)) {}

  std::string_view detail() const {
    const auto message = std::string_view{message_};
    const auto pos = message.find(": ");
    return pos == std::string_view::npos ? message : message.substr(pos + 2);
  }

private:
  std::string message_;
};

template <typename... Args>
[[noreturn]] inline void panic(std::source_location location, Args&&... args) {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":"
     << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  }
  else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

template <typename... Args>
class runtime_panic {
public:
  explicit runtime_panic(
    Args&&... args,
    std::source_location location = std::source_location::current()
  ) {
    panic(location, std::forward<Args>(args)...);
  }

  [[noreturn]] ~runtime_panic() { std::terminate(); }
};

template <typename... Args>
class runtime_assert {
public:
  template <typename T>
  explicit runtime_assert(
    T&& condition,
    Args&&... args,
    std::source_location location = std::source_location::current()
  ) {
    if (!condition) [[unlikely]] {
      panic(location, std::forward<Args>(args)...);
    }
  }
};

template <typename T, typename... Args>
runtime_assert(T&&, Args&&...) -> runtime_assert<Args...>;

template <typename... Args>
runtime_panic(Args&&...) -> runtime_panic<Args...>;

}

#ifdef __CUDACC__

namespace device::PDL {

template <bool kUsePDL>
LOGSSMS_DEVICE void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
LOGSSMS_DEVICE void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

}

namespace host {

inline void CUDA_CHECK(
  ::cudaError_t error,
  std::source_location location = std::source_location::current()
) {
  if (error != ::cudaSuccess) [[unlikely]] {
    panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline void CUDA_CHECK(
  std::source_location location = std::source_location::current()
) {
  CUDA_CHECK(::cudaGetLastError(), location);
}

template <auto F>
inline void set_smem_once(std::size_t smem_size) {
  static const auto max_smem_size = [&] {
    CUDA_CHECK(::cudaFuncSetAttribute(
      F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
    ));
    return smem_size;
  }();
  runtime_assert(
    smem_size <= max_smem_size,
    "Dynamic shared memory size exceeds the previously set maximum size: ",
    max_smem_size, " bytes"
  );
}

class LaunchKernel {
public:
  explicit LaunchKernel(
    dim3 grid_dim,
    dim3 block_dim,
    DLDevice device,
    std::size_t dynamic_shared_mem_bytes = 0
  ) noexcept : config_(
    make_config(
      grid_dim,
      block_dim,
      resolve_device(device),
      dynamic_shared_mem_bytes
    )
  ) {}

  explicit LaunchKernel(
    dim3 grid_dim,
    dim3 block_dim,
    cudaStream_t stream,
    std::size_t dynamic_shared_mem_bytes = 0
  ) noexcept : config_(
    make_config(
      grid_dim, block_dim, stream, dynamic_shared_mem_bytes
    )
  ) {}

  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  static cudaStream_t resolve_device(DLDevice device) {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  template <typename Kernel, typename... Args>
  void operator()(Kernel&& kernel, Args&&... args) const {
    CUDA_CHECK(::cudaLaunchKernelEx(
      &config_,
      std::forward<Kernel>(kernel),
      std::forward<Args>(args)...
    ));
  }

  LaunchKernel& with_attr(bool use_pdl) {
    if (use_pdl) {
      attr_.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      attr_.val.programmaticStreamSerializationAllowed = 1;
      config_.attrs = &attr_;
      config_.numAttrs = 1;
    }
    else {
      config_.attrs = nullptr;
      config_.numAttrs = 0;
    }
    return *this;
  }

private:
  static cudaLaunchConfig_t make_config(
    dim3 grid_dim,
    dim3 block_dim,
    cudaStream_t stream,
    std::size_t smem
  ) {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim          = grid_dim;
    config.blockDim         = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream           = stream;
    return config;
  }

  cudaLaunchConfig_t  config_{};
  cudaLaunchAttribute attr_{};
};

}

#endif

}
