#pragma once

#include <algorithm>
#include <array>
#include <vector>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <source_location>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <logssms/utils.hpp>

namespace logssms::host {

namespace stdr = std::ranges;
namespace stdv = std::views;

namespace detail {

struct SizeRef;
struct DTypeRef;
struct DeviceRef;

inline constexpr auto kAnyDeviceID = -1;
inline constexpr auto kAnySize     = static_cast<int64_t>(-1);
inline constexpr auto kNullSize    = std::numeric_limits<int64_t>::min();
inline constexpr auto kNullDType   = static_cast<DLDataTypeCode>(18u);
inline constexpr auto kNullDevice  = static_cast<DLDeviceType>(-1);

template <typename T>
struct dtype_trait;

template <std::integral T>
struct dtype_trait<T> {
  static constexpr auto value = DLDataType{
    .code  = std::is_signed_v<T> ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt,
    .bits  = static_cast<std::uint8_t>(sizeof(T) * 8),
    .lanes = 1
  };
};

template <std::floating_point T>
struct dtype_trait<T> {
  static constexpr auto value = DLDataType{
    .code  = DLDataTypeCode::kDLFloat,
    .bits  = static_cast<std::uint8_t>(sizeof(T) * 8),
    .lanes = 1
  };
};

template <DLDeviceType Code>
struct device_trait {
  static constexpr auto value = DLDevice{
    .device_type = Code,
    .device_id   = kAnyDeviceID
  };
};

template <typename... Ts>
inline constexpr auto kDTypeList =
    std::array<DLDataType, sizeof...(Ts)>{dtype_trait<Ts>::value...};

template <DLDeviceType... Codes>
inline constexpr auto kDeviceList =
    std::array<DLDevice, sizeof...(Codes)>{device_trait<Codes>::value...};

template <typename T>
struct PrintAbleSpan {
  explicit PrintAbleSpan(std::span<const T> data) : data(data) {}
  explicit PrintAbleSpan(const std::vector<T>& data) : data(data) {}
  std::span<const T> data;
};

// define DLDataType comparison and printing in root namespace
inline constexpr auto kDeviceStringMap = [] {
  constexpr auto map =
      std::array<std::pair<DLDeviceType, std::string_view>, 16>{
          std::pair{DLDeviceType::kDLCPU, "cpu"},
          std::pair{DLDeviceType::kDLCUDA, "cuda"},
          std::pair{DLDeviceType::kDLCUDAHost, "cuda_host"},
          std::pair{DLDeviceType::kDLOpenCL, "opencl"},
          std::pair{DLDeviceType::kDLVulkan, "vulkan"},
          std::pair{DLDeviceType::kDLMetal, "metal"},
          std::pair{DLDeviceType::kDLVPI, "vpi"},
          std::pair{DLDeviceType::kDLROCM, "rocm"},
          std::pair{DLDeviceType::kDLROCMHost, "rocm_host"},
          std::pair{DLDeviceType::kDLExtDev, "ext_dev"},
          std::pair{DLDeviceType::kDLCUDAManaged, "cuda_managed"},
          std::pair{DLDeviceType::kDLOneAPI, "oneapi"},
          std::pair{DLDeviceType::kDLWebGPU, "webgpu"},
          std::pair{DLDeviceType::kDLHexagon, "hexagon"},
          std::pair{DLDeviceType::kDLMAIA, "maia"},
          std::pair{DLDeviceType::kDLTrn, "trn"},
      };
  constexpr auto max_type = stdr::max(map | stdv::keys);
  auto result = std::array<std::string_view, max_type + 1>{};
  for (const auto& [code, name] : map) {
    result[static_cast<std::size_t>(code)] = name;
  }
  return result;
}();

struct PrintableDevice {
  DLDevice device;
};

inline std::ostream& operator<<(std::ostream& os, DLDevice device) {
  const auto& mapping = kDeviceStringMap;
  const auto entry = static_cast<std::size_t>(device.device_type);
  runtime_assert(entry < mapping.size());
  const auto name = mapping[entry];
  runtime_assert(!name.empty(), "Unknown device: ", int(device.device_type));
  os << name;
  if (device.device_id != kAnyDeviceID)
    os << "[" << device.device_id << "]";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, PrintableDevice pd) {
  return os << pd.device;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, PrintAbleSpan<T> span) {
  os << "[";
  for (std::size_t i = 0; i < span.data.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << span.data[i];
  }
  os << "]";
  return os;
}
} // namespace detail

inline bool operator==(DLDevice lhs, DLDevice rhs) noexcept {
  return lhs.device_type == rhs.device_type && (
         lhs.device_id == rhs.device_id ||
         lhs.device_id == detail::kAnyDeviceID ||
         rhs.device_id == detail::kAnyDeviceID);
}

struct SymbolicSize {
public:
  SymbolicSize(std::string_view annotation = {})
      : value_(detail::kNullSize)
      , annotation_(annotation) {}
  ~SymbolicSize() = default;

  SymbolicSize(const SymbolicSize&) = default;
  SymbolicSize& operator=(const SymbolicSize&) = default;

  std::string_view get_name() const {
    return annotation_;
  }

  void set_value(int64_t value) {
    runtime_assert(!this->has_value(), "Size value already set");
    value_ = value;
  }

  bool has_value() const noexcept {
    return value_ != detail::kNullSize;
  }

  std::optional<int64_t> get_value() const {
    return this->has_value() ? std::optional{value_} : std::nullopt;
  }

  int64_t unwrap() const {
    runtime_assert(this->has_value(), "Size value is not set");
    return value_;
  }

  void verify(int64_t size, const char* prefix, int64_t dim) {
    if (this->has_value()) {
      runtime_assert(
        value_ == size,
        "Size mismatch for ", _name(prefix, dim),
        ": expected ", value_, " but got ", size
      );
    }
    else {
      this->set_value(size);
    }
  }

  std::string value_or_name(const char* prefix, int64_t dim) const {
    if (const auto value = this->get_value()) {
      return std::to_string(*value);
    }
    else {
      return _name(prefix, dim);
    }
  }

private:
  std::string _name(const char *prefix, int64_t dim) const {
    std::ostringstream os;
    if (annotation_.empty()) {
      os << prefix << '#' << dim;
    }
    else {
      os << annotation_ << '(' << prefix << '#' << dim << ')';
    }
    return std::move(os).str();
  }

  std::int64_t     value_;
  std::string_view annotation_;
};

struct SymbolicDType {
public:
  SymbolicDType() : value_({detail::kNullDType, 0, 0}) {}
  ~SymbolicDType() = default;

  void set_value(DLDataType value) {
    runtime_assert(!this->has_value(), "Dtype value already set");
    runtime_assert(
        check_(value),
        "Dtype value [", value, "] not in the allowed options: ",
        detail::PrintAbleSpan{options_}
    );
    value_ = value;
  }

  bool has_value() const {
    return value_.code != detail::kNullDType;
  }

  std::optional<DLDataType> get_value() const {
    return this->has_value() ? std::optional{value_} : std::nullopt;
  }

  DLDataType unwrap() const {
    runtime_assert(this->has_value(), "Dtype value is not set");
    return value_;
  }

  void set_options(std::span<const DLDataType> options) {
    options_.assign(options.begin(), options.end());
  }

  template <typename... Ts>
  void set_options() {
    const auto& options = detail::kDTypeList<Ts...>;
    options_.assign(options.begin(), options.end());
  }

  void verify(DLDataType dtype) {
    if (this->has_value()) {
      runtime_assert(
        value_ == dtype,
        "DType mismatch: expected ", value_, " but got ", dtype
      );
    }
    else {
      this->set_value(dtype);
    }
  }

private:
  bool check_(DLDataType value) const {
    return stdr::empty(options_) ||
           stdr::find(options_, value) != stdr::end(options_);
  }

  std::vector<DLDataType> options_;
  DLDataType              value_;
};

struct SymbolicDevice {
public:
  SymbolicDevice() : value_({detail::kNullDevice, detail::kAnyDeviceID}) {}
  ~SymbolicDevice() = default;

  void set_value(DLDevice value) {
    runtime_assert(!this->has_value(), "Device value already set");
    runtime_assert(
        check_(value),
        "Device value [", detail::PrintableDevice{value},
        "] not in the allowed options: ", detail::PrintAbleSpan{options_}
    );
    value_ = value;
  }

  bool has_value() const {
    return value_.device_type != detail::kNullDevice;
  }

  std::optional<DLDevice> get_value() const {
    return this->has_value() ? std::optional{value_} : std::nullopt;
  }

  DLDevice unwrap() const {
    runtime_assert(this->has_value(), "Device value is not set");
    return value_;
  }

  void set_options(std::span<const DLDevice> options) {
    options_.assign(options.begin(), options.end());
  }

  template <DLDeviceType... Codes>
  void set_options() {
    const auto& options = detail::kDeviceList<Codes...>;
    options_.assign(options.begin(), options.end());
  }

  void verify(DLDevice device) {
    if (this->has_value()) {
      runtime_assert(
        value_ == device,
        "Device mismatch: expected ",
        detail::PrintableDevice{value_}, " but got ",
        detail::PrintableDevice{device}
      );
    }
    else {
      this->set_value(device);
    }
  }

private:
  bool check_(DLDevice value) const {
    if (options_.empty())
      return true;
    for (const auto& opt : options_) {
      if (opt == value)
        return true;
    }
    return false;
  }

  std::vector<DLDevice> options_;
  DLDevice              value_;
};

struct TensorMatcher;

namespace detail {

template <std::default_initializable T>
struct BaseRef {
  explicit BaseRef() : cache_(), ref_(&cache_) {}
  BaseRef(T& value) : cache_(), ref_(&value) {}

  BaseRef(const BaseRef& other)
      : cache_()
      , ref_(other.owns_cache() ? &cache_ : other.ref_) {
    if (other.owns_cache()) {
      cache_ = other.cache_;
    }
  }

  BaseRef(BaseRef&& other) noexcept(
    std::is_nothrow_default_constructible_v<T> &&
    std::is_nothrow_move_assignable_v<T>
  ) : cache_()
    , ref_(other.owns_cache() ? &cache_ : other.ref_) {
    if (other.owns_cache()) {
      cache_ = std::move(other.cache_);
    }
  }

  BaseRef& operator=(const BaseRef& other) {
    if (this != &other) {
      if (other.owns_cache()) {
        cache_ = other.cache_;
        ref_ = &cache_;
      }
      else {
        ref_ = other.ref_;
      }
    }
    return *this;
  }

  BaseRef& operator=(BaseRef&& other) noexcept(
    std::is_nothrow_move_assignable_v<T>
  ) {
    if (this != &other) {
      if (other.owns_cache()) {
        cache_ = std::move(other.cache_);
        ref_ = &cache_;
      }
      else {
        ref_ = other.ref_;
      }
    }
    return *this;
  }

  T* operator->() const {
    return ref_;
  }

  T& operator*()  const {
    return *ref_;
  }

  void rebind(const BaseRef& other) {
    *this = other;
  }

  bool owns_cache() const noexcept {
    return ref_ == &cache_;
  }

  T  cache_;
  T* ref_;
};

struct SizeRef : BaseRef<SymbolicSize> {
  using BaseRef::BaseRef;

  SizeRef(int64_t value) : BaseRef() {
    if (value != kAnySize) {
      ref_->set_value(value);
    }
    // otherwise, match any size
  }
};

struct DTypeRef : BaseRef<SymbolicDType> {
  using BaseRef::BaseRef;

  DTypeRef(DLDataType options) : BaseRef() {
    ref_->set_value(options);
  }

  DTypeRef(std::initializer_list<DLDataType> options) : BaseRef() {
    ref_->set_options(options);
  }

  DTypeRef(std::span<const DLDataType> options) : BaseRef() {
    ref_->set_options(options);
  }
};

struct DeviceRef : BaseRef<SymbolicDevice> {
  using BaseRef::BaseRef;

  DeviceRef(DLDevice options) : BaseRef() {
    ref_->set_value(options);
  }

  DeviceRef(std::initializer_list<DLDevice> options) : BaseRef() {
    ref_->set_options(options);
  }

  DeviceRef(std::span<const DLDevice> options) : BaseRef() {
    ref_->set_options(options);
  }
};

} // namespace detail

struct TensorMatcher {
private:
  using SizeRef   = detail::SizeRef;
  using DTypeRef  = detail::DTypeRef;
  using DeviceRef = detail::DeviceRef;
  using Loc_t     = std::source_location;

public:
  explicit TensorMatcher(std::initializer_list<SizeRef> shape)
      : shape_(shape), strides_(), dtype_(), device_() {}
  ~TensorMatcher() = default;

  TensorMatcher(const TensorMatcher&)            = delete;
  TensorMatcher &operator=(const TensorMatcher&) = delete;

  TensorMatcher&& with_strides(std::initializer_list<SizeRef> strides) && {
    runtime_assert(
      strides_.size() == 0,
      "Strides already specified"
    );
    runtime_assert(
      shape_.size() == strides.size(),
      "Strides size must match shape size"
    );
    strides_ = strides;
    return std::move(*this);
  }

  TensorMatcher&& with_dtype(DTypeRef&& dtype) && {
    _init_dtype();
    dtype_.rebind(dtype);
    return std::move(*this);
  }

  template <typename... Ts>
  TensorMatcher&& with_dtype(DTypeRef&& dtype) && {
    _init_dtype();
    if constexpr (sizeof...(Ts)) {
      dtype->template set_options<Ts...>();
    }
    dtype_.rebind(dtype);
    return std::move(*this);
  }

  template <typename... Ts>
    requires(sizeof...(Ts) > 0)
  TensorMatcher&& with_dtype() && {
    _init_dtype();
    dtype_->set_options<Ts...>();
    return std::move(*this);
  }

  TensorMatcher&& with_device(DeviceRef&& device) && {
    _init_device();
    device_.rebind(device);
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  TensorMatcher&& with_device(DeviceRef&& device) && {
    _init_device();
    if constexpr (sizeof...(Codes)) {
      device->template set_options<Codes...>();
    }
    device_.rebind(device);
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
    requires(sizeof...(Codes) > 0)
  TensorMatcher&& with_device() && {
    _init_device();
    device_->set_options<Codes...>();
    return std::move(*this);
  }

  const TensorMatcher&& verify(
    tvm::ffi::TensorView view,
    Loc_t loc = Loc_t::current()
  ) const && {
    try {
      this->_verify_impl(view);
    }
    catch (PanicError& e) {
      std::ostringstream oss;
      oss << "Tensor match failed for " << this->debug_str() << " at "
          << loc.file_name() << ":" << loc.line()
          << "\n- Root cause: " << e.detail();
      throw PanicError(std::move(oss).str());
    }
    return std::move(*this);
  }

  std::string debug_str() const {
    auto oss = std::ostringstream{};
    oss << "Tensor<";
    std::size_t dim = 0;
    for (const auto& size_ref : shape_) {
      if (dim > 0) {
        oss << ", ";
      }
      oss << size_ref->value_or_name("shape", dim++);
    }
    oss << ">";
    if (strides_.size() > 0) {
      oss << " [strides=<";
      dim = 0;
      for (const auto& stride_ref : strides_) {
        if (dim > 0) {
          oss << ", ";
        }
        oss << stride_ref->value_or_name("stride", dim++);
      }
      oss << ">]";
    }
    return std::move(oss).str();
  }

private:
  void _verify_impl(tvm::ffi::TensorView view) const {
    const auto dim = static_cast<std::size_t>(view.dim());
    runtime_assert(
      dim == shape_.size(),
      "Tensor dimension mismatch: expected ", shape_.size(), " but got ", dim
    );
    for (std::size_t i = 0; i < dim; i++) {
      shape_[i]->verify(view.size(i), "shape", i);
    }
    if (this->_has_strides()) {
      for (std::size_t i = 0; i < dim; i++) {
        if (view.size(i) != 1 || !strides_[i]->has_value()) {
          // skip stride check for size 1 dimension
          strides_[i]->verify(view.stride(i), "stride", i);
        }
      }
    }
    else {
      runtime_assert(
        view.is_contiguous(),
        "Tensor is not contiguous as expected"
      );
    }
    // since we may double verify, we will force to check
    dtype_->verify(view.dtype());
    device_->verify(view.device());
  }

  void _init_dtype() {
    runtime_assert(!has_dtype_, "DType already specified");
    has_dtype_ = true;
  }

  void _init_device() {
    runtime_assert(!has_device_, "Device already specified");
    has_device_ = true;
  }

  bool _has_strides() const {
    return !strides_.empty();
  }

  std::vector<SizeRef> shape_;
  std::vector<SizeRef> strides_;
  DTypeRef             dtype_;
  DeviceRef            device_;
  bool                 has_dtype_  = false;
  bool                 has_device_ = false;
};

} // namespace logssms::host
