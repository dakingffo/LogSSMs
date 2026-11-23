#include <limits>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <map>
#include <list>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace logssms {
namespace {

using Int = int64_t;
using Vector = std::vector<Int>;
using List = tvm::ffi::Array<Int>;
using Rows = tvm::ffi::Array<List>;
using Result = tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>;

void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

const Int* cpu_ints(tvm::ffi::TensorView tensor) {
  check(tensor.ndim() == 1 && tensor.is_contiguous()
        && tensor.device().device_type == kDLCPU && tensor.dtype().code == kDLInt
        && tensor.dtype().bits == 64 && tensor.dtype().lanes == 1,
        "Planning metadata must be contiguous CPU int64");
  return static_cast<const Int*>(tensor.data_ptr());
}

struct Entry {
  Int slot, begin, end, span;
  bool eligible;
};

std::vector<Entry> entries(const List& row) {
  check(row.size() % 5 == 0, "Invalid LogState entry metadata");
  std::vector<Entry> result;
  result.reserve(row.size() / 5);
  for (size_t i = 0; i < row.size(); i += 5) {
    result.push_back({row[i], row[i + 1], row[i + 2], row[i + 3], bool(row[i + 4])});
  }
  return result;
}

Rows batch_branch_points(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs, Int granularity) {
  check(granularity > 0, "Invalid LogState granularity");
  std::vector<Int> order(inputs.size()), lengths(inputs.size());
  std::vector<const int32_t*> ints32(inputs.size());
  std::vector<const Int*> ints64(inputs.size());
  std::iota(order.begin(), order.end(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& t = inputs[i];
    check(t.ndim() == 1 && t.is_contiguous() && t.device().device_type == kDLCPU
          && t.dtype().code == kDLInt && (t.dtype().bits == 32 || t.dtype().bits == 64),
          "Branch tokens must be contiguous 1-D CPU int32/int64 tensors");
    lengths[i] = t.size(0) / granularity * granularity;
    if (t.dtype().bits == 32) {
      ints32[i] = static_cast<const int32_t*>(t.data_ptr());
    }
    else {
      ints64[i] = static_cast<const Int*>(t.data_ptr());
    }
  }
  auto token = [&](Int i, Int j) -> Int {
    return ints32[i] ? ints32[i][j] : ints64[i][j];
  };
  std::sort(order.begin(), order.end(), [&](Int a, Int b) {
    for (Int j = 0, n = std::min(lengths[a], lengths[b]); j < n; ++j) {
      const auto x = token(a, j), y = token(b, j);
      if (x != y) {
        return x < y;
      }
    }
    return lengths[a] != lengths[b] ? lengths[a] < lengths[b] : a < b;
  });
  std::vector<std::pair<Int, Int>> stack;
  Rows result;
  for (Int i = 0; i < static_cast<Int>(order.size()); ++i) {
    Int depth = 0, begin = i;
    if (i + 1 < static_cast<Int>(order.size())) {
      const auto a = order[i], b = order[i + 1];
      while (depth < std::min(lengths[a], lengths[b]) && token(a, depth) == token(b, depth)) {
        ++depth;
      }
      depth = depth / granularity * granularity;
    }
    while (!stack.empty() && stack.back().first > depth) {
      const auto branch_depth = stack.back().first;
      begin = stack.back().second;
      stack.pop_back();
      Vector requests(order.begin() + begin, order.begin() + i + 1);
      std::sort(requests.begin(), requests.end());
      requests.insert(requests.begin(), branch_depth);
      result.push_back(List(requests.begin(), requests.end()));
    }
    if (depth && (stack.empty() || stack.back().first < depth)) {
      stack.emplace_back(depth, begin);
    }
  }
  return result;
}

Rows plan_materialize(const Rows& paths, const Rows& boundaries,
                      const Rows& candidates, Int granularity, Int log_count) {
  check(paths.size() == boundaries.size() && granularity > 0 && log_count > 0, "Invalid materialize metadata");
  struct PathIndex {
    std::vector<Entry> values;
    Vector checkpoints, boundaries;
    std::vector<std::pair<Int, Int>> logs;
  };
  std::vector<PathIndex> unpacked(paths.size());
  for (size_t p = 0; p < paths.size(); ++p) {
    auto& index = unpacked[p];
    index.values = entries(paths[p]);
    index.boundaries.assign(boundaries[p].begin(), boundaries[p].end());
    std::sort(index.boundaries.begin(), index.boundaries.end());
    std::vector<std::pair<Int, Int>> logs;
    for (const auto& value : index.values) {
      if (value.span >= 0) {
        index.checkpoints.push_back(value.end);
      } else {
        logs.emplace_back(value.begin, value.end);
      }
    }
    std::sort(index.checkpoints.begin(), index.checkpoints.end());
    std::sort(logs.begin(), logs.end());
    for (const auto& interval : logs) {
      if (!index.logs.empty() && interval.first <= index.logs.back().second) {
        index.logs.back().second = std::max(index.logs.back().second, interval.second);
      } else {
        index.logs.push_back(interval);
      }
    }

  }
  Rows result;
  for (Int candidate = 0; candidate < static_cast<Int>(candidates.size()); ++candidate) {
    const auto& row = candidates[candidate];
    check(row.size() == 3 && row[0] >= 0 && row[0] < static_cast<Int>(paths.size()),
          "Invalid materialize candidate path");
    const auto& index = unpacked[row[0]];
    check(row[1] >= 0 && row[1] < static_cast<Int>(index.values.size()), "Invalid candidate entry");
    const auto& entry = index.values[row[1]];
    if (!entry.eligible || entry.span >= 0
        || entry.end - entry.begin != granularity) {
      continue;
    }
    const auto boundary = std::upper_bound(index.boundaries.begin(), index.boundaries.end(), entry.begin);
    if (boundary != index.boundaries.end() && *boundary < entry.end) {
      continue;
    }
    if (!row[2]) {
      result.push_back(List({candidate}));
      continue;
    }
    auto checkpoint = std::upper_bound(index.checkpoints.begin(), index.checkpoints.end(), entry.begin);
    const Int base = checkpoint == index.checkpoints.begin() ? 0 : *std::prev(checkpoint);
    auto log = std::upper_bound(
        index.logs.begin(), index.logs.end(), base,
        [](Int position, const auto& interval) { return position < interval.first; });
    if (log != index.logs.begin() && std::prev(log)->second >= entry.end) {
      result.push_back(List({candidate}));
    }
  }
  if (log_count == 1) {
    return result;
  }
  Rows grouped;
  Vector group;
  Int path = -1, previous_end = -1;
  for (const auto& accepted : result) {
    const auto candidate = accepted[0];
    const auto& source = candidates[candidate];
    const auto& entry = unpacked[source[0]].values[source[1]];
    const auto& index = unpacked[source[0]];
    if (source[0] != path || entry.begin != previous_end
        || std::binary_search(index.boundaries.begin(), index.boundaries.end(), entry.begin)) {
      group.clear();
    }
    path = source[0];
    previous_end = entry.end;
    group.push_back(candidate);
    if (static_cast<Int>(group.size()) == log_count) {
      grouped.push_back(List(group.begin(), group.end()));
      group.clear();
    }
  }
  return grouped;
}


Rows read_rows(tvm::ffi::TensorView tensor) {
  const auto data = cpu_ints(tensor);
  Rows result;
  for (Int i = 0; i < tensor.size(0);) {
    const auto size = data[i++];
    check(size >= 0 && size <= tensor.size(0) - i, "Invalid packed planning row");
    result.push_back(List(data + i, data + i + size));
    i += size;
  }
  return result;
}

tvm::ffi::Tensor make_int_tensor(Int size) {
  return tvm::ffi::Tensor::FromEnvAlloc(
      TVMFFIEnvTensorAlloc, tvm::ffi::ShapeView(&size, 1),
      DLDataType{kDLInt, 64, 1}, DLDevice{kDLCPU, 0});
}

tvm::ffi::Tensor write_rows(const Rows& rows) {
  Int size = 0;
  for (const auto& row : rows) {
    size += 1 + row.size();
  }
  auto result = make_int_tensor(size);
  auto data = static_cast<Int*>(result.data_ptr());
  for (const auto& row : rows) {
    *data++ = row.size();
    data = std::copy(row.begin(), row.end(), data);
  }
  return result;
}

Rows branch_protections(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs,
                        const Rows& forced, Int granularity) {

  std::map<Vector, std::set<Int>> candidates;
  auto prefix = [&](Int request, Int position) {
    check(request >= 0 && request < static_cast<Int>(inputs.size()), "Invalid branch request");
    const auto& tensor = inputs[request];
    check(position >= 0 && position <= tensor.size(0), "Invalid branch endpoint");
    Vector key(position);
    for (Int i = 0; i < position; ++i) {
      key[i] = tensor.dtype().bits == 32 ? static_cast<const int32_t*>(tensor.data_ptr())[i]
                                        : static_cast<const Int*>(tensor.data_ptr())[i];
    }
    return key;
  };
  for (const auto& row : batch_branch_points(inputs, granularity)) {
    auto& requests = candidates[prefix(row[1], row[0])];
    requests.insert(row.begin() + 1, row.end());
  }
  for (const auto& row : forced) {
    check(row.size() == 2 && row[1] % granularity == 0, "Invalid forced branch");
    candidates[prefix(row[0], row[1])].insert(row[0]);
  }
  std::vector<Vector> sorted;
  for (const auto& [key, requests] : candidates) {
    if (!key.empty()) {
      Vector row{static_cast<Int>(key.size())};
      row.insert(row.end(), requests.begin(), requests.end());
      sorted.push_back(std::move(row));
    }
  }
  std::sort(sorted.begin(), sorted.end());
  Rows result;
  for (const auto& row : sorted) {
    result.push_back(List(row.begin(), row.end()));
  }
  return result;
}

constexpr Int kEvictionBlocked = 1;

template <typename NodeRows>
class EvictionQueue {
public:
  explicit EvictionQueue(const NodeRows& nodes) : nodes_(nodes) { reset(); }

  void reset() {
    children_.assign(nodes_.size(), 0);
    alive_.assign(nodes_.size(), true);
    indices_.clear();
    leaves_.clear();
    for (Int i = 0; i < static_cast<Int>(nodes_.size()); ++i) {
      if (nodes_[i].empty()) {
        alive_[i] = false;
        continue;
      }
      check(nodes_[i].size() == 4 && nodes_[i][2] >= 0 && nodes_[i][2] <= kEvictionBlocked,
            "Invalid eviction node");
      check(indices_.emplace(nodes_[i][0], i).second, "Duplicate eviction node");
    }
    for (const auto& node : nodes_) {
      if (!node.empty() && indices_.contains(node[1])) {
        ++children_[indices_.at(node[1])];
      }
    }
    for (Int i = 0; i < static_cast<Int>(nodes_.size()); ++i) {
      if (alive_[i] && !children_[i] && nodes_[i][2] != kEvictionBlocked) {
        push(i);
      }
    }
  }

  Int take(Int index) {
    check(leaves_.erase(index) == 1, "Eviction node is not an unlocked leaf");
    alive_[index] = false;
    const auto parent = indices_.find(nodes_[index][1]);
    if (parent != indices_.end() && !--children_[parent->second] && nodes_[parent->second][2] != kEvictionBlocked) {
      push(parent->second);
      return parent->second;
    }
    return -1;
  }

  void update(Int index) {
    if (alive_[index] && !children_[index] && nodes_[index][2] != kEvictionBlocked)
      push(index);
    else leaves_.erase(index);
  }
  bool is_leaf(Int index) const { return leaves_.contains(index); }
  bool can_materialize(Int index) const { return alive_.at(index) && nodes_[index][2] == 0; }
  const std::unordered_set<Int>& leaves() const { return leaves_; }

private:
  void push(Int index) {
    leaves_.insert(index);
  }

  const NodeRows& nodes_;
  Vector children_;
  std::vector<bool> alive_;
  std::unordered_map<Int, Int> indices_;
  std::unordered_set<Int> leaves_;
};

class EvictionPlanner {
  using NodeRows = std::vector<Vector>;
  struct Density {
    double tail = 0, atom = 0, g = 1;
    bool default_order = false;
    std::vector<double> probability{0.};
    std::vector<std::pair<double, double>> past_tail_prefix;
    double prefix(double position) const {
      auto i = static_cast<Int>(std::floor((tail - position) / g));
      double mass = probability[std::clamp<Int>(i, 0, probability.size() - 1)];
      if (!past_tail_prefix.empty()) {
        const auto end = std::lower_bound(past_tail_prefix.begin(), past_tail_prefix.end(),
                                          position, [](const auto &point, double value) {
                                            return point.first < value;
                                          });
        const auto before = end == past_tail_prefix.begin() ? 0. : std::prev(end)->second;
        mass += past_tail_prefix.back().second - before;
      }
      return mass;
    }
    double mass(double begin, double end) const {
      return std::max(0., prefix(begin) - prefix(end));
    }
  };
  struct Sum {
    double value = 0., error = 0.;
    void add(double x) {
      const double result = value + x;
      error += std::abs(value) >= std::abs(x) ? (value - result) + x : (x - result) + value;
      value = result;
    }
    double total() const { return error && std::isfinite(error) ? value + error : value; }
  };

  struct ActiveDensity {
    Int tail, g, bins;
    std::vector<double> prefix{0.};
    std::vector<std::pair<Int, double>> past;

    ActiveDensity(const double *p, const double *end) : tail(p[0]), g(p[1]), bins(p[2]) {
      check(g > 0 && bins >= 1 && p[7] >= 0 && p[7] <= (end - p - 8) / 2,
            "Invalid active density");
      const auto prior = p[6];
      Sum total, past_mass;
      std::vector<double> masses;
      masses.reserve(bins);
      for (Int j = 1; j <= bins; ++j) {
        const double mass = prior > 0. ? 1. / (j * j) : g * p[3] * std::pow(j * g, -p[4]);
        masses.push_back(mass);
        total.add(mass);
      }
      for (Int j = 0; j < static_cast<Int>(p[7]); ++j) {
        const auto pos = p[8 + 2 * j], probability = p[9 + 2 * j];
        past.emplace_back(pos, probability);
        past_mass.add(probability);
      }
      const auto scale = prior > 0. ? prior / total.total() :
          std::min(1., std::max(0., 1. - p[5] - past_mass.total()) /
                       std::max(1e-300, total.total()));
      prefix.reserve(bins + 1);
      for (auto mass : masses)
        prefix.push_back(prefix.back() + mass * scale);
    }

    double mass(Int begin, Int end) const {
      auto bucket = [&](Int position) {
        return std::min(bins, std::max(Int{0}, tail - position) / g);
      };
      Sum atoms;
      for (const auto &[position, probability] : past)
        if (begin <= position && position < end) atoms.add(probability);
      return prefix[bucket(begin)] - prefix[bucket(end)] + atoms.total();
    }
  };

  struct ActiveCandidate {
    const Int *begin, *end;
    Int frames;
    double cost;
    double score() const { return cost / frames; }
  };

  static std::vector<ActiveCandidate> active_costs(tvm::ffi::TensorView geometry,
                                                   tvm::ffi::TensorView density) {
    const auto *packed = cpu_ints(geometry);
    const auto size = geometry.size(0);
    if (!size)
      return {};
    check(size >= 2 && packed[0] >= 0 && packed[0] <= (size - 2) / 8 &&
              packed[1] >= 0 && packed[1] <= size - 2 - 8 * packed[0],
          "Invalid active reclamation geometry");
    check(density.ndim() == 1 && density.is_contiguous() &&
              density.device().device_type == kDLCPU && density.dtype().code == kDLFloat &&
              density.dtype().bits == 64,
          "Invalid active density parameters");
    const auto *cursor = static_cast<const double *>(density.data_ptr());
    const auto *model_end = cursor + density.size(0);
    std::vector<ActiveDensity> models;
    while (cursor < model_end) {
      check(model_end - cursor >= 8, "Truncated active density");
      models.emplace_back(cursor, model_end);
      cursor += 8 + 2 * static_cast<Int>(cursor[7]);
    }
    const auto count = packed[0];
    const auto *dep = packed + 2 + count * 8;
    const auto *dep_end = dep + packed[1];
    const auto *decay = dep_end;
    std::vector<ActiveCandidate> result;
    result.reserve(count);
    for (Int i = 0; i < count; ++i) {
      const auto *row = packed + 2 + i * 8;
      check(row[0] >= 0 && row[0] < static_cast<Int>(models.size()) &&
                row[6] >= dep - (packed + 2 + count * 8) && row[6] <= packed[1] &&
                row[7] % 2 == 0 && row[7] >= decay - dep_end && row[7] <= packed + size - dep_end,
            "Invalid active reclamation offsets");
      const auto &model = models[row[0]];
      Int frames = row[5];
      for (; decay < dep_end + row[7]; decay += 2)
        frames += decay[0] == decay[1];
      check(frames > 0, "Invalid active reclamation capacity");
      double cost = 0.;
      if (row[4]) {
        Sum increments;
        for (auto x = row[1] + model.g; x < row[3]; x += model.g)
          increments.add((x - row[1]) * model.mass(x, x + model.g));
        cost = row[4] / 64. + increments.total();
      } else {
        cost = (row[2] - row[1]) * model.mass(row[2], row[3]);
      }
      const auto *next = packed + 2 + count * 8 + row[6];
      result.push_back({dep, next, frames, cost});
      dep = next;
    }
    check(dep == dep_end && decay == packed + size, "Unexpected active reclamation metadata");
    return result;
  }

  struct Candidate {
    double phase, score;
    Int timestamp, id, kind, version;
    double cost_score = 0.;
    Int depth = 0, owner = 0, begin = 0;
    bool operator>(const Candidate &o) const {
      return std::tie(phase, score, timestamp, kind, depth, owner, begin, id) >
             std::tie(o.phase, o.score, o.timestamp, o.kind, o.depth, o.owner,
                      o.begin, o.id);
    }
  };
  class CandidateQueue
      : public std::priority_queue<Candidate, std::vector<Candidate>,
                                   std::greater<Candidate>> {
    using Base = std::priority_queue<Candidate, std::vector<Candidate>,
                                     std::greater<Candidate>>;
    bool collecting = false;

  public:
    void begin_update() {
      this->c.clear();
      collecting = true;
    }
    void push(const Candidate &candidate) {
      if (collecting)
        this->c.push_back(candidate);
      else
        Base::push(candidate);
    }
    void finish_update() {
      std::make_heap(this->c.begin(), this->c.end(), this->comp);
      collecting = false;
    }
  };

  static constexpr size_t kSlotRecordSize = 6;
  struct SlotRecord {
    Int frames, decay, begin, end;
    bool checkpoint;
  };

  template <typename Row>
  static SlotRecord read_slot(const Row &row, size_t offset = 0) {
    check(row.size() >= offset + kSlotRecordSize, "Truncated eviction slot");
    const auto frames = row[offset + 1], decay = row[offset + 2], kind = row[offset + 5];
    check(frames >= 0 && (kind == 1 || (kind == 0 && decay >= 0)), "Invalid eviction slot");
    return {frames, decay, row[offset + 3], row[offset + 4], kind == 1};
  }

  NodeRows nodes, kv, node_slots;
  List config;
  EvictionQueue<NodeRows> queue;
  std::vector<double> model_storage;
  const double *model_data = nullptr;
  std::unordered_map<Int, SlotRecord> slot_info;
  std::unordered_map<Int, Vector> actions;
  std::unordered_map<Int, Int> decay_occupied, decay_released, pages,
      page_delta;
  Int free = 0, decay = 0, kv_free = 0;
  Vector selected, released, page_order, compressed;
  std::unordered_set<Int> released_set;
  std::unordered_map<Int, Density> densities;
  Vector model_ids, node_version;
  std::vector<Vector> ordered_slots, order_keys;
  std::unordered_map<Int, Int> previous, slot_node, successors, action_version;
  std::unordered_map<Int, std::vector<Int>> kv_users, decay_users, owner_nodes;
  std::unordered_map<Int, Int> pending_depths;
  CandidateQueue candidates, default_candidates, leaf_candidates;
  NodeRows materializations;
  std::vector<size_t> materialization_starts;
  std::vector<Vector> node_materializations;

  bool fits() {
    const auto required_decay =
        (std::max(Int{0}, config[7] - decay) + config[5] - 1) / config[5];
    const auto log_frames = config[9];
    return config[6] + config[7] * log_frames + config[8] * config[2] +
               (config[7] * config[4] + config[8]) * config[3] +
               required_decay <=
           free + kv_free;
  }
  void release(Int slot) {
    check(slot_info.contains(slot), "Missing eviction slot");
    check(released_set.insert(slot).second, "Duplicate eviction slot release");
    const auto &record = slot_info.at(slot);
    released.push_back(slot);
    free += record.frames;
    if (record.decay >= 0) {
      check(decay_occupied.contains(record.decay),
            "Missing decay frame occupancy");
      if (++decay_released[record.decay] == decay_occupied.at(record.decay)) {
        ++free;
        decay -= config[5] - 1;
      } else {
        ++decay;
      }
    }
  }
  void prune(Int index) {
    selected.push_back(index);
    check(kv[index].size() % 3 == 0, "Invalid eviction KV references");
    for (size_t i = 0; i < kv[index].size(); i += 3) {
      const auto frame = kv[index][i];
      auto page = pages.try_emplace(frame, 0).first;
      if (!page_delta.contains(frame))
        page_order.push_back(frame);
      auto &count = page->second;
      count += kv[index][i + 1];
      page_delta[frame] += kv[index][i + 1];
      check(count <= kv[index][i + 2], "Eviction exceeds cached KV references");
      kv_free += count == kv[index][i + 2];
    }
    for (const auto slot : node_slots[index]) {
      if (released_set.contains(slot)) {
        continue;
      }
      check(slot_info.contains(slot), "Missing eviction slot");
      release(slot);
    }
  }
  tvm::ffi::Tensor result(const Vector *materialize = nullptr, Int active_action = -1,
                          const ActiveCandidate *active = nullptr) {
    std::sort(released.begin(), released.end());
    const auto action_size = materialize ? materialize->size() + 1 : active ? 5 : 0;
    auto packed = make_int_tensor(4 + selected.size() + 2 * page_order.size() +
                                  released.size() + compressed.size() + action_size);
    auto data = static_cast<Int *>(packed.data_ptr());
    auto write = [&](const Vector &row) {
      *data++ = row.size();
      data = std::copy(row.begin(), row.end(), data);
    };
    write(selected);
    *data++ = 2 * page_order.size();
    for (const auto frame : page_order) {
      *data++ = frame;
      *data++ = page_delta.at(frame);
    }
    write(released);
    write(compressed);
    if (materialize)
      write(*materialize);
    else if (active) {
      *data++ = 4;
      *data++ = -2;
      *data++ = active_action;
      *data++ = active->frames;
      *data = std::bit_cast<Int>(active->cost);
    }
    return packed;
  }
  Int successor(Int slot) {
    while (slot >= 0 && successors.contains(slot))
      slot = successors.at(slot);
    return slot;
  }
  double recovery_before(Int index, Int slot) {
    if (previous.contains(slot))
      return static_cast<double>(slot_info.at(previous.at(slot)).end);
    const auto end = slot_info.at(slot).end;
    double restored = model_data[index * 10 + 7];
    for (auto preceding : ordered_slots[index]) {
      if (released_set.contains(preceding))
        continue;
      const auto &record = slot_info.at(preceding);
      if (record.end >= end)
        break;
      if (record.checkpoint || record.begin == restored)
        restored = record.end;
    }
    return restored;
  }
  Int slot_credit(Int slot) {
    if (released_set.contains(slot))
      return Int{0};
    const auto &record = slot_info.at(slot);
    Int credit = record.frames;
    if (record.decay >= 0 &&
        decay_released[record.decay] + 1 == decay_occupied.at(record.decay))
      ++credit;
    return credit;
  }
  Int prune_credit(Int index) {
    Int credit = 0;
    std::unordered_map<Int, Int> decays;
    for (auto slot : node_slots[index])
      if (!released_set.contains(slot)) {
        const auto &record = slot_info.at(slot);
        credit += record.frames;
        if (record.decay >= 0)
          ++decays[record.decay];
      }
    for (const auto &[frame, count] : decays)
      credit += decay_released[frame] + count == decay_occupied.at(frame);
    for (size_t j = 0; j < kv[index].size(); j += 3) {
      const auto found = pages.find(kv[index][j]);
      const auto count = found == pages.end() ? 0 : found->second;
      credit += count + kv[index][j + 1] == kv[index][j + 2];
    }
    return credit;
  }
  void push_node(Int index) {
    if (index < 0 || !queue.is_leaf(index))
      return;
    const auto version = ++node_version[index];
    const auto &order = order_keys[index];
    Int checkpoints = 0;
    for (auto slot : node_slots[index]) {
      checkpoints += !released_set.contains(slot) && actions.contains(slot);
    }
    const auto *p = model_data + index * 10;
    const auto &density = densities.at(model_ids[index]);
    double cost = 0., restored = p[7];
    for (auto slot : ordered_slots[index]) {
      if (released_set.contains(slot))
        continue;
      const auto &record = slot_info.at(slot);
      if (record.checkpoint || record.begin == restored) {
        cost += std::max(0., record.end - restored) *
                (density.atom + density.prefix(record.end));
        restored = record.end;
      }
    }
    const auto credit = prune_credit(index);
    const auto score = credit ? cost / credit : std::numeric_limits<double>::infinity();
    leaf_candidates.push({checkpoints > 1 ? 1. : 0., score, nodes[index][3], index, 0,
                          version, score, order[0], order[1], order[2]});
  }
  void push_action(Int slot) {
    const auto version = ++action_version[slot];
    if (!actions.contains(slot) || released_set.contains(slot))
      return;
    const auto &action = actions.at(slot);
    const auto right = successor(action[1]);
    if (right < 0 || released_set.contains(right))
      return;
    const auto &left_record = slot_info.at(slot),
               &right_record = slot_info.at(right);
    const auto left_span = left_record.end - left_record.begin;
    const auto right_span = right_record.end - right_record.begin;
    const auto &density = densities.at(model_ids[action[3]]);
    const auto distance = left_record.end - recovery_before(action[3], slot);
    const auto cost = std::max(0., distance) * density.mass(left_record.end, right_record.end);
    const auto credit = slot_credit(slot);
    const auto score = credit ? cost / credit : std::numeric_limits<double>::infinity();
    if (density.default_order)
      default_candidates.push(
          {1., static_cast<double>(std::max(left_span, right_span)), action[2],
           slot, 1, version, score});
    else
      candidates.push({0., score, action[2], slot, 1, version, score});
  }
  bool valid(const Candidate &c) {
    if (c.kind == 0)
      return c.version == node_version[c.id] && queue.is_leaf(c.id);
    if (!actions.contains(c.id) || c.version != action_version[c.id] || released_set.contains(c.id))
      return false;
    const auto right = successor(actions.at(c.id)[1]);
    return right >= 0 && !released_set.contains(right);
  }

  std::optional<Candidate> pressure_materialization() {
    std::optional<Candidate> best;
    for (Int i = 0; i < static_cast<Int>(materializations.size()); ++i) {
      const auto &row = materializations[i];
      if (row.empty() || released_set.contains(row[0]) || released_set.contains(row.back()))
        continue;
      const auto start = materialization_starts[i];
      if (std::any_of(row.begin() + 2, row.begin() + start, [&](Int source) {
            return released_set.contains(-source - 2);
          }))
        continue;
      const auto index = slot_node.at(row.back());
      const auto begin = slot_info.at(row[start]).begin, end = slot_info.at(row.back()).end;
      const auto &density = densities.at(model_ids[index]);
      const bool absorb = row[1] < 0;
      bool feasible = true;
      Int credit = absorb ? 0 : -(config[2] + config[3]);
      // Reconstruction runs over cached rank updates instead of the model.
      // Express its launch and replay work in equivalent prefill tokens.
      const auto replay_begin = row[0] < 0 ? 0 : slot_info.at(row[0]).end;
      double cost = (end - replay_begin) / 64.;
      if (absorb) {
        const auto owner = slot_node.at(row[0]);
        if (nodes[owner][2] != 0)
          continue;
        cost += std::max(0., begin - recovery_before(owner, row[0])) * density.mass(begin, end);
      }
      std::unordered_map<Int, Int> records;
      for (size_t j = start; j < row.size(); ++j) {
        const auto slot = row[j];
        if (released_set.contains(slot) || nodes[slot_node.at(slot)][2] != 0) {
          feasible = false;
          break;
        }
        const auto &record = slot_info.at(slot);
        credit += record.frames;
        ++records[record.decay];
        if (record.end < end)
          cost += (record.end - begin) * density.mass(record.end, record.end + density.g);
      }
      if (!feasible || credit <= 0)
        continue;
      for (const auto &[frame, count] : records)
        credit += decay_released[frame] + count == decay_occupied.at(frame);
      const auto &order = order_keys[index];
      Candidate candidate{0., cost / credit, nodes[index][3], i, 2, 0,
                          cost / credit, order[0], order[1], order[2]};
      if (!best || *best > candidate)
        best = candidate;
    }
    return best;
  }

  template <typename Indices>
  void load_models(const Indices &indices, tvm::ffi::TensorView models) {
    check(models.ndim() == 1 && models.is_contiguous() &&
              models.device().device_type == kDLCPU &&
              models.dtype().code == kDLFloat && models.dtype().bits == 64,
          "Invalid eviction density models");
    const auto model_size = static_cast<Int>(indices.size() * 10);
    check(models.size(0) >= model_size && (models.size(0) - model_size) % 3 == 0,
          "Invalid eviction tail probabilities");
    const auto *input = static_cast<const double *>(models.data_ptr());
    for (size_t row = 0; row < indices.size(); ++row) {
      const auto index = indices[row];
      check(index >= 0 && index < static_cast<Int>(nodes.size()),
            "Invalid eviction model index");
      std::copy_n(input + row * 10, 10, model_storage.data() + index * 10);
    }
    densities.clear();
    for (const auto i : indices) {
      const auto *p = model_data + i * 10;
      const auto id = static_cast<Int>(p[0]);
      model_ids[i] = id;
      if (densities.contains(id))
        continue;
      Density d;
      d.default_order = p[9] != 0.;
      d.tail = p[2];
      d.atom = std::clamp(p[5], 0., 1.);
      d.g = p[6];
      check(d.g > 0 && std::isfinite(p[3]) && std::isfinite(p[4]),
            "Invalid hit density");
      const auto count =
          std::max<Int>(1, static_cast<Int>((d.tail - p[1]) / d.g));
      for (Int j = 1; j <= count; ++j) {
        const auto a = j * d.g;
        const auto mass = std::exp(p[4] - p[3] * std::log(a)) * d.g;
        d.probability.push_back(d.probability.back() + mass);
      }
      densities.emplace(id, std::move(d));
    }
    for (auto i = model_size; i < models.size(0); i += 3) {
      const auto owner = static_cast<Int>(input[i]);
      auto &density = densities.at(owner);
      const auto position = input[i + 1], probability = input[i + 2];
      auto &points = density.past_tail_prefix;
      check(position >= 0 && position < density.tail && probability > 0 &&
                std::isfinite(probability) && (points.empty() || points.back().first < position),
            "Invalid past tail probability");
      const auto previous = points.empty() ? 0. : points.back().second;
      points.emplace_back(position, previous + probability);
    }
    for (auto &[id, d] : densities) {
      const double past_mass = d.past_tail_prefix.empty() ? 0. : d.past_tail_prefix.back().second;
      const auto scale =
          std::min(1., std::max(0., 1. - d.atom - past_mass) /
                        std::max(1e-300, d.probability.back()));
      for (auto &v : d.probability)
        v *= scale;
    }
  }

  size_t materialization_start(const Vector &row) const {
    check(row.size() >= 3 &&
              (row[1] < 0 || slot_node.contains(row[1])) &&
              slot_node.contains(row.back()) &&
              (row[0] < 0 || (slot_info.contains(row[0]) && slot_info.at(row[0]).checkpoint)),
          "Invalid pressure materialization");
    check(row[1] >= 0 || (row[0] >= 0 && slot_node.at(row[0]) == slot_node.at(row.back())),
          "Log tail absorption must stay inside one node");
    auto cursor = row[0] < 0 ? 0 : slot_info.at(row[0]).end;
    size_t start = 2;
    for (; start < row.size() && row[start] < 0; ++start) {
      const auto &record = slot_info.at(-row[start] - 2);
      check(row[1] >= 0 && !record.checkpoint && record.begin == cursor,
            "Invalid ancestor Log recovery");
      cursor = record.end;
    }
    for (size_t j = start; j < row.size(); ++j) {
      const auto &record = slot_info.at(row[j]);
      check(!record.checkpoint && record.begin == cursor, "Noncontiguous pressure materialization");
      cursor = record.end;
    }
    if (row[1] >= 0) {
      const auto &tail = slot_info.at(row[1]);
      check(!tail.checkpoint && tail.end >= cursor, "Invalid pressure materialization tail boundary");
    }
    return start;
  }

public:
  EvictionPlanner(std::array<NodeRows, 7> groups, const List &capacity,
                  tvm::ffi::TensorView models, const Rows &order)
      : nodes(std::move(groups[0])), kv(std::move(groups[1])),
        node_slots(std::move(groups[2])), config(capacity), queue(nodes),
        node_version(nodes.size()), ordered_slots(nodes.size()),
        materializations(std::move(groups[6])) {
    auto &slots = groups[3];
    const auto &occupied = groups[4];
    auto &checkpoint_actions = groups[5];
    for (const auto &row : order)
      order_keys.emplace_back(row.begin(), row.end());
    check(order_keys.size() == nodes.size(),
          "Invalid eviction ordering metadata");
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i)
      owner_nodes[order_keys[i][1]].push_back(i);
    check(nodes.size() == kv.size() && nodes.size() == node_slots.size() &&
              config.size() == 10 && config[5] > 0,
          "Invalid eviction capacity metadata");
    for (auto &row : slots) {
      const auto record = read_slot(row);
      slot_info.emplace(row[0], record);
    }
    for (const auto &row : occupied) {
      check(row.size() == 2 && row[1] > 0 && row[1] <= config[5],
            "Invalid decay occupancy");
      decay_occupied.emplace(row[0], row[1]);
    }
    model_ids.resize(nodes.size());
    model_storage.resize(nodes.size() * 10);
    model_data = model_storage.data();
    Vector indices(nodes.size());
    std::iota(indices.begin(), indices.end(), 0);
    load_models(indices, models);
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      bool ordered = true;
      auto end = std::numeric_limits<Int>::min();
      for (auto slot : node_slots[i]) {
        slot_node.emplace(slot, i);
        const auto next_end = slot_info.at(slot).end;
        ordered &= end < next_end;
        end = next_end;
      }
      auto &ordered_row = ordered_slots[i];
      ordered_row = node_slots[i];
      if (!ordered)
        std::sort(ordered_row.begin(), ordered_row.end(), [&](Int a, Int b) {
          return slot_info.at(a).end < slot_info.at(b).end;
        });
    }
    node_materializations.resize(nodes.size());
    for (size_t id = 0; id < materializations.size(); ++id) {
      const auto &row = materializations[id];
      materialization_starts.push_back(materialization_start(row));
      node_materializations[slot_node.at(row.back())].push_back(id);
    }
    for (auto &row : checkpoint_actions) {
      check(row.size() >= 4, "Invalid checkpoint eviction action");
      check(slot_node.contains(row[0]) && slot_node.at(row[0]) == row[3],
            "Checkpoint action has an invalid owner");
      check(row[1] < 0 || (slot_node.contains(row[1]) &&
                           slot_node.at(row[0]) == slot_node.at(row[1])),
            "Checkpoint merging must stay inside one node");
      const auto slot = row[0], right = row[1];
      actions.emplace(slot, std::move(row));
      if (right >= 0)
        previous[right] = slot;
    }
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      for (size_t j = 0; j < kv[i].size(); j += 3)
        kv_users[kv[i][j]].push_back(i);
      for (auto slot : node_slots[i])
        if (slot_info.at(slot).decay >= 0)
          decay_users[slot_info.at(slot).decay].push_back(i);
    }
    for (const auto index : queue.leaves())
      push_node(index);
    for (const auto &[slot, action] : actions)
      push_action(slot);
  }
  bool replace_materialized_node(Int index, Int timestamp, const Rows &records,
                                 const Rows &checkpoint_rows, const Rows &frontiers,
                                 const Rows &groups) {
    if (!pages.empty() || !queue.can_materialize(index))
      return false;

    remove_node_slots(index);
    install_node_slots(index, records);
    install_checkpoint_actions(checkpoint_rows);
    size_t first = 0;
    for (const auto &row : frontiers) {
      check(row.size() == 3 && row[0] >= 0 && row[0] < static_cast<Int>(nodes.size()) &&
                row[2] >= 0 && row[2] <= static_cast<Int>(groups.size() - first),
            "Invalid replacement recovery frontier");
      const auto owner = row[0];
      set_materializations(owner, groups, first, row[2]);
      first += row[2];
      if (model_storage[owner * 10 + 7] != row[1]) {
        model_storage[owner * 10 + 7] = row[1];
        push_node(owner);
        for (const auto slot : node_slots[owner])
          if (actions.contains(slot)) push_action(slot);
      }
    }
    check(first == groups.size(), "Unexpected replacement materializations");

    nodes[index][3] = timestamp;
    push_node(index);
    for (const auto slot : node_slots[index])
      if (actions.contains(slot))
        push_action(slot);
    return true;
  }

  void remove_node_slots(Int index) {
    auto &slots = node_slots[index];
    for (const auto slot : slots) {
      const auto frame = slot_info.at(slot).decay;
      if (frame >= 0)
        std::erase(decay_users.at(frame), index);
      ++action_version[slot];
      actions.erase(slot);
      successors.erase(slot);
      previous.erase(slot);
      released_set.erase(slot);
      slot_node.erase(slot);
      slot_info.erase(slot);
    }
    slots.clear();
    ordered_slots[index].clear();
  }

  template <typename Records>
  void install_node_slots(Int index, const Records &records) {
    auto &slots = node_slots[index];
    for (const auto &row : records) {
      const auto record = read_slot(row);
      const auto slot = row[0], frame = record.decay;
      check(!slot_node.contains(slot),
            "Invalid replacement slot ownership");
      slots.push_back(slot);
      slot_node.emplace(slot, index);
      slot_info.emplace(slot, record);
      if (frame >= 0)
        decay_users[frame].push_back(index);
    }
    ordered_slots[index] = slots;
  }

  template <typename Checkpoints>
  void install_checkpoint_actions(const Checkpoints &checkpoint_rows) {
    for (const auto &row : checkpoint_rows) {
      check(row.size() == 4 && slot_node.at(row[0]) == row[3] &&
                (row[1] < 0 || slot_node.at(row[1]) == row[3]),
            "Invalid replacement checkpoint action");
      actions.emplace(row[0], Vector(row.begin(), row.end()));
      if (row[1] >= 0)
        previous[row[1]] = row[0];
    }
  }

  void set_materializations(Int index, const Rows &groups, size_t first, size_t count) {
    auto &ids = node_materializations[index];
    while (ids.size() < count) {
      ids.push_back(materializations.size());
      materializations.emplace_back();
      materialization_starts.push_back(0);
    }
    for (size_t i = 0; i < count; ++i) {
      const auto id = ids[i];
      const auto &source = groups[first + i];
      auto &row = materializations[id];
      row.assign(source.begin(), source.end());
      materialization_starts[id] = materialization_start(row);
      check(slot_node.at(row.back()) == index, "Invalid replacement materialization owner");
    }
    for (size_t i = count; i < ids.size(); ++i)
      materializations[ids[i]].clear();
  }

  void rebase(Int size, const List &removed, const Rows &rows, const List &indices,
              tvm::ffi::TensorView models, const Rows &order) {
    check(size >= static_cast<Int>(nodes.size()) && !rows.empty() &&
              order.size() == indices.size(),
          "Invalid eviction rebase");
    const auto &header = rows[0];
    check(header.size() == 5, "Invalid eviction rebase header");
    const auto nodes_count = header[0], actions_count = header[1], groups_count = header[2];
    const auto occupancy_count = header[3], kv_count = header[4];
    check(nodes_count >= 0 && actions_count >= 0 && groups_count >= 0 &&
              occupancy_count >= 0 && kv_count >= 0 &&
              1 + 4 * nodes_count + actions_count + groups_count + occupancy_count + kv_count ==
                  static_cast<Int>(rows.size()),
          "Invalid eviction rebase metadata");
    nodes.resize(size);
    node_slots.resize(size);
    ordered_slots.resize(size);
    node_materializations.resize(size);
    kv.resize(size);
    order_keys.resize(size);
    node_version.resize(size);
    model_ids.resize(size);
    model_storage.resize(size * 10);
    model_data = model_storage.data();

    auto clear_node = [&](Int index) {
      check(index >= 0 && index < size, "Invalid eviction rebase node");
      remove_node_slots(index);
      for (size_t j = 0; j < kv[index].size(); j += 3)
        std::erase(kv_users.at(kv[index][j]), index);
      kv[index].clear();
      nodes[index].clear();
      set_materializations(index, {}, 0, 0);
    };
    for (const auto index : removed)
      clear_node(index);
    for (Int i = 0; i < nodes_count; ++i) {
      const auto &row = rows[1 + i * 4];
      check(row.size() == 4, "Invalid replacement topology");
      clear_node(row[0]);
    }
    for (Int i = 0; i < nodes_count; ++i) {
      const auto first = 1 + i * 4;
      const auto &row = rows[first];
      const auto index = row[0];
      nodes[index].assign(row.begin(), row.end());
      const auto &kv_row = rows[first + 1];
      check(kv_row.size() % 3 == 0, "Invalid replacement KV metadata");
      kv[index].assign(kv_row.begin(), kv_row.end());
      for (size_t j = 0; j < kv_row.size(); j += 3)
        kv_users[kv_row[j]].push_back(index);
      const auto &slot_row = rows[first + 2];
      check(slot_row.size() % kSlotRecordSize == 0, "Invalid replacement slot metadata");
      for (size_t j = 0; j < slot_row.size(); j += kSlotRecordSize) {
        const auto record = read_slot(slot_row, j);
        const auto slot = slot_row[j], frame = record.decay;
        check(slot_node.emplace(slot, index).second, "Duplicate replacement slot");
        node_slots[index].push_back(slot);
        slot_info.emplace(slot, record);
        if (frame >= 0)
          decay_users[frame].push_back(index);
      }
      ordered_slots[index] = node_slots[index];
    }
    const auto action_first = 1 + 4 * nodes_count;
    const auto group_first = action_first + actions_count;
    const auto occupied_first = group_first + groups_count;
    const auto kv_first = occupied_first + occupancy_count;
    install_checkpoint_actions(Rows(rows.begin() + action_first, rows.begin() + group_first));
    size_t group = group_first;
    for (Int i = 0; i < nodes_count; ++i) {
      const auto &count = rows[4 + i * 4];
      check(count.size() == 1 && count[0] >= 0 &&
                count[0] <= static_cast<Int>(occupied_first - group),
            "Invalid replacement group count");
      set_materializations(rows[1 + i * 4][0], rows, group, count[0]);
      group += count[0];
    }
    check(group == static_cast<size_t>(occupied_first), "Unexpected replacement groups");
    decay_occupied.clear();
    for (Int i = occupied_first; i < kv_first; ++i) {
      const auto &row = rows[i];
      decay_occupied.emplace(row[0], row[1]);
    }
    for (Int i = kv_first; i < static_cast<Int>(rows.size()); ++i) {
      const auto &row = rows[i];
      const auto found = kv_users.find(row[0]);
      if (found == kv_users.end())
        continue;
      for (const auto index : found->second)
        for (size_t j = 0; j < kv[index].size(); j += 3)
          if (kv[index][j] == row[0]) {
            kv[index][j + 2] = row[1];
            break;
          }
    }
    pages.clear();
    decay_released.clear();
    released_set.clear();
    successors.clear();
    previous.clear();
    for (const auto &[slot, action] : actions)
      if (action[1] >= 0) previous[action[1]] = slot;
    pending_depths.clear();
    owner_nodes.clear();
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto index = indices[i];
      const auto &row = order[i];
      order_keys[index].assign(row.begin(), row.end());
      owner_nodes[row[1]].push_back(index);
    }
    queue.reset();
    refresh_models(indices, models);
  }

  void update_locks(Int num_blocked, const Rows &rows) {
    check(num_blocked > 0 && num_blocked <= static_cast<Int>(rows.size()),
          "Invalid eviction lock update");
    for (Int i = 0; i < num_blocked; ++i) {
      const auto &row = rows[i];
      check(row.size() == 2 && row[0] >= 0 && row[0] < static_cast<Int>(nodes.size()),
            "Invalid eviction lock metadata");
      const auto index = row[0];
      nodes[index][2] = row[1];
      queue.update(index);
      for (const auto slot : node_slots[index]) {
        ++action_version[slot];
        actions.erase(slot);
        previous.erase(slot);
        successors.erase(slot);
      }
    }
    for (Int i = num_blocked; i < static_cast<Int>(rows.size()); ++i) {
      const auto &row = rows[i];
      check(row.size() == 4 && slot_node.contains(row[0]) && slot_node.at(row[0]) == row[3],
            "Invalid checkpoint lock metadata");
      actions.emplace(row[0], Vector(row.begin(), row.end()));
      if (row[1] >= 0) previous[row[1]] = row[0];
    }
    for (Int i = 0; i < num_blocked; ++i) {
      const auto index = rows[i][0];
      push_node(index);
      for (const auto slot : node_slots[index])
        if (actions.contains(slot)) push_action(slot);
    }
  }
  void refresh_models(const List &indices, tvm::ffi::TensorView models) {
    load_models(indices, models);
    for (const auto &[owner, depth] : pending_depths)
      for (const auto index : owner_nodes[owner])
        order_keys[index][0] = depth;
    pending_depths.clear();
    candidates.begin_update();
    default_candidates.begin_update();
    leaf_candidates.begin_update();
    for (const auto index : queue.leaves())
      push_node(index);
    for (const auto &[slot, action] : actions)
      push_action(slot);
    candidates.finish_update();
    default_candidates.finish_update();
    leaf_candidates.finish_update();
  }
  template <typename Occupancy>
  tvm::ffi::Tensor plan(const List &capacity, const Occupancy &occupancy,
                       tvm::ffi::TensorView active_geometry, tvm::ffi::TensorView active_models) {
    check(capacity.size() == 10, "Invalid eviction capacity metadata");
    config = capacity;
    free = config[0];
    decay = config[1];
    kv_free = 0;
    selected.clear();
    released.clear();
    compressed.clear();
    page_order.clear();
    page_delta.clear();
    for (const auto &row : occupancy) {
      const auto frame = row[0], count = row[1] + decay_released[frame];
      if (decay_occupied.at(frame) == count)
        continue;
      decay_occupied[frame] = count;
      for (const auto user : decay_users[frame]) {
        push_node(user);
        for (const auto slot : node_slots[user])
          if (!released_set.contains(slot) && actions.contains(slot))
            push_action(slot);
      }
    }
    for (const auto &[owner, depth] : pending_depths)
      for (const auto index : owner_nodes[owner]) {
        order_keys[index][0] = depth;
        push_node(index);
      }
    pending_depths.clear();
    if (fits())
      return result();
    const auto active = active_costs(active_geometry, active_models);
    std::optional<Candidate> active_action;
    auto valid_active = [&](const ActiveCandidate &candidate) {
      return std::none_of(candidate.begin, candidate.end, [&](Int slot) {
        return released_set.contains(slot);
      });
    };
    auto discard_stale = [&](CandidateQueue &heap) {
      while (!heap.empty() && !valid(heap.top()))
        heap.pop();
    };
    while (true) {
      const auto materialize = pressure_materialization();
      discard_stale(candidates);
      discard_stale(default_candidates);
      if (!active_action || !valid_active(active[active_action->id])) {
        active_action.reset();
        for (Int i = 0; i < static_cast<Int>(active.size()); ++i) {
          if (!valid_active(active[i]))
            continue;
          const auto score = active[i].score();
          Candidate candidate{0., score, std::numeric_limits<Int>::max(), i, 3, 0, score};
          if (!active_action || *active_action > candidate)
            active_action = candidate;
        }
      }
      auto conversion = materialize;
      if (active_action && (!conversion || active_action->score < conversion->score))
        conversion = active_action;
      CandidateQueue *selected_queue = nullptr;
      std::optional<Candidate> merge;
      if (!candidates.empty()) {
        selected_queue = &candidates;
        merge = candidates.top();
      }
      if (!default_candidates.empty()) {
        auto next = default_candidates.top();
        next.phase = 0.;
        next.score = next.cost_score;
        if (!merge || *merge > next) {
          selected_queue = &default_candidates;
          merge = next;
        }
      }
      if (conversion && (!merge || !(*conversion > *merge))) {
        return conversion->kind == 3 ? result(nullptr, conversion->id, &active[conversion->id]) :
                                       result(&materializations[conversion->id]);
      }
      if (!selected_queue) {
        discard_stale(leaf_candidates);
        if (leaf_candidates.empty())
          break;
        selected_queue = &leaf_candidates;
      }
      const auto c = selected_queue->top();
      selected_queue->pop();
      if (c.kind == 0) {
        if (c.version != node_version[c.id] || !queue.is_leaf(c.id))
          continue;
        const auto parent = queue.take(c.id);
        prune(c.id);
        if (parent >= 0 && order_keys[c.id][1] >= 0 &&
            order_keys[parent][1] == order_keys[c.id][1])
          pending_depths[order_keys[c.id][1]] = order_keys[c.id][2];
        push_node(parent);
        for (size_t j = 0; j < kv[c.id].size(); j += 3)
          for (const auto user : kv_users[kv[c.id][j]])
            push_node(user);
        for (const auto slot : node_slots[c.id]) {
          if (actions.contains(slot)) {
            const auto right = successor(actions.at(slot)[1]);
            successors[slot] = -1;
            if (previous.contains(slot))
              push_action(previous.at(slot));
            if (right >= 0)
              previous.erase(right);
          }
          const auto frame = slot_info.at(slot).decay;
          if (frame >= 0)
            for (const auto user : decay_users[frame])
              push_node(user);
        }
      } else {
        if (!actions.contains(c.id) || c.version != action_version[c.id] || released_set.contains(c.id))
          continue;
        const auto right = successor(actions.at(c.id)[1]);
        if (right < 0 || released_set.contains(right))
          continue;
        compressed.insert(compressed.end(), {c.id, right});
        slot_info.at(right).begin = slot_info.at(c.id).begin;
        successors[c.id] = right;
        release(c.id);
        if (previous.contains(c.id)) {
          previous[right] = previous.at(c.id);
          push_action(previous.at(c.id));
        } else
          previous.erase(right);
        push_action(right);
        push_node(slot_node.at(c.id));
        push_node(slot_node.at(right));
      }
      if (fits()) {
        return result();
      }
    }
    if (!active.empty() && (!selected.empty() || !released.empty()))
      return result();
    return make_int_tensor(0);
  }
};

class RecycleBin {
public:
  explicit RecycleBin(Int capacity) : capacity_(capacity) {
    check(capacity >= 0, "Invalid recycle capacity");
  }

  Int remember(const tvm::ffi::Array<tvm::ffi::Tensor> &parts,
               tvm::ffi::TensorView entry_rows, const List &protected_ends,
               const List &stats) {
    if (!capacity_) {
      return 0;
    }
    auto tokens = read_prefix(parts);
    std::vector<std::array<Int, 2>> logs;
    for (const auto &row : read_rows(entry_rows)) {
      check(row.size() == 2 && row[0] >= 0 && row[0] < row[1] &&
                row[1] <= static_cast<Int>(tokens.size()),
            "Invalid recycled Log range");
      logs.push_back({row[0], row[1]});
    }
    if (logs.empty()) {
      return 0;
    }
    Vector protected_values(protected_ends.begin(), protected_ends.end());
    std::sort(protected_values.begin(), protected_values.end());
    logs = merge_logs({}, logs);
    auto it = records_.lower_bound(tokens);
    while (it != records_.end() && it->first.size() >= tokens.size() &&
           std::equal(tokens.begin(), tokens.end(), it->first.begin())) {
      if (it->first.size() == tokens.size()) {
        ++it;
        continue;
      }
      auto &record = it->second;

      auto refreshed = merge_logs(record.logs, logs);
      Vector ends = record.protected_ends;
      ends.insert(ends.end(), protected_values.begin(), protected_values.end());
      std::sort(ends.begin(), ends.end());
      ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
      used_ -= bytes(it->first, record);
      ids_.erase(record.id);
      record.logs = std::move(refreshed);
      set_stats(record,
                merge_owners(record.stats, Vector(stats.begin(), stats.end())));
      record.protected_ends = std::move(ends);
      record.id = ++serial_;
      *record.lru = record.id;
      ids_[record.id] = it;
      used_ += bytes(it->first, record);
      ++it;
    }
    Record record{++serial_,
                  logs,
                  protected_values,
                  Vector(stats.begin(), stats.end()),
                  {}};
    if (const auto old = records_.find(tokens); old != records_.end()) {
      record.logs = merge_logs(old->second.logs, logs);
      record.protected_ends.insert(record.protected_ends.end(),
                                   old->second.protected_ends.begin(),
                                   old->second.protected_ends.end());
      std::sort(record.protected_ends.begin(), record.protected_ends.end());
      record.protected_ends.erase(std::unique(record.protected_ends.begin(),
                                              record.protected_ends.end()),
                                  record.protected_ends.end());
      record.stats = merge_owners(old->second.stats, record.stats);
    }
    if (bytes(tokens, record) > capacity_) {
      for (size_t i = 2; i < record.stats.size(); i += 3)
        if (!owner_refs_.contains(record.stats[i]))
          retired_.push_back(record.stats[i]);
      trim();
      return 0;
    }
    if (const auto old = records_.find(tokens); old != records_.end()) {
      erase(old);
    }
    auto [inserted, fresh] =
        records_.emplace(std::move(tokens), std::move(record));
    check(fresh, "Duplicate recycled prefix");
    lru_.push_back(inserted->second.id);
    inserted->second.lru = std::prev(lru_.end());
    ids_[inserted->second.id] = inserted;
    retain(inserted->second.stats);
    used_ += bytes(inserted->first, inserted->second);
    trim();
    return inserted->second.id;
  }

  Vector match(tvm::ffi::TensorView query, Int start, Int granularity, bool touch_match) {
    check(query.ndim() == 1 && query.is_contiguous() &&
              query.device().device_type == kDLCPU &&
              query.dtype().code == kDLInt &&
              (query.dtype().bits == 32 || query.dtype().bits == 64) &&
              start >= 0 && granularity > 0,
          "Invalid recycle query");
    Vector tokens(query.size(0));
    for (Int i = 0; i < query.size(0); ++i) {
      tokens[i] = query.dtype().bits == 32
                      ? static_cast<const int32_t *>(query.data_ptr())[i]
                      : static_cast<const Int *>(query.data_ptr())[i];
    }
    auto right = records_.lower_bound(tokens);
    auto best = records_.end();
    Int matched = start;
    auto consider = [&](auto candidate) {
      const auto limit = std::min(tokens.size(), candidate->first.size());
      const auto pair = std::mismatch(tokens.begin(), tokens.begin() + limit,
                                      candidate->first.begin());
      const auto length =
          (pair.first - tokens.begin()) / granularity * granularity;
      for (const auto &log : candidate->second.logs) {
        const auto end = std::min(length, log[1]) / granularity * granularity;
        if (end > matched && end > log[0]) {
          matched = end;
          best = candidate;
        }
      }
      return length;
    };

    auto left = right;
    while (left != records_.begin()) {
      --left;
      if (consider(left) <= matched) {
        break;
      }
    }
    while (right != records_.end()) {
      if (consider(right) <= matched) {
        break;
      }
      ++right;
    }
    if (best == records_.end()) {
      return {};
    }
    if (touch_match)
      lru_.splice(lru_.end(), lru_, best->second.lru);
    return {best->second.id, matched};
  }

  Rows get(Int id) const {
    check(ids_.contains(id), "Expired recycled record");
    const auto it = ids_.at(id);
    Rows result({List(it->first.begin(), it->first.end()),
                 List(it->second.protected_ends.begin(),
                      it->second.protected_ends.end()),
                 List(it->second.stats.begin(), it->second.stats.end())});
    for (const auto &log : it->second.logs) {
      result.push_back(List(log.begin(), log.end()));
    }
    return result;
  }

  void discard(Int id) {
    if (ids_.contains(id)) {
      erase(ids_.at(id));
    }
  }

  void touch(Int id) {
    if (ids_.contains(id)) {
      lru_.splice(lru_.end(), lru_, ids_.at(id)->second.lru);
    }
  }

  Vector usage() const { return {used_, static_cast<Int>(records_.size())}; }

  Vector retired_owners() {
    Vector result;
    for (auto id : retired_)
      if (!owner_refs_.contains(id))
        result.push_back(id);
    retired_.clear();
    return result;
  }

private:
  static Vector merge_owners(const Vector &old, const Vector &added) {
    std::vector<std::array<Int, 3>> rows;
    for (const auto *values : {&old, &added}) {
      check(values->size() % 3 == 0, "Invalid recycled owner ranges");
      for (size_t i = 0; i < values->size(); i += 3)
        rows.push_back({(*values)[i], (*values)[i + 1], (*values)[i + 2]});
    }
    std::stable_sort(rows.begin(), rows.end(),
                     [](const auto &a, const auto &b) { return a[0] < b[0]; });
    Vector result;
    for (auto row : rows) {
      if (!result.empty())
        row[0] = std::max(row[0], result[result.size() - 2]);
      if (row[0] < row[1])
        result.insert(result.end(), row.begin(), row.end());
    }
    return result;
  }
  void retain(const Vector &stats) {
    for (size_t i = 2; i < stats.size(); i += 3)
      ++owner_refs_[stats[i]];
  }
  void release(const Vector &stats) {
    for (size_t i = 2; i < stats.size(); i += 3) {
      auto it = owner_refs_.find(stats[i]);
      check(it != owner_refs_.end(), "Missing recycled owner reference");
      if (!--it->second) {
        retired_.push_back(it->first);
        owner_refs_.erase(it);
      }
    }
  }
  template <typename RecordType>
  void set_stats(RecordType &record, Vector stats) {
    release(record.stats);
    retain(stats);
    record.stats = std::move(stats);
  }
  static std::vector<std::array<Int, 2>>
  merge_logs(std::vector<std::array<Int, 2>> logs,
             const std::vector<std::array<Int, 2>> &added) {
    logs.insert(logs.end(), added.begin(), added.end());
    std::sort(logs.begin(), logs.end());
    std::vector<std::array<Int, 2>> result;
    for (const auto &log : logs) {
      if (!result.empty() && log[0] < result.back()[1]) {
        result.back()[1] = std::max(result.back()[1], log[1]);
      } else {
        result.push_back(log);
      }
    }
    return result;
  }

  static Vector read_prefix(const tvm::ffi::Array<tvm::ffi::Tensor> &parts) {
    Vector tokens;
    for (const auto &part : parts) {
      check(part.ndim() == 1 && part.is_contiguous() &&
                part.device().device_type == kDLCPU &&
                part.dtype().code == kDLInt &&
                (part.dtype().bits == 32 || part.dtype().bits == 64),
            "Recycle tokens must be contiguous CPU int32/int64");
      for (Int i = 0; i < part.size(0); ++i) {
        tokens.push_back(part.dtype().bits == 32
                             ? static_cast<const int32_t *>(part.data_ptr())[i]
                             : static_cast<const Int *>(part.data_ptr())[i]);
      }
    }
    return tokens;
  }

  struct Record {
    Int id;
    std::vector<std::array<Int, 2>> logs;
    Vector protected_ends;
    Vector stats;
    std::list<Int>::iterator lru;
  };
  using Records = std::map<Vector, Record>;

  static Int bytes(const Vector &tokens, const Record &record) {

    return 256 + 8 * tokens.size() + 16 * record.logs.size() +
           8 * (record.protected_ends.size() + record.stats.size());
  }

  void erase(Records::iterator it) {
    used_ -= bytes(it->first, it->second);
    release(it->second.stats);
    ids_.erase(it->second.id);
    lru_.erase(it->second.lru);
    records_.erase(it);
  }

  void trim() {
    while (used_ > capacity_) {
      erase(ids_.at(lru_.front()));
    }
  }

  Int capacity_, used_ = 0, serial_ = 0;
  std::unordered_map<Int, Int> owner_refs_;
  Vector retired_;
  Records records_;
  std::list<Int> lru_;
  std::unordered_map<Int, Records::iterator> ids_;
};

Result make_recycle_bin(Int capacity) {


  auto bin = std::make_shared<RecycleBin>(capacity);
  Result result;
  result.Set("remember", tvm::ffi::Function::FromTyped(
    [bin](const tvm::ffi::Array<tvm::ffi::Tensor>& parts, tvm::ffi::TensorView entries,
          const List& ends, const List& stats) {
      return bin->remember(parts, entries, ends, stats);
    }));
  result.Set("match", tvm::ffi::Function::FromTyped(
    [bin](tvm::ffi::TensorView query, Int start, Int granularity, bool touch_match) {
      const auto values = bin->match(query, start, granularity, touch_match);
      return List(values.begin(), values.end());
    }));


  result.Set("get", tvm::ffi::Function::FromTyped(
    [bin](Int id, tvm::ffi::TensorView) { return write_rows(bin->get(id)); }));
  result.Set("discard", tvm::ffi::Function::FromTyped(
    [bin](Int id) { bin->discard(id); }));
  result.Set("touch", tvm::ffi::Function::FromTyped(
    [bin](Int id) { bin->touch(id); }));
  result.Set("retired_owners", tvm::ffi::Function::FromTyped(
    [bin]() { const auto ids=bin->retired_owners(); return List(ids.begin(),ids.end()); }));
  result.Set("usage", tvm::ffi::Function::FromTyped(
    [bin]() {
      const auto values = bin->usage();
      return List(values.begin(), values.end());
    }));
  return result;
}

struct Planner {
  static tvm::ffi::Tensor protections(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs,
                                     tvm::ffi::TensorView forced, Int granularity) {
    return write_rows(branch_protections(inputs, read_rows(forced), granularity));
  }

  static std::array<std::vector<Vector>, 7> eviction_groups(
      tvm::ffi::TensorView packed, const List &counts) {
    check(counts.size() == 7, "Invalid eviction metadata groups");
    const auto begin = cpu_ints(packed);
    const auto end = begin + packed.shape()[0];
    auto cursor = begin;
    std::array<std::vector<Vector>, 7> groups;
    for (size_t group = 0; group < counts.size(); ++group) {
      check(counts[group] >= 0, "Invalid eviction row count");
      groups[group].reserve(counts[group]);
      for (Int row = 0; row < counts[group]; ++row) {
        check(cursor < end, "Truncated eviction metadata");
        const auto size = *cursor++;
        check(size >= 0 && size <= end - cursor, "Invalid eviction row size");
        groups[group].emplace_back(cursor, cursor + size);
        cursor += size;
      }
    }
    check(cursor == end, "Unexpected eviction metadata");
    return groups;
  }
  static Result make_eviction(tvm::ffi::TensorView packed, const List &counts,
                             const List &config, tvm::ffi::TensorView models,
                             tvm::ffi::TensorView order) {
    auto planner = std::make_shared<EvictionPlanner>(
        eviction_groups(packed, counts), config, models, read_rows(order));
    Result result;
    result.Set("plan", tvm::ffi::Function::FromTyped(
      [planner](const List &capacity, tvm::ffi::TensorView occupied,
                tvm::ffi::TensorView active_geometry, tvm::ffi::TensorView active_models,
                Int num_blocked, tvm::ffi::TensorView locks,
                const List &model_indices, tvm::ffi::TensorView models) {
        if (num_blocked)
          planner->update_locks(num_blocked, read_rows(locks));
        if (!model_indices.empty())
          planner->refresh_models(model_indices, models);
        return planner->plan(capacity, read_rows(occupied), active_geometry, active_models);
      }));
    result.Set("rebase", tvm::ffi::Function::FromTyped(
      [planner](Int size, const List &removed, tvm::ffi::TensorView geometry,
                const List &indices, tvm::ffi::TensorView models, tvm::ffi::TensorView order,
                const List &capacity, tvm::ffi::TensorView active_geometry,
                tvm::ffi::TensorView active_models) {
        planner->rebase(size, removed, read_rows(geometry), indices, models, read_rows(order));
        return planner->plan(capacity, Rows{}, active_geometry, active_models);
      }));
    result.Set("replace_materialized_node", tvm::ffi::Function::FromTyped(
      [planner](Int index, Int timestamp, tvm::ffi::TensorView records,
                tvm::ffi::TensorView actions, tvm::ffi::TensorView frontiers,
                tvm::ffi::TensorView groups) {
        return planner->replace_materialized_node(index, timestamp, read_rows(records),
            read_rows(actions), read_rows(frontiers), read_rows(groups));
      }));
    return result;
  }

  static tvm::ffi::Tensor materialize(tvm::ffi::TensorView paths, tvm::ffi::TensorView boundaries,
                                    tvm::ffi::TensorView candidates, Int granularity, Int log_count) {
    return write_rows(plan_materialize(read_rows(paths), read_rows(boundaries), read_rows(candidates), granularity, log_count));
  }


};

}
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_materialize, logssms::Planner::materialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(branch_protections, logssms::Planner::protections);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_recycle_bin, logssms::make_recycle_bin);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_eviction_planner, logssms::Planner::make_eviction);
