#include <limits>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <map>
#include <list>
#include <memory>
#include <numeric>
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

tvm::ffi::Tensor write_rows(const Rows& rows) {
  Int size = 0;
  for (const auto& row : rows) {
    size += 1 + row.size();
  }
  auto result = tvm::ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc, tvm::ffi::ShapeView(&size, 1),
    DLDataType{kDLInt, 64, 1}, DLDevice{kDLCPU, 0}
  );
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
constexpr Int kEvictionExpand = 2;

template <typename NodeRows>
class EvictionQueue {
public:
  explicit EvictionQueue(const NodeRows& nodes) : nodes_(nodes), children_(nodes.size()) {
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      check(nodes[i].size() == 5 && nodes[i][2] >= 0 && nodes[i][2] <= kEvictionExpand,
            "Invalid eviction node");
      check(indices_.emplace(nodes[i][0], i).second, "Duplicate eviction node");
    }
    for (const auto& node : nodes) {
      if (indices_.contains(node[1])) {
        ++children_[indices_.at(node[1])];
      }
    }
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      if (!children_[i] && nodes[i][2] != kEvictionBlocked) {
        push(i);
      }
    }
  }

  Int take(Int index) {
    check(leaves_.erase(index) == 1, "Eviction node is not an unlocked leaf");
    const auto parent = indices_.find(nodes_[index][1]);
    if (parent != indices_.end() && !--children_[parent->second] && nodes_[parent->second][2] != kEvictionBlocked) {
      push(parent->second);
      return parent->second;
    }
    return -1;
  }

  bool is_leaf(Int index) const { return leaves_.contains(index); }
  const std::unordered_set<Int>& leaves() const { return leaves_; }

private:
  void push(Int index) {
    leaves_.insert(index);
  }

  const NodeRows& nodes_;
  Vector children_;
  std::unordered_map<Int, Int> indices_;
  std::unordered_set<Int> leaves_;
};

class EvictionPlanner {
  using NodeRows = std::vector<Vector>;
  struct Density {
    double tail = 0, atom = 0, g = 1;
    bool default_order = false;
    std::vector<double> probability{0.};
    double prefix(double position) const {
      auto i = static_cast<Int>(std::floor((tail - position) / g));
      return probability[std::clamp<Int>(i, 0, probability.size() - 1)];
    }
    double mass(double begin, double end) const {
      return std::max(0., prefix(begin) - prefix(end));
    }
  };
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

  NodeRows nodes, kv, node_slots;
  List config;
  EvictionQueue<NodeRows> queue;
  std::vector<double> model_storage;
  const double *model_data = nullptr;
  bool online;
  std::unordered_map<Int, Vector> slot_info, actions;
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
  CandidateQueue candidates, default_candidates;

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
    free += record[1];
    if (record[2] >= 0) {
      check(decay_occupied.contains(record[2]),
            "Missing decay frame occupancy");
      if (++decay_released[record[2]] == decay_occupied.at(record[2])) {
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
  Rows result() {
    Vector page_counts;
    for (const auto frame : page_order) {
      page_counts.insert(page_counts.end(), {frame, page_delta.at(frame)});
    }
    std::sort(released.begin(), released.end());
    return Rows({List(selected.begin(), selected.end()),
                 List(page_counts.begin(), page_counts.end()),
                 List(released.begin(), released.end()),
                 List(compressed.begin(), compressed.end())});
  }
  Int successor(Int slot) {
    while (slot >= 0 && successors.contains(slot))
      slot = successors.at(slot);
    return slot;
  }
  double recovery_before(Int index, Int slot) {
    if (previous.contains(slot))
      return static_cast<double>(slot_info.at(previous.at(slot))[4]);
    const auto end = slot_info.at(slot)[4];
    double restored = model_data[index * 10 + 7];
    for (auto preceding : ordered_slots[index]) {
      if (released_set.contains(preceding))
        continue;
      const auto &record = slot_info.at(preceding);
      if (record[4] >= end)
        break;
      if ((nodes[index][3] != 0 && record[2] < 0) || record[3] == restored)
        restored = record[4];
    }
    return restored;
  }
  Int slot_credit(Int slot) {
    if (released_set.contains(slot))
      return Int{0};
    const auto &record = slot_info.at(slot);
    Int credit = record[1];
    if (record[2] >= 0 &&
        decay_released[record[2]] + 1 == decay_occupied.at(record[2]))
      ++credit;
    return credit;
  }
  Int prune_credit(Int index) {
    Int credit = 0;
    std::unordered_map<Int, Int> decays;
    for (auto slot : node_slots[index])
      if (!released_set.contains(slot)) {
        const auto &record = slot_info.at(slot);
        credit += record[1];
        if (record[2] >= 0)
          ++decays[record[2]];
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
      if (checkpoints > 1)
        return;
    }
    const bool log = nodes[index][3] == 0 && !node_slots[index].empty();
    double phase = (log || nodes[index][2] == kEvictionExpand) ? 0. : 2.,
           score = 0.;
    const auto default_phase = phase;
    if (online) {
      phase = 0.;
      const auto *p = model_data + index * 10;
      const auto &density = densities.at(model_ids[index]);
      double cost = 0., restored = p[7];
      for (auto slot : ordered_slots[index]) {
        if (released_set.contains(slot))
          continue;
        const auto &record = slot_info.at(slot);
        if ((nodes[index][3] != 0 && record[2] < 0) || record[3] == restored) {
          cost += std::max(0., record[4] - restored) *
                  (density.atom + density.prefix(record[4]));
          restored = record[4];
        }
      }
      const auto credit = prune_credit(index);
      score = credit ? cost / credit : std::numeric_limits<double>::infinity();
    }
    if (online && densities.at(model_ids[index]).default_order)
      default_candidates.push({default_phase, 0., nodes[index][4], index, 0,
                               version, score, order[0], order[1], order[2]});
    else
      candidates.push({phase, score, nodes[index][4], index, 0, version, score,
                       order[0], order[1], order[2]});
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
    const auto left_span = left_record[4] - left_record[3];
    const auto right_span = right_record[4] - right_record[3];
    double phase = 1., score = std::max(left_span, right_span);
    if (online) {
      phase = 0.;
      const auto distance = left_record[4] - recovery_before(action[3], slot);
      const auto cost =
          std::max(0., distance) * densities.at(model_ids[action[3]])
                                       .mass(left_record[4], right_record[4]);
      const auto credit = slot_credit(slot);
      score = credit ? cost / credit : std::numeric_limits<double>::infinity();
    }
    if (online && densities.at(model_ids[action[3]]).default_order)
      default_candidates.push(
          {1., static_cast<double>(std::max(left_span, right_span)), action[2],
           slot, 1, version, score});
    else
      candidates.push({phase, score, action[2], slot, 1, version, score});
  }
  bool valid(const Candidate &c) {
    if (c.kind == 0)
      return c.version == node_version[c.id] && queue.is_leaf(c.id);
    if (c.version != action_version[c.id] || released_set.contains(c.id))
      return false;
    const auto right = successor(actions.at(c.id)[1]);
    return right >= 0 && !released_set.contains(right);
  }

  template <typename Indices>
  void load_models(const Indices &indices, tvm::ffi::TensorView models) {
    check(models.ndim() == 1 && models.is_contiguous() &&
              models.device().device_type == kDLCPU &&
              models.dtype().code == kDLFloat && models.dtype().bits == 64 &&
              models.size(0) == static_cast<Int>(indices.size() * 10),
          "Invalid eviction density models");
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
      const auto scale =
          std::min(1., (1. - d.atom) / std::max(1e-300, d.probability.back()));
      for (auto &v : d.probability)
        v *= scale;
      densities.emplace(id, std::move(d));
    }
  }

public:
  EvictionPlanner(std::array<NodeRows, 6> groups, const List &capacity,
                  tvm::ffi::TensorView models, const Rows &order)
      : nodes(std::move(groups[0])), kv(std::move(groups[1])),
        node_slots(std::move(groups[2])), config(capacity), queue(nodes),
        node_version(nodes.size()), ordered_slots(nodes.size()) {
    const auto &slots = groups[3];
    const auto &occupied = groups[4];
    const auto &checkpoint_actions = groups[5];
    for (const auto &row : order)
      order_keys.emplace_back(row.begin(), row.end());
    if (order_keys.empty())
      for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i)
        order_keys.push_back({0, -1, i});
    check(order_keys.size() == nodes.size(),
          "Invalid eviction ordering metadata");
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i)
      owner_nodes[order_keys[i][1]].push_back(i);
    check(nodes.size() == kv.size() && nodes.size() == node_slots.size() &&
              config.size() == 10 && config[5] > 0,
          "Invalid eviction capacity metadata");
    for (const auto &row : slots) {
      check(row.size() == 5 && row[1] >= 0, "Invalid eviction slot");
      slot_info.emplace(row[0], Vector(row.begin(), row.end()));
    }
    for (const auto &row : occupied) {
      check(row.size() == 2 && row[1] > 0 && row[1] <= config[5],
            "Invalid decay occupancy");
      decay_occupied.emplace(row[0], row[1]);
    }
    online = models.size(0) != 0;
    model_ids.resize(nodes.size());
    if (online) {
      model_storage.resize(nodes.size() * 10);
      model_data = model_storage.data();
      Vector indices(nodes.size());
      std::iota(indices.begin(), indices.end(), 0);
      load_models(indices, models);
    }
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      for (auto slot : node_slots[i]) {
        slot_node.emplace(slot, i);
        ordered_slots[i].push_back(slot);
      }
      std::sort(ordered_slots[i].begin(), ordered_slots[i].end(),
                [&](Int a, Int b) {
                  return slot_info.at(a)[4] < slot_info.at(b)[4];
                });
    }
    for (const auto &row : checkpoint_actions) {
      check(row.size() >= 4, "Invalid checkpoint eviction action");
      check(slot_node.contains(row[0]) && slot_node.at(row[0]) == row[3],
            "Checkpoint action has an invalid owner");
      check(row[1] < 0 || (slot_node.contains(row[1]) &&
                           slot_node.at(row[0]) == slot_node.at(row[1])),
            "Checkpoint merging must stay inside one node");
      actions.emplace(row[0], Vector(row.begin(), row.end()));
      if (row[1] >= 0)
        previous[row[1]] = row[0];
    }
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      for (size_t j = 0; j < kv[i].size(); j += 3)
        kv_users[kv[i][j]].push_back(i);
      for (auto slot : node_slots[i])
        if (slot_info.at(slot)[2] >= 0)
          decay_users[slot_info.at(slot)[2]].push_back(i);
    }
    for (const auto index : queue.leaves())
      push_node(index);
    for (const auto &[slot, action] : actions)
      push_action(slot);
  }
  void refresh_models(const List &indices, tvm::ffi::TensorView models) {
    check(online, "Cannot refresh a default eviction planner");
    load_models(indices, models);
    for (const auto &[owner, depth] : pending_depths)
      for (const auto index : owner_nodes[owner])
        order_keys[index][0] = depth;
    pending_depths.clear();
    candidates.begin_update();
    default_candidates.begin_update();
    for (const auto index : queue.leaves())
      push_node(index);
    for (const auto &[slot, action] : actions)
      push_action(slot);
    candidates.finish_update();
    default_candidates.finish_update();
  }
  template <typename Occupancy>
  Rows plan(const List &capacity, const Occupancy &occupancy) {
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
    while (true) {
      while (!candidates.empty() && !valid(candidates.top()))
        candidates.pop();
      while (!default_candidates.empty() && !valid(default_candidates.top()))
        default_candidates.pop();
      if (candidates.empty() && default_candidates.empty())
        break;
      bool use_default = candidates.empty();
      if (!default_candidates.empty() && !candidates.empty()) {
        auto next = default_candidates.top();
        next.phase = 0.;
        next.score = next.cost_score;
        use_default = candidates.top() > next;
      }
      auto &selected_queue = use_default ? default_candidates : candidates;
      const auto c = selected_queue.top();
      selected_queue.pop();
      if (c.kind == 0) {
        if (c.version != node_version[c.id] || !queue.is_leaf(c.id))
          continue;
        if (nodes[c.id][2] == kEvictionExpand)
          return Rows({List({-1})});
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
          const auto frame = slot_info.at(slot)[2];
          if (frame >= 0)
            for (const auto user : decay_users[frame])
              push_node(user);
        }
      } else {
        if (c.version != action_version[c.id] || released_set.contains(c.id))
          continue;
        const auto right = successor(actions.at(c.id)[1]);
        if (right < 0 || released_set.contains(right))
          continue;
        compressed.insert(compressed.end(), {c.id, right});
        slot_info.at(right)[3] = slot_info.at(c.id)[3];
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
      if (fits())
        return result();
    }
    return {};
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

  Vector match(tvm::ffi::TensorView query, Int start, Int granularity) {
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
    [bin](tvm::ffi::TensorView query, Int start, Int granularity) {
      const auto values = bin->match(query, start, granularity);
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

  static std::array<std::vector<Vector>, 6> eviction_groups(
      tvm::ffi::TensorView packed, const List &counts) {
    check(counts.size() == 6, "Invalid eviction metadata groups");
    const auto begin = cpu_ints(packed);
    const auto end = begin + packed.shape()[0];
    auto cursor = begin;
    std::array<std::vector<Vector>, 6> groups;
    for (size_t group = 0; group < groups.size(); ++group) {
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
  static tvm::ffi::Tensor eviction(tvm::ffi::TensorView packed, const List &counts,
                                 const List &config, tvm::ffi::TensorView models) {
    auto groups = eviction_groups(packed, counts);
    auto occupied = groups[4];
    EvictionPlanner planner(std::move(groups), config, models, {});
    return write_rows(planner.plan(config, occupied));
  }
  static Result make_eviction(tvm::ffi::TensorView packed, const List &counts,
                             const List &config, tvm::ffi::TensorView models,
                             tvm::ffi::TensorView order) {
    auto planner = std::make_shared<EvictionPlanner>(
        eviction_groups(packed, counts), config, models, read_rows(order));
    Result result;
    result.Set("plan", tvm::ffi::Function::FromTyped(
      [planner](const List &capacity, tvm::ffi::TensorView occupied) {
        return write_rows(planner->plan(capacity, read_rows(occupied)));
      }));
    result.Set("refresh_models", tvm::ffi::Function::FromTyped(
      [planner](const List &indices, tvm::ffi::TensorView models) {
        planner->refresh_models(indices, models);
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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_eviction, logssms::Planner::eviction);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_recycle_bin, logssms::make_recycle_bin);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_eviction_planner, logssms::Planner::make_eviction);
