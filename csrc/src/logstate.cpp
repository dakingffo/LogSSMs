#include <algorithm>
#include <array>
#include <cstdint>
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


  Int slot, begin, end, span, physical;
  bool eligible = true, protected_entry = false, fragmented = false;
};

std::vector<Entry> entries(const List& row) {
  check(row.size() % 8 == 0, "Invalid LogState entry metadata");
  std::vector<Entry> result;
  result.reserve(row.size() / 8);
  for (size_t i = 0; i < row.size(); i += 8) {
    result.push_back({row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 4],
                      bool(row[i + 5]), bool(row[i + 6]), bool(row[i + 7])});
  }
  return result;
}

bool protected_interval(const std::vector<Entry>& path, const List& boundaries,
                        Int begin, Int end) {
  for (const auto position : boundaries) {
    if (begin < position && position < end) {
      return true;
    }
  }
  for (const auto& entry : path) {
    if (entry.protected_entry && begin <= entry.begin && entry.end <= end) {
      return true;
    }
  }
  return false;
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
                      const Rows& candidates, Int page_size) {
  check(paths.size() == boundaries.size() && page_size > 0, "Invalid materialize metadata");
  std::vector<std::vector<Entry>> unpacked;
  for (const auto& path : paths) {
    unpacked.push_back(entries(path));
  }
  Rows result;
  for (Int candidate = 0; candidate < static_cast<Int>(candidates.size()); ++candidate) {
    const auto& row = candidates[candidate];
    check(row.size() == 2 && row[0] >= 0 && row[0] < static_cast<Int>(paths.size()),
          "Invalid materialize candidate path");
    const auto& path = unpacked[row[0]];
    check(row[1] >= 0 && row[1] < static_cast<Int>(path.size()), "Invalid candidate entry");
    const auto& entry = path[row[1]];
    if (!entry.eligible || entry.span >= 0 || entry.fragmented
        || (entry.end - entry.begin) % page_size
        || protected_interval(path, boundaries[row[0]], entry.begin, entry.end)) {
      continue;
    }
    Int base_slot = -1, cursor = 0;
    for (const auto& value : path) {
      if (value.span >= 0 && value.end <= entry.begin
          && (base_slot == -1 || value.end > cursor)) {
        base_slot = value.slot;
        cursor = value.end;
      }
    }
    Vector output{candidate, base_slot};
    for (Int i = 0; i < static_cast<Int>(path.size()); ++i) {
      const auto& value = path[i];
      if (value.span >= 0 || value.end <= cursor || value.begin > cursor) {
        continue;
      }
      output.push_back(i);
      cursor = std::min(entry.end, value.end);
      if (cursor == entry.end) {
        break;
      }
    }
    if (cursor == entry.end) {
      result.push_back(List(output.begin(), output.end()));
    }
  }
  return result;
}

Result plan_coalesce(const Rows& paths, const Rows& boundaries, const Rows& locks,
                     const Rows& anchors, const List& materialized,
                     Int page_size, bool diagnostics) {
  check(paths.size() == boundaries.size() && paths.size() == locks.size()
        && page_size > 0, "Invalid coalesce metadata");
  std::unordered_set<Int> projected(materialized.begin(), materialized.end()), used;
  std::map<std::string, Int> counts;
  Rows groups;
  auto add = [&](const std::string& key, Int value = 1) {
    if (diagnostics) {
      counts[key] += value;
    }
  };
  for (const auto& anchor : anchors) {

    check(anchor.size() == 4 && anchor[0] >= 0 && anchor[0] < static_cast<Int>(paths.size())
          && anchor[2] >= 2, "Invalid coalesce anchor");
    const auto index = anchor[0];
    const auto path = entries(paths[index]);
    std::vector<Entry> selected;
    add("merge_anchors");
    for (auto entry : path) {
      if (entry.end > anchor[1]) {
        add("entries_beyond_match_endpoint");
        continue;
      }
      if (!entry.eligible) {
        add("entries_not_eligible_this_round");
        continue;
      }
      if (used.contains(entry.slot)) {
        continue;
      }
      if (projected.contains(entry.slot)) {
        entry.span = 1;
      }
      if (entry.span < 0) {
        add("logs_not_materialized");
      }
      else {
        selected.push_back(entry);
      }
    }
    add("checkpoint_candidates", selected.size());
    if (selected.empty()) {
      add("anchors_without_eligible_entries");
      continue;
    }
    add(anchor[3] ? "anchors_with_stats" : "anchors_without_stats");
    add("anchors_intensity_" + std::to_string(anchor[2]));
    auto blocked = [&](Int begin, Int end) {
      return protected_interval(path, boundaries[index], begin, end);
    };
    auto locked = [&](Int position) {
      return std::find(locks[index].begin(), locks[index].end(), position) != locks[index].end();
    };
    std::vector<std::vector<Entry>> segments;
    std::vector<Entry> segment;
    for (const auto& entry : selected) {
      const bool entry_blocked = blocked(entry.begin, entry.end);
      const bool gap = !segment.empty() && segment.back().end != entry.begin;
      const bool lock = !segment.empty() && locked(segment.back().end);
      const bool protection = !segment.empty() && blocked(segment.back().begin, entry.end);
      add("protected_entries", entry_blocked);
      add("gaps", gap);
      add("lock_boundaries", lock);
      add("protected_boundaries", protection);
      if (entry_blocked || gap || lock || protection) {
        if (!segment.empty()) {
          segments.push_back(std::move(segment));
          segment.clear();
        }
      }
      if (!entry_blocked) {
        segment.push_back(entry);
      }
    }
    if (!segment.empty()) {
      segments.push_back(std::move(segment));
    }
    add("segments", segments.size());
    auto merge = [&](const std::vector<Entry>& group) {
      add("candidate_groups");
      for (const auto& entry : group) {
        if (used.contains(entry.slot)) {
          add("groups_already_planned");
          return;
        }
      }
      std::string reason;
      if (std::any_of(group.begin(), group.end(), [&](const auto& e) { return e.end - e.begin != page_size * e.span; })) {
        reason = "span_mismatch";
      }
      else {
        for (size_t i = 1; i < group.size(); ++i) {
          if (group[i - 1].span > group[i].span) {
            reason = "decreasing_span";
            break;
          }
          if (group[i - 1].end != group[i].begin) {
            reason = "discontinuity";
            break;
          }
          if (locked(group[i - 1].end)) {
            reason = "lock_boundary";
            break;
          }
        }
        if (reason.empty() && blocked(group.front().begin, group.back().end)) {
          reason = "protected_interval";
        }
      }
      if (!reason.empty()) {
        add("rejected_" + reason);
        return;
      }
      Vector output{group.front().begin, group.back().end, 0};
      for (const auto& entry : group) {
        output[2] += entry.span;
        output.push_back(entry.slot);
        output.push_back(entry.span);
        used.insert(entry.slot);
      }
      groups.push_back(List(output.begin(), output.end()));
      add("planned_groups");
      add("planned_slot_releases", group.size() - 1);
      const bool equal = std::all_of(group.begin(), group.end(), [&](const auto& e) { return e.span == group[0].span; });
      add(equal ? "same_span_groups" : "cross_span_groups");
      add("planned_covered_tokens", group.back().end - group.front().begin);
      add("planned_arity_" + std::to_string(group.size()));
    };
    for (const auto& values : segments) {
      add("segments_without_merge_pair", values.size() < 2);
      for (size_t begin = 0; begin < values.size();) {
        size_t end = begin + 1;
        while (end < values.size() && end - begin < static_cast<size_t>(anchor[2])
               && values[end - 1].span <= values[end].span) {
          ++end;
        }
        if (end - begin >= 2) {
          merge(std::vector<Entry>(values.begin() + begin, values.begin() + end));
        }
        begin = end;
      }
    }
  }
  tvm::ffi::Map<tvm::ffi::String, Int> diag;
  for (const auto& [key, value] : counts) {
    diag.Set(key, value);
  }
  Result result;
  result.Set("groups", groups);
  result.Set("diagnostics", diag);
  return result;
}

struct Segment {
  Int slot, end;
  Vector outputs;
};

struct Chain {
  Int base;
  std::vector<Segment> segments;
};

Rows schedule_construction_waves(tvm::ffi::TensorView chain_rows, tvm::ffi::TensorView plans) {


  std::vector<Chain> chains;
  std::unordered_map<Int, Int> ends, writers;
  const auto plan_data = cpu_ints(plans);
  check(plans.size(0) % 2 == 0, "Invalid materialize output");
  for (Int i = 0; i < plans.size(0); i += 2) {
    ends[plan_data[i]] = plan_data[i + 1];
  }
  const auto data = cpu_ints(chain_rows);
  for (Int i = 0; i < chain_rows.size(0);) {
    const auto size = data[i++];
    check(size >= 1 && size <= chain_rows.size(0) - i, "Invalid construction chain");
    const auto end = i + size;
    Chain chain{data[i++], {}};
    while (i < end) {
      check(i + 3 <= end && data[i + 2] >= 0
            && data[i + 2] <= end - i - 3, "Invalid construction segment");
      const auto n = data[i + 2];
      chain.segments.push_back({data[i], data[i + 1], Vector(data + i + 3, data + i + 3 + n)});
      i += 3 + n;
    }
    chains.push_back(std::move(chain));
  }
  for (Int i = 0; i < static_cast<Int>(chains.size()); ++i) {
    for (const auto& entry : chains[i].segments) {
      for (const auto slot : entry.outputs) {
        if (ends.contains(slot)) {
          check(!writers.contains(slot), "Checkpoint has multiple writers");
          writers[slot] = i;
        }
      }
    }
  }
  check(writers.size() == ends.size(), "Missing Checkpoint writer");
  Vector starts(chains.size(), 0);
  for (Int i = 0; i < static_cast<Int>(chains.size()); ++i) {
    bool has_outputs = false;
    for (Int j = 0; j < static_cast<Int>(chains[i].segments.size()); ++j) {
      const auto& entry = chains[i].segments[j];
      if (writers.contains(entry.slot) && writers[entry.slot] != i
          && entry.end == ends[entry.slot] && !has_outputs && entry.outputs.empty()) {
        starts[i] = j + 1;
      }
      has_outputs |= !entry.outputs.empty();
    }
    if (starts[i]) {
      chains[i].base = chains[i].segments[starts[i] - 1].slot;
    }
  }
  std::vector<std::set<Int>> successors(chains.size());
  Vector indegree(chains.size());
  auto depend = [&](Int before, Int after) {
    if (before != after && successors[before].insert(after).second) {
      ++indegree[after];
    }
  };
  for (Int i = 0; i < static_cast<Int>(chains.size()); ++i) {

    if (writers.contains(chains[i].base)) {
      depend(writers[chains[i].base], i);
    }
    for (size_t j = starts[i]; j < chains[i].segments.size(); ++j) {
      const auto slot = chains[i].segments[j].slot;
      if (writers.contains(slot)) {
        depend(i, writers[slot]);
      }
    }
  }
  Rows result;
  Vector ready;
  for (Int i = 0; i < static_cast<Int>(chains.size()); ++i) {
    if (!indegree[i]) {
      ready.push_back(i);
    }
  }
  Int completed = 0, wave = 0;
  while (!ready.empty()) {
    std::sort(ready.begin(), ready.end());
    Vector next;
    for (const auto i : ready) {
      result.push_back(List({wave, i, starts[i], chains[i].base}));
      ++completed;
      for (const auto after : successors[i]) {
        if (!--indegree[after]) {
          next.push_back(after);
        }
      }
    }
    ready = std::move(next);
    ++wave;
  }
  check(completed == static_cast<Int>(chains.size()), "Cyclic LogState construction dependencies");
  return result;
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

tvm::ffi::Tensor write_vector(const Vector& values) {
  Int size = values.size();
  auto result = tvm::ffi::Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc, tvm::ffi::ShapeView(&size, 1),
    DLDataType{kDLInt, 64, 1}, DLDevice{kDLCPU, 0}
  );
  std::copy(values.begin(), values.end(), static_cast<Int*>(result.data_ptr()));
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

struct ConstructionSegment {
  Int entry, begin, end;
  Vector outputs;
};

struct ConstructionChain {
  Int base;
  bool materialize;
  std::vector<ConstructionSegment> segments;
};

Rows construction_plan(tvm::ffi::TensorView entry_rows, tvm::ffi::TensorView restore_rows,
                       tvm::ffi::TensorView material_rows) {
  const auto entries = read_rows(entry_rows);
  const auto restores = read_rows(restore_rows);
  const auto materials = read_rows(material_rows);
  for (const auto& entry : entries) {
    check(entry.size() == 4 && entry[1] < entry[2], "Invalid construction entry");
  }
  auto entry_at = [&](Int index) -> List {
    check(index >= 0 && index < static_cast<Int>(entries.size()), "Invalid construction entry index");
    return entries[index];
  };
  for (const auto& row : restores) {
    check(row.size() >= 2, "Invalid restore task");
    for (size_t i = 2; i < row.size(); ++i) {
      entry_at(row[i]);
    }
  }
  Vector order(materials.size());
  std::iota(order.begin(), order.end(), 0);
  std::unordered_set<Int> remaining;
  for (const auto& row : materials) {
    check(row.size() >= 4, "Invalid materialize task");
    check(remaining.insert(row[0]).second, "Duplicate materialize slot");
    for (size_t i = 4; i < row.size(); ++i) {
      entry_at(row[i]);
    }
  }
  std::stable_sort(order.begin(), order.end(), [&](Int a, Int b) {
    return materials[a][2] > materials[b][2];
  });


  auto covered = [&](const Vector& a, const Vector& b) {
    std::unordered_map<Int, std::vector<std::pair<Int, Int>>> by_slot;
    for (const auto i : b) {
      const auto& e = entry_at(i);
      by_slot[e[0]].emplace_back(e[1], e[2]);
    }
    for (const auto i : a) {
      const auto& e = entry_at(i);
      const auto found = by_slot.find(e[0]);
      if (found == by_slot.end() || !std::any_of(found->second.begin(), found->second.end(),
          [&](const auto& range) { return range.first <= e[1] && e[2] <= range.second; })) {
        return false;
      }
    }
    return true;
  };
  std::vector<bool> used(restores.size());
  std::vector<ConstructionChain> chains;
  for (const auto index : order) {
    const auto& plan = materials[index];
    if (!remaining.contains(plan[0])) {
      continue;
    }
    Vector logs(plan.begin() + 4, plan.end());
    std::map<Int, Vector> events;
    bool has_target = false;
    for (const auto i : logs) {
      const auto& e = entry_at(i);
      if (remaining.contains(e[0])) {
        events[e[2]].push_back(e[0]);
        has_target |= e[0] == plan[0];
      }
    }
    check(has_target, "Missing materialize target in replay logs");
    for (const auto i : logs) {
      remaining.erase(entry_at(i)[0]);
    }
    for (size_t i = 0; i < restores.size(); ++i) {
      const auto& task = restores[i];
      if (used[i] || task[0] != plan[3] || task.size() == 2) {
        continue;
      }
      Vector task_logs(task.begin() + 2, task.end());
      const auto source_covered = covered(logs, task_logs);
      if (!source_covered && !covered(task_logs, logs)) {
        continue;
      }
      if (source_covered) {
        logs = std::move(task_logs);
      }
      events[entry_at(task[task.size() - 1])[2]].push_back(task[1]);
      used[i] = true;
    }
    ConstructionChain chain{plan[3], true, {}};
    for (const auto i : logs) {
      const auto& e = entry_at(i);
      Int begin = e[1];
      for (auto it = events.upper_bound(begin); it != events.end() && it->first <= e[2]; ++it) {
        chain.segments.push_back({i, begin, it->first, it->second});
        begin = it->first;
      }
      if (begin < e[2]) {
        chain.segments.push_back({i, begin, e[2], {}});
      }
    }
    chains.push_back(std::move(chain));
  }
  check(remaining.empty(), "Unplanned materialize slots");
  for (size_t i = 0; i < restores.size(); ++i) {
    if (used[i]) {
      continue;
    }
    const auto& task = restores[i];
    ConstructionChain chain{task[0], false, {}};
    for (size_t j = 2; j < task.size(); ++j) {
      const auto& e = entry_at(task[j]);
      chain.segments.push_back({task[j], e[1], e[2], j + 1 == task.size() ? Vector{task[1]} : Vector{}});
    }
    chains.push_back(std::move(chain));
  }
  Rows packed_chains;
  Vector plan_ends;
  for (const auto& chain : chains) {
    Vector row{chain.base};
    for (const auto& segment : chain.segments) {
      row.insert(row.end(), {entry_at(segment.entry)[0], segment.end,
                             static_cast<Int>(segment.outputs.size())});
      row.insert(row.end(), segment.outputs.begin(), segment.outputs.end());
    }
    packed_chains.push_back(List(row.begin(), row.end()));
  }
  for (const auto& plan : materials) {
    plan_ends.insert(plan_ends.end(), {plan[0], plan[2]});
  }
  auto chain_tensor = write_rows(packed_chains);
  auto ends_tensor = write_vector(plan_ends);
  const auto wave_rows = schedule_construction_waves(chain_tensor, ends_tensor);
  Rows result;
  for (const auto& wave : wave_rows) {
    const auto& chain = chains[wave[1]];
    Vector row{wave[0], wave[3], Int(chain.materialize)};
    for (size_t j = wave[2]; j < chain.segments.size(); ++j) {
      const auto& segment = chain.segments[j];
      row.insert(row.end(), {segment.entry, segment.begin, segment.end,
                             static_cast<Int>(segment.outputs.size())});
      row.insert(row.end(), segment.outputs.begin(), segment.outputs.end());
    }
    result.push_back(List(row.begin(), row.end()));
  }
  return result;
}

class EvictionQueue {
public:
  explicit EvictionQueue(const Rows& nodes) : nodes_(nodes), children_(nodes.size()) {
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      check(nodes[i].size() == 5, "Invalid eviction node");
      check(indices_.emplace(nodes[i][0], i).second, "Duplicate eviction node");
    }
    for (const auto& node : nodes) {
      if (indices_.contains(node[1])) {
        ++children_[indices_.at(node[1])];
      }
    }
    for (Int i = 0; i < static_cast<Int>(nodes.size()); ++i) {
      if (!children_[i] && !nodes[i][2]) {
        push(i);
      }
    }
  }

  Int pop() {
    if (heap_.empty()) {
      return -1;
    }
    const auto index = std::get<3>(heap_.top());
    heap_.pop();
    const auto parent = indices_.find(nodes_[index][1]);
    if (parent != indices_.end() && !--children_[parent->second] && !nodes_[parent->second][2]) {
      push(parent->second);
    }
    return index;
  }

  bool next_is_log() const {
    return !heap_.empty() && !std::get<0>(heap_.top());
  }

private:
  void push(Int index) {
    const auto& row = nodes_[index];
    heap_.emplace(row[3], row[4], row[0], index);
  }

  Rows nodes_;
  Vector children_;
  std::unordered_map<Int, Int> indices_;
  using Priority = std::tuple<Int, Int, Int, Int>;
  std::priority_queue<Priority, std::vector<Priority>, std::greater<Priority>> heap_;
};

Rows eviction_order(const Rows& nodes) {
  EvictionQueue queue(nodes);
  Vector order;
  for (Int index; (index = queue.pop()) >= 0;) {
    order.push_back(index);
  }
  return Rows({List(order.begin(), order.end())});
}

Rows plan_eviction(const Rows& nodes, const Rows& kv, const Rows& node_slots,
                   const Rows& slots, const Rows& occupied, const List& config,
                   const Rows& checkpoint_pairs) {


  check(nodes.size() == kv.size() && nodes.size() == node_slots.size()
        && config.size() == 10 && config[5] > 0, "Invalid eviction capacity metadata");
  std::unordered_map<Int, List> slot_info;
  for (const auto& row : slots) {
    check(row.size() == 5 && row[1] > 0, "Invalid eviction slot");
    slot_info.emplace(row[0], row);
  }
  std::unordered_map<Int, Int> decay_occupied, decay_released, references, pages;
  for (const auto& row : occupied) {
    check(row.size() == 2 && row[1] > 0 && row[1] <= config[5], "Invalid decay occupancy");
    decay_occupied.emplace(row[0], row[1]);
  }
  Int free = config[0], decay = config[1], kv_free = 0;
  auto fits = [&]() {
    const auto required_decay = (std::max(Int{0}, config[7] - decay) + config[5] - 1) / config[5];
    const auto log_frames = config[9];
    return config[6] + config[7] * log_frames + config[8] * config[2]
           + (config[7] * config[4] + config[8]) * config[3] + required_decay <= free + kv_free;
  };
  EvictionQueue queue(nodes);
  Vector selected, released, page_order, compressed;
  std::unordered_set<Int> released_set;
  auto release = [&](Int slot) {
    check(slot_info.contains(slot), "Missing eviction slot");
    check(released_set.insert(slot).second, "Duplicate eviction slot release");
    const auto& record = slot_info.at(slot);
    released.push_back(slot);
    free += record[2];
    if (record[3] >= 0) {
      check(decay_occupied.contains(record[3]), "Missing decay frame occupancy");
      if (++decay_released[record[3]] == decay_occupied.at(record[3])) {
        ++free;
        decay -= config[5] - 1;
      }
      else {
        ++decay;
      }
    }
  };
  auto prune = [&](Int index) {
    selected.push_back(index);
    check(kv[index].size() % 3 == 0, "Invalid eviction KV references");
    for (size_t i = 0; i < kv[index].size(); i += 3) {
      const auto frame = kv[index][i];
      auto [page, inserted] = pages.try_emplace(frame, 0);
      if (inserted) {
        page_order.push_back(frame);
      }
      auto& count = page->second;
      count += kv[index][i + 1];
      check(count <= kv[index][i + 2], "Eviction exceeds cached KV references");
      kv_free += count == kv[index][i + 2];
    }
    for (const auto slot : node_slots[index]) {
      if (released_set.contains(slot)) {
        continue;
      }
      check(slot_info.contains(slot), "Missing eviction slot");
      const auto& record = slot_info.at(slot);
      const auto refs = ++references[slot];
      check(refs <= record[1], "Eviction exceeds cached slot references");
      if (refs != record[1]) {
        continue;
      }
      release(slot);
    }
  };
  auto result = [&]() {
    Vector page_counts;
    for (const auto frame : page_order) {
      page_counts.insert(page_counts.end(), {frame, pages.at(frame)});
    }
    std::sort(released.begin(), released.end());
    return Rows({List(selected.begin(), selected.end()),
                 List(page_counts.begin(), page_counts.end()),
                 List(released.begin(), released.end()),
                 List(compressed.begin(), compressed.end())});
  };


  while (queue.next_is_log()) {
    prune(queue.pop());
    if (fits()) {
      return result();
    }
  }
  std::vector<Vector> pairs;
  for (const auto& row : checkpoint_pairs) {
    check(row.size() == 4 && row[0] != row[1], "Invalid checkpoint eviction pair");
    pairs.emplace_back(row.begin(), row.end());
  }
  std::sort(pairs.begin(), pairs.end(), [](const Vector& left, const Vector& right) {
    return std::tie(left[3], left[2], left[0]) < std::tie(right[3], right[2], right[0]);
  });


  std::vector<Vector> ordered;
  ordered.reserve(pairs.size());
  while (!pairs.empty()) {
    if (pairs.size() == 1) {
      ordered.push_back(std::move(pairs.front()));
      break;
    }
    std::vector<Vector> retained;
    retained.reserve((pairs.size() + 1) / 2);
    for (size_t i = 0; i < pairs.size(); ++i) {
      if (i % 2) {
        ordered.push_back(std::move(pairs[i]));
      }
      else {
        retained.push_back(std::move(pairs[i]));
      }
    }
    pairs = std::move(retained);
  }
  std::unordered_map<Int, Int> successors;
  for (const auto& pair : ordered) {
    if (released_set.contains(pair[0])) {
      continue;
    }
    auto right = pair[1];


    while (successors.contains(right)) {
      right = successors.at(right);
    }
    check(slot_info.contains(pair[0]) && slot_info.contains(right),
          "Missing checkpoint eviction slot");
    check(slot_info.at(pair[0])[1] == 1 && slot_info.at(right)[1] == 1,
          "Cannot compress shared checkpoint slots");
    compressed.insert(compressed.end(), {pair[0], right});
    successors.emplace(pair[0], right);
    release(pair[0]);
    if (fits()) {
      return result();
    }
  }


  for (Int index; (index = queue.pop()) >= 0;) {
    prune(index);
    if (fits()) {
      return result();
    }
  }
  return {};
}

class RecycleBin {
public:
  explicit RecycleBin(Int capacity) : capacity_(capacity) {
    check(capacity >= 0, "Invalid recycle capacity");
  }

  Int remember(const tvm::ffi::Array<tvm::ffi::Tensor>& parts,
               tvm::ffi::TensorView entry_rows, const List& protected_ends, const List& stats) {
    if (!capacity_) {
      return 0;
    }
    auto tokens = read_prefix(parts);
    std::vector<std::array<Int, 2>> logs;
    for (const auto& row : read_rows(entry_rows)) {
      check(row.size() == 2 && row[0] >= 0 && row[0] < row[1]
            && row[1] <= static_cast<Int>(tokens.size()), "Invalid recycled Log range");
      logs.push_back({row[0], row[1]});
    }
    if (logs.empty()) {
      return 0;
    }
    Vector protected_values(protected_ends.begin(), protected_ends.end());
    std::sort(protected_values.begin(), protected_values.end());
    logs = merge_logs({}, logs);
    auto it = records_.lower_bound(tokens);
    while (it != records_.end() && it->first.size() >= tokens.size()
           && std::equal(tokens.begin(), tokens.end(), it->first.begin())) {
      if (it->first.size() == tokens.size()) {
        ++it;
        continue;
      }
      auto& record = it->second;


      auto refreshed = merge_logs(record.logs, logs);
      Vector ends = record.protected_ends;
      ends.insert(ends.end(), protected_values.begin(), protected_values.end());
      std::sort(ends.begin(), ends.end());
      ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
      used_ -= bytes(it->first, record);
      ids_.erase(record.id);
      record.logs = std::move(refreshed);
      record.protected_ends = std::move(ends);
      record.id = ++serial_;
      *record.lru = record.id;
      ids_[record.id] = it;
      used_ += bytes(it->first, record);
      ++it;
    }
    Record record{++serial_, logs, protected_values, Vector(stats.begin(), stats.end()), {}};
    if (const auto old = records_.find(tokens); old != records_.end()) {
      record.logs = merge_logs(old->second.logs, logs);
      record.protected_ends.insert(record.protected_ends.end(),
                                  old->second.protected_ends.begin(), old->second.protected_ends.end());
      std::sort(record.protected_ends.begin(), record.protected_ends.end());
      record.protected_ends.erase(std::unique(record.protected_ends.begin(), record.protected_ends.end()),
                                  record.protected_ends.end());
      if (record.stats.empty()) {
        record.stats = old->second.stats;
      }
    }
    if (bytes(tokens, record) > capacity_) {
      trim();
      return 0;
    }
    if (const auto old = records_.find(tokens); old != records_.end()) {
      erase(old);
    }
    auto [inserted, fresh] = records_.emplace(std::move(tokens), std::move(record));
    check(fresh, "Duplicate recycled prefix");
    lru_.push_back(inserted->second.id);
    inserted->second.lru = std::prev(lru_.end());
    ids_[inserted->second.id] = inserted;
    used_ += bytes(inserted->first, inserted->second);
    trim();
    return inserted->second.id;
  }

  Vector match(tvm::ffi::TensorView query, Int start, Int granularity) {
    check(query.ndim() == 1 && query.is_contiguous() && query.device().device_type == kDLCPU
          && query.dtype().code == kDLInt && (query.dtype().bits == 32 || query.dtype().bits == 64)
          && start >= 0 && granularity > 0, "Invalid recycle query");
    Vector tokens(query.size(0));
    for (Int i = 0; i < query.size(0); ++i) {
      tokens[i] = query.dtype().bits == 32 ? static_cast<const int32_t*>(query.data_ptr())[i]
                                         : static_cast<const Int*>(query.data_ptr())[i];
    }
    auto right = records_.lower_bound(tokens);
    auto best = records_.end();
    Int matched = start;
    auto consider = [&](auto candidate) {
      const auto limit = std::min(tokens.size(), candidate->first.size());
      const auto pair = std::mismatch(tokens.begin(), tokens.begin() + limit, candidate->first.begin());
      const auto length = (pair.first - tokens.begin()) / granularity * granularity;
      for (const auto& log : candidate->second.logs) {
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
                 List(it->second.protected_ends.begin(), it->second.protected_ends.end()),
                 List(it->second.stats.begin(), it->second.stats.end())});
    for (const auto& log : it->second.logs) {
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

private:
  static std::vector<std::array<Int, 2>> merge_logs(
      std::vector<std::array<Int, 2>> logs, const std::vector<std::array<Int, 2>>& added) {
    logs.insert(logs.end(), added.begin(), added.end());
    std::sort(logs.begin(), logs.end());
    std::vector<std::array<Int, 2>> result;
    for (const auto& log : logs) {
      if (!result.empty() && log[0] < result.back()[1]) {
        result.back()[1] = std::max(result.back()[1], log[1]);
      }
      else {
        result.push_back(log);
      }
    }
    return result;
  }

  static Vector read_prefix(const tvm::ffi::Array<tvm::ffi::Tensor>& parts) {
    Vector tokens;
    for (const auto& part : parts) {
      check(part.ndim() == 1 && part.is_contiguous() && part.device().device_type == kDLCPU
            && part.dtype().code == kDLInt && (part.dtype().bits == 32 || part.dtype().bits == 64),
            "Recycle tokens must be contiguous CPU int32/int64");
      for (Int i = 0; i < part.size(0); ++i) {
        tokens.push_back(part.dtype().bits == 32 ? static_cast<const int32_t*>(part.data_ptr())[i]
                                               : static_cast<const Int*>(part.data_ptr())[i]);
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

  static Int bytes(const Vector& tokens, const Record& record) {

    return 256 + 8 * tokens.size() + 16 * record.logs.size()
           + 8 * (record.protected_ends.size() + record.stats.size());
  }

  void erase(Records::iterator it) {
    used_ -= bytes(it->first, it->second);
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
  result.Set("usage", tvm::ffi::Function::FromTyped(
    [bin]() {
      const auto values = bin->usage();
      return List(values.begin(), values.end());
    }));
  return result;
}

struct Planner {
  static tvm::ffi::Tensor branches(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs, Int granularity) {
    return write_rows(batch_branch_points(inputs, granularity));
  }

  static tvm::ffi::Tensor protections(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs,
                                     tvm::ffi::TensorView forced, Int granularity) {
    return write_rows(branch_protections(inputs, read_rows(forced), granularity));
  }

  static tvm::ffi::Tensor construction(tvm::ffi::TensorView entries, tvm::ffi::TensorView restores,
                                     tvm::ffi::TensorView materialize) {
    return write_rows(construction_plan(entries, restores, materialize));
  }

  static tvm::ffi::Tensor eviction(tvm::ffi::TensorView nodes, tvm::ffi::TensorView kv,
                                 tvm::ffi::TensorView node_slots, tvm::ffi::TensorView slots,
                                 tvm::ffi::TensorView occupied, const List& config,
                                 tvm::ffi::TensorView checkpoint_pairs) {
    return write_rows(plan_eviction(read_rows(nodes), read_rows(kv), read_rows(node_slots),
                                    read_rows(slots), read_rows(occupied), config,
                                    read_rows(checkpoint_pairs)));
  }

  static tvm::ffi::Tensor eviction_candidates(tvm::ffi::TensorView nodes) {
    return write_rows(eviction_order(read_rows(nodes)));
  }

  static tvm::ffi::Tensor materialize(tvm::ffi::TensorView paths, tvm::ffi::TensorView boundaries,
                                    tvm::ffi::TensorView candidates, Int page_size) {
    return write_rows(plan_materialize(read_rows(paths), read_rows(boundaries), read_rows(candidates), page_size));
  }

  static Result coalesce(tvm::ffi::TensorView paths, tvm::ffi::TensorView boundaries,
                         tvm::ffi::TensorView locks, tvm::ffi::TensorView anchors,
                         const List& materialized, Int page_size, bool diagnostics) {
    auto result = plan_coalesce(read_rows(paths), read_rows(boundaries), read_rows(locks),
                                read_rows(anchors), materialized, page_size, diagnostics);
    result.Set("groups", write_rows(result.at("groups").cast<Rows>()));
    return result;
  }

  static tvm::ffi::Tensor cover(tvm::ffi::TensorView path, Int begin, Int end) {
    const auto rows = read_rows(path);
    check(rows.size() == 1 && begin <= end, "Invalid log coverage interval");
    const auto values = entries(rows[0]);
    Vector order;
    for (Int i = 0; i < static_cast<Int>(values.size()); ++i) {
      if (values[i].span < 0) {
        order.push_back(i);
      }
    }
    std::stable_sort(order.begin(), order.end(), [&](Int a, Int b) {
      return std::tie(values[a].begin, values[a].end) < std::tie(values[b].begin, values[b].end);
    });
    Vector selected;
    Int cursor = begin;
    for (const auto i : order) {
      const auto& entry = values[i];
      if (entry.end <= cursor || entry.begin > cursor) {
        continue;
      }
      selected.push_back(i);
      cursor = std::min(end, entry.end);
      if (cursor == end) {
        break;
      }
    }
    selected.insert(selected.begin(), cursor);
    return write_rows(Rows({List(selected.begin(), selected.end())}));
  }
};

}
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_branch_points, logssms::Planner::branches);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_materialize, logssms::Planner::materialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_coalesce, logssms::Planner::coalesce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cover_logs, logssms::Planner::cover);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(branch_protections, logssms::Planner::protections);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(construction_plan, logssms::Planner::construction);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_eviction, logssms::Planner::eviction);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eviction_candidates, logssms::Planner::eviction_candidates);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_recycle_bin, logssms::make_recycle_bin);
