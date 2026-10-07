#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
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
  // Flat path row: slot, begin, end, span (-1 for Log), physical begin,
  // round eligibility, slot protection, physical-slot fragmentation.
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
  std::iota(order.begin(), order.end(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& t = inputs[i];
    check(t.ndim() == 1 && t.is_contiguous() && t.device().device_type == kDLCPU
          && t.dtype().code == kDLInt && (t.dtype().bits == 32 || t.dtype().bits == 64),
          "Branch tokens must be contiguous 1-D CPU int32/int64 tensors");
    lengths[i] = t.size(0) / granularity * granularity;
  }
  auto token = [&](Int i, Int j) -> Int {
    const auto& t = inputs[i];
    return t.dtype().bits == 32 ? static_cast<const int32_t*>(t.data_ptr())[j]
                                : static_cast<const int64_t*>(t.data_ptr())[j];
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
    // Anchor: path index, matched endpoint, maximum arity, has statistics.
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
  // Borrow packed CPU buffers only for this call. No Node/device ownership.
  // A row is [base, (Log slot, end, output count, outputs...)...].
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
    // New-checkpoint consumers follow its writer; old-Log consumers precede it.
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

struct Planner {
  static tvm::ffi::Tensor branches(const tvm::ffi::Array<tvm::ffi::Tensor>& inputs, Int granularity) {
    return write_rows(batch_branch_points(inputs, granularity));
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

  static tvm::ffi::Tensor waves(tvm::ffi::TensorView chains, tvm::ffi::TensorView plans) {
    return write_rows(schedule_construction_waves(chains, plans));
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

} // namespace
} // namespace logssms

TVM_FFI_DLL_EXPORT_TYPED_FUNC(batch_branch_points, logssms::Planner::branches);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_materialize, logssms::Planner::materialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_coalesce, logssms::Planner::coalesce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(schedule_construction_waves, logssms::Planner::waves);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cover_logs, logssms::Planner::cover);
