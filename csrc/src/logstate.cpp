#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include "logstate/allocator.hpp"
#include "logstate/eviction_planner.hpp"
#include "logstate/radix_tree.hpp"
#include "logstate/reuse_model.hpp"

namespace logssms {
namespace {

using namespace logssms::logstate;
using List = tvm::ffi::Array<Int>;
using Rows = tvm::ffi::Array<List>;
using Result = tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>;

template <class Fn>
void reg(Result& r, const char* name, Fn&& fn) {
  r.Set(name, tvm::ffi::Function::FromTyped(std::forward<Fn>(fn)));
}

struct CPUNDAlloc {
  void AllocData(DLTensor* t) {
    size_t bytes = 1;
    for (int i = 0; i < t->ndim; ++i) bytes *= static_cast<size_t>(t->shape[i]);
    bytes *= static_cast<size_t>((t->dtype.bits * t->dtype.lanes + 7) / 8);
    t->data = bytes ? std::malloc(bytes) : nullptr;
  }
  void FreeData(DLTensor* t) { std::free(t->data); }
};

tvm::ffi::Tensor make_i32_tensor(Int n) {
  return tvm::ffi::Tensor::FromNDAlloc(CPUNDAlloc(), tvm::ffi::ShapeView(&n, 1),
                                       DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0});
}
tvm::ffi::Tensor make_i32_2d(Int r, Int c) {
  std::array<Int, 2> shape{r, c};
  return tvm::ffi::Tensor::FromNDAlloc(CPUNDAlloc(), tvm::ffi::ShapeView(shape.data(), 2),
                                       DLDataType{kDLInt, 32, 1}, DLDevice{kDLCPU, 0});
}
const int32_t* cpu_i32(tvm::ffi::TensorView t, std::vector<int32_t>& scratch) {
  check(t.ndim() == 1 && t.is_contiguous() && t.device().device_type == kDLCPU &&
            t.dtype().code == kDLInt && (t.dtype().bits == 32 || t.dtype().bits == 64),
        "Expected contiguous 1-D CPU int32/int64 tensor");
  if (t.dtype().bits == 32) return static_cast<const int32_t*>(t.data_ptr());
  const auto* p64 = static_cast<const int64_t*>(t.data_ptr());
  scratch.resize(static_cast<size_t>(t.size(0)));
  for (size_t i = 0; i < scratch.size(); ++i) scratch[i] = static_cast<int32_t>(p64[i]);
  return scratch.data();
}
tvm::ffi::Tensor to_i32_tensor(const std::vector<int32_t>& v) {
  auto t = make_i32_tensor(static_cast<Int>(v.size()));
  if (!v.empty()) std::memcpy(t.data_ptr(), v.data(), v.size() * sizeof(int32_t));
  return t;
}
std::vector<int32_t> from_i32_tensor(tvm::ffi::TensorView t) {
  std::vector<int32_t> scratch;
  const int32_t n = static_cast<int32_t>(t.size(0));
  if (n == 0) return {};
  const int32_t* p = cpu_i32(t, scratch);
  return {p, p + n};
}
List to_list(const std::vector<int32_t>& v) {
  std::vector<Int> tmp(v.begin(), v.end());
  return List(tmp.begin(), tmp.end());
}
std::vector<int32_t> from_list(const List& l) {
  std::vector<int32_t> out;
  out.reserve(l.size());
  for (Int v : l) out.push_back(static_cast<int32_t>(v));
  return out;
}

struct AllocRegistry {
  std::mutex mu;
  int64_t next_id = 1;
  std::unordered_map<int64_t, std::weak_ptr<AllocatorCore>> map;
  int64_t insert(const std::shared_ptr<AllocatorCore>& p) {
    std::lock_guard<std::mutex> lk(mu);
    map[next_id] = p;
    return next_id++;
  }
  std::shared_ptr<AllocatorCore> get(int64_t id) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = map.find(id);
    check(it != map.end(), "Invalid AllocatorCore handle");
    auto sp = it->second.lock();
    check(sp != nullptr, "Expired AllocatorCore handle");
    return sp;
  }
};
AllocRegistry& alloc_registry() { static AllocRegistry reg; return reg; }

std::vector<std::vector<int32_t>> compute_branch_protections(
    const std::vector<std::pair<const int32_t*, int32_t>>& inputs,
    const std::vector<std::pair<int32_t, int32_t>>& forced, int32_t g) {
  check(g > 0, "Invalid LogState granularity");
  const int32_t n = static_cast<int32_t>(inputs.size());
  std::vector<int32_t> order(n), lengths(n);
  std::iota(order.begin(), order.end(), 0);
  for (int32_t i = 0; i < n; ++i) lengths[i] = (inputs[i].second / g) * g;
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    int32_t m = std::min(lengths[a], lengths[b]);
    const int32_t *pa = inputs[a].first, *pb = inputs[b].first;
    for (int32_t j = 0; j < m; ++j) if (pa[j] != pb[j]) return pa[j] < pb[j];
    return lengths[a] != lengths[b] ? lengths[a] < lengths[b] : a < b;
  });
  std::map<std::vector<int32_t>, std::set<int32_t>> cands;
  auto make_key = [&](int32_t req, int32_t pos) {
    check(req >= 0 && req < n && pos >= 0 && pos <= inputs[req].second, "Invalid branch endpoint");
    return std::vector<int32_t>(inputs[req].first, inputs[req].first + pos);
  };
  std::vector<std::pair<int32_t, int32_t>> stack;
  for (int32_t i = 0; i < n; ++i) {
    int32_t depth = 0, begin = i;
    if (i + 1 < n) {
      int32_t a = order[i], b = order[i + 1], m = std::min(lengths[a], lengths[b]);
      while (depth < m && inputs[a].first[depth] == inputs[b].first[depth]) ++depth;
      depth = (depth / g) * g;
    }
    while (!stack.empty() && stack.back().first > depth) {
      int32_t bd = stack.back().first;
      begin = stack.back().second;
      stack.pop_back();
      auto& s = cands[make_key(order[begin], bd)];
      for (int32_t k = begin; k <= i; ++k) s.insert(order[k]);
    }
    if (depth > 0 && (stack.empty() || stack.back().first < depth)) stack.emplace_back(depth, begin);
  }
  for (const auto& [req, pos] : forced) {
    check(pos % g == 0, "Invalid forced branch");
    cands[make_key(req, pos)].insert(req);
  }
  std::vector<std::vector<int32_t>> out;
  for (const auto& [key, reqs] : cands) {
    if (key.empty()) continue;
    std::vector<int32_t> row{static_cast<int32_t>(key.size())};
    row.insert(row.end(), reqs.begin(), reqs.end());
    out.push_back(std::move(row));
  }
  std::sort(out.begin(), out.end());
  return out;
}

struct PendingMatch {
  int32_t position = 0, match_node = 0, trace_node = 0, trace_len = 0, trace_input_len = 0;
  int64_t trace_rev = -1;
};

class LogStateCore {
 public:
  std::shared_ptr<AllocatorCore> alloc;
  RadixTreeCore tree;
  ReuseModelCore reuse;
  EvictionPlannerCore planner;
  bool enable_hit_materialization = false;

  std::vector<ActiveRequestState> req_states;
  std::unordered_map<int32_t, PendingMatch> pending;
  std::unordered_map<int64_t, std::vector<int32_t>> protected_prefixes;
  std::unordered_set<int64_t> sampled_queries;

  int32_t dummy_kv_frame = -1, dummy_log_slot = -1, dummy_work_slot = -1, pressure_work_slot = -1;
  int64_t stat_eviction_rounds = 0, stat_pressure_dropped_checkpoints = 0;
  int64_t stat_pressure_pruned_checkpoints = 0, stat_evicted_kv_tokens = 0, stat_released_slots_eviction = 0;

  LogStateCore(std::shared_ptr<AllocatorCore> alloc_, int32_t g, int32_t ps,
               bool enable_merge_adj, int32_t min_sampling_len, double max_exp,
               int32_t half_life, bool enable_hit_mat)
      : alloc(std::move(alloc_)),
        tree(g, ps, alloc->materialize_log_count, alloc.get()),
        reuse(g, min_sampling_len, half_life, max_exp, enable_merge_adj),
        planner(tree, *alloc, reuse),
        enable_hit_materialization(enable_hit_mat) {
    if (enable_merge_adj) {
      tree.analysis_factory = [this](int32_t o, int64_t gid) { return reuse.new_chain(o, gid); };
      tree.chain_group_fn = [this](int32_t cid) { return reuse.chain(cid).group_id; };
      tree.chain_unused_fn = [this](int32_t cid) { const auto& c = reuse.chain(cid); return c.tail == c.origin; };
    }
  }

  void ensure_req_slot(int32_t row) { if (row >= static_cast<int32_t>(req_states.size())) req_states.resize(row + 1); }
  static int32_t div_ceil(int32_t a, int32_t b) { return (a + b - 1) / b; }

  int32_t decode_reservation(int32_t begin, int32_t end, bool has_ckpt) const {
    int32_t kv = std::max<int32_t>(0, div_ceil(end, tree.page_size) - div_ceil(begin, tree.page_size));
    int32_t logs = std::min(kv, alloc->materialize_log_count + 1);
    return std::min(alloc->required_frames(kv, logs, (logs < kv && !has_ckpt) ? 1 : 0, 0),
                    alloc->required_frames(kv, kv, 0, 0));
  }

  int32_t prefill_peak(int32_t begin, int32_t prompt_end, int32_t end, int32_t prompt_frames, bool is_new) const {
    int32_t pp = std::max<int32_t>(0, div_ceil(prompt_end, tree.page_size) - div_ceil(begin, tree.page_size));
    int32_t dp = std::max<int32_t>(0, div_ceil(end, tree.page_size) - div_ceil(prompt_end, tree.page_size));
    int32_t w = alloc->materialize_log_count + 1, tr = std::min(dp, w);
    int32_t tr_peak = prompt_frames + alloc->required_frames(tr, tr, 0, 0);
    if (dp <= w) return tr_peak;
    int32_t cf = alloc->payload_frames + alloc->conv_frames;
    int32_t steady = pp + (1 + static_cast<int32_t>(is_new)) * cf + alloc->required_frames(dp, w, 0, 0);
    return std::min(prompt_frames + alloc->required_frames(dp, dp, 0, 0), std::max(tr_peak, steady));
  }

  int32_t reservation_size(int32_t token_len, int32_t handle_cached_len, int32_t row, int32_t end_at) const {
    const bool is_new = (row < 0);
    int32_t begin = is_new ? handle_cached_len
                           : (req_states[row].live_granules.empty()
                                  ? std::max<int32_t>(0, req_states[row].request_end - token_len)
                                  : req_states[row].live_granules.back().end);
    int32_t end = is_new ? (begin + token_len) : req_states[row].request_end;
    if (end_at >= 0) end = std::min(end, end_at);
    if (begin >= end) return 0;
    int32_t kv = div_ceil(end, tree.page_size) - (begin / tree.page_size);
    return alloc->required_frames(kv, kv, is_new ? 1 : 0, 0);
  }

  int32_t admission_capacity() const { return alloc->projected_free_capacity().first + tree.evictable_kv_frames(); }
  int32_t available_size() const { return alloc->admissible_pages(tree.evictable_kv_frames()) * tree.page_size; }

  void release_cached_kv(const std::vector<std::pair<int32_t, int32_t>>& kv_rel) {
    std::vector<int32_t> freed;
    for (const auto& [f, cnt] : kv_rel) {
      tree.add_cached_page_tokens(f, -cnt);
      if (tree.cached_page_tokens(f) == 0) freed.push_back(f);
    }
    if (!freed.empty()) alloc->release_kv(freed);
  }
};

Result make_allocator_ffi(Int num_frames, Int frame_bytes, Int payload_bytes,
                          Int conv_snapshot_bytes, Int decay_bytes, Int num_granules, Int log_payload_bytes) {
  auto alloc = std::make_shared<AllocatorCore>(
      static_cast<int32_t>(num_frames), frame_bytes, payload_bytes,
      conv_snapshot_bytes, decay_bytes, static_cast<int32_t>(num_granules), log_payload_bytes);
  Result r;
  r.Set("_id", alloc_registry().insert(alloc));
  r.Set("num_frames", static_cast<Int>(alloc->num_frames));
  r.Set("frame_bytes", static_cast<Int>(alloc->frame_bytes));
  r.Set("payload_frames", static_cast<Int>(alloc->payload_frames));
  r.Set("log_payload_frames", static_cast<Int>(alloc->log_payload_frames));
  r.Set("descriptor_frames", static_cast<Int>(alloc->descriptor_frames));
  r.Set("checkpoint_log_count", static_cast<Int>(alloc->checkpoint_log_count));
  r.Set("conv_frames", static_cast<Int>(alloc->conv_frames));
  r.Set("decay_records_per_frame", static_cast<Int>(alloc->decay_records_per_frame));
  r.Set("num_granules", static_cast<Int>(alloc->num_granules));
  r.Set("materialize_log_count", static_cast<Int>(alloc->materialize_log_count));

  reg(r, "free_frames", [alloc]() -> Int { return alloc->free_frames_count(); });
  reg(r, "free_decay_records", [alloc]() -> Int { return alloc->free_decay_records(); });
  reg(r, "set_reclaimable", [alloc](Int s, bool v) { alloc->set_reclaimable(static_cast<int32_t>(s), v); });
  reg(r, "required_frames", [alloc](Int kv, Int l, Int st, Int fd) -> Int {
    return alloc->required_frames(static_cast<int32_t>(kv), static_cast<int32_t>(l),
                                  static_cast<int32_t>(st), static_cast<int32_t>(fd));
  });
  reg(r, "projected_free_capacity", [alloc]() -> List {
    auto [ff, fd] = alloc->projected_free_capacity();
    return List({static_cast<Int>(ff), static_cast<Int>(fd)});
  });
  reg(r, "lock_frame_cost", [alloc](const List& s) -> Int { return alloc->lock_frame_cost(from_list(s)); });
  reg(r, "admissible_pages", [alloc](Int ekv) -> Int { return alloc->admissible_pages(static_cast<int32_t>(ekv)); });
  reg(r, "reserve", [alloc](Int kv, Int l, Int st) -> Result {
    auto res = alloc->reserve(static_cast<int32_t>(kv), static_cast<int32_t>(l), static_cast<int32_t>(st));
    Result out;
    out.Set("ok", res.ok);
    if (res.ok) {
      out.Set("kv", to_i32_tensor(res.kv_frames));
      out.Set("logs", to_i32_tensor(res.log_slots));
      out.Set("works", to_i32_tensor(res.work_slots));
    }
    return out;
  });
  reg(r, "checkpoint_frame_capacity", [alloc](const List& s) -> Int { return alloc->checkpoint_frame_capacity(from_list(s)); });
  reg(r, "checkpoint_frame_savings", [alloc](const List& s) -> Int { return alloc->checkpoint_frame_savings(from_list(s)); });
  reg(r, "materialize_group", [alloc](const List& s) -> Result {
    int64_t rid = alloc->materialize_group(from_list(s));
    Result out;
    out.Set("id", rid);
    out.Set("frames", to_i32_tensor(alloc->retired_frames(rid)));
    return out;
  });
  reg(r, "release_retirement", [alloc](Int rid) { alloc->release_retirement(rid); });
  reg(r, "release_kv", [alloc](const List& f) { alloc->release_kv(from_list(f)); });
  reg(r, "release_slot", [alloc](Int s) { alloc->release_slot(static_cast<int32_t>(s)); });
  reg(r, "release_slots", [alloc](const List& sl) { for (Int s : sl) alloc->release_slot(static_cast<int32_t>(s)); });
  reg(r, "seed_checkpoint_slot", [alloc](Int s) { alloc->seed_checkpoint_slot(static_cast<int32_t>(s)); });
  reg(r, "pack_descriptors", [alloc](const List& sl) -> Result {
    auto sids = from_list(sl);
    const Int n = static_cast<Int>(sids.size());
    auto idx_t = to_i32_tensor(sids);
    auto p_t = make_i32_tensor(n * alloc->descriptor_frames);
    auto c_t = make_i32_tensor(n * alloc->num_granules * alloc->conv_frames);
    auto d_t = make_i32_tensor(n * 2);
    alloc->pack_descriptors(sids, static_cast<int32_t*>(p_t.data_ptr()),
                            static_cast<int32_t*>(c_t.data_ptr()), static_cast<int32_t*>(d_t.data_ptr()));
    Result out;
    out.Set("indices", idx_t); out.Set("payload", p_t); out.Set("conv", c_t); out.Set("decay", d_t);
    return out;
  });
  reg(r, "slot_info", [alloc](Int slot) -> tvm::ffi::Tensor {
    const auto& sr = alloc->slot(static_cast<int32_t>(slot));
    const Int np = static_cast<Int>(sr.payload.size()), nc = static_cast<Int>(sr.conv.size());
    auto t = make_i32_tensor(5 + np + nc);
    auto* p = static_cast<int32_t*>(t.data_ptr());
    p[0] = static_cast<int32_t>(sr.kind); p[1] = sr.conv_granules;
    p[2] = sr.decay.frame; p[3] = sr.decay.index; p[4] = static_cast<int32_t>(np);
    if (np > 0) std::memcpy(p + 5, sr.payload.data(), static_cast<size_t>(np) * sizeof(int32_t));
    if (nc > 0) std::memcpy(p + 5 + np, sr.conv.data(), static_cast<size_t>(nc) * sizeof(int32_t));
    return t;
  });
  reg(r, "usage_stats", [alloc]() -> Result {
    auto u = alloc->usage_stats();
    Result out;
    out.Set("total_frames", static_cast<Int>(u.total_frames)); out.Set("free_frames", static_cast<Int>(u.free_frames));
    out.Set("kv_frames", static_cast<Int>(u.kv_frames)); out.Set("payload_frames", static_cast<Int>(u.payload_frames));
    out.Set("conv_frames", static_cast<Int>(u.conv_frames)); out.Set("decay_frames", static_cast<Int>(u.decay_frames));
    out.Set("free_decay_records", static_cast<Int>(u.free_decay_records)); out.Set("log_slots", static_cast<Int>(u.log_slots));
    out.Set("checkpoint_slots", static_cast<Int>(u.checkpoint_slots)); out.Set("work_slots", static_cast<Int>(u.work_slots));
    out.Set("log_payload_conv", static_cast<Int>(u.log_payload_conv));
    out.Set("checkpoint_payload_conv", static_cast<Int>(u.checkpoint_payload_conv));
    out.Set("work_payload_conv", static_cast<Int>(u.work_payload_conv));
    return out;
  });
  reg(r, "allocated_kv", [alloc]() { return to_i32_tensor(alloc->allocated_kv_list()); });
  reg(r, "allocated_slots", [alloc]() { return to_i32_tensor(alloc->allocated_slot_list()); });
  reg(r, "reclaimable_slots", [alloc]() { return to_i32_tensor(alloc->reclaimable_slot_list()); });
  reg(r, "check_integrity", [alloc]() { alloc->check_integrity(); });
  return r;
}

Result make_core_ffi(Int alloc_id, Int g, Int ps, bool enable_merge_adj, Int min_sampling_len,
                     double max_exp, Int half_life, bool enable_hit_mat) {
  auto core = std::make_shared<LogStateCore>(
      alloc_registry().get(alloc_id), static_cast<int32_t>(g), static_cast<int32_t>(ps),
      enable_merge_adj, static_cast<int32_t>(min_sampling_len), max_exp,
      static_cast<int32_t>(half_life), enable_hit_mat);
  Result r;

  reg(r, "set_virtual_clock", [core](bool en, Int ns) { core->tree.set_virtual_clock(en, ns); });
  reg(r, "advance_virtual_clock", [core](Int ns) { core->tree.advance_virtual_clock(ns); });
  reg(r, "set_fixed_clock", [core](Int ts) { core->tree.set_fixed_clock(ts); });
  reg(r, "clear_fixed_clock", [core]() { core->tree.clear_fixed_clock(); });
  reg(r, "set_reserved_dummies", [core](Int kv, Int ls, Int ws, Int pw) {
    core->dummy_kv_frame = static_cast<int32_t>(kv); core->dummy_log_slot = static_cast<int32_t>(ls);
    core->dummy_work_slot = static_cast<int32_t>(ws); core->pressure_work_slot = static_cast<int32_t>(pw);
  });
  reg(r, "admission_capacity", [core]() -> Int { return core->admission_capacity(); });
  reg(r, "available_size", [core]() -> Int { return core->available_size(); });
  reg(r, "size_info", [core]() -> List { return List({core->tree.evictable_size, core->tree.locked_size}); });
  reg(r, "pin_recovery", [core](tvm::ffi::TensorView s, bool rel) {
    if (s.size(0) > 0) core->tree.pin_recovery(from_i32_tensor(s), rel);
  });
  reg(r, "lock_node", [core](Int nid, bool unl, Int pos) {
    if (nid > 0) core->tree.lock(static_cast<int32_t>(nid), unl, static_cast<int32_t>(pos));
  });
  reg(r, "unlock_handle", [core](Int nid, Int clen, tvm::ffi::TensorView rs) {
    if (rs.size(0) > 0) core->tree.pin_recovery(from_i32_tensor(rs), true);
    if (clen > 0 && nid > 0 && nid < core->tree.num_nodes_capacity() && core->tree.node(static_cast<int32_t>(nid)).alive)
      core->tree.lock(static_cast<int32_t>(nid), true, static_cast<int32_t>(clen));
  });

  reg(r, "match_prefix", [core](tvm::ffi::TensorView input_ids) -> Result {
    std::vector<int32_t> scratch;
    const int32_t* toks = cpu_i32(input_ids, scratch);
    const int32_t usable = (static_cast<int32_t>(input_ids.size(0)) / core->tree.granularity) * core->tree.granularity;
    auto tr = core->tree.trace(toks, usable);
    auto rec = core->tree.recovery_at(tr.node_id, tr.matched_len);
    int32_t rec_node = 0;
    if (rec.position > 0) {
      rec_node = tr.node_id;
      while (rec_node > 0 && rec.position <= core->tree.node(rec_node).begin()) rec_node = core->tree.node(rec_node).parent;
    }
    auto kv_t = make_i32_tensor(rec.position);
    if (rec.position > 0) core->tree.copy_matched_kv(rec_node, rec.position, static_cast<int32_t*>(kv_t.data_ptr()));
    std::vector<int32_t> rec_slots, ls, lb, le;
    if (rec.base_checkpoint_slot >= 0) rec_slots.push_back(rec.base_checkpoint_slot);
    for (const auto& e : rec.logs) { rec_slots.push_back(e.slot); ls.push_back(e.slot); lb.push_back(e.begin); le.push_back(e.end); }
    if (rec.conv_slot >= 0) rec_slots.push_back(rec.conv_slot);
    std::sort(rec_slots.begin(), rec_slots.end());
    rec_slots.erase(std::unique(rec_slots.begin(), rec_slots.end()), rec_slots.end());
    int32_t tnode = tr.node_id;
    while (tnode > 0 && tr.matched_len <= core->tree.node(tnode).begin()) tnode = core->tree.node(tnode).parent;

    Result out;
    out.Set("matched_len", static_cast<Int>(tr.matched_len));
    out.Set("trace_node", static_cast<Int>(tnode));
    out.Set("trace_is_exact", tr.is_exact);
    out.Set("trace_has_children", !core->tree.node(tnode).is_leaf());
    out.Set("trace_rev", core->tree.revision);
    out.Set("trace_input_len", static_cast<Int>(usable));
    out.Set("cached_len", static_cast<Int>(rec.position));
    out.Set("node_id", static_cast<Int>(rec_node));
    out.Set("base_ckpt", static_cast<Int>(rec.base_checkpoint_slot));
    out.Set("log_slots", to_i32_tensor(ls));
    out.Set("log_begins", to_i32_tensor(lb));
    out.Set("log_ends", to_i32_tensor(le));
    out.Set("conv_slot", static_cast<Int>(rec.conv_slot));
    out.Set("conv_granule", static_cast<Int>(rec.conv_granule));
    out.Set("recovery_slots", to_i32_tensor(rec_slots));
    out.Set("kv_indices", kv_t);
    out.Set("source_owner", static_cast<Int>(rec_node > 0 ? core->tree.node(rec_node).analysis_owner : -1));
    return out;
  });

  reg(r, "acquire_prefill",
      [core](Int uid, Int tnode, Int mlen, bool texact, bool tchildren, Int rnode, Int clen, Int bckpt,
             tvm::ffi::TensorView rslots_t, Int total_len, bool has_adm,
             const List& dec_rows_l, const List& dec_conf_l, Int gid, Int src_owner) -> Result {
        auto rslots = from_i32_tensor(rslots_t);
        Result out;
        for (int32_t s : rslots) if (!core->tree.has_slot(s)) { out.Set("admitted", false); return out; }
        auto dec_rows = from_list(dec_rows_l), dec_conf = from_list(dec_conf_l);
        for (size_t i = 0; i < dec_rows.size(); ++i) {
          if (dec_rows[i] >= 0 && dec_rows[i] < static_cast<int32_t>(core->req_states.size()))
            core->req_states[dec_rows[i]].confirmed_decode_prefix_len = dec_conf[i];
        }
        if (!has_adm) {
          int32_t avail = core->admission_capacity();
          if (core->alloc->free_frames_count() < total_len && total_len <= avail) {
            auto [kvf, lks] = core->tree.lock_footprint(static_cast<int32_t>(rnode), static_cast<int32_t>(clen));
            for (int32_t s : rslots) if (core->tree.is_evictable_slot(s)) lks.push_back(s);
            std::sort(lks.begin(), lks.end()); lks.erase(std::unique(lks.begin(), lks.end()), lks.end());
            avail -= kvf + core->alloc->lock_frame_cost(lks);
          }
          if (total_len > avail) { out.Set("admitted", false); return out; }
        }
        core->tree.pin_recovery(rslots, false);
        if (clen > 0) core->tree.lock(static_cast<int32_t>(rnode), false, static_cast<int32_t>(clen));
        int32_t avail = core->admission_capacity();
        bool ok = (total_len <= avail) ||
                  (has_adm && !dec_rows.empty() &&
                   total_len <= avail + core->planner.active_reclaimable_frames(core->req_states, dec_rows, core->protected_prefixes));
        if (!ok) {
          if (clen > 0) core->tree.lock(static_cast<int32_t>(rnode), true, static_cast<int32_t>(clen));
          core->tree.pin_recovery(rslots, true);
          out.Set("admitted", false);
          return out;
        }
        int32_t fn = static_cast<int32_t>(rnode);
        if (clen > 0 && core->tree.node(fn).depth > clen) fn = core->tree.node_at_depth(fn, static_cast<int32_t>(clen));
        if (mlen > 0 && (!texact || tchildren)) {
          int32_t tn = static_cast<int32_t>(tnode);
          if (core->tree.node(tn).alive && mlen <= core->tree.node(tn).depth) {
            core->tree.protect_at(tn, static_cast<int32_t>(mlen));
            core->protected_prefixes[uid].push_back(static_cast<int32_t>(mlen));
          }
        }
        if (clen > 0) core->tree.touch_path(fn, static_cast<int32_t>(bckpt));
        int32_t owner = -1;
        if (core->reuse.enabled) {
          if (core->sampled_queries.insert(uid).second)
            core->reuse.record_query(core->tree, static_cast<int32_t>(tnode), static_cast<int32_t>(mlen), gid);
          owner = (src_owner > 0) ? static_cast<int32_t>(src_owner)
                                  : core->reuse.new_chain(static_cast<int32_t>(clen), gid);
        }
        out.Set("admitted", true);
        out.Set("node_id", static_cast<Int>(fn));
        out.Set("analysis_owner", static_cast<Int>(owner));
        return out;
      });

  reg(r, "begin_prefill_admission",
      [core](const List& dr, const List& dcan, const List& ir, const List& ipe, const List& iml, const List& irem) -> Int {
        int32_t res = 0;
        for (size_t i = 0; i < dr.size(); ++i) {
          if (!dcan[i]) continue;
          const auto& st = core->req_states[static_cast<int32_t>(dr[i])];
          int32_t b = st.live_granules.empty() ? st.handle_cached_len : st.live_granules.back().end;
          bool rec_ok = (st.compacted_end > 0 || st.handle_cached_len == 0) ||
                        (core->tree.recovery_at(st.handle_node_id, st.handle_cached_len).position == st.handle_cached_len);
          if (rec_ok) res += core->decode_reservation(b, st.request_end, st.compacted_end > 0);
          else {
            int32_t pages = std::max<int32_t>(0, LogStateCore::div_ceil(st.request_end, core->tree.page_size) -
                                                     LogStateCore::div_ceil(b, core->tree.page_size));
            res += core->alloc->required_frames(pages, pages, 0, 0);
          }
        }
        for (size_t i = 0; i < ir.size(); ++i) {
          int32_t row = static_cast<int32_t>(ir[i]);
          const auto& st = core->req_states[row];
          int32_t b = st.live_granules.empty() ? st.handle_cached_len : st.live_granules.back().end;
          int32_t pe = static_cast<int32_t>(ipe[i]);
          int32_t prompt = core->reservation_size(static_cast<int32_t>(irem[i]), 0, row, pe);
          res += core->prefill_peak(b, pe, static_cast<int32_t>(iml[i]), prompt, false);
        }
        return res;
      });

  reg(r, "request_prefill_peak", [core](Int prev_row, Int hclen, Int ilen, Int olen) -> Int {
    const bool is_new = (prev_row < 0);
    int32_t b = is_new ? static_cast<int32_t>(hclen)
                       : (core->req_states[static_cast<int32_t>(prev_row)].live_granules.empty()
                              ? core->req_states[static_cast<int32_t>(prev_row)].handle_cached_len
                              : core->req_states[static_cast<int32_t>(prev_row)].live_granules.back().end);
    int32_t end = static_cast<int32_t>(ilen + olen);
    int32_t prompt = core->reservation_size(end - b, static_cast<int32_t>(hclen),
                                            static_cast<int32_t>(prev_row), static_cast<int32_t>(ilen));
    return core->prefill_peak(b, static_cast<int32_t>(ilen), end, prompt, is_new);
  });

  reg(r, "init_request",
      [core](Int row_i, Int uid, Int wslot, Int clen, Int req_end, Int hnid, Int owner,
             const List& prots, Int tnode, Int tlen, Int tilen, Int trev, Int, Int, Int) {
        int32_t row = static_cast<int32_t>(row_i);
        core->ensure_req_slot(row);
        auto& st = core->req_states[row];
        check(!st.active, "table_idx already active");
        st = ActiveRequestState{};
        st.active = true; st.table_idx = row; st.req_uid = uid;
        st.work_slot = static_cast<int32_t>(wslot);
        st.work_committed_pos = st.handle_cached_len = static_cast<int32_t>(clen);
        st.request_end = static_cast<int32_t>(req_end);
        st.handle_node_id = static_cast<int32_t>(hnid);
        st.analysis_owner = static_cast<int32_t>(owner);
        st.protected_endpoints = from_list(prots);
        core->pending[row] = {static_cast<int32_t>(clen), static_cast<int32_t>(hnid),
                              static_cast<int32_t>(tnode), static_cast<int32_t>(tlen),
                              static_cast<int32_t>(tilen), trev};
      });

  reg(r, "try_allocate_batch",
      [core](const List& rows, const List& dev_lens, const List& max_lens, const List& clens, Int pt_w_i) -> Result {
        const int32_t ps = core->tree.page_size, pt_w = static_cast<int32_t>(pt_w_i);
        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> info;
        std::vector<int32_t> fresh_works;
        int32_t needed = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
          int32_t row = static_cast<int32_t>(rows[i]), dl = static_cast<int32_t>(dev_lens[i]);
          int32_t ml = static_cast<int32_t>(max_lens[i]), cl = static_cast<int32_t>(clens[i]);
          auto& st = core->req_states[row];
          st.request_end = ml;
          int32_t b = st.live_granules.empty() ? cl : st.live_granules.back().end;
          if (dl > b) {
            int32_t count = 1 + (std::max<int32_t>(0, dl - b - (ps - (b % ps))) + ps - 1) / ps;
            info.emplace_back(row, b, count, ml);
            needed += count;
          }
          auto pit = core->pending.find(row);
          if (cl == 0 && (pit == core->pending.end() || pit->second.position == 0)) fresh_works.push_back(st.work_slot);
        }
        Result out;
        out.Set("fresh_work_slots", to_i32_tensor(fresh_works));
        out.Set("needed", static_cast<Int>(needed));
        if (needed == 0) { out.Set("ok", true); return out; }
        auto res = core->alloc->reserve(needed, needed, 0);
        if (!res.ok) { out.Set("ok", false); return out; }
        auto regions_t = make_i32_2d(needed, 5);
        auto* rptr = static_cast<int32_t*>(regions_t.data_ptr());
        int32_t off = 0;
        for (const auto& [row, begin, count, ml] : info) {
          auto& lgs = core->req_states[row].live_granules;
          for (int32_t k = 0, lb = begin; k < count; ++k) {
            int32_t le = ((lb / ps) + 1) * ps, kvo = res.kv_frames[off] * ps, ss = res.log_slots[off];
            lgs.push_back({lb, le, kvo, ss, -1});
            int32_t* regp = rptr + off++ * 5;
            regp[0] = row; regp[1] = lb; regp[2] = std::min({le, ml, pt_w}); regp[3] = kvo; regp[4] = ss;
            lb = le;
          }
        }
        auto p_t = make_i32_tensor(needed * core->alloc->descriptor_frames);
        auto c_t = make_i32_tensor(needed * core->alloc->num_granules * core->alloc->conv_frames);
        auto d_t = make_i32_tensor(needed * 2);
        core->alloc->pack_descriptors(res.log_slots, static_cast<int32_t*>(p_t.data_ptr()),
                                      static_cast<int32_t*>(c_t.data_ptr()), static_cast<int32_t*>(d_t.data_ptr()));
        out.Set("ok", true); out.Set("regions", regions_t);
        out.Set("slot_indices", to_i32_tensor(res.log_slots));
        out.Set("payload", p_t); out.Set("conv", c_t); out.Set("decay", d_t);
        return out;
      });

  reg(r, "plan_and_apply_eviction",
      [core](Int np, Int ns, Int nw, const List& dr_l, const List& dc_l, const List& dcan_l) -> Result {
        auto dr = from_list(dr_l), dc = from_list(dc_l);
        std::vector<int32_t> active_rows;
        for (size_t i = 0; i < dr.size(); ++i) {
          if (dr[i] >= 0 && dr[i] < static_cast<int32_t>(core->req_states.size()) && core->req_states[dr[i]].active) {
            core->req_states[dr[i]].confirmed_decode_prefix_len = dc[i];
            if (dcan_l[i]) { core->req_states[dr[i]].is_decode_active = true; active_rows.push_back(dr[i]); }
          }
        }
        for (const auto& st : core->req_states) {
          if (st.active && st.is_decode_active &&
              std::find(active_rows.begin(), active_rows.end(), st.table_idx) == active_rows.end())
            active_rows.push_back(st.table_idx);
        }
        std::sort(active_rows.begin(), active_rows.end());
        std::vector<ActiveCompactionAction> active_cands;
        if (!active_rows.empty())
          active_cands = core->planner.compute_active_candidates(core->req_states, active_rows, core->protected_prefixes, true);
        auto plan = core->planner.plan(static_cast<int32_t>(np), static_cast<int32_t>(ns), static_cast<int32_t>(nw), active_cands);
        Result out;
        out.Set("ok", plan.ok);
        if (!plan.ok) return out;

        int64_t ev_toks = 0, pr_ckpts = 0;
        for (const auto& [l, rr] : plan.checkpoint_drops) core->tree.drop_checkpoint(l, rr);
        for (int32_t nid : plan.pruned_nodes) {
          const auto& n = core->tree.node(nid);
          ev_toks += n.length;
          pr_ckpts += static_cast<int64_t>(n.checkpoint_slots.size());
          core->tree.prune_leaf(nid);
        }
        core->release_cached_kv(plan.kv_releases);
        ++core->stat_eviction_rounds;
        core->stat_pressure_dropped_checkpoints += static_cast<int64_t>(plan.checkpoint_drops.size());
        core->stat_pressure_pruned_checkpoints += pr_ckpts;
        core->stat_evicted_kv_tokens += ev_toks;
        core->stat_released_slots_eviction += static_cast<int64_t>(plan.retired_slots.size());

        out.Set("retired_slots", to_i32_tensor(plan.retired_slots));
        out.Set("has_materialization", !plan.materialization.empty());
        if (!plan.materialization.empty()) {
          const auto& row = plan.materialization;
          int32_t base = row[0]; bool absorb = (row[1] < 0);
          size_t fs = 2; while (fs < row.size() && row[fs] < 0) ++fs;
          std::vector<int32_t> ls, ll, slots;
          for (size_t j = 2; j < fs; ++j) {
            int32_t s = -row[j] - 2; const auto& e = core->tree.entry_for_slot(s);
            ls.push_back(s); ll.push_back(e.end - e.begin);
          }
          int32_t fb = core->tree.entry_for_slot(row[fs]).begin, le = core->tree.entry_for_slot(row.back()).end;
          int32_t onid = core->tree.slot_owner(row.back()); int64_t ots = core->tree.node(onid).timestamp;
          for (size_t j = fs; j < row.size(); ++j) {
            int32_t s = row[j]; const auto& e = core->tree.entry_for_slot(s);
            ls.push_back(s); ll.push_back(e.end - e.begin); slots.push_back(s);
          }
          int32_t begin = absorb ? core->tree.entry_for_slot(base).begin : fb;
          int32_t dest = absorb ? base : slots.back();
          for (int32_t s : slots) core->tree.detach_entry(s);
          if (absorb) core->tree.detach_entry(base);
          Result mcmd;
          mcmd.Set("base", static_cast<Int>(base)); mcmd.Set("absorb", absorb);
          mcmd.Set("owner_nid", static_cast<Int>(onid)); mcmd.Set("owner_ts", ots);
          mcmd.Set("dest", static_cast<Int>(dest)); mcmd.Set("begin", static_cast<Int>(begin));
          mcmd.Set("end", static_cast<Int>(le));
          mcmd.Set("slots", to_i32_tensor(slots)); mcmd.Set("log_slots", to_i32_tensor(ls));
          mcmd.Set("log_lens", to_i32_tensor(ll));
          out.Set("mat_cmd", mcmd);
        }
        out.Set("has_active", plan.active_action.has_value());
        if (plan.active_action.has_value()) {
          const auto& act = *plan.active_action;
          auto& st = core->req_states[act.table_idx]; auto& lgs = st.live_granules;
          std::vector<int32_t> sources, ls, ll;
          for (size_t i = 0; i + 1 < act.indices.size(); ++i) {
            sources.push_back(lgs[act.indices[i]].state_slot);
            lgs[act.indices[i]].state_slot = lgs[act.indices[i]].checkpoint_begin = -1;
          }
          int32_t last_s = lgs[act.indices.back()].state_slot;
          int32_t fb = lgs[act.indices.front()].begin, le = lgs[act.indices.back()].end;
          lgs[act.indices.back()].checkpoint_begin = act.begin;
          st.compacted_end = std::max(st.compacted_end, le);
          for (const auto& [s, b, e] : act.logs) { ls.push_back(s); ll.push_back(e - b); }
          Result acmd;
          acmd.Set("table_idx", static_cast<Int>(act.table_idx)); acmd.Set("base", static_cast<Int>(act.base_slot));
          acmd.Set("first_begin", static_cast<Int>(fb)); acmd.Set("last_end", static_cast<Int>(le));
          acmd.Set("sources", to_i32_tensor(sources)); acmd.Set("last_slot", static_cast<Int>(last_s));
          acmd.Set("log_slots", to_i32_tensor(ls)); acmd.Set("log_lens", to_i32_tensor(ll));
          out.Set("active_cmd", acmd);
        }
        return out;
      });

  reg(r, "finish_pressure_materialization", [core](Int onid, Int dest, Int b, Int e, Int ots) {
    StateEntry cp{static_cast<int32_t>(dest), EntryKind::kCheckpoint, static_cast<int32_t>(b),
                  static_cast<int32_t>(e), static_cast<int32_t>((e - b) / core->tree.granularity), 0};
    core->tree.attach_entry(static_cast<int32_t>(onid), cp, ots);
  });

  reg(r, "prepare_decode_replay_counts", [core](const List& rows, const List& clens, Int pad_sz, Int rw) -> List {
    std::vector<Int> counts;
    counts.reserve(static_cast<size_t>(pad_sz));
    for (size_t i = 0; i < rows.size(); ++i) {
      auto& st = core->req_states[static_cast<int32_t>(rows[i])];
      int32_t cl = static_cast<int32_t>(clens[i]);
      const auto& lgs = st.live_granules;
      check(!lgs.empty(), "Decode token has no LogState granule");
      int32_t cur = std::min<int32_t>(st.live_granule_cursor, static_cast<int32_t>(lgs.size()) - 1), idx = -1;
      if (lgs[cur].begin <= cl && cl < lgs[cur].end) idx = cur;
      else if (lgs.back().begin <= cl && cl < lgs.back().end) idx = static_cast<int32_t>(lgs.size()) - 1;
      else {
        int32_t j = (cl < lgs[cur].begin) ? 0 : cur;
        while (j < static_cast<int32_t>(lgs.size()) && lgs[j].end <= cl) ++j;
        if (j < static_cast<int32_t>(lgs.size()) && lgs[j].begin <= cl) idx = j;
      }
      check(idx >= 0, "Decode token has no LogState granule");
      st.live_granule_cursor = idx;
      int32_t cnt = cl - st.work_committed_pos;
      check(cnt >= 0 && cnt < rw, "Invalid replay count");
      counts.push_back(cnt);
      if ((cl - lgs[idx].begin + 1) % rw == 0) st.work_committed_pos = cl + 1;
    }
    counts.resize(static_cast<size_t>(pad_sz), 0);
    return List(counts.begin(), counts.end());
  });

  reg(r, "prepare_prefill_protections",
      [core](const List& rows, const List& uids, const tvm::ffi::Array<tvm::ffi::Tensor>& inputs) {
        const int32_t gr = core->tree.granularity;
        std::vector<std::vector<int32_t>> scratches(inputs.size());
        std::vector<std::pair<const int32_t*, int32_t>> views;
        std::vector<RadixTrace> traces;
        std::vector<std::pair<int32_t, int32_t>> forced;
        for (size_t i = 0; i < inputs.size(); ++i) {
          const int32_t* p = cpu_i32(inputs[i], scratches[i]);
          int32_t len = static_cast<int32_t>(inputs[i].size(0)), usable = (len / gr) * gr;
          views.emplace_back(p, len);
          int32_t row = static_cast<int32_t>(rows[i]);
          RadixTrace tr;
          auto pit = core->pending.find(row);
          if (pit != core->pending.end() && pit->second.trace_rev == core->tree.revision &&
              usable >= pit->second.trace_input_len &&
              (pit->second.trace_len < pit->second.trace_input_len ||
               usable == pit->second.trace_input_len || core->tree.node(pit->second.trace_node).is_leaf())) {
            int32_t nid = pit->second.trace_node, ml = pit->second.trace_len;
            int32_t in_n = ml - core->tree.node(nid).begin();
            tr = {nid, ml, in_n, nid == 0 || in_n == core->tree.node(nid).length};
          } else {
            tr = core->tree.trace(p, usable);
          }
          traces.push_back(tr);
          if (tr.matched_len > 0 && (!tr.is_exact || (tr.matched_len <= usable && !core->tree.node(tr.node_id).is_leaf())))
            forced.emplace_back(static_cast<int32_t>(i), tr.matched_len);
        }
        for (const auto& prow : compute_branch_protections(views, forced, gr)) {
          int32_t pos = prow[0], r0 = prow[1];
          if (pos <= traces[r0].matched_len) {
            if (auto tp = core->tree.trace_prefix(traces[r0], pos); tp.matched_len == pos && tp.node_id > 0)
              core->tree.protect_at(tp.node_id, pos);
          }
          for (size_t k = 1; k < prow.size(); ++k) core->protected_prefixes[uids[prow[k]]].push_back(pos);
        }
      });

  reg(r, "prepare_prefill_batch",
      [core](const List& rows, const List& clens, const List& dlens, const List&,
             const tvm::ffi::Array<tvm::ffi::Tensor>& inputs) -> Result {
        const int32_t gr = core->tree.granularity;
        std::vector<Int> first_offsets;
        bool all_offsets = true;
        for (size_t i = 0; i < rows.size(); ++i) {
          auto& st = core->req_states[static_cast<int32_t>(rows[i])];
          int32_t cl = static_cast<int32_t>(clens[i]);
          bool found = false;
          for (const auto& lg : st.live_granules) {
            if (lg.begin <= cl && cl < lg.end) { first_offsets.push_back(cl - lg.begin); found = true; break; }
          }
          if (!found) { all_offsets = false; break; }
        }
        for (size_t i = 0; i < rows.size(); ++i)
          core->req_states[static_cast<int32_t>(rows[i])].work_committed_pos = static_cast<int32_t>(dlens[i]);

        std::vector<std::vector<int32_t>> mat_groups;
        if (core->enable_hit_materialization) {
          std::vector<std::vector<int32_t>> scratches(inputs.size());
          std::unordered_set<int32_t> nodes;
          std::unordered_map<int32_t, int32_t> limits, cur_pins;
          for (size_t i = 0; i < inputs.size(); ++i) {
            const int32_t* p = cpu_i32(inputs[i], scratches[i]);
            auto tr = core->tree.trace(p, (static_cast<int32_t>(inputs[i].size(0)) / gr) * gr);
            if (tr.matched_len == 0) continue;
            bool is_ep = true;
            for (int32_t cur = tr.node_id; cur > 0; cur = core->tree.node(cur).parent, is_ep = false) {
              nodes.insert(cur);
              int32_t lim = is_ep ? tr.matched_len : core->tree.node(cur).depth;
              auto it = limits.find(cur);
              if (it == limits.end() || lim > it->second) limits[cur] = lim;
            }
          }
          for (size_t i = 0; i < rows.size(); ++i) {
            auto pit = core->pending.find(static_cast<int32_t>(rows[i]));
            if (pit == core->pending.end() || pit->second.position == 0) continue;
            auto rec = core->tree.recovery_at(pit->second.match_node, pit->second.position);
            std::unordered_set<int32_t> uniq;
            if (rec.base_checkpoint_slot >= 0) uniq.insert(rec.base_checkpoint_slot);
            for (const auto& le : rec.logs) uniq.insert(le.slot);
            if (rec.conv_slot >= 0) uniq.insert(rec.conv_slot);
            for (int32_t s : uniq) ++cur_pins[s];
          }
          const int32_t log_cnt = core->alloc->materialize_log_count;
          for (int32_t nid : nodes) {
            const auto& n = core->tree.node(nid);
            int32_t lim = limits[nid];
            std::vector<int32_t> bounds;
            for (const auto& [p, c] : n.endpoint_refs) if (c > 0) bounds.push_back(p);
            std::sort(bounds.begin(), bounds.end());
            std::vector<int32_t> grp;
            int32_t prev_end = -1;
            for (const auto& e : n.entries) {
              if (e.end > lim) break;
              int32_t refs = core->tree.recovery_slot_ref(e.slot);
              if (refs > cur_pins[e.slot] || refs > 1 || e.kind != EntryKind::kLog || e.end - e.begin != gr) {
                grp.clear(); prev_end = -1; continue;
              }
              if (e.begin != prev_end || std::binary_search(bounds.begin(), bounds.end(), e.begin)) grp.clear();
              if (grp.empty() && core->tree.recovery_at(nid, e.begin).position != e.begin) { prev_end = -1; continue; }
              prev_end = e.end;
              grp.push_back(e.slot);
              if (static_cast<int32_t>(grp.size()) == log_cnt) {
                if (core->alloc->checkpoint_frame_savings(grp) > 0) mat_groups.push_back(grp);
                grp.clear();
              }
            }
          }
          std::sort(mat_groups.begin(), mat_groups.end(), [&](const auto& a, const auto& b) {
            const auto &ea = core->tree.entry_for_slot(a.back()), &eb = core->tree.entry_for_slot(b.back());
            return std::tie(ea.end, ea.begin, a.back()) < std::tie(eb.end, eb.begin, b.back());
          });
        }
        Rows mat_rows;
        for (const auto& mg : mat_groups) mat_rows.push_back(to_list(mg));
        Result out;
        out.Set("has_first_offsets", all_offsets);
        if (all_offsets) out.Set("first_offsets", List(first_offsets.begin(), first_offsets.end()));
        out.Set("materialize_groups", mat_rows);
        return out;
      });

  reg(r, "commit_hit_materializations", [core](const Rows& groups) -> Rows {
    Rows accepted;
    const int32_t gr = core->tree.granularity;
    for (const auto& gl : groups) {
      auto slots = from_list(gl);
      bool ok = !slots.empty();
      for (int32_t s : slots) if (!core->tree.has_slot(s) || core->tree.recovery_slot_ref(s) > 0) { ok = false; break; }
      if (!ok) continue;
      int32_t b = core->tree.entry_for_slot(slots.front()).begin, e = core->tree.entry_for_slot(slots.back()).end, pe = b;
      for (int32_t s : slots) {
        const auto& en = core->tree.entry_for_slot(s);
        if (en.kind != EntryKind::kLog || en.end - en.begin != gr || en.begin != pe) { ok = false; break; }
        pe = en.end;
      }
      if (!ok) continue;
      int32_t owner = core->tree.slot_owner(slots.back());
      for (int32_t s : slots) core->tree.detach_entry(s);
      core->tree.attach_entry(owner, {slots.back(), EntryKind::kCheckpoint, b, e, (e - b) / gr, 0});
      accepted.push_back(gl);
    }
    return accepted;
  });

  reg(r, "get_restore_tasks", [core](const List& rows, bool pop_pending) -> Result {
    std::vector<int32_t> trows, ws, bs, cs, cg, fls, fll, lo{0}, fps, po{0};
    for (size_t i = 0; i < rows.size(); ++i) {
      int32_t row = static_cast<int32_t>(rows[i]);
      auto pit = core->pending.find(row);
      if (pit == core->pending.end() || pit->second.position == 0) {
        if (pop_pending) core->pending.erase(row);
        continue;
      }
      auto rec = core->tree.recovery_at(pit->second.match_node, pit->second.position);
      check(rec.position == pit->second.position, "Lost recovery during prefill restore");
      trows.push_back(row); ws.push_back(core->req_states[row].work_slot);
      bs.push_back(rec.base_checkpoint_slot); cs.push_back(rec.conv_slot); cg.push_back(rec.conv_granule);
      std::vector<int32_t> pins;
      if (rec.base_checkpoint_slot >= 0) pins.push_back(rec.base_checkpoint_slot);
      for (const auto& le : rec.logs) { fls.push_back(le.slot); fll.push_back(le.end - le.begin); pins.push_back(le.slot); }
      lo.push_back(static_cast<int32_t>(fls.size()));
      if (pop_pending && !pins.empty()) core->tree.pin_recovery(pins, false);
      fps.insert(fps.end(), pins.begin(), pins.end());
      po.push_back(static_cast<int32_t>(fps.size()));
      if (pop_pending) core->pending.erase(row);
    }
    Result out;
    out.Set("task_rows", to_i32_tensor(trows)); out.Set("work_slots", to_i32_tensor(ws));
    out.Set("base_slots", to_i32_tensor(bs)); out.Set("conv_slots", to_i32_tensor(cs));
    out.Set("conv_granules", to_i32_tensor(cg)); out.Set("log_slots", to_i32_tensor(fls));
    out.Set("log_lens", to_i32_tensor(fll)); out.Set("log_offsets", to_i32_tensor(lo));
    out.Set("pin_slots", to_i32_tensor(fps)); out.Set("pin_offsets", to_i32_tensor(po));
    return out;
  });

  reg(r, "recovery_for_slot_end", [core](Int rs) -> Result {
    int32_t s = static_cast<int32_t>(rs);
    auto rec = core->tree.recovery_at(core->tree.slot_owner(s), core->tree.entry_for_slot(s).end);
    std::vector<int32_t> ls, ll;
    for (const auto& le : rec.logs) { ls.push_back(le.slot); ll.push_back(le.end - le.begin); }
    Result out;
    out.Set("base", static_cast<Int>(rec.base_checkpoint_slot));
    out.Set("log_slots", to_i32_tensor(ls)); out.Set("log_lens", to_i32_tensor(ll));
    return out;
  });

  reg(r, "cache_req",
      [core](Int row_i, Int uid, tvm::ffi::TensorView input_ids, Int clen_i, bool finished,
             Int old_nid, Int old_c_i, tvm::ffi::TensorView old_rslots_t, Int old_owner,
             const List& old_prot, bool can_decode) -> Result {
        int32_t row = static_cast<int32_t>(row_i), cl = static_cast<int32_t>(clen_i), old_c = static_cast<int32_t>(old_c_i);
        const int32_t gr = core->tree.granularity, pgs = core->tree.page_size;
        auto& st = core->req_states[row]; auto& granules = st.live_granules;
        std::vector<int32_t> prot = from_list(old_prot);
        if (auto pit = core->protected_prefixes.find(uid); pit != core->protected_prefixes.end())
          prot.insert(prot.end(), pit->second.begin(), pit->second.end());
        std::sort(prot.begin(), prot.end()); prot.erase(std::unique(prot.begin(), prot.end()), prot.end());

        std::vector<int32_t> cidx, cend;
        for (int32_t i = 0; i < static_cast<int32_t>(granules.size()); ++i)
          if (granules[i].end <= cl) { cidx.push_back(i); cend.push_back(granules[i].end); }
        if (finished) {
          for (int32_t i = 0; i < static_cast<int32_t>(granules.size()); ++i) {
            const auto& lg = granules[i];
            if (lg.end <= cl || lg.checkpoint_begin >= 0) continue;
            int32_t bp = 0;
            for (int32_t p : prot) if (lg.begin < p && p <= cl && cl < lg.end) bp = std::max(bp, p);
            if (bp > 0) { cidx.push_back(i); cend.push_back(bp); }
          }
        }
        int32_t ins_len = old_c;
        for (size_t k = 0; k < cidx.size() && granules[cidx[k]].begin == ins_len; ++k) ins_len = cend[k];
        check(ins_len % gr == 0, "Unaligned insert_len");

        std::vector<KvFrameSpan> spans; spans.reserve(cidx.size());
        std::vector<int32_t> kv_synth(std::max<int32_t>(ins_len, 0), 0);
        if (old_c > 0 && old_nid > 0 && old_nid < core->tree.num_nodes_capacity() && core->tree.node(static_cast<int32_t>(old_nid)).alive)
          core->tree.copy_matched_kv(static_cast<int32_t>(old_nid), std::min(old_c, ins_len), kv_synth.data());
        for (size_t k = 0; k < cidx.size(); ++k) {
          const auto& lg = granules[cidx[k]]; int32_t ge = cend[k];
          spans.push_back({lg.begin, ge, lg.kv_offset / pgs});
          for (int32_t pos = lg.begin, lim = std::min(ge, ins_len); pos < lim; ++pos) kv_synth[pos] = lg.kv_offset + (pos - lg.begin);
        }
        std::vector<int32_t> tok_scratch;
        const int32_t* toks = cpu_i32(input_ids, tok_scratch);
        auto ins = core->tree.insert(toks, kv_synth.data(), ins_len, ins_len, static_cast<int32_t>(old_owner), &spans);
        int32_t covered = core->tree.recovery_at(ins.node_id, ins.inserted_len).position;

        std::vector<int32_t> rel_kv, rel_slots;
        std::vector<uint8_t> removed(granules.size(), 0);
        int32_t last_adopted_end = -1, adopted = 0;
        for (size_t k = 0; k < cidx.size(); ++k) {
          int32_t gi = cidx[k], ge = cend[k]; const auto& lg = granules[gi];
          if (old_c <= lg.begin && ge <= ins.inserted_len) {
            removed[gi] = 1; ++adopted; last_adopted_end = ge;
            int32_t lb = std::max(lg.begin, covered);
            if (lg.checkpoint_begin >= 0 && lb < ge) {
              int32_t cb = std::max(lg.checkpoint_begin, (covered / gr) * gr);
              core->tree.attach_entry(ins.node_id, {lg.state_slot, EntryKind::kCheckpoint, cb, lg.end, (lg.end - cb) / gr, 0});
              covered = lg.end;
            } else if (lg.state_slot >= 0 && lb < ge) {
              core->tree.attach_entry(ins.node_id, {lg.state_slot, EntryKind::kLog, lb, ge, -1, 0});
              covered = ge;
            } else if (lg.state_slot >= 0) rel_slots.push_back(lg.state_slot);
            if (int32_t kb = std::max(lg.begin, ins.previous_len); kb < ge)
              core->tree.add_cached_page_tokens(lg.kv_offset / pgs, ge - kb);
            else rel_kv.push_back(lg.kv_offset / pgs);
          } else if (ge <= ins.previous_len) {
            removed[gi] = 1; rel_kv.push_back(lg.kv_offset / pgs);
            if (lg.state_slot >= 0) rel_slots.push_back(lg.state_slot);
          }
        }
        for (int32_t p : prot) if (p > 0 && p <= ins.inserted_len) core->tree.protect_at(core->tree.trace(toks, p).node_id, p);
        int32_t new_owner = core->tree.node(ins.node_id).analysis_owner;
        if (new_owner > 0) core->reuse.publish(new_owner, ins.inserted_len);
        int32_t new_node = (ins.inserted_len == 0) ? 0
                           : (ins.inserted_len < core->tree.node(ins.node_id).depth
                                  ? core->tree.node_at_depth(ins.node_id, ins.inserted_len)
                                  : ins.node_id);
        auto new_rec = core->tree.recovery_at(new_node, ins.inserted_len);
        check(new_rec.position == ins.inserted_len, "Cannot recover newly cached handle");
        auto new_kv_t = make_i32_tensor(ins.inserted_len);
        if (ins.inserted_len > 0) core->tree.copy_matched_kv(new_node, ins.inserted_len, static_cast<int32_t*>(new_kv_t.data_ptr()));

        if (old_rslots_t.size(0) > 0) core->tree.pin_recovery(from_i32_tensor(old_rslots_t), true);
        if (old_c > 0 && old_nid > 0 && old_nid < core->tree.num_nodes_capacity() && core->tree.node(static_cast<int32_t>(old_nid)).alive)
          core->tree.lock(static_cast<int32_t>(old_nid), true, old_c);
        core->pending.erase(row);

        std::vector<LiveGranule> rem;
        for (size_t i = 0; i < granules.size(); ++i) if (!removed[i]) rem.push_back(granules[i]);
        int32_t cc_src = -1, cc_gran = -1, cc_dst = -1;
        if (!finished && adopted > 0 && cl == last_adopted_end) {
          cc_src = new_rec.conv_slot; cc_gran = new_rec.conv_granule; cc_dst = st.work_slot;
        }
        if (finished) {
          core->protected_prefixes.erase(uid); core->sampled_queries.erase(uid);
          for (const auto& lg : rem) { rel_kv.push_back(lg.kv_offset / pgs); if (lg.state_slot >= 0) rel_slots.push_back(lg.state_slot); }
          rel_slots.push_back(st.work_slot);
          st.active = st.is_decode_active = false; st.live_granules.clear(); st.input_ids_prefix.clear();
        } else {
          if (ins.inserted_len > 0) core->tree.lock(new_node, false, ins.inserted_len);
          st.handle_node_id = new_node; st.handle_cached_len = ins.inserted_len;
          st.cached_recovery_rev = -1; st.analysis_owner = new_owner;
          st.protected_endpoints = prot; st.is_decode_active = can_decode;
          st.live_granules = std::move(rem); st.live_granule_cursor = 0; st.compacted_end = 0;
          for (const auto& lg : st.live_granules) if (lg.checkpoint_begin >= 0) st.compacted_end = std::max(st.compacted_end, lg.end);
          if (can_decode) st.input_ids_prefix.assign(toks, toks + cl);
        }
        std::vector<int32_t> nls, nlb, nle;
        for (const auto& le : new_rec.logs) { nls.push_back(le.slot); nlb.push_back(le.begin); nle.push_back(le.end); }
        Result out;
        out.Set("inserted_len", static_cast<Int>(ins.inserted_len));
        out.Set("new_node_id", static_cast<Int>(new_node)); out.Set("new_owner", static_cast<Int>(new_owner));
        out.Set("new_kv_indices", new_kv_t); out.Set("new_base_ckpt", static_cast<Int>(new_rec.base_checkpoint_slot));
        out.Set("new_conv_slot", static_cast<Int>(new_rec.conv_slot)); out.Set("new_conv_granule", static_cast<Int>(new_rec.conv_granule));
        out.Set("new_log_slots", to_i32_tensor(nls)); out.Set("new_log_begins", to_i32_tensor(nlb));
        out.Set("new_log_ends", to_i32_tensor(nle)); out.Set("new_prot_endpoints", to_i32_tensor(prot));
        out.Set("released_kv_frames", to_i32_tensor(rel_kv)); out.Set("released_slots", to_i32_tensor(rel_slots));
        out.Set("conv_copy_src", static_cast<Int>(cc_src)); out.Set("conv_copy_gran", static_cast<Int>(cc_gran));
        out.Set("conv_copy_dst", static_cast<Int>(cc_dst));
        return out;
      });

  reg(r, "free_to_cached", [core](Int row_i, Int cl_i) -> Result {
    auto& st = core->req_states[static_cast<int32_t>(row_i)];
    int32_t cl = static_cast<int32_t>(cl_i), pgs = core->tree.page_size;
    std::vector<LiveGranule> keep;
    std::vector<int32_t> rel_kv, rel_slots, clear_ranges;
    for (const auto& lg : st.live_granules) {
      if (lg.begin < cl) keep.push_back(lg);
      else {
        rel_kv.push_back(lg.kv_offset / pgs);
        if (lg.state_slot >= 0) rel_slots.push_back(lg.state_slot);
        clear_ranges.push_back(lg.begin); clear_ranges.push_back(lg.end);
      }
    }
    st.live_granules = std::move(keep); st.live_granule_cursor = 0;
    if (st.compacted_end > cl) {
      st.compacted_end = 0;
      for (const auto& lg : st.live_granules) if (lg.checkpoint_begin >= 0) st.compacted_end = std::max(st.compacted_end, lg.end);
    }
    Result out;
    out.Set("released_kv_frames", to_i32_tensor(rel_kv));
    out.Set("released_slots", to_i32_tensor(rel_slots));
    out.Set("clear_ranges", to_i32_tensor(clear_ranges));
    return out;
  });

  reg(r, "check_integrity", [core](const List& retiring_list) {
    core->tree.check_integrity();
    core->alloc->check_integrity();
    int32_t live_cnt = 0;
    for (const auto& st : core->req_states) if (st.active) live_cnt += static_cast<int32_t>(st.live_granules.size());
    check(live_cnt == 0, "LogState cache has request-owned pages while idle.");
    int64_t total_kv = 0;
    std::unordered_set<int32_t> exp_kv, exp_slots;
    if (core->dummy_kv_frame >= 0) exp_kv.insert(core->dummy_kv_frame);
    for (int32_t f = 0; f < core->alloc->num_frames; ++f) {
      int32_t c = core->tree.cached_page_tokens(f);
      total_kv += c;
      if (c > 0) exp_kv.insert(f);
    }
    check(total_kv == core->tree.evictable_size + core->tree.locked_size, "Cached KV token accounting mismatch.");
    for (int32_t s : {core->dummy_log_slot, core->dummy_work_slot, core->pressure_work_slot})
      if (s >= 0) exp_slots.insert(s);
    for (int32_t s : from_list(retiring_list)) exp_slots.insert(s);
    for (const auto& st : core->req_states) if (st.active && st.work_slot >= 0) exp_slots.insert(st.work_slot);
    for (int32_t nid = 1; nid < core->tree.num_nodes_capacity(); ++nid) {
      if (const auto& n = core->tree.node(nid); n.alive)
        for (const auto& e : n.entries) exp_slots.insert(e.slot);
    }
    check(core->alloc->allocated_kv_list().size() == exp_kv.size(), "Allocator KV ownership mismatch");
    check(core->alloc->allocated_slot_list().size() == exp_slots.size(), "Allocator slot ownership mismatch");
  });

  reg(r, "eviction_stats", [core]() -> Result {
    Result out;
    out.Set("eviction_rounds", core->stat_eviction_rounds);
    out.Set("pressure_dropped_checkpoints", core->stat_pressure_dropped_checkpoints);
    out.Set("pressure_pruned_checkpoints", core->stat_pressure_pruned_checkpoints);
    out.Set("evicted_kv_tokens", core->stat_evicted_kv_tokens);
    out.Set("released_state_slots_from_eviction", core->stat_released_slots_eviction);
    return out;
  });

  reg(r, "tree_layout_snapshot", [core]() -> Result {
    std::vector<int32_t> cp_spans, log_runs;
    int64_t ncp = 0, nlog = 0;
    for (int32_t nid = 1; nid < core->tree.num_nodes_capacity(); ++nid) {
      const auto& n = core->tree.node(nid);
      if (!n.alive) continue;
      int32_t run = 0, last = -1;
      for (const auto& e : n.entries) {
        if (e.kind == EntryKind::kCheckpoint) {
          ++ncp; cp_spans.push_back(e.end - e.begin);
          if (run > 0) { log_runs.push_back(run); run = 0; }
        } else {
          ++nlog;
          if (run > 0 && e.begin != last) { log_runs.push_back(run); run = 0; }
          run += e.end - e.begin;
        }
        last = e.end;
      }
      if (run > 0) log_runs.push_back(run);
    }
    Result out;
    out.Set("checkpoint", ncp); out.Set("log", nlog);
    out.Set("checkpoint_spans", to_i32_tensor(cp_spans)); out.Set("log_run_spans", to_i32_tensor(log_runs));
    return out;
  });

  reg(r, "tree_nodes_summary", [core]() -> Result {
    Rows nodes_meta, entry_rows;
    tvm::ffi::Array<tvm::ffi::Tensor> keys;
    for (int32_t nid = 1; nid < core->tree.num_nodes_capacity(); ++nid) {
      const auto& n = core->tree.node(nid);
      if (!n.alive) continue;
      nodes_meta.push_back(List({static_cast<Int>(nid), static_cast<Int>(n.parent), n.uuid,
                                 static_cast<Int>(n.begin()), static_cast<Int>(n.depth),
                                 static_cast<Int>(n.is_protected), static_cast<Int>(n.ref_count),
                                 n.timestamp, static_cast<Int>(n.is_leaf())}));
      std::vector<Int> er;
      for (const auto& e : n.entries) {
        er.insert(er.end(), {e.slot, static_cast<Int>(e.kind), e.begin, e.end, e.span_granules});
      }
      entry_rows.push_back(List(er.begin(), er.end()));
      std::vector<int32_t> path; core->tree.path_nodes(nid, path);
      auto kt = make_i32_tensor(n.depth);
      auto* dst = static_cast<int32_t*>(kt.data_ptr());
      int32_t off = 0;
      for (int32_t pid : path) {
        const auto& pn = core->tree.node(pid);
        std::memcpy(dst + off, pn.token_data(), static_cast<size_t>(pn.length) * sizeof(int32_t));
        off += pn.length;
      }
      keys.push_back(kt);
    }
    Result out;
    out.Set("nodes", nodes_meta); out.Set("entries", entry_rows); out.Set("prefix_keys", keys);
    return out;
  });

  reg(r, "tree_insert_raw", [core](tvm::ffi::TensorView tokens, tvm::ffi::TensorView kv, const List& ef, Int cl) -> Int {
    std::vector<int32_t> ts, ks;
    auto ins = core->tree.insert(cpu_i32(tokens, ts), cpu_i32(kv, ks), static_cast<int32_t>(tokens.size(0)),
                                 static_cast<int32_t>(cl), -1, nullptr);
    for (size_t i = 0; i < ef.size(); i += 5) {
      core->tree.attach_entry(ins.node_id, {static_cast<int32_t>(ef[i]), static_cast<EntryKind>(ef[i + 1]),
                                            static_cast<int32_t>(ef[i + 2]), static_cast<int32_t>(ef[i + 3]),
                                            static_cast<int32_t>(ef[i + 4]), 0});
    }
    return ins.node_id;
  });
  reg(r, "tree_reset", [core]() { core->tree.reset(); core->reuse.reset(); });
  return r;
}

}  // namespace
}  // namespace logssms

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_allocator, logssms::make_allocator_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_core, logssms::make_core_ffi);
