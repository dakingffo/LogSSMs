#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "allocator.hpp"
#include "radix_tree.hpp"
#include "reuse_model.hpp"

namespace logssms::logstate {

struct NeumaierSum {
  double value = 0.0, error = 0.0;
  void add(double x) {
    const double r = value + x;
    error += std::abs(value) >= std::abs(x) ? (value - r) + x : (x - r) + value;
    value = r;
  }
  double total() const { return (error != 0.0 && std::isfinite(error)) ? value + error : value; }
};

struct LiveGranule {
  int32_t begin = 0, end = 0, kv_offset = 0;
  int32_t state_slot = -1, checkpoint_begin = -1;
};

struct ActiveRequestState {
  bool active = false;
  int32_t table_idx = -1;
  int64_t req_uid = -1;
  int32_t work_slot = -1;
  int32_t work_committed_pos = 0;
  int32_t request_end = 0;
  int32_t handle_cached_len = 0;
  int32_t handle_node_id = 0;
  int32_t analysis_owner = -1;
  int32_t compacted_end = 0;
  int32_t live_granule_cursor = 0;
  bool is_decode_active = false;
  int32_t confirmed_decode_prefix_len = 0;
  std::vector<LiveGranule> live_granules;
  std::vector<int32_t> protected_endpoints;
  std::vector<int32_t> input_ids_prefix;
  int64_t cached_recovery_rev = -1;
  StateRecovery cached_recovery;
};

struct ActiveCompactionAction {
  int32_t table_idx = -1;
  std::vector<int32_t> indices;
  int32_t begin = 0, boundary = 0, end = 0, base_slot = -1;
  std::vector<std::tuple<int32_t, int32_t, int32_t>> logs;
  std::vector<int32_t> tree_sources;
  std::vector<std::pair<int32_t, int32_t>> decay_counts;
  int32_t base_frames = 0, replay = 0, frames = 0;
  double cost = 0.0;
  double score() const { return cost / static_cast<double>(frames); }
};

struct EvictionPlanResult {
  bool ok = false;
  std::vector<int32_t> pruned_nodes;
  std::vector<std::pair<int32_t, int32_t>> kv_releases;
  std::vector<int32_t> retired_slots;
  std::vector<std::pair<int32_t, int32_t>> checkpoint_drops;
  std::vector<int32_t> materialization;
  std::optional<ActiveCompactionAction> active_action;
};

class EvictionPlannerCore {
 public:
  RadixTreeCore& tree;
  AllocatorCore& alloc;
  ReuseModelCore& reuse;

  EvictionPlannerCore(RadixTreeCore& tree_, AllocatorCore& alloc_, ReuseModelCore& reuse_)
      : tree(tree_), alloc(alloc_), reuse(reuse_) {}

  std::vector<ActiveCompactionAction> compute_active_candidates(
      std::vector<ActiveRequestState>& req_states,
      const std::vector<int32_t>& decode_table_idxs,
      const std::unordered_map<int64_t, std::vector<int32_t>>& req_protected_prefixes,
      bool score_candidates) {
    std::vector<ActiveCompactionAction> out;
    const int32_t g = tree.granularity;
    const int32_t mat_count = alloc.materialize_log_count;
    const int32_t ckpt_frames = alloc.payload_frames + alloc.conv_frames;

    for (int32_t row : decode_table_idxs) {
      if (row < 0 || row >= static_cast<int32_t>(req_states.size())) continue;
      auto& st = req_states[row];
      if (!st.active || !st.is_decode_active) continue;
      const int32_t completed = st.confirmed_decode_prefix_len;
      const auto& granules = st.live_granules;
      auto it = std::upper_bound(
          granules.begin(), granules.end(), completed,
          [](int32_t val, const LiveGranule& lg) { return val < lg.end; });
      const int32_t eligible = static_cast<int32_t>(it - granules.begin());
      if (eligible <= 0 || (st.compacted_end == 0 && eligible < mat_count)) continue;

      std::vector<int32_t> prot = st.protected_endpoints;
      auto pit = req_protected_prefixes.find(st.req_uid);
      if (pit != req_protected_prefixes.end()) {
        prot.insert(prot.end(), pit->second.begin(), pit->second.end());
      }
      if (st.handle_node_id > 0 && st.handle_node_id < tree.num_nodes_capacity()) {
        const auto& hn = tree.node(st.handle_node_id);
        if (hn.alive && (hn.depth > st.handle_cached_len || !hn.is_leaf()) &&
            !st.input_ids_prefix.empty()) {
          int32_t trace_len = std::min<int32_t>(
              static_cast<int32_t>(st.input_ids_prefix.size()), (completed / g) * g);
          auto tr = tree.trace(st.input_ids_prefix.data(), trace_len);
          for (int32_t cur = tr.node_id; cur > 0; cur = tree.node(cur).parent) {
            if (tree.node(cur).is_protected) prot.push_back(tree.node(cur).depth);
          }
        }
      }
      std::sort(prot.begin(), prot.end());
      prot.erase(std::unique(prot.begin(), prot.end()), prot.end());
      auto is_prot = [&](int32_t p) { return std::binary_search(prot.begin(), prot.end(), p); };
      auto has_prot_inside = [&](int32_t b, int32_t e) {
        auto lb = std::upper_bound(prot.begin(), prot.end(), b);
        return lb != prot.end() && *lb < e;
      };

      if (st.cached_recovery_rev != tree.revision) {
        st.cached_recovery = tree.recovery_at(st.handle_node_id, st.handle_cached_len);
        st.cached_recovery_rev = tree.revision;
      }
      const auto& rec = st.cached_recovery;
      int32_t base = rec.base_checkpoint_slot;
      std::vector<int32_t> tree_sources;
      if (base >= 0) tree_sources.push_back(base);
      std::vector<std::tuple<int32_t, int32_t, int32_t>> logs;
      logs.reserve(rec.logs.size() + mat_count);
      for (const auto& le : rec.logs) {
        tree_sources.push_back(le.slot);
        logs.emplace_back(le.slot, 0, le.end - le.begin);
      }
      int32_t cursor = rec.position, prev_ckpt = -1;
      std::vector<int32_t> group;
      bool mat_found = false;

      const size_t start_idx = out.size();
      for (int32_t idx = 0; idx < eligible; ++idx) {
        const auto& entry = granules[idx];
        if (entry.state_slot < 0) continue;
        if (entry.checkpoint_begin >= 0) {
          if (alloc.slot(entry.state_slot).kind == SlotKind::kCheckpoint) {
            base = entry.state_slot;
            cursor = entry.end;
            tree_sources.clear();
            logs.clear();
          }
          group.clear();
          if (prev_ckpt >= 0) {
            const auto& left = granules[prev_ckpt];
            if (left.end == entry.checkpoint_begin && !is_prot(left.end)) {
              ActiveCompactionAction act;
              act.table_idx = row;
              act.indices = {prev_ckpt, idx};
              act.begin = left.checkpoint_begin;
              act.boundary = left.end;
              act.end = entry.end;
              act.base_frames = act.frames = ckpt_frames;
              out.push_back(std::move(act));
            }
          }
          prev_ckpt = idx;
          continue;
        }
        if (mat_found || entry.begin != cursor) continue;
        cursor = entry.end;
        logs.emplace_back(entry.state_slot, 0, entry.end - entry.begin);
        if (is_prot(entry.begin)) group.clear();
        if (entry.end - entry.begin != g || has_prot_inside(entry.begin, entry.end)) {
          group.clear();
          continue;
        }
        group.push_back(idx);
        if (static_cast<int32_t>(group.size()) != mat_count) continue;

        std::vector<int32_t> slot_ids;
        slot_ids.reserve(group.size());
        for (int32_t gi : group) slot_ids.push_back(granules[gi].state_slot);

        ActiveCompactionAction act;
        act.table_idx = row;
        act.indices = group;
        act.begin = granules[group[0]].begin;
        act.boundary = granules[group[0]].end;
        act.end = entry.end;
        act.base_slot = base;
        act.logs = logs;
        act.tree_sources = tree_sources;
        act.base_frames = alloc.checkpoint_frame_capacity(slot_ids) - alloc.payload_frames;
        for (int32_t sid : slot_ids) {
          int32_t df = alloc.slot(sid).decay.frame;
          if (df < 0) continue;
          add_pair_count_(act.decay_counts, df);
        }
        act.frames = act.base_frames;
        for (const auto& [df, cnt] : act.decay_counts) {
          if (cnt == alloc.decay_count(df)) ++act.frames;
        }
        for (const auto& [_, lb, le] : act.logs) act.replay += (le - lb);
        out.push_back(std::move(act));

        if (st.compacted_end == 0) break;
        mat_found = true;
      }

      if (score_candidates && out.size() > start_idx) {
        const auto& density = reuse.active_density(st.analysis_owner, completed);
        for (size_t k = start_idx; k < out.size(); ++k) {
          auto& act = out[k];
          if (act.replay > 0) {
            NeumaierSum inc;
            for (int32_t x = act.begin + g; x < act.end; x += g) {
              inc.add(static_cast<double>(x - act.begin) *
                      density.mass(static_cast<double>(x), static_cast<double>(x + g)));
            }
            act.cost = static_cast<double>(act.replay) * 1e-12 + inc.total();
          } else {
            act.cost = static_cast<double>(act.boundary - act.begin) *
                       density.mass(static_cast<double>(act.boundary), static_cast<double>(act.end));
          }
        }
      }
    }
    return out;
  }

  int32_t active_reclaimable_frames(
      std::vector<ActiveRequestState>& req_states,
      const std::vector<int32_t>& decode_table_idxs,
      const std::unordered_map<int64_t, std::vector<int32_t>>& req_protected_prefixes) {
    auto cands = compute_active_candidates(req_states, decode_table_idxs, req_protected_prefixes, false);
    int32_t frames = 0;
    std::unordered_map<int32_t, std::unordered_set<int32_t>> used;
    for (const auto& act : cands) {
      bool blocked = false;
      for (int32_t ts : act.tree_sources) {
        if (tree.is_evictable_slot(ts)) { blocked = true; break; }
      }
      if (blocked) continue;
      auto& u = used[act.table_idx];
      bool disjoint = true;
      for (int32_t idx : act.indices) {
        if (u.contains(idx)) { disjoint = false; break; }
      }
      if (disjoint) {
        frames += act.frames;
        for (int32_t idx : act.indices) u.insert(idx);
      }
    }
    return frames;
  }

  // Persistent-heap incremental zero-serialization eviction planner.
  EvictionPlanResult plan(
      int32_t needed_pages, int32_t needed_states, int32_t needed_works,
      const std::vector<ActiveCompactionAction>& active_cands) {
    EvictionPlanResult res;
    if (reuse.enabled) reuse.refresh(&tree);

    int32_t free_f = alloc.free_frames_count();
    int32_t free_d = alloc.free_decay_records();
    int32_t kv_free = 0;
    auto fits = [&]() -> bool {
      const int32_t req_decay =
          (std::max<int32_t>(0, needed_states - free_d) + alloc.decay_records_per_frame - 1) /
          alloc.decay_records_per_frame;
      return needed_pages + needed_states * alloc.log_payload_frames +
                 needed_works * alloc.payload_frames +
                 (needed_states * alloc.num_granules + needed_works) * alloc.conv_frames +
                 req_decay <= free_f + kv_free;
    };
    if (fits()) { res.ok = true; return res; }

    // Incremental sync of persistent flat arrays & heaps (O(dirty_nodes)!)
    sync_incremental_();

    // Start clean simulation generation (O(1)!)
    ++sim_gen_;
    touched_kv_frames_.clear();
    sim_dirty_leaves_.clear();
    sim_dirty_ckpts_.clear();
    sim_dirty_mats_.clear();

    auto mark_sim_leaf = [&](int32_t nid) { sim_dirty_leaves_.push_back(nid); push_leaf_(nid); };
    auto mark_sim_ckpt = [&](int32_t s) { sim_dirty_ckpts_.push_back(s); push_ckpt_(s); };
    auto mark_sim_mat = [&](int32_t mid) { sim_dirty_mats_.push_back(mid); push_mat_(mid); };

    auto release_sim_slot = [&](int32_t s) {
      sim_slot_released_gen_[s] = sim_gen_;
      res.retired_slots.push_back(s);
      const auto& sr = alloc.slot(s);
      free_f += sr.frame_count();
      if (sr.decay.valid()) {
        int32_t df = sr.decay.frame;
        if (inc_decay_released_(df) == alloc.decay_count(df)) {
          ++free_f;
          free_d -= alloc.decay_records_per_frame - 1;
        } else {
          ++free_d;
        }
        for (int32_t u : decay_users_[df]) {
          for (int32_t mid : node_mat_ids_[u]) mark_sim_mat(mid);
          mark_sim_leaf(u);
        }
      }
    };

    auto valid_active = [&](const ActiveCompactionAction& a) {
      for (int32_t ts : a.tree_sources) {
        if (is_slot_released_(ts)) return false;
      }
      return true;
    };
    std::optional<Cand> best_active;
    auto refresh_active = [&]() {
      if (best_active && valid_active(active_cands[best_active->id])) return;
      best_active.reset();
      for (int32_t i = 0; i < static_cast<int32_t>(active_cands.size()); ++i) {
        if (!valid_active(active_cands[i])) continue;
        double sc = active_cands[i].score();
        Cand c{0.0, sc, std::numeric_limits<int64_t>::max(), 3, 0, 0, 0, i, 0, sc};
        if (!best_active || *best_active > c) best_active = c;
      }
    };

    while (true) {
      while (!mat_heap_.empty() && !valid_mat_(mat_heap_.top())) {
        if (mat_heap_.top().version == mat_ver_[mat_heap_.top().id]) {
          sim_dirty_mats_.push_back(mat_heap_.top().id);
        }
        mat_heap_.pop();
      }
      while (!ckpt_heap_.empty() && !valid_ckpt_(ckpt_heap_.top())) {
        if (ckpt_heap_.top().version == ckpt_ver_[ckpt_heap_.top().id]) {
          sim_dirty_ckpts_.push_back(ckpt_heap_.top().id);
        }
        ckpt_heap_.pop();
      }
      while (!leaf_heap_.empty() && !valid_leaf_(leaf_heap_.top())) {
        if (leaf_heap_.top().version == leaf_ver_[leaf_heap_.top().id]) {
          sim_dirty_leaves_.push_back(leaf_heap_.top().id);
        }
        leaf_heap_.pop();
      }
      refresh_active();

      std::optional<Cand> conversion;
      if (!mat_heap_.empty()) conversion = mat_heap_.top();
      if (best_active && (!conversion || best_active->score < conversion->score)) {
        conversion = best_active;
      }

      int selected_kind = -1;
      std::optional<Cand> merge;
      if (!ckpt_heap_.empty()) {
        selected_kind = 1;
        merge = ckpt_heap_.top();
        merge->phase = 0.0;
        merge->score = merge->cost_score;
      }
      if (!leaf_heap_.empty() && (!merge || *merge > leaf_heap_.top())) {
        selected_kind = 0;
        merge = leaf_heap_.top();
      }

      if (conversion && (!merge || !(*conversion > *merge))) {
        if (conversion->kind == 3) res.active_action = active_cands[conversion->id];
        else res.materialization = pressure_mats_[conversion->id].row;
        res.ok = true;
        break;
      }
      if (selected_kind < 0) break;

      if (selected_kind == 0) {
        Cand c = leaf_heap_.top();
        leaf_heap_.pop();
        if (!valid_leaf_(c)) continue;
        int32_t nid = c.id;
        sim_dirty_leaves_.push_back(nid);
        sim_node_gen_[nid] = sim_gen_;
        sim_node_alive_[nid] = 0;
        sim_children_left_[nid] = 0;

        int32_t pid = tree.node(nid).parent;
        if (pid > 0) {
          if (sim_node_gen_[pid] != sim_gen_) {
            sim_node_gen_[pid] = sim_gen_;
            sim_node_alive_[pid] = 1;
            sim_children_left_[pid] = static_cast<int32_t>(tree.node(pid).children.size());
          }
          --sim_children_left_[pid];
        }

        res.pruned_nodes.push_back(nid);
        for (const auto& [f, cnt] : tree.node(nid).kv_frame_counts) {
          if (add_kv_released_(f, cnt) == tree.cached_page_tokens(f)) ++kv_free;
          for (int32_t u : kv_users_[f]) mark_sim_leaf(u);
        }
        for (const auto& e : tree.node(nid).entries) {
          if (!is_slot_released_(e.slot)) {
            if (ckpt_has_action_[e.slot]) {
              int32_t r = get_ckpt_succ_(e.slot);
              set_ckpt_succ_(e.slot, -1);
              int32_t prev = get_ckpt_prev_(e.slot);
              if (prev >= 0) mark_sim_ckpt(prev);
              if (r >= 0) set_ckpt_prev_(r, -1);
              sim_dirty_ckpts_.push_back(e.slot);
            }
            release_sim_slot(e.slot);
          }
        }
        if (pid > 0) mark_sim_leaf(pid);
      } else {
        Cand c = ckpt_heap_.top();
        ckpt_heap_.pop();
        if (!valid_ckpt_(c)) continue;
        int32_t left = c.id;
        int32_t right = get_ckpt_succ_(left);
        sim_dirty_ckpts_.push_back(left);
        res.checkpoint_drops.emplace_back(left, right);
        set_slot_begin_(right, get_slot_begin_(left));
        set_ckpt_succ_(left, right);
        release_sim_slot(left);
        int32_t prev = get_ckpt_prev_(left);
        if (prev >= 0) {
          set_ckpt_prev_(right, prev);
          mark_sim_ckpt(prev);
        } else {
          set_ckpt_prev_(right, -1);
        }
        mark_sim_ckpt(right);
        int32_t owner = tree.slot_owner(left);
        for (int32_t mid : node_mat_ids_[owner]) mark_sim_mat(mid);
        mark_sim_leaf(owner);
      }

      if (fits()) { res.ok = true; break; }
    }

    if (!res.ok && !active_cands.empty() &&
        (!res.pruned_nodes.empty() || !res.retired_slots.empty())) {
      res.ok = true;
    }

    std::sort(res.retired_slots.begin(), res.retired_slots.end());
    res.kv_releases.reserve(touched_kv_frames_.size());
    for (int32_t f : touched_kv_frames_) {
      res.kv_releases.emplace_back(f, sim_kv_released_[f]);
    }

    // Revert simulation overlay in O(1) and restore only touched persistent heap entries!
    ++sim_gen_;
    restore_sim_dirty_(sim_dirty_mats_, [&](int32_t mid) { push_mat_(mid); });
    restore_sim_dirty_(sim_dirty_ckpts_, [&](int32_t s) { push_ckpt_(s); });
    restore_sim_dirty_(sim_dirty_leaves_, [&](int32_t nid) { push_leaf_(nid); });
    return res;
  }

 private:
  struct Cand {
    double phase, score;
    int64_t timestamp;
    int32_t kind, depth;
    int64_t owner;
    int32_t begin, id, version;
    double cost_score;
    bool operator>(const Cand& o) const {
      return std::tie(phase, score, timestamp, kind, depth, owner, begin, id) >
             std::tie(o.phase, o.score, o.timestamp, o.kind, o.depth, o.owner, o.begin, o.id);
    }
  };
  using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;

  struct PressureMat {
    int32_t owner_node = -1, base_slot = -1;
    size_t log_start = 2;
    bool absorb = false;
    std::vector<int32_t> row;
  };
  struct EvalMat { bool feasible = false; int32_t credit = 0; double score = 0.0; };

  int64_t synced_reset_rev_ = -1;
  int64_t synced_rev_ = -1, synced_lock_rev_ = -1, synced_access_rev_ = -1, synced_model_ver_ = -1;

  // Persistent per-node state
  std::vector<uint8_t> node_in_planner_, node_blocked_;
  std::vector<int32_t> recovery_begin_, recovery_out_;
  std::vector<int64_t> node_ts_;
  std::vector<std::vector<int32_t>> node_kv_frames_, node_decay_frames_, node_ckpt_slots_, node_mat_ids_;

  // Persistent pressure mats with freelist
  std::vector<PressureMat> pressure_mats_;
  std::vector<uint8_t> mat_alive_;
  std::vector<int32_t> free_mat_ids_;

  // Persistent per-slot & per-frame indices
  std::vector<uint8_t> ckpt_has_action_;
  std::vector<int32_t> ckpt_action_right_, ckpt_prev_, slot_begin_, slot_end_;
  std::vector<int64_t> ckpt_ts_;
  std::vector<std::vector<int32_t>> kv_users_, decay_users_;

  // Persistent min-heaps + monotonic version stamps
  std::vector<int32_t> leaf_ver_, ckpt_ver_, mat_ver_;
  MinHeap leaf_heap_, ckpt_heap_, mat_heap_;

  // Generation-stamped simulation overlay
  int64_t sim_gen_ = 0;
  std::vector<int64_t> sim_slot_released_gen_, sim_decay_gen_, sim_kv_gen_, sim_node_gen_;
  std::vector<int64_t> sim_succ_gen_, sim_prev_gen_, sim_begin_gen_;
  std::vector<int32_t> sim_decay_released_, sim_kv_released_, sim_children_left_;
  std::vector<int32_t> sim_ckpt_succ_, sim_ckpt_prev_, sim_slot_begin_;
  std::vector<uint8_t> sim_node_alive_;

  // Reusable scratch buffers (zero per-call heap allocs)
  std::vector<int32_t> touched_kv_frames_, sim_dirty_leaves_, sim_dirty_ckpts_, sim_dirty_mats_;
  std::vector<std::pair<int32_t, int32_t>> tmp_decays_, tmp_records_;
  std::vector<std::vector<int32_t>> tmp_groups_;

  static void add_pair_count_(std::vector<std::pair<int32_t, int32_t>>& v, int32_t key) {
    for (auto& kv : v) {
      if (kv.first == key) { ++kv.second; return; }
    }
    v.emplace_back(key, 1);
  }
  static void erase_unordered_(std::vector<int32_t>& v, int32_t val) {
    for (size_t i = 0; i < v.size(); ++i) {
      if (v[i] == val) { v[i] = v.back(); v.pop_back(); return; }
    }
  }
  template <typename Fn>
  static void restore_sim_dirty_(std::vector<int32_t>& ids, Fn&& fn) {
    if (ids.empty()) return;
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (int32_t id : ids) fn(id);
  }

  // Simulation-aware accessors (when sim_gen_ was bumped at exit, all revert to real state in O(1)!)
  bool is_slot_released_(int32_t s) const { return sim_slot_released_gen_[s] == sim_gen_; }
  int32_t get_decay_released_(int32_t f) const { return sim_decay_gen_[f] == sim_gen_ ? sim_decay_released_[f] : 0; }
  int32_t inc_decay_released_(int32_t f) {
    if (sim_decay_gen_[f] != sim_gen_) { sim_decay_gen_[f] = sim_gen_; sim_decay_released_[f] = 0; }
    return ++sim_decay_released_[f];
  }
  int32_t get_kv_released_(int32_t f) const { return sim_kv_gen_[f] == sim_gen_ ? sim_kv_released_[f] : 0; }
  int32_t add_kv_released_(int32_t f, int32_t cnt) {
    if (sim_kv_gen_[f] != sim_gen_) {
      sim_kv_gen_[f] = sim_gen_;
      sim_kv_released_[f] = 0;
      touched_kv_frames_.push_back(f);
    }
    return sim_kv_released_[f] += cnt;
  }
  int32_t get_children_left_(int32_t nid) const {
    return sim_node_gen_[nid] == sim_gen_ ? sim_children_left_[nid]
                                          : static_cast<int32_t>(tree.node(nid).children.size());
  }
  bool is_node_alive_(int32_t nid) const {
    return sim_node_gen_[nid] == sim_gen_ ? (sim_node_alive_[nid] != 0) : tree.node(nid).alive;
  }
  bool is_sim_leaf_(int32_t nid) const {
    return nid > 0 && nid < static_cast<int32_t>(node_blocked_.size()) &&
           is_node_alive_(nid) && get_children_left_(nid) == 0 && !node_blocked_[nid];
  }
  int32_t get_ckpt_succ_(int32_t s) const {
    int32_t cur = ckpt_action_right_[s];
    while (cur >= 0 && sim_succ_gen_[cur] == sim_gen_) cur = sim_ckpt_succ_[cur];
    return cur;
  }
  int32_t get_ckpt_prev_(int32_t s) const { return sim_prev_gen_[s] == sim_gen_ ? sim_ckpt_prev_[s] : ckpt_prev_[s]; }
  void set_ckpt_prev_(int32_t s, int32_t p) { sim_prev_gen_[s] = sim_gen_; sim_ckpt_prev_[s] = p; }
  void set_ckpt_succ_(int32_t s, int32_t r) { sim_succ_gen_[s] = sim_gen_; sim_ckpt_succ_[s] = r; }
  int32_t get_slot_begin_(int32_t s) const { return sim_begin_gen_[s] == sim_gen_ ? sim_slot_begin_[s] : slot_begin_[s]; }
  void set_slot_begin_(int32_t s, int32_t b) { sim_begin_gen_[s] = sim_gen_; sim_slot_begin_[s] = b; }

  double recovery_before_(int32_t nid, int32_t slot) const {
    int32_t prev = get_ckpt_prev_(slot);
    if (prev >= 0) return static_cast<double>(slot_end_[prev]);
    const int32_t end = slot_end_[slot];
    double restored = static_cast<double>(recovery_begin_[nid]);
    for (const auto& e : tree.node(nid).entries) {
      if (is_slot_released_(e.slot)) continue;
      if (e.end >= end) break;
      if (e.kind == EntryKind::kCheckpoint || e.begin == restored) {
        restored = static_cast<double>(e.end);
      }
    }
    return restored;
  }

  int32_t slot_credit_(int32_t s) const {
    if (is_slot_released_(s)) return 0;
    const auto& sr = alloc.slot(s);
    int32_t credit = sr.frame_count();
    if (sr.decay.valid() && get_decay_released_(sr.decay.frame) + 1 == alloc.decay_count(sr.decay.frame)) {
      ++credit;
    }
    return credit;
  }

  int32_t prune_credit_(int32_t nid) {
    int32_t credit = 0;
    tmp_decays_.clear();
    for (const auto& e : tree.node(nid).entries) {
      if (is_slot_released_(e.slot)) continue;
      const auto& sr = alloc.slot(e.slot);
      credit += sr.frame_count();
      if (sr.decay.valid()) add_pair_count_(tmp_decays_, sr.decay.frame);
    }
    for (const auto& [f, cnt] : tmp_decays_) {
      if (get_decay_released_(f) + cnt == alloc.decay_count(f)) ++credit;
    }
    for (const auto& [f, cnt] : tree.node(nid).kv_frame_counts) {
      if (get_kv_released_(f) + cnt == tree.cached_page_tokens(f)) ++credit;
    }
    return credit;
  }

  bool has_node_merge_(int32_t nid) const {
    for (int32_t s : tree.node(nid).checkpoint_slots) {
      if (is_slot_released_(s) || !ckpt_has_action_[s]) continue;
      int32_t r = get_ckpt_succ_(s);
      if (r >= 0 && !is_slot_released_(r)) return true;
    }
    return false;
  }

  EvalMat eval_pressure_mat_(int32_t mid) {
    if (mid < 0 || mid >= static_cast<int32_t>(mat_alive_.size()) || !mat_alive_[mid]) return {};
    const auto& pm = pressure_mats_[mid];
    if (pm.row.empty()) return {};
    if ((pm.base_slot >= 0 && is_slot_released_(pm.base_slot)) || is_slot_released_(pm.row.back())) return {};
    for (size_t j = 2; j < pm.log_start; ++j) {
      if (is_slot_released_(-pm.row[j] - 2)) return {};
    }
    if (pm.absorb && node_blocked_[tree.slot_owner(pm.base_slot)]) return {};
    const int32_t nid = pm.owner_node;
    const auto& density = reuse.density_for_node(tree, nid);
    const int32_t begin = slot_begin_[pm.row[pm.log_start]];
    const int32_t end = slot_end_[pm.row.back()];
    int32_t credit = pm.absorb ? 0 : -(alloc.payload_frames + alloc.conv_frames);
    const int32_t replay_begin = (pm.base_slot < 0) ? 0 : slot_end_[pm.base_slot];
    double cost = static_cast<double>(end - replay_begin) * 1e-12;
    if (pm.absorb) {
      int32_t base_owner = tree.slot_owner(pm.base_slot);
      cost += std::max(0.0, static_cast<double>(begin) - recovery_before_(base_owner, pm.base_slot)) *
              density.mass(static_cast<double>(begin), static_cast<double>(end));
    }
    tmp_records_.clear();
    for (size_t j = pm.log_start; j < pm.row.size(); ++j) {
      int32_t s = pm.row[j];
      if (is_slot_released_(s) || node_blocked_[tree.slot_owner(s)]) return {};
      const auto& sr = alloc.slot(s);
      credit += sr.frame_count();
      if (sr.decay.valid()) add_pair_count_(tmp_records_, sr.decay.frame);
      int32_t se = slot_end_[s];
      if (se < end) {
        cost += static_cast<double>(se - begin) *
                density.mass(static_cast<double>(se), static_cast<double>(se) + density.g);
      }
    }
    for (const auto& [df, cnt] : tmp_records_) {
      if (get_decay_released_(df) + cnt == alloc.decay_count(df)) ++credit;
    }
    if (credit <= 0) return {};
    return {true, credit, cost / static_cast<double>(credit)};
  }

  bool has_node_materialization_(int32_t nid) {
    for (int32_t mid : node_mat_ids_[nid]) {
      if (eval_pressure_mat_(mid).feasible) return true;
    }
    return false;
  }

  void push_leaf_(int32_t nid) {
    if (nid <= 0 || nid >= static_cast<int32_t>(leaf_ver_.size())) return;
    const int32_t ver = ++leaf_ver_[nid];
    if (!is_sim_leaf_(nid) || has_node_merge_(nid) || has_node_materialization_(nid)) return;
    const auto& n = tree.node(nid);
    const auto& density = reuse.density_for_node(tree, nid);
    double cost = 0.0, restored = static_cast<double>(recovery_begin_[nid]);
    for (const auto& e : n.entries) {
      if (is_slot_released_(e.slot)) continue;
      if (e.kind == EntryKind::kCheckpoint || e.begin == restored) {
        cost += std::max(0.0, static_cast<double>(e.end) - restored) *
                density.mass(static_cast<double>(e.end), std::numeric_limits<double>::infinity());
        restored = static_cast<double>(e.end);
      }
    }
    const int32_t credit = prune_credit_(nid);
    const double score = credit > 0 ? cost / static_cast<double>(credit)
                                    : std::numeric_limits<double>::infinity();
    leaf_heap_.push({0.0, score, node_ts_[nid], 0, n.depth, n.uuid, n.begin(), nid, ver, score});
  }

  void push_ckpt_(int32_t slot) {
    if (slot < 0 || slot >= static_cast<int32_t>(ckpt_ver_.size())) return;
    const int32_t ver = ++ckpt_ver_[slot];
    if (!ckpt_has_action_[slot] || is_slot_released_(slot)) return;
    const int32_t right = get_ckpt_succ_(slot);
    if (right < 0 || is_slot_released_(right)) return;
    const int32_t nid = tree.slot_owner(slot);
    const int32_t left_span = slot_end_[slot] - get_slot_begin_(slot);
    const int32_t right_span = slot_end_[right] - get_slot_begin_(right);
    const auto& density = reuse.density_for_node(tree, nid);
    const double dist = static_cast<double>(slot_end_[slot]) - recovery_before_(nid, slot);
    const double cost = std::max(0.0, dist) *
                        density.mass(static_cast<double>(slot_end_[slot]),
                                     static_cast<double>(slot_end_[right]));
    const int32_t credit = slot_credit_(slot);
    const double score = credit > 0 ? cost / static_cast<double>(credit)
                                    : std::numeric_limits<double>::infinity();
    ckpt_heap_.push({1.0, static_cast<double>(std::max(left_span, right_span)),
                     ckpt_ts_[slot], 1, 0, 0, 0, slot, ver, score});
  }

  void push_mat_(int32_t mid) {
    if (mid < 0 || mid >= static_cast<int32_t>(mat_ver_.size())) return;
    const int32_t ver = ++mat_ver_[mid];
    auto ev = eval_pressure_mat_(mid);
    if (!ev.feasible) return;
    const int32_t nid = pressure_mats_[mid].owner_node;
    const auto& n = tree.node(nid);
    mat_heap_.push({0.0, ev.score, node_ts_[nid], 2, n.depth, n.uuid, n.begin(), mid, ver, ev.score});
  }

  bool valid_leaf_(const Cand& c) {
    return c.version == leaf_ver_[c.id] && is_sim_leaf_(c.id) &&
           !has_node_merge_(c.id) && !has_node_materialization_(c.id);
  }
  bool valid_ckpt_(const Cand& c) const {
    if (c.version != ckpt_ver_[c.id] || !ckpt_has_action_[c.id] || is_slot_released_(c.id)) return false;
    int32_t r = get_ckpt_succ_(c.id);
    return r >= 0 && !is_slot_released_(r);
  }
  bool valid_mat_(const Cand& c) {
    return c.version == mat_ver_[c.id] && eval_pressure_mat_(c.id).feasible;
  }

  void ensure_capacity_(int32_t cap_nodes, int32_t cap_slots, int32_t cap_frames) {
    if (static_cast<int32_t>(node_in_planner_.size()) < cap_nodes) {
      node_in_planner_.resize(cap_nodes, 0);
      node_blocked_.resize(cap_nodes, 0);
      recovery_begin_.resize(cap_nodes, 0);
      recovery_out_.resize(cap_nodes, 0);
      node_ts_.resize(cap_nodes, 0);
      node_kv_frames_.resize(cap_nodes);
      node_decay_frames_.resize(cap_nodes);
      node_ckpt_slots_.resize(cap_nodes);
      node_mat_ids_.resize(cap_nodes);
      leaf_ver_.resize(cap_nodes, 0);
      sim_node_gen_.resize(cap_nodes, 0);
      sim_node_alive_.resize(cap_nodes, 0);
      sim_children_left_.resize(cap_nodes, 0);
    }
    if (static_cast<int32_t>(ckpt_has_action_.size()) < cap_slots) {
      ckpt_has_action_.resize(cap_slots, 0);
      ckpt_action_right_.resize(cap_slots, -1);
      ckpt_prev_.resize(cap_slots, -1);
      ckpt_ts_.resize(cap_slots, 0);
      slot_begin_.resize(cap_slots, 0);
      slot_end_.resize(cap_slots, 0);
      ckpt_ver_.resize(cap_slots, 0);
      sim_slot_released_gen_.resize(cap_slots, 0);
      sim_succ_gen_.resize(cap_slots, 0);
      sim_ckpt_succ_.resize(cap_slots, -1);
      sim_prev_gen_.resize(cap_slots, 0);
      sim_ckpt_prev_.resize(cap_slots, -1);
      sim_begin_gen_.resize(cap_slots, 0);
      sim_slot_begin_.resize(cap_slots, 0);
    }
    if (static_cast<int32_t>(kv_users_.size()) < cap_frames) {
      kv_users_.resize(cap_frames);
      decay_users_.resize(cap_frames);
      sim_decay_gen_.resize(cap_frames, 0);
      sim_decay_released_.resize(cap_frames, 0);
      sim_kv_gen_.resize(cap_frames, 0);
      sim_kv_released_.resize(cap_frames, 0);
    }
  }

  void unregister_node_(int32_t nid) {
    if (nid <= 0 || nid >= static_cast<int32_t>(node_in_planner_.size()) || !node_in_planner_[nid]) return;
    for (int32_t f : node_kv_frames_[nid]) erase_unordered_(kv_users_[f], nid);
    node_kv_frames_[nid].clear();
    for (int32_t f : node_decay_frames_[nid]) erase_unordered_(decay_users_[f], nid);
    node_decay_frames_[nid].clear();
    for (int32_t s : node_ckpt_slots_[nid]) {
      int32_t r = ckpt_action_right_[s];
      if (r >= 0 && ckpt_prev_[r] == s) ckpt_prev_[r] = -1;
      ckpt_has_action_[s] = 0;
      ckpt_action_right_[s] = -1;
      ++ckpt_ver_[s];
    }
    node_ckpt_slots_[nid].clear();
    for (int32_t mid : node_mat_ids_[nid]) {
      mat_alive_[mid] = 0;
      ++mat_ver_[mid];
      free_mat_ids_.push_back(mid);
    }
    node_mat_ids_[nid].clear();
    node_in_planner_[nid] = 0;
    ++leaf_ver_[nid];
  }

  bool register_node_(int32_t nid) {
    unregister_node_(nid);
    if (nid <= 0 || !tree.node(nid).alive) return false;
    const auto& n = tree.node(nid);
    const int32_t rec_in = (n.parent > 0) ? recovery_out_[n.parent] : 0;
    recovery_begin_[nid] = rec_in;
    node_blocked_[nid] = tree.node_is_blocked(nid) ? 1 : 0;
    node_ts_[nid] = tree.node_eviction_timestamp(nid);
    node_in_planner_[nid] = 1;

    node_kv_frames_[nid].reserve(n.kv_frame_counts.size());
    for (const auto& [f, _] : n.kv_frame_counts) {
      node_kv_frames_[nid].push_back(f);
      kv_users_[f].push_back(nid);
    }
    int32_t last_cp_end = -1;
    for (const auto& e : n.entries) {
      slot_begin_[e.slot] = e.begin;
      slot_end_[e.slot] = e.end;
      const auto& sr = alloc.slot(e.slot);
      if (sr.decay.valid()) {
        int32_t df = sr.decay.frame;
        if (decay_users_[df].empty() || decay_users_[df].back() != nid) {
          if (std::find(node_decay_frames_[nid].begin(), node_decay_frames_[nid].end(), df) ==
              node_decay_frames_[nid].end()) {
            node_decay_frames_[nid].push_back(df);
            decay_users_[df].push_back(nid);
          }
        }
      }
      if (e.kind == EntryKind::kCheckpoint) {
        last_cp_end = e.end;
        if (tree.recovery_slot_ref(e.slot) == 0) {
          int32_t right = tree.checkpoint_successor(e.slot);
          ckpt_has_action_[e.slot] = 1;
          ckpt_action_right_[e.slot] = right;
          ckpt_ts_[e.slot] = e.checkpoint_ts;
          node_ckpt_slots_[nid].push_back(e.slot);
          if (right >= 0) ckpt_prev_[right] = e.slot;
        }
      }
    }
    int32_t rec_out = (last_cp_end >= 0) ? last_cp_end : rec_in;
    auto it = std::upper_bound(
        n.entries.begin(), n.entries.end(), rec_out,
        [](int32_t val, const StateEntry& e) { return val < e.end; });
    for (; it != n.entries.end(); ++it) {
      if (it->kind == EntryKind::kLog && it->begin == rec_out) rec_out = it->end;
    }
    const bool rec_changed = (rec_out != recovery_out_[nid]);
    recovery_out_[nid] = rec_out;

    tmp_groups_.clear();
    tree.pressure_materializations_for_node(nid, alloc.materialize_log_count, tmp_groups_);
    for (auto& g_row : tmp_groups_) {
      PressureMat pm;
      pm.owner_node = nid;
      pm.base_slot = g_row[0];
      pm.absorb = (g_row[1] < 0);
      size_t st = 2;
      while (st < g_row.size() && g_row[st] < 0) ++st;
      pm.log_start = st;
      pm.row = std::move(g_row);
      int32_t mid;
      if (!free_mat_ids_.empty()) {
        mid = free_mat_ids_.back();
        free_mat_ids_.pop_back();
        pressure_mats_[mid] = std::move(pm);
        mat_alive_[mid] = 1;
      } else {
        mid = static_cast<int32_t>(pressure_mats_.size());
        pressure_mats_.push_back(std::move(pm));
        mat_alive_.push_back(1);
        mat_ver_.push_back(0);
      }
      node_mat_ids_[nid].push_back(mid);
    }
    return rec_changed;
  }

  void rebuild_heaps_from_state_() {
    leaf_heap_ = MinHeap{};
    ckpt_heap_ = MinHeap{};
    mat_heap_ = MinHeap{};
    for (int32_t mid = 0; mid < static_cast<int32_t>(mat_alive_.size()); ++mid) {
      if (mat_alive_[mid]) push_mat_(mid);
    }
    const int32_t cap_nodes = static_cast<int32_t>(node_in_planner_.size());
    for (int32_t nid = 1; nid < cap_nodes; ++nid) {
      if (!node_in_planner_[nid]) continue;
      for (int32_t s : node_ckpt_slots_[nid]) push_ckpt_(s);
    }
    for (int32_t lid : tree.leaves()) push_leaf_(lid);
  }

  void sync_incremental_() {
    const int32_t cap_nodes = tree.num_nodes_capacity();
    const int32_t cap_slots = alloc.num_slot_capacity();
    const int32_t cap_frames = alloc.num_frames;
    ensure_capacity_(cap_nodes, cap_slots, cap_frames);

    // Cold start or full tree.reset()
    if (synced_reset_rev_ != tree.reset_revision || synced_rev_ < 0) {
      for (int32_t nid = 1; nid < static_cast<int32_t>(node_in_planner_.size()); ++nid) {
        unregister_node_(nid);
      }
      pressure_mats_.clear();
      mat_alive_.clear();
      mat_ver_.clear();
      free_mat_ids_.clear();
      std::vector<int32_t> stack = {0};
      while (!stack.empty()) {
        int32_t nid = stack.back();
        stack.pop_back();
        if (nid > 0) register_node_(nid);
        for (int32_t c : tree.node(nid).children) stack.push_back(c);
      }
      ++sim_gen_;
      rebuild_heaps_from_state_();
      tree.clear_dirty();
      tree.clear_dirty_kv_frames();
      alloc.clear_dirty_decay_frames();
      synced_reset_rev_ = tree.reset_revision;
      synced_rev_ = tree.revision;
      synced_lock_rev_ = tree.lock_revision;
      synced_access_rev_ = tree.access_revision;
      synced_model_ver_ = reuse.model_version;
      return;
    }

    const bool struct_or_lock =
        (synced_rev_ != tree.revision || synced_lock_rev_ != tree.lock_revision);
    const bool access_changed = (synced_access_rev_ != tree.access_revision);
    const bool model_changed = (synced_model_ver_ != reuse.model_version);
    const bool frames_dirty =
        (!alloc.dirty_decay_frames().empty() || !tree.dirty_kv_frames().empty());

    if (!struct_or_lock && !access_changed && !model_changed && !frames_dirty) return;

    ++sim_gen_;  // ensure real-state view while updating persistent heaps
    std::vector<int32_t> repush_nodes;

    for (int32_t id : tree.removed_nodes()) unregister_node_(id);

    if (struct_or_lock) {
      std::vector<int32_t> dirty;
      dirty.reserve(tree.dirty_nodes().size());
      std::unordered_set<int32_t> seen;
      seen.reserve(tree.dirty_nodes().size() * 2 + 4);
      for (int32_t id : tree.dirty_nodes()) {
        if (id > 0 && id < cap_nodes && tree.node(id).alive && seen.insert(id).second) {
          dirty.push_back(id);
        }
      }
      // Unregister ALL dirty nodes first so slots moved by split_at are never clobbered!
      for (int32_t nid : dirty) unregister_node_(nid);
      // Sort strictly top-down by depth so parents update recovery_out_ before children.
      std::sort(dirty.begin(), dirty.end(), [&](int32_t a, int32_t b) {
        return tree.node(a).depth < tree.node(b).depth;
      });
      const bool rev_changed = (synced_rev_ != tree.revision);
      for (size_t i = 0; i < dirty.size(); ++i) {
        int32_t nid = dirty[i];
        bool rec_changed = register_node_(nid);
        repush_nodes.push_back(nid);
        if ((rec_changed || rev_changed) && !tree.node(nid).children.empty()) {
          // Propagate to any non-dirty descendants in strict depth order
          std::vector<int32_t> sub;
          for (int32_t cid : tree.node(nid).children) {
            if (tree.node(cid).alive && seen.insert(cid).second) sub.push_back(cid);
          }
          for (size_t j = 0; j < sub.size(); ++j) {
            int32_t sid = sub[j];
            unregister_node_(sid);
            bool rc = register_node_(sid);
            repush_nodes.push_back(sid);
            if (rc || rev_changed) {
              for (int32_t gc : tree.node(sid).children) {
                if (tree.node(gc).alive && seen.insert(gc).second) sub.push_back(gc);
              }
            }
          }
        }
      }
    } else if (access_changed) {
      for (int32_t nid : tree.dirty_nodes()) {
        if (nid <= 0 || nid >= cap_nodes || !tree.node(nid).alive || !node_in_planner_[nid]) continue;
        node_ts_[nid] = tree.node_eviction_timestamp(nid);
        for (int32_t s : tree.node(nid).checkpoint_slots) {
          ckpt_ts_[s] = tree.entry_for_slot(s).checkpoint_ts;
        }
        repush_nodes.push_back(nid);
      }
    }

    for (int32_t df : alloc.dirty_decay_frames()) {
      if (df >= 0 && df < static_cast<int32_t>(decay_users_.size())) {
        for (int32_t u : decay_users_[df]) repush_nodes.push_back(u);
      }
    }
    for (int32_t kf : tree.dirty_kv_frames()) {
      if (kf >= 0 && kf < static_cast<int32_t>(kv_users_.size())) {
        for (int32_t u : kv_users_[kf]) repush_nodes.push_back(u);
      }
    }

    tree.clear_dirty();
    tree.clear_dirty_kv_frames();
    alloc.clear_dirty_decay_frames();

    const bool heaps_bloated =
        leaf_heap_.size() > static_cast<size_t>(cap_nodes) * 4 + 1024 ||
        ckpt_heap_.size() > static_cast<size_t>(cap_slots) * 2 + 1024 ||
        mat_heap_.size() > pressure_mats_.size() * 4 + 1024;
    if (model_changed || heaps_bloated) {
      rebuild_heaps_from_state_();
    } else if (!repush_nodes.empty()) {
      std::sort(repush_nodes.begin(), repush_nodes.end());
      repush_nodes.erase(std::unique(repush_nodes.begin(), repush_nodes.end()), repush_nodes.end());
      for (int32_t nid : repush_nodes) {
        if (!node_in_planner_[nid]) continue;
        for (int32_t mid : node_mat_ids_[nid]) push_mat_(mid);
        for (int32_t s : node_ckpt_slots_[nid]) push_ckpt_(s);
        push_leaf_(nid);
      }
    }

    synced_rev_ = tree.revision;
    synced_lock_rev_ = tree.lock_revision;
    synced_access_rev_ = tree.access_revision;
    synced_model_ver_ = reuse.model_version;
  }
};

}  // namespace logssms::logstate
