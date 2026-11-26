#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "allocator.hpp"

namespace logssms::logstate {

enum class EntryKind : uint8_t { kLog = 0, kCheckpoint = 1 };

struct StateEntry {
  int32_t slot = -1;
  EntryKind kind = EntryKind::kLog;
  int32_t begin = 0, end = 0, span_granules = -1;
  int64_t checkpoint_ts = 0;
};

struct KvFrameSpan { int32_t begin = 0, end = 0, frame = 0; };
struct TokenKvBuf { std::vector<int32_t> tokens, kv; };
struct RadixTrace { int32_t node_id = 0, matched_len = 0, matched_in_node = 0; bool is_exact = true; };

struct StateRecovery {
  int32_t position = 0, base_checkpoint_slot = -1;
  std::vector<StateEntry> logs;
  int32_t conv_slot = -1, conv_granule = -1, lock_node_id = 0;
};

struct RadixNode {
  int32_t id = 0, parent = -1, depth = 0, length = 0, ref_count = 0;
  int64_t uuid = 0, timestamp = 0;
  bool alive = false, is_protected = false;
  std::shared_ptr<TokenKvBuf> buf;
  int32_t buf_offset = 0, analysis_owner = -1, leaf_pos = -1;

  std::vector<KvFrameSpan> kv_spans;
  std::vector<std::pair<int32_t, int32_t>> kv_frame_counts;  // sorted by frame
  std::vector<StateEntry> entries;                           // sorted by end
  std::vector<int32_t> checkpoint_slots;                     // sorted by end
  std::vector<int32_t> children;
  std::unordered_map<uint64_t, std::vector<int32_t>> child_hash;
  std::vector<std::pair<int32_t, int32_t>> endpoint_refs;

  int32_t begin() const { return depth - length; }
  bool is_root() const { return parent < 0; }
  bool is_leaf() const { return children.empty(); }
  bool is_locked() const { return ref_count > 0; }
  const int32_t* token_data() const { return buf ? buf->tokens.data() + buf_offset : nullptr; }
  const int32_t* kv_data() const { return buf ? buf->kv.data() + buf_offset : nullptr; }
};

inline uint64_t hash_granule(const int32_t* data, int32_t g) {
  uint64_t h = 14695981039346656037ULL;
  for (int32_t i = 0; i < g; ++i) { h ^= static_cast<uint32_t>(data[i]); h *= 1099511628211ULL; }
  return h;
}

class RadixTreeCore {
 public:
  int32_t granularity = 16, page_size = 16, checkpoint_log_count = 1;
  AllocatorCore* allocator = nullptr;
  int64_t evictable_size = 0, locked_size = 0;
  int64_t revision = 0, lock_revision = 0, access_revision = 0, topology_revision = 0, reset_revision = 0;

  std::function<int32_t(int32_t, int64_t)> analysis_factory;
  std::function<int64_t(int32_t)> chain_group_fn;
  std::function<bool(int32_t)> chain_unused_fn;

  RadixTreeCore(int32_t g_, int32_t ps_, int32_t ckpt_cnt_, AllocatorCore* alloc_)
      : granularity(g_), page_size(ps_), checkpoint_log_count(ckpt_cnt_), allocator(alloc_) {
    check(granularity > 0 && page_size > 0 && checkpoint_log_count > 0, "Invalid RadixTreeCore config");
    const int32_t nf = allocator ? allocator->num_frames : 0;
    kv_frame_refs_.assign(nf, 0);
    locked_kv_refs_.assign(nf, 0);
    cached_page_tokens_.assign(nf, 0);
    dirty_kv_flag_.assign(nf, 0);
    init_root_();
  }

  void set_virtual_clock(bool en, int64_t ns = 0) { use_virtual_clock_ = en; virtual_now_ns_ = ns; if (!en) virtual_last_ts_ = -1; }
  void advance_virtual_clock(int64_t ns) { use_virtual_clock_ = true; virtual_now_ns_ = ns; }
  void set_fixed_clock(int64_t ts) { use_fixed_clock_ = true; fixed_ts_ = ts; }
  void clear_fixed_clock() { use_fixed_clock_ = false; }

  int64_t now_ns() {
    if (use_fixed_clock_) return fixed_ts_;
    if (use_virtual_clock_) { virtual_last_ts_ = std::max(virtual_now_ns_, virtual_last_ts_ + 1); return virtual_last_ts_; }
    return std::chrono::steady_clock::now().time_since_epoch().count();
  }

  void reset() {
    if (allocator) {
      for (int32_t s = 0; s < static_cast<int32_t>(evictable_slot_.size()); ++s)
        if (evictable_slot_[s]) allocator->set_reclaimable(s, false);
    }
    nodes_.clear(); free_node_ids_.clear(); leaves_.clear();
    slot_owner_.clear(); slot_entry_.clear();
    recovery_slot_refs_.clear(); unprunable_log_refs_.clear(); evictable_slot_.clear();
    std::fill(kv_frame_refs_.begin(), kv_frame_refs_.end(), 0);
    std::fill(locked_kv_refs_.begin(), locked_kv_refs_.end(), 0);
    std::fill(cached_page_tokens_.begin(), cached_page_tokens_.end(), 0);
    std::fill(dirty_kv_flag_.begin(), dirty_kv_flag_.end(), 0);
    dirty_kv_frames_.clear(); dirty_nodes_.clear(); dirty_node_flag_.clear(); removed_nodes_.clear();
    evictable_kv_count_ = 0; evictable_size = locked_size = 0;
    ++revision; ++topology_revision; ++reset_revision;
    init_root_();
  }

  int32_t num_nodes_capacity() const { return static_cast<int32_t>(nodes_.size()); }
  const RadixNode& node(int32_t id) const { return nodes_[id]; }
  const std::vector<int32_t>& leaves() const { return leaves_; }
  int32_t evictable_kv_frames() const { return evictable_kv_count_; }
  bool has_slot(int32_t s) const { return s >= 0 && s < static_cast<int32_t>(slot_owner_.size()) && slot_owner_[s] >= 0; }
  int32_t slot_owner(int32_t s) const { check(has_slot(s), "Slot not owned by tree"); return slot_owner_[s]; }
  const StateEntry& entry_for_slot(int32_t s) const { check(has_slot(s), "Slot not owned by tree"); return slot_entry_[s]; }
  int32_t recovery_slot_ref(int32_t s) const {
    return (s >= 0 && s < static_cast<int32_t>(recovery_slot_refs_.size())) ? recovery_slot_refs_[s] : 0;
  }
  bool is_evictable_slot(int32_t s) const {
    return s >= 0 && s < static_cast<int32_t>(evictable_slot_.size()) && evictable_slot_[s] != 0;
  }
  int32_t cached_page_tokens(int32_t f) const {
    return (f >= 0 && f < static_cast<int32_t>(cached_page_tokens_.size())) ? cached_page_tokens_[f] : 0;
  }
  void add_cached_page_tokens(int32_t f, int32_t delta) {
    ensure_frame_(f);
    cached_page_tokens_[f] += delta;
    check(cached_page_tokens_[f] >= 0, "Negative cached_page_tokens");
    if (delta != 0 && !dirty_kv_flag_[f]) { dirty_kv_flag_[f] = 1; dirty_kv_frames_.push_back(f); }
  }
  const std::vector<int32_t>& dirty_nodes() const { return dirty_nodes_; }
  const std::vector<int32_t>& removed_nodes() const { return removed_nodes_; }
  const std::vector<int32_t>& dirty_kv_frames() const { return dirty_kv_frames_; }
  void clear_dirty() {
    for (int32_t id : dirty_nodes_) if (id < static_cast<int32_t>(dirty_node_flag_.size())) dirty_node_flag_[id] = 0;
    dirty_nodes_.clear(); removed_nodes_.clear();
  }
  void clear_dirty_kv_frames() {
    for (int32_t f : dirty_kv_frames_) if (f < static_cast<int32_t>(dirty_kv_flag_.size())) dirty_kv_flag_[f] = 0;
    dirty_kv_frames_.clear();
  }

  RadixTrace trace(const int32_t* tokens, int32_t len) const {
    const int32_t usable = len / granularity * granularity;
    int32_t matched = 0, cur = 0;
    while (matched < usable) {
      int32_t child = find_child_(cur, tokens + matched);
      if (child < 0) break;
      const auto& cn = nodes_[child];
      int32_t m = compare_tokens_(cn.token_data(), cn.length, tokens + matched, usable - matched) / granularity * granularity;
      matched += m;
      if (m != cn.length) return {child, matched, m, false};
      cur = child;
    }
    return {cur, matched, nodes_[cur].length, true};
  }

  RadixTrace trace_prefix(const RadixTrace& t, int32_t pos) const {
    check(pos >= 0 && pos <= t.matched_len, "Invalid trace_prefix position");
    int32_t cur = t.node_id;
    while (cur > 0 && pos <= nodes_[cur].begin()) cur = nodes_[cur].parent;
    int32_t in_n = pos - nodes_[cur].begin();
    return {cur, pos, in_n, cur == 0 || in_n == nodes_[cur].length};
  }

  StateRecovery recovery_at(int32_t node_id, int32_t pos) const {
    check(pos >= 0 && pos <= nodes_[node_id].depth, "Invalid recovery position");
    int32_t ep = node_id;
    while (ep > 0 && pos <= nodes_[ep].begin()) ep = nodes_[ep].parent;
    return recovery_for_endpoint_(ep, pos);
  }

  void path_nodes(int32_t node_id, std::vector<int32_t>& out) const {
    out.clear();
    for (int32_t cur = node_id; cur > 0; cur = nodes_[cur].parent) out.push_back(cur);
    std::reverse(out.begin(), out.end());
  }

  void copy_matched_kv(int32_t node_id, int32_t matched_len, int32_t* out_kv) const {
    if (matched_len <= 0) return;
    std::vector<int32_t> path;
    path_nodes(node_id, path);
    int32_t written = 0;
    for (int32_t nid : path) {
      const auto& n = nodes_[nid];
      int32_t take = std::min(n.length, matched_len - written);
      if (take <= 0) break;
      std::memcpy(out_kv + written, n.kv_data(), static_cast<size_t>(take) * sizeof(int32_t));
      written += take;
    }
  }

  void touch_path(int32_t node_id, int32_t base_ckpt) {
    ++access_revision;
    const int64_t tic = now_ns();
    for (int32_t cur = node_id; cur > 0; cur = nodes_[cur].parent) { nodes_[cur].timestamp = tic; mark_dirty_(cur); }
    if (base_ckpt >= 0 && has_slot(base_ckpt)) {
      int32_t owner = slot_owner_[base_ckpt], end = slot_entry_[base_ckpt].end;
      slot_entry_[base_ckpt].checkpoint_ts = tic;
      auto& entries = nodes_[owner].entries;
      auto it = std::lower_bound(entries.begin(), entries.end(), end, [](const StateEntry& e, int32_t v) { return e.end < v; });
      for (; it != entries.end() && it->end == end; ++it) {
        if (it->slot == base_ckpt) { it->checkpoint_ts = tic; break; }
      }
      mark_dirty_(owner);
    }
  }

  int32_t split_at(int32_t node_id, int32_t pos) {
    check(node_id > 0 && nodes_[node_id].alive && pos > 0 && pos < nodes_[node_id].length && pos % granularity == 0,
          "Invalid split_at");
    const int32_t parent_id = nodes_[node_id].parent;
    const int32_t split_depth = nodes_[node_id].begin() + pos;
    const int32_t prefix_id = alloc_node_(nodes_[node_id].timestamp);
    auto& orig = nodes_[node_id];
    auto& pref = nodes_[prefix_id];
    pref.analysis_owner = orig.analysis_owner;
    pref.buf = orig.buf;
    pref.buf_offset = orig.buf_offset;
    pref.length = pos;
    pref.parent = parent_id;
    pref.depth = nodes_[parent_id].depth + pos;

    std::vector<KvFrameSpan> left_spans, right_spans;
    for (const auto& sp : orig.kv_spans) {
      if (sp.end <= pos) left_spans.push_back(sp);
      else if (sp.begin >= pos) right_spans.push_back({sp.begin - pos, sp.end - pos, sp.frame});
      else { left_spans.push_back({sp.begin, pos, sp.frame}); right_spans.push_back({0, sp.end - pos, sp.frame}); }
    }
    pref.kv_spans = std::move(left_spans);
    orig.kv_spans = std::move(right_spans);
    for (const auto& sp : pref.kv_spans) add_kv_count_(pref.kv_frame_counts, sp.frame, sp.end - sp.begin);
    orig.kv_frame_counts.clear();
    for (const auto& sp : orig.kv_spans) add_kv_count_(orig.kv_frame_counts, sp.frame, sp.end - sp.begin);

    size_t li = 0, ri = 0;
    const int32_t ld = orig.is_locked() ? 1 : 0;
    while (li < pref.kv_frame_counts.size() && ri < orig.kv_frame_counts.size()) {
      if (pref.kv_frame_counts[li].first == orig.kv_frame_counts[ri].first) {
        update_kv_frame_(pref.kv_frame_counts[li].first, 1, ld); ++li; ++ri;
      } else if (pref.kv_frame_counts[li].first < orig.kv_frame_counts[ri].first) ++li;
      else ++ri;
    }

    replace_child_(parent_id, node_id, prefix_id);
    pref.ref_count = orig.ref_count;
    std::vector<std::pair<int32_t, int32_t>> pref_ep, suf_ep;
    int32_t pref_ep_sum = 0;
    for (const auto& [p, c] : orig.endpoint_refs) {
      if (p <= split_depth) { pref_ep.emplace_back(p, c); pref_ep_sum += c; }
      else suf_ep.emplace_back(p, c);
    }
    const int32_t suffix_refs = orig.ref_count - pref_ep_sum;
    pref.endpoint_refs = std::move(pref_ep);
    orig.endpoint_refs = std::move(suf_ep);

    auto old_entries = std::move(orig.entries);
    orig.entries.clear();
    orig.checkpoint_slots.clear();
    for (auto& e : old_entries) {
      unregister_entry_(e, node_id, false);
      if (e.end <= split_depth) {
        register_entry_(prefix_id, e, false);
      } else {
        if (e.kind == EntryKind::kCheckpoint && e.begin < split_depth) {
          e.begin = split_depth;
          e.span_granules = (e.end - split_depth) / granularity;
        }
        register_entry_(node_id, e, false);
      }
    }

    orig.buf_offset += pos;
    orig.length -= pos;
    orig.parent = prefix_id;
    add_child_(prefix_id, node_id);
    const bool was_locked = orig.is_locked();
    orig.ref_count = suffix_refs;
    update_size_class_(node_id, was_locked);

    refresh_checkpoint_neighbors_(parent_id);
    refresh_checkpoint_neighbors_(prefix_id);
    refresh_checkpoint_neighbors_(node_id);
    mark_dirty_(parent_id); mark_dirty_(prefix_id); mark_dirty_(node_id);
    ++revision; ++topology_revision;
    return prefix_id;
  }

  int32_t node_at_depth(int32_t node_id, int32_t depth) {
    int32_t cur = node_id;
    while (cur > 0 && depth <= nodes_[cur].begin()) cur = nodes_[cur].parent;
    check(cur == 0 ? depth == 0 : (nodes_[cur].begin() < depth && depth <= nodes_[cur].depth), "Invalid node_at_depth");
    return (cur > 0 && depth < nodes_[cur].depth) ? split_at(cur, depth - nodes_[cur].begin()) : cur;
  }

  struct InsertResult { int32_t node_id = 0, previous_len = 0, inserted_len = 0; };

  InsertResult insert(const int32_t* tokens, const int32_t* kv_indices, int32_t total_len,
                      int32_t cache_len, int32_t analysis_owner,
                      const std::vector<KvFrameSpan>* explicit_spans = nullptr) {
    const int32_t usable = total_len / granularity * granularity;
    const int32_t inserted = (cache_len < 0) ? usable : cache_len;
    check(inserted >= 0 && inserted <= usable && inserted % granularity == 0, "Invalid insert length");
    int32_t cur = 0, matched = 0;
    const int64_t tic = now_ns();
    while (matched < usable) {
      int32_t child = find_child_(cur, tokens + matched);
      if (child < 0) break;
      const auto& cn = nodes_[child];
      int32_t m = compare_tokens_(cn.token_data(), cn.length, tokens + matched, usable - matched) / granularity * granularity;
      matched += m;
      if (m != cn.length) { cur = split_at(child, m); nodes_[cur].timestamp = tic; mark_dirty_(cur); break; }
      cur = child;
      nodes_[cur].timestamp = tic;
      mark_dirty_(cur);
    }

    const int32_t prev_len = matched;
    if (inserted == 0) cur = 0;
    else if (inserted < prev_len) cur = node_at_depth(cur, inserted);
    else if (prev_len < inserted) {
      const int32_t delta = inserted - prev_len;
      std::vector<KvFrameSpan> suffix_spans;
      if (explicit_spans && !explicit_spans->empty()) {
        int32_t covered = prev_len;
        for (const auto& sp : *explicit_spans) {
          int32_t b = std::max(sp.begin, prev_len), e = std::min(sp.end, inserted);
          if (b >= e) continue;
          check(b == covered, "Noncontiguous explicit kv_frame_spans");
          if (!suffix_spans.empty() && suffix_spans.back().frame == sp.frame && suffix_spans.back().end == b - prev_len)
            suffix_spans.back().end = e - prev_len;
          else suffix_spans.push_back({b - prev_len, e - prev_len, sp.frame});
          covered = e;
        }
        check(covered == inserted, "Incomplete explicit kv_frame_spans");
      } else {
        for (int32_t i = prev_len; i < inserted; ++i) {
          int32_t f = kv_indices[i] / page_size, rel = i - prev_len;
          if (!suffix_spans.empty() && suffix_spans.back().frame == f) suffix_spans.back().end = rel + 1;
          else suffix_spans.push_back({rel, rel + 1, f});
        }
      }

      if (cur > 0 && nodes_[cur].is_leaf() && !nodes_[cur].is_protected) {
        auto& n = nodes_[cur];
        if (n.analysis_owner < 0 && analysis_factory) n.analysis_owner = analysis_factory(n.begin(), -1);
        const int32_t old_len = n.length;
        if (!n.buf || n.buf.use_count() > 1 || n.buf_offset != 0) {
          auto nb = std::make_shared<TokenKvBuf>();
          int32_t cap = std::max(old_len + delta, std::max(1, old_len) * 2);
          nb->tokens.reserve(cap); nb->kv.reserve(cap);
          if (old_len > 0) {
            nb->tokens.assign(n.token_data(), n.token_data() + old_len);
            nb->kv.assign(n.kv_data(), n.kv_data() + old_len);
          }
          n.buf = std::move(nb); n.buf_offset = 0;
        }
        n.buf->tokens.insert(n.buf->tokens.end(), tokens + prev_len, tokens + inserted);
        n.buf->kv.insert(n.buf->kv.end(), kv_indices + prev_len, kv_indices + inserted);
        n.length += delta; n.depth += delta;
        for (const auto& sp : suffix_spans) {
          if (!n.kv_spans.empty() && n.kv_spans.back().frame == sp.frame && n.kv_spans.back().end == old_len + sp.begin)
            n.kv_spans.back().end = old_len + sp.end;
          else n.kv_spans.push_back({old_len + sp.begin, old_len + sp.end, sp.frame});
          if (add_kv_count_(n.kv_frame_counts, sp.frame, sp.end - sp.begin)) {
            update_kv_frame_(sp.frame, 1, n.is_locked() ? 1 : 0);
          }
        }
        if (n.is_locked()) locked_size += delta; else evictable_size += delta;
        refresh_checkpoint_neighbors_(cur);
        mark_dirty_(cur);
      } else {
        const int32_t parent_id = cur, new_id = alloc_node_(now_ns());
        auto& nn = nodes_[new_id];
        nn.buf = std::make_shared<TokenKvBuf>();
        nn.buf->tokens.assign(tokens + prev_len, tokens + inserted);
        nn.buf->kv.assign(kv_indices + prev_len, kv_indices + inserted);
        nn.length = delta; nn.parent = parent_id; nn.depth = nodes_[parent_id].depth + delta;
        nn.kv_spans = std::move(suffix_spans);
        for (const auto& sp : nn.kv_spans) add_kv_count_(nn.kv_frame_counts, sp.frame, sp.end - sp.begin);
        add_child_(parent_id, new_id);

        auto& pn = nodes_[parent_id];
        if (parent_id > 0 && pn.children.size() == 1) {
          nn.analysis_owner = pn.analysis_owner;
        } else {
          if (analysis_owner >= 0 && chain_unused_fn && chain_unused_fn(analysis_owner)) nn.analysis_owner = analysis_owner;
          else if (analysis_factory) {
            int64_t grp = (analysis_owner >= 0 && chain_group_fn) ? chain_group_fn(analysis_owner) : -1;
            nn.analysis_owner = analysis_factory(prev_len, grp);
          }
          if (parent_id > 0 && pn.children.size() == 2) {
            for (int32_t sh = parent_id, own = pn.analysis_owner;
                 own >= 0 && sh > 0 && nodes_[sh].analysis_owner == own; sh = nodes_[sh].parent) {
              nodes_[sh].analysis_owner = -1; mark_dirty_(sh);
            }
          }
        }
        remove_leaf_(parent_id); add_leaf_(new_id);
        if (parent_id > 0 && pn.children.size() > 1) protect(parent_id, true);
        cur = new_id;
        for (const auto& [f, _] : nodes_[new_id].kv_frame_counts) update_kv_frame_(f, 1, 0);
        evictable_size += delta;
        refresh_checkpoint_neighbors_(new_id);
        mark_dirty_(parent_id); mark_dirty_(new_id);
      }
      ++revision; ++topology_revision;
    }
    return {cur, prev_len, inserted};
  }

  void attach_entry(int32_t leaf_id, StateEntry entry, int64_t checkpoint_ts = -1) {
    check(entry.end <= nodes_[leaf_id].depth && entry.begin % granularity == 0 && entry.end % granularity == 0 && !has_slot(entry.slot),
          "Invalid attach_entry");
    if (entry.kind == EntryKind::kLog) check(entry.end - entry.begin == granularity, "Invalid Log entry length");
    else check(entry.span_granules > 0 && entry.end - entry.begin == entry.span_granules * granularity, "Invalid Checkpoint span");
    int32_t owner = find_owner_(leaf_id, entry.end);
    if (entry.kind == EntryKind::kCheckpoint && entry.begin < nodes_[owner].begin()) {
      entry.begin = nodes_[owner].begin();
      entry.span_granules = (entry.end - entry.begin) / granularity;
    }
    check(entry.begin >= nodes_[owner].begin(), "Entry begin precedes owner");
    if (entry.kind == EntryKind::kCheckpoint) entry.checkpoint_ts = (checkpoint_ts < 0) ? now_ns() : checkpoint_ts;
    register_entry_(owner, entry);
    mark_dirty_(owner);
    ++revision;
  }

  StateEntry detach_entry(int32_t slot) {
    int32_t owner = slot_owner(slot), target_end = slot_entry_[slot].end;
    auto& entries = nodes_[owner].entries;
    auto it = std::lower_bound(entries.begin(), entries.end(), target_end, [](const StateEntry& e, int32_t v) { return e.end < v; });
    for (; it != entries.end() && it->end == target_end; ++it) {
      if (it->slot == slot) {
        StateEntry e = *it;
        entries.erase(it);
        if (e.kind == EntryKind::kCheckpoint) {
          auto& cps = nodes_[owner].checkpoint_slots;
          if (auto cit = std::find(cps.begin(), cps.end(), slot); cit != cps.end()) cps.erase(cit);
        }
        unregister_entry_(e, owner);
        mark_dirty_(owner);
        ++revision;
        return e;
      }
    }
    throw std::runtime_error("Slot not found in owner entries");
  }

  void lock(int32_t node_id, bool unlock, int32_t position = -1) {
    ++lock_revision;
    int32_t cur = node_id;
    if (position < 0) position = nodes_[cur].depth;
    while (cur > 0 && position <= nodes_[cur].begin()) cur = nodes_[cur].parent;
    if (cur > 0) {
      auto& eps = nodes_[cur].endpoint_refs;
      auto it = std::lower_bound(eps.begin(), eps.end(), position, [](const auto& a, int32_t p) { return a.first < p; });
      if (unlock) {
        check(it != eps.end() && it->first == position && it->second > 0, "Unlocking unreferenced endpoint");
        if (--it->second == 0) eps.erase(it);
      } else {
        if (it != eps.end() && it->first == position) ++it->second; else eps.insert(it, {position, 1});
      }
    }
    for (; cur > 0; cur = nodes_[cur].parent) {
      auto& n = nodes_[cur];
      bool was = n.is_locked();
      if (unlock) { check(n.ref_count > 0, "Negative ref_count"); --n.ref_count; }
      else ++n.ref_count;
      update_size_class_(cur, was);
    }
  }

  std::pair<int32_t, std::vector<int32_t>> lock_footprint(int32_t node_id, int32_t position = -1) const {
    std::vector<int32_t> frames, slots;
    int32_t cur = node_id;
    if (position >= 0) while (cur > 0 && position <= nodes_[cur].begin()) cur = nodes_[cur].parent;
    for (; cur > 0; cur = nodes_[cur].parent) {
      const auto& n = nodes_[cur];
      if (n.is_locked()) continue;
      for (const auto& [f, _] : n.kv_frame_counts) if (locked_kv_refs_[f] == 0) frames.push_back(f);
      for (const auto& e : n.entries) {
        if (e.end <= (position < 0 ? n.depth : position) && is_evictable_slot(e.slot) &&
            (e.kind == EntryKind::kLog || checkpoint_successor(e.slot) < 0)) slots.push_back(e.slot);
      }
    }
    std::sort(frames.begin(), frames.end()); frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    std::sort(slots.begin(), slots.end()); slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    return {static_cast<int32_t>(frames.size()), std::move(slots)};
  }

  void pin_recovery(const std::vector<int32_t>& slots, bool release = false) {
    ++lock_revision;
    std::vector<int32_t> owners;
    for (int32_t s : slots) {
      ensure_slot_(s);
      int32_t refs = recovery_slot_refs_[s] + (release ? -1 : 1);
      check(refs >= 0 && (refs == 0 || has_slot(s)), "Invalid pin_recovery");
      recovery_slot_refs_[s] = refs;
      update_slot_reclaimability_(s);
      if (has_slot(s)) {
        int32_t owner = slot_owner_[s];
        if (refs == (release ? 0 : 1)) mark_dirty_(owner);
        if (std::find(owners.begin(), owners.end(), owner) == owners.end()) owners.push_back(owner);
      }
    }
    for (int32_t owner : owners) refresh_checkpoint_neighbors_(owner);
  }

  void protect(int32_t node_id, bool val = true) {
    check(node_id > 0 && nodes_[node_id].alive, "Cannot protect root or dead node");
    if (nodes_[node_id].is_protected != val) {
      nodes_[node_id].is_protected = val;
      ++revision;
      refresh_checkpoint_neighbors_(node_id);
      mark_dirty_(node_id);
    }
  }
  void protect_at(int32_t node_id, int32_t pos) { protect(node_at_depth(node_id, pos), true); }

  int32_t checkpoint_successor(int32_t slot) const {
    if (!has_slot(slot)) return -1;
    const auto& se = slot_entry_[slot];
    if (se.kind != EntryKind::kCheckpoint) return -1;
    const auto& n = nodes_[slot_owner_[slot]];
    if (n.is_protected && se.end == n.depth) return -1;
    auto it = std::lower_bound(n.entries.begin(), n.entries.end(), se.end, [](const StateEntry& e, int32_t v) { return e.end < v; });
    for (; it != n.entries.end() && it->end == se.end; ++it) {
      if (it->slot == slot) {
        ++it;
        if (it != n.entries.end() && it->kind == EntryKind::kCheckpoint &&
            it->begin == se.end && recovery_slot_ref(it->slot) == 0) return it->slot;
        return -1;
      }
    }
    return -1;
  }

  StateEntry drop_checkpoint(int32_t left_slot, int32_t right_slot) {
    const StateEntry left = entry_for_slot(left_slot), right = entry_for_slot(right_slot);
    const int32_t owner = slot_owner(right_slot);
    check(left.kind == EntryKind::kCheckpoint && right.kind == EntryKind::kCheckpoint &&
              left.end == right.begin && owner == slot_owner(left_slot) &&
              !(nodes_[owner].is_protected && left.end == nodes_[owner].depth) &&
              recovery_slot_ref(left_slot) == 0 && recovery_slot_ref(right_slot) == 0,
          "Invalid drop_checkpoint");
    const int64_t rts = right.checkpoint_ts;
    detach_entry(left_slot); detach_entry(right_slot);
    StateEntry merged = right;
    merged.begin = left.begin;
    merged.span_granules = left.span_granules + right.span_granules;
    attach_entry(owner, merged, rts);
    return left;
  }

  void prune_leaf(int32_t node_id, std::vector<int32_t>* released = nullptr) {
    auto& n = nodes_[node_id];
    check(node_id > 0 && n.alive && n.is_leaf() && !n.is_locked(), "Node is not an unlocked leaf");
    evictable_size -= n.length;
    for (const auto& [f, _] : n.kv_frame_counts) update_kv_frame_(f, -1, 0);
    auto entries = std::move(n.entries);
    n.entries.clear(); n.checkpoint_slots.clear();
    for (const auto& e : entries) {
      unregister_entry_(e, node_id, false);
      if (released) released->push_back(e.slot);
    }
    const int32_t parent_id = n.parent, old_owner = n.analysis_owner;
    remove_child_(parent_id, node_id); remove_leaf_(node_id);
    n.alive = false; n.buf.reset(); n.kv_spans.clear(); n.kv_frame_counts.clear(); n.endpoint_refs.clear();
    free_node_ids_.push_back(node_id); removed_nodes_.push_back(node_id);

    auto& pn = nodes_[parent_id];
    if (parent_id > 0 && pn.is_leaf()) {
      if (pn.analysis_owner < 0 && analysis_factory) {
        int64_t grp = (old_owner >= 0 && chain_group_fn) ? chain_group_fn(old_owner) : -1;
        pn.analysis_owner = analysis_factory(pn.begin(), grp);
      }
      add_leaf_(parent_id);
    }
    refresh_checkpoint_neighbors_(parent_id);
    mark_dirty_(parent_id);
    ++revision; ++topology_revision;
  }

  bool node_is_blocked(int32_t nid) const {
    const auto& n = nodes_[nid];
    if (n.is_root() || n.is_locked()) return true;
    for (const auto& e : n.entries) if (recovery_slot_ref(e.slot) > 0) return true;
    return false;
  }

  int64_t node_eviction_timestamp(int32_t nid) const {
    const auto& n = nodes_[nid];
    int64_t ts = n.timestamp;
    bool has_cp = false;
    for (int32_t s : n.checkpoint_slots) {
      int64_t cts = slot_entry_[s].checkpoint_ts;
      if (!has_cp || cts < ts) { ts = cts; has_cp = true; }
    }
    return ts;
  }

  void pressure_materializations_for_node(int32_t nid, int32_t count, std::vector<std::vector<int32_t>>& out) const {
    const auto& n = nodes_[nid];
    const auto& entries = n.entries;
    if (entries.empty()) return;
    const int32_t m = static_cast<int32_t>(entries.size());
    std::vector<int32_t> run_ends(m + 1);
    for (int32_t i = 0; i <= m; ++i) run_ends[i] = i;
    for (int32_t i = m - 1; i >= 0; --i) {
      if (entries[i].kind == EntryKind::kLog) {
        int32_t end = i + 1;
        run_ends[i] = (end < m && entries[i].end == entries[end].begin) ? run_ends[end] : end;
      }
    }
    if (const int32_t leading = run_ends[0]; leading >= count && entries[0].begin == n.begin()) {
      auto rec = recovery_at(n.parent, n.begin());
      if (rec.position == n.begin()) {
        std::vector<int32_t> row;
        row.reserve(2 + rec.logs.size() + count);
        row.push_back(rec.base_checkpoint_slot);
        row.push_back(entries[(leading / count) * count - 1].slot);
        for (const auto& le : rec.logs) row.push_back(-le.slot - 2);
        for (int32_t i = 0; i < count; ++i) row.push_back(entries[i].slot);
        out.push_back(std::move(row));
      }
    }
    for (int32_t i = 0; i < m; ++i) {
      if (entries[i].kind != EntryKind::kCheckpoint) continue;
      int32_t first = i + 1, end = run_ends[first];
      if (first >= end || entries[i].end != entries[first].begin) continue;
      int32_t complete = ((end - first) / count) * count;
      if (complete > 0) {
        std::vector<int32_t> row{entries[i].slot, entries[first + complete - 1].slot};
        for (int32_t j = first; j < first + count; ++j) row.push_back(entries[j].slot);
        out.push_back(std::move(row));
      } else if (end == m && entries[end - 1].end == n.depth) {
        std::vector<int32_t> row{entries[i].slot, -1};
        for (int32_t j = first; j < end; ++j) row.push_back(entries[j].slot);
        out.push_back(std::move(row));
      }
    }
  }

  void check_integrity() const {
    int64_t ev_sz = 0, lk_sz = 0;
    std::vector<int32_t> kv_refs(kv_frame_refs_.size(), 0), lk_refs(locked_kv_refs_.size(), 0), stack = {0};
    int32_t leaf_cnt = 0;
    while (!stack.empty()) {
      int32_t id = stack.back(); stack.pop_back();
      const auto& n = nodes_[id];
      check(n.alive, "Dead node reachable");
      if (n.is_root()) {
        check(n.ref_count == 1 && n.length == 0, "Invalid root");
      } else {
        if (n.is_leaf()) { check(n.leaf_pos >= 0 && leaves_[n.leaf_pos] == id, "Leaf stale"); ++leaf_cnt; }
        else check(n.leaf_pos == -1, "Non-leaf in leaf index");
        check(n.length > 0 && n.length % granularity == 0 && n.ref_count >= 0, "Invalid node");
        for (const auto& [f, _] : n.kv_frame_counts) { ++kv_refs[f]; if (n.is_locked()) ++lk_refs[f]; }
        int32_t prev_end = -1;
        for (const auto& e : n.entries) {
          check(n.begin() <= e.begin && e.begin < e.end && e.end <= n.depth && e.end >= prev_end &&
                    has_slot(e.slot) && slot_owner_[e.slot] == id,
                "Invalid entry");
          prev_end = e.end;
        }
        (n.is_locked() ? lk_sz : ev_sz) += n.length;
      }
      for (int32_t c : n.children) stack.push_back(c);
    }
    check(leaf_cnt == static_cast<int32_t>(leaves_.size()) && ev_sz == evictable_size && lk_sz == locked_size,
          "Tree accounting mismatch");
    int32_t ev_kv = 0;
    for (size_t f = 0; f < kv_refs.size(); ++f) {
      check(kv_refs[f] == kv_frame_refs_[f] && lk_refs[f] == locked_kv_refs_[f], "KV ref mismatch");
      if (kv_refs[f] > 0 && lk_refs[f] == 0) ++ev_kv;
    }
    check(ev_kv == evictable_kv_count_, "Evictable KV count mismatch");
  }

 private:
  std::vector<RadixNode> nodes_;
  std::vector<int32_t> free_node_ids_, leaves_, slot_owner_, recovery_slot_refs_, unprunable_log_refs_;
  std::vector<int32_t> kv_frame_refs_, locked_kv_refs_, cached_page_tokens_, dirty_nodes_, removed_nodes_, dirty_kv_frames_;
  std::vector<StateEntry> slot_entry_;
  std::vector<uint8_t> evictable_slot_, dirty_node_flag_, dirty_kv_flag_;
  int64_t next_uuid_ = 0, virtual_now_ns_ = 0, virtual_last_ts_ = -1, fixed_ts_ = 0;
  int32_t evictable_kv_count_ = 0;
  bool use_virtual_clock_ = false, use_fixed_clock_ = false;

  static bool add_kv_count_(std::vector<std::pair<int32_t, int32_t>>& cnts, int32_t f, int32_t len) {
    auto it = std::lower_bound(cnts.begin(), cnts.end(), f, [](const auto& a, int32_t v) { return a.first < v; });
    if (it != cnts.end() && it->first == f) { it->second += len; return false; }
    cnts.insert(it, {f, len});
    return true;
  }

  void init_root_() {
    nodes_.clear();
    RadixNode root;
    root.id = 0; root.parent = -1; root.uuid = next_uuid_++; root.timestamp = now_ns();
    root.ref_count = 1; root.alive = true;
    nodes_.push_back(std::move(root));
    dirty_node_flag_.assign(1, 0);
  }
  int32_t alloc_node_(int64_t ts) {
    int32_t id;
    if (!free_node_ids_.empty()) { id = free_node_ids_.back(); free_node_ids_.pop_back(); nodes_[id] = RadixNode{}; }
    else { id = static_cast<int32_t>(nodes_.size()); nodes_.emplace_back(); dirty_node_flag_.push_back(0); }
    auto& n = nodes_[id];
    n.id = id; n.uuid = next_uuid_++; n.timestamp = ts; n.alive = true;
    return id;
  }
  void mark_dirty_(int32_t nid) {
    if (nid < 0) return;
    if (nid >= static_cast<int32_t>(dirty_node_flag_.size())) dirty_node_flag_.resize(nid + 1, 0);
    if (!dirty_node_flag_[nid]) { dirty_node_flag_[nid] = 1; dirty_nodes_.push_back(nid); }
  }
  void ensure_slot_(int32_t s) {
    if (s >= static_cast<int32_t>(slot_owner_.size())) {
      const size_t n = static_cast<size_t>(s) + 1;
      slot_owner_.resize(n, -1); slot_entry_.resize(n);
      recovery_slot_refs_.resize(n, 0); unprunable_log_refs_.resize(n, 0); evictable_slot_.resize(n, 0);
    }
  }
  void ensure_frame_(int32_t f) {
    if (f >= static_cast<int32_t>(kv_frame_refs_.size())) {
      const size_t n = static_cast<size_t>(f) + 1;
      kv_frame_refs_.resize(n, 0); locked_kv_refs_.resize(n, 0);
      cached_page_tokens_.resize(n, 0); dirty_kv_flag_.resize(n, 0);
    }
  }
  int32_t compare_tokens_(const int32_t* a, int32_t al, const int32_t* b, int32_t bl) const {
    int32_t n = std::min(al, bl), i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
  }
  int32_t find_child_(int32_t pid, const int32_t* key) const {
    const auto& pn = nodes_[pid];
    if (pn.children.empty()) return -1;
    const size_t bytes = static_cast<size_t>(granularity) * sizeof(int32_t);
    if (pn.child_hash.empty()) {
      for (int32_t cid : pn.children) if (std::memcmp(nodes_[cid].token_data(), key, bytes) == 0) return cid;
      return -1;
    }
    auto it = pn.child_hash.find(hash_granule(key, granularity));
    if (it == pn.child_hash.end()) return -1;
    for (int32_t cid : it->second) if (std::memcmp(nodes_[cid].token_data(), key, bytes) == 0) return cid;
    return -1;
  }
  void add_child_(int32_t pid, int32_t cid) {
    auto& pn = nodes_[pid];
    pn.children.push_back(cid);
    if (!pn.child_hash.empty()) pn.child_hash[hash_granule(nodes_[cid].token_data(), granularity)].push_back(cid);
    else if (pn.children.size() > 8) {
      for (int32_t c : pn.children) pn.child_hash[hash_granule(nodes_[c].token_data(), granularity)].push_back(c);
    }
  }
  void remove_child_(int32_t pid, int32_t cid) {
    auto& pn = nodes_[pid];
    if (auto it = std::find(pn.children.begin(), pn.children.end(), cid); it != pn.children.end()) pn.children.erase(it);
    if (!pn.child_hash.empty()) {
      if (auto hit = pn.child_hash.find(hash_granule(nodes_[cid].token_data(), granularity)); hit != pn.child_hash.end()) {
        auto& v = hit->second;
        if (auto vit = std::find(v.begin(), v.end(), cid); vit != v.end()) v.erase(vit);
        if (v.empty()) pn.child_hash.erase(hit);
      }
    }
  }
  void replace_child_(int32_t pid, int32_t old_id, int32_t new_id) {
    auto& pn = nodes_[pid];
    for (int32_t& c : pn.children) if (c == old_id) { c = new_id; break; }
    if (!pn.child_hash.empty()) {
      if (auto hit = pn.child_hash.find(hash_granule(nodes_[new_id].token_data(), granularity)); hit != pn.child_hash.end()) {
        for (int32_t& c : hit->second) if (c == old_id) { c = new_id; break; }
      }
    }
  }
  void add_leaf_(int32_t nid) {
    if (nodes_[nid].leaf_pos >= 0) return;
    nodes_[nid].leaf_pos = static_cast<int32_t>(leaves_.size());
    leaves_.push_back(nid);
  }
  void remove_leaf_(int32_t nid) {
    int32_t pos = nodes_[nid].leaf_pos;
    if (pos < 0) return;
    int32_t last = leaves_.back();
    leaves_[pos] = last; nodes_[last].leaf_pos = pos;
    leaves_.pop_back(); nodes_[nid].leaf_pos = -1;
  }
  int32_t find_owner_(int32_t leaf_id, int32_t end) const {
    int32_t cur = leaf_id;
    while (cur > 0 && nodes_[cur].parent > 0 && end <= nodes_[nodes_[cur].parent].depth) cur = nodes_[cur].parent;
    check(nodes_[cur].begin() < end && end <= nodes_[cur].depth, "Owner not found");
    return cur;
  }

  StateRecovery recovery_for_endpoint_(int32_t ep, int32_t matched_len) const {
    StateRecovery rec;
    if (matched_len <= 0 || ep == 0) return rec;
    std::vector<int32_t> chain;
    int32_t base_slot = -1, base_end = 0, base_owner = -1;
    for (int32_t cur = ep; cur > 0; cur = nodes_[cur].parent) {
      chain.push_back(cur);
      const auto& cps = nodes_[cur].checkpoint_slots;
      bool found = false;
      for (int32_t i = static_cast<int32_t>(cps.size()) - 1; i >= 0; --i) {
        if (const auto& e = slot_entry_[cps[i]]; e.end <= matched_len) {
          base_slot = e.slot; base_end = e.end; base_owner = cur; found = true; break;
        }
      }
      if (found) break;
    }
    int32_t cursor = base_end, last_log_owner = -1;
    for (int32_t i = static_cast<int32_t>(chain.size()) - 1; i >= 0 && cursor < matched_len; --i) {
      int32_t nid = chain[i];
      const auto& entries = nodes_[nid].entries;
      auto it = std::upper_bound(entries.begin(), entries.end(), cursor, [](int32_t v, const StateEntry& e) { return v < e.end; });
      for (; it != entries.end() && cursor < matched_len && it->end <= matched_len; ++it) {
        if (it->kind != EntryKind::kLog || it->begin > cursor) continue;
        check(it->begin == cursor, "Misaligned Log entry");
        rec.logs.push_back(*it); last_log_owner = nid; cursor = it->end;
      }
    }
    rec.position = cursor; rec.base_checkpoint_slot = base_slot;
    if (!rec.logs.empty()) { rec.conv_slot = rec.logs.back().slot; rec.conv_granule = 0; rec.lock_node_id = last_log_owner; }
    else if (base_slot >= 0) { rec.conv_slot = base_slot; rec.conv_granule = 0; rec.lock_node_id = base_owner; }
    return rec;
  }

  void update_kv_frame_(int32_t f, int32_t rd, int32_t ld) {
    ensure_frame_(f);
    evictable_kv_count_ -= (kv_frame_refs_[f] > 0 && locked_kv_refs_[f] == 0) ? 1 : 0;
    kv_frame_refs_[f] += rd; locked_kv_refs_[f] += ld;
    check(locked_kv_refs_[f] >= 0 && locked_kv_refs_[f] <= kv_frame_refs_[f], "Invalid KV refs");
    evictable_kv_count_ += (kv_frame_refs_[f] > 0 && locked_kv_refs_[f] == 0) ? 1 : 0;
  }

  void update_size_class_(int32_t nid, bool was_locked) {
    auto& n = nodes_[nid];
    if (was_locked == n.is_locked()) return;
    mark_dirty_(nid);
    const int32_t d = n.is_locked() ? 1 : -1;
    for (const auto& [f, _] : n.kv_frame_counts) update_kv_frame_(f, 0, d);
    for (const auto& e : n.entries) {
      if (e.kind == EntryKind::kLog) {
        ensure_slot_(e.slot);
        unprunable_log_refs_[e.slot] += d;
        check(unprunable_log_refs_[e.slot] >= 0, "Negative unprunable_log_refs");
      }
      update_slot_reclaimability_(e.slot);
    }
    evictable_size -= d * n.length;
    locked_size += d * n.length;
  }

  bool slot_reclaimable_(int32_t s) const {
    if (!has_slot(s) || unprunable_log_refs_[s] > 0 || recovery_slot_refs_[s] > 0) return false;
    if (slot_entry_[s].kind == EntryKind::kCheckpoint && nodes_[slot_owner_[s]].is_locked()) {
      return checkpoint_successor(s) >= 0;
    }
    return true;
  }

  void update_slot_reclaimability_(int32_t s) {
    ensure_slot_(s);
    bool r = slot_reclaimable_(s);
    if (r == (evictable_slot_[s] != 0)) return;
    evictable_slot_[s] = r ? 1 : 0;
    if (allocator) allocator->set_reclaimable(s, r);
  }

  void register_entry_(int32_t owner_id, const StateEntry& entry, bool refresh_neighbors = true) {
    ensure_slot_(entry.slot);
    auto& n = nodes_[owner_id];
    auto& entries = n.entries;
    if (entries.empty() || entries.back().end <= entry.end) entries.push_back(entry);
    else entries.insert(std::upper_bound(entries.begin(), entries.end(), entry.end,
                                         [](int32_t v, const StateEntry& e) { return v < e.end; }), entry);
    slot_owner_[entry.slot] = owner_id;
    slot_entry_[entry.slot] = entry;
    if (entry.kind == EntryKind::kCheckpoint) {
      auto& cps = n.checkpoint_slots;
      if (cps.empty() || slot_entry_[cps.back()].end <= entry.end) cps.push_back(entry.slot);
      else cps.insert(std::upper_bound(cps.begin(), cps.end(), entry.end,
                                       [&](int32_t v, int32_t s) { return v < slot_entry_[s].end; }), entry.slot);
    }
    if (n.is_locked() && entry.kind == EntryKind::kLog) ++unprunable_log_refs_[entry.slot];
    update_slot_reclaimability_(entry.slot);
    if (refresh_neighbors) refresh_entry_neighbors_(owner_id, entry.end);
  }

  void unregister_entry_(const StateEntry& entry, int32_t owner_id, bool refresh_neighbors = true) {
    if (nodes_[owner_id].is_locked() && entry.kind == EntryKind::kLog) {
      check(unprunable_log_refs_[entry.slot] > 0, "Negative unprunable_log_refs");
      --unprunable_log_refs_[entry.slot];
    }
    slot_owner_[entry.slot] = -1;
    update_slot_reclaimability_(entry.slot);
    if (refresh_neighbors) refresh_entry_neighbors_(owner_id, entry.end);
  }

  void refresh_ancestor_tail_(int32_t nid) {
    for (int32_t p = nodes_[nid].parent; p > 0; p = nodes_[p].parent) {
      if (!nodes_[p].entries.empty()) {
        if (const auto& last = nodes_[p].entries.back(); last.kind == EntryKind::kCheckpoint)
          update_slot_reclaimability_(last.slot);
        break;
      }
    }
  }

  void refresh_entry_neighbors_(int32_t nid, int32_t pos) {
    const auto& cps = nodes_[nid].checkpoint_slots;
    if (!cps.empty()) {
      auto it = std::lower_bound(cps.begin(), cps.end(), pos - granularity * 2,
                                 [&](int32_t s, int32_t v) { return slot_entry_[s].end < v; });
      for (; it != cps.end(); ++it) {
        const auto& e = slot_entry_[*it];
        if (e.end > pos + granularity * 2) { if (e.begin == pos) update_slot_reclaimability_(e.slot); break; }
        update_slot_reclaimability_(e.slot);
      }
    }
    refresh_ancestor_tail_(nid);
  }

  void refresh_checkpoint_neighbors_(int32_t nid) {
    if (nid <= 0 || !nodes_[nid].alive) return;
    for (int32_t s : nodes_[nid].checkpoint_slots) update_slot_reclaimability_(s);
    refresh_ancestor_tail_(nid);
  }
};

}  // namespace logssms::logstate
