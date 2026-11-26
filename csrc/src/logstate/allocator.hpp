#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace logssms::logstate {

using Int = int64_t;

inline void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

enum class SlotKind : uint8_t { kFree = 0, kLog = 1, kWork = 2, kCheckpoint = 3 };

struct DecayRecord {
  int32_t frame = -1, index = -1;
  bool valid() const { return frame >= 0; }
};

struct SlotRecord {
  SlotKind kind = SlotKind::kFree;
  bool reclaimable = false;
  std::vector<int32_t> payload, conv;  // conv flat: conv_granules * conv_frames
  int32_t conv_granules = 0;
  DecayRecord decay;
  int32_t frame_count() const { return static_cast<int32_t>(payload.size() + conv.size()); }
};

struct RetiredResources {
  int64_t id = 0;
  std::vector<int32_t> frames;
  std::vector<DecayRecord> decays;
};

struct ReserveResult {
  bool ok = false;
  std::vector<int32_t> kv_frames, log_slots, work_slots;
};

struct UsageStats {
  int32_t total_frames = 0, free_frames = 0, kv_frames = 0, payload_frames = 0, conv_frames = 0;
  int32_t decay_frames = 0, free_decay_records = 0, log_slots = 0, checkpoint_slots = 0;
  int32_t work_slots = 0, log_payload_conv = 0, checkpoint_payload_conv = 0, work_payload_conv = 0;
};

class AllocatorCore {
 public:
  int32_t num_frames = 0, payload_frames = 0, log_payload_frames = 0, descriptor_frames = 0;
  int32_t checkpoint_log_count = 0, conv_frames = 0, decay_records_per_frame = 0;
  int32_t num_granules = 0, materialize_log_count = 0;
  int64_t frame_bytes = 0;

  AllocatorCore(int32_t num_frames_, int64_t frame_bytes_, int64_t payload_bytes,
                int64_t conv_snapshot_bytes, int64_t decay_bytes,
                int32_t num_granules_, int64_t log_payload_bytes = -1)
      : num_frames(num_frames_), num_granules(num_granules_), frame_bytes(frame_bytes_) {
    check(num_frames > 0 && frame_bytes > 0 && payload_bytes > 0 &&
              conv_snapshot_bytes > 0 && decay_bytes > 0 && num_granules > 0,
          "Invalid LogState allocator configuration");
    check(decay_bytes <= frame_bytes, "decay records must fit within one frame");
    payload_frames = static_cast<int32_t>((payload_bytes + frame_bytes - 1) / frame_bytes);
    log_payload_frames = log_payload_bytes < 0
                             ? payload_frames
                             : static_cast<int32_t>((log_payload_bytes + frame_bytes - 1) / frame_bytes);
    check(log_payload_frames > 0, "Invalid log_payload_frames");
    descriptor_frames = std::max(payload_frames, log_payload_frames);
    checkpoint_log_count = (payload_frames + log_payload_frames - 1) / log_payload_frames;
    conv_frames = static_cast<int32_t>((conv_snapshot_bytes + frame_bytes - 1) / frame_bytes);
    decay_records_per_frame = static_cast<int32_t>(frame_bytes / decay_bytes);
    materialize_log_count = std::max(
        checkpoint_log_count,
        (payload_frames + conv_frames) / (log_payload_frames + num_granules * conv_frames) + 1);
    words_per_frame_ = (decay_records_per_frame + 63) / 64;
    free_frames_.resize(num_frames);
    for (int32_t i = 0; i < num_frames; ++i) free_frames_[i] = num_frames - 1 - i;
    is_kv_.assign(num_frames, 0);
    decay_masks_.assign(static_cast<size_t>(num_frames) * words_per_frame_, 0);
    reclaimable_decay_masks_.assign(static_cast<size_t>(num_frames) * words_per_frame_, 0);
    decay_counts_.assign(num_frames, 0);
    reclaimable_decay_counts_.assign(num_frames, 0);
    open_decay_pos_.assign(num_frames, -1);
    dirty_decay_flag_.assign(num_frames, 0);
    slots_.reserve(num_frames);
  }

  int32_t free_frames_count() const { return static_cast<int32_t>(free_frames_.size()); }
  int32_t free_decay_records() const { return free_decay_records_; }
  int32_t decay_count(int32_t f) const { return (f >= 0 && f < num_frames) ? decay_counts_[f] : 0; }
  bool has_slot(int32_t s) const { return s >= 0 && s < static_cast<int32_t>(slots_.size()) && slots_[s].kind != SlotKind::kFree; }
  const SlotRecord& slot(int32_t s) const { check(has_slot(s), "Unknown LogState slot"); return slots_[s]; }
  SlotRecord& slot_mut(int32_t s) { check(has_slot(s), "Unknown LogState slot"); return slots_[s]; }
  int32_t num_slot_capacity() const { return static_cast<int32_t>(slots_.size()); }
  const std::vector<int32_t>& dirty_decay_frames() const { return dirty_decay_frames_; }
  void clear_dirty_decay_frames() {
    for (int32_t f : dirty_decay_frames_) dirty_decay_flag_[f] = 0;
    dirty_decay_frames_.clear();
  }

  void set_reclaimable(int32_t sid, bool reclaimable) {
    auto& rec = slot_mut(sid);
    if (rec.reclaimable == reclaimable) return;
    rec.reclaimable = reclaimable;
    const int32_t d = reclaimable ? 1 : -1;
    reclaimable_slots_count_ += d;
    reclaimable_frames_ += d * rec.frame_count();
    if (rec.decay.valid()) set_decay_reclaimable_(rec.decay, reclaimable);
  }

  int32_t required_frames(int32_t kv, int32_t logs, int32_t states, int32_t free_decay_override = -1) const {
    const int32_t avail = free_decay_override < 0 ? free_decay_records_ : free_decay_override;
    const int32_t df = (std::max<int32_t>(0, logs - avail) + decay_records_per_frame - 1) / decay_records_per_frame;
    return kv + logs * log_payload_frames + states * payload_frames + (logs * num_granules + states) * conv_frames + df;
  }

  std::pair<int32_t, int32_t> projected_free_capacity() const {
    return {free_frames_count() + reclaimable_frames_ + reclaimable_decay_frames_,
            free_decay_records_ + reclaimable_decay_records_ - reclaimable_decay_frames_ * decay_records_per_frame};
  }

  int32_t lock_frame_cost(const std::vector<int32_t>& slot_ids) const {
    int32_t frames = 0;
    std::vector<int32_t> counted;
    for (int32_t sid : slot_ids) {
      if (!has_slot(sid) || !slots_[sid].reclaimable) continue;
      const auto& rec = slots_[sid];
      frames += rec.frame_count();
      if (rec.decay.valid() && reclaimable_decay_counts_[rec.decay.frame] == decay_counts_[rec.decay.frame] &&
          std::find(counted.begin(), counted.end(), rec.decay.frame) == counted.end()) {
        counted.push_back(rec.decay.frame);
      }
    }
    return frames + static_cast<int32_t>(counted.size());
  }

  int32_t admissible_pages(int32_t evictable_kv_frames) const {
    auto [ff, fdr] = projected_free_capacity();
    ff += evictable_kv_frames;
    const int32_t pf = 1 + log_payload_frames + num_granules * conv_frames;
    const int32_t wf = payload_frames + conv_frames;
    int32_t lo = 0, hi = std::max<int32_t>(0, (ff - wf) / pf);
    while (lo < hi) {
      int32_t mid = (lo + hi + 1) / 2;
      if (required_frames(mid, mid, 1, fdr) <= ff) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  ReserveResult reserve(int32_t kv, int32_t logs, int32_t states) {
    check(kv >= 0 && logs >= 0 && states >= 0, "Invalid reserve counts");
    ReserveResult res;
    if (required_frames(kv, logs, states) > free_frames_count()) return res;
    res.ok = true;
    res.kv_frames.reserve(kv);
    for (int32_t i = 0; i < kv; ++i) {
      int32_t f = take_frame_();
      is_kv_[f] = 1;
      ++kv_count_;
      res.kv_frames.push_back(f);
    }
    res.log_slots.reserve(logs);
    for (int32_t i = 0; i < logs; ++i) res.log_slots.push_back(new_slot_(SlotKind::kLog));
    res.work_slots.reserve(states);
    for (int32_t i = 0; i < states; ++i) res.work_slots.push_back(new_slot_(SlotKind::kWork));
    return res;
  }

  int32_t checkpoint_frame_capacity(const std::vector<int32_t>& slot_ids) const {
    int32_t total = 0;
    for (int32_t sid : slot_ids) total += slot(sid).frame_count();
    return total - conv_frames;
  }

  int32_t checkpoint_frame_savings(const std::vector<int32_t>& slot_ids) const {
    int32_t savings = checkpoint_frame_capacity(slot_ids) - payload_frames;
    std::vector<std::pair<int32_t, int32_t>> counts;
    for (int32_t sid : slot_ids) {
      const auto& d = slot(sid).decay;
      if (!d.valid()) continue;
      bool found = false;
      for (auto& kv : counts) {
        if (kv.first == d.frame) { ++kv.second; found = true; break; }
      }
      if (!found) counts.emplace_back(d.frame, 1);
    }
    for (const auto& [f, cnt] : counts) {
      if (cnt == decay_counts_[f]) ++savings;
    }
    return savings;
  }

  int64_t materialize_group(const std::vector<int32_t>& slot_ids) {
    check(!slot_ids.empty(), "Empty materialization group");
    check(checkpoint_frame_capacity(slot_ids) >= payload_frames, "Insufficient group frames for checkpoint");
    check(checkpoint_frame_savings(slot_ids) >= 0, "Materialization must cover its expansion");

    std::vector<int32_t> was_reclaimable;
    for (int32_t sid : slot_ids) {
      if (slot(sid).reclaimable) { was_reclaimable.push_back(sid); set_reclaimable(sid, false); }
    }
    const auto& last_rec = slot(slot_ids.back());
    check(static_cast<int32_t>(last_rec.conv.size()) >= conv_frames, "Missing terminal conv frames");
    std::vector<int32_t> terminal(last_rec.conv.end() - conv_frames, last_rec.conv.end());

    std::vector<int32_t> frames;
    frames.reserve(slot_ids.size() * (log_payload_frames + num_granules * conv_frames));
    for (auto it = slot_ids.rbegin(); it != slot_ids.rend(); ++it) {
      const auto& p = slot(*it).payload;
      frames.insert(frames.end(), p.begin(), p.end());
    }
    for (int32_t sid : slot_ids) {
      for (int32_t f : slot(sid).conv) {
        if (std::find(terminal.begin(), terminal.end(), f) == terminal.end()) frames.push_back(f);
      }
    }
    std::vector<int32_t> new_payload(frames.begin(), frames.begin() + payload_frames);
    std::vector<int32_t> surplus(frames.begin() + payload_frames, frames.end());
    std::vector<DecayRecord> decays;
    for (int32_t sid : slot_ids) {
      if (slot(sid).decay.valid()) decays.push_back(slot(sid).decay);
    }
    for (size_t i = 0; i + 1 < slot_ids.size(); ++i) {
      auto& r = slot_mut(slot_ids[i]);
      r.kind = SlotKind::kCheckpoint;
      r.payload.clear();
      r.conv.clear();
      r.conv_granules = 0;
      r.decay = {-1, -1};
    }
    {
      auto& r = slot_mut(slot_ids.back());
      r.kind = SlotKind::kCheckpoint;
      r.payload = std::move(new_payload);
      r.conv = std::move(terminal);
      r.conv_granules = 1;
      r.decay = {-1, -1};
    }

    int64_t rid = ++next_retire_id_;
    RetiredResources ret{rid, std::move(surplus), std::move(decays)};
    reclaimable_frames_ += static_cast<int32_t>(ret.frames.size());
    for (const auto& d : ret.decays) set_decay_reclaimable_(d, true);
    retired_.emplace(rid, std::move(ret));
    for (int32_t sid : was_reclaimable) set_reclaimable(sid, true);
    return rid;
  }

  const std::vector<int32_t>& retired_frames(int64_t rid) const {
    auto it = retired_.find(rid);
    check(it != retired_.end(), "Unknown retirement ID");
    return it->second.frames;
  }

  void release_retirement(int64_t rid) {
    auto it = retired_.find(rid);
    check(it != retired_.end(), "Unknown retirement ID");
    auto ret = std::move(it->second);
    retired_.erase(it);
    reclaimable_frames_ -= static_cast<int32_t>(ret.frames.size());
    for (int32_t f : ret.frames) free_frames_.push_back(f);
    for (const auto& d : ret.decays) { set_decay_reclaimable_(d, false); release_decay_(d); }
  }

  void release_kv(const std::vector<int32_t>& frames) {
    for (int32_t f : frames) {
      check(f >= 0 && f < num_frames && is_kv_[f], "Releasing unallocated KV frame");
      is_kv_[f] = 0;
      --kv_count_;
      free_frames_.push_back(f);
    }
  }

  void release_slot(int32_t sid) {
    set_reclaimable(sid, false);
    auto& rec = slot_mut(sid);
    for (int32_t f : rec.payload) free_frames_.push_back(f);
    for (int32_t f : rec.conv) free_frames_.push_back(f);
    if (rec.decay.valid()) release_decay_(rec.decay);
    rec.kind = SlotKind::kFree;
    rec.payload.clear();
    rec.conv.clear();
    rec.conv_granules = 0;
    rec.decay = {-1, -1};
    free_slot_ids_.push_back(sid);
    --allocated_slots_count_;
  }

  void seed_checkpoint_slot(int32_t sid) {
    auto& rec = slot_mut(sid);
    check(rec.kind == SlotKind::kWork && !rec.decay.valid(), "seed_checkpoint_slot expects work slot");
    rec.kind = SlotKind::kCheckpoint;
  }

  void pack_descriptors(const std::vector<int32_t>& slot_ids,
                        int32_t* payload_out, int32_t* conv_out, int32_t* decay_out) const {
    const int32_t cw = num_granules * conv_frames;
    for (size_t i = 0; i < slot_ids.size(); ++i) {
      const auto& rec = slot(slot_ids[i]);
      int32_t* pr = payload_out + i * descriptor_frames;
      std::fill_n(pr, descriptor_frames, -1);
      std::copy(rec.payload.begin(), rec.payload.end(), pr);
      int32_t* cr = conv_out + i * cw;
      std::fill_n(cr, cw, -1);
      if (rec.kind == SlotKind::kLog || rec.kind == SlotKind::kWork) {
        std::copy(rec.conv.begin(), rec.conv.end(), cr);
      } else if (rec.kind == SlotKind::kCheckpoint && !rec.conv.empty()) {
        std::copy(rec.conv.begin(), rec.conv.end(), cr + (num_granules - 1) * conv_frames);
      }
      int32_t* dr = decay_out + i * 2;
      dr[0] = rec.decay.valid() ? rec.decay.frame : -1;
      dr[1] = rec.decay.valid() ? rec.decay.index : -1;
    }
  }

  UsageStats usage_stats() const {
    UsageStats u;
    u.total_frames = num_frames;
    u.free_frames = free_frames_count();
    u.kv_frames = kv_count_;
    u.decay_frames = active_decay_frames_;
    u.free_decay_records = free_decay_records_;
    for (const auto& rec : slots_) {
      if (rec.kind == SlotKind::kFree) continue;
      int32_t pf = static_cast<int32_t>(rec.payload.size()), cf = static_cast<int32_t>(rec.conv.size());
      u.payload_frames += pf;
      u.conv_frames += cf;
      if (rec.kind == SlotKind::kLog) { ++u.log_slots; u.log_payload_conv += pf + cf; }
      else if (rec.kind == SlotKind::kCheckpoint) { ++u.checkpoint_slots; u.checkpoint_payload_conv += pf + cf; }
      else if (rec.kind == SlotKind::kWork) { ++u.work_slots; u.work_payload_conv += pf + cf; }
    }
    for (const auto& [_, ret] : retired_) u.payload_frames += static_cast<int32_t>(ret.frames.size());
    return u;
  }

  void check_integrity() const {
    std::vector<uint8_t> seen(num_frames, 0);
    int32_t owned = 0, kv_chk = 0, d_rec = 0, alloc_s = 0, d_bits = 0, fd_chk = 0, act_df = 0;
    auto mark = [&](int32_t f) {
      check(f >= 0 && f < num_frames && !seen[f], "Invalid or duplicate frame ownership");
      seen[f] = 1;
      ++owned;
    };
    for (int32_t f : free_frames_) mark(f);
    for (int32_t f = 0; f < num_frames; ++f) {
      if (is_kv_[f]) { mark(f); ++kv_chk; }
      if (decay_counts_[f] > 0) mark(f);
    }
    check(kv_chk == kv_count_, "KV frame count mismatch");
    for (const auto& [_, ret] : retired_) {
      for (int32_t f : ret.frames) mark(f);
      d_rec += static_cast<int32_t>(ret.decays.size());
    }
    for (const auto& rec : slots_) {
      if (rec.kind == SlotKind::kFree) continue;
      ++alloc_s;
      for (int32_t f : rec.payload) mark(f);
      for (int32_t f : rec.conv) mark(f);
      check(rec.decay.valid() == (rec.kind == SlotKind::kLog), "Decay record must belong only to Log slot");
      if (rec.decay.valid()) {
        check(get_decay_bit_(decay_masks_, rec.decay.frame, rec.decay.index), "Missing decay bit");
        ++d_rec;
      }
    }
    check(alloc_s == allocated_slots_count_, "Allocated slot count mismatch");
    for (int32_t f = 0; f < num_frames; ++f) {
      if (decay_counts_[f] > 0) {
        ++act_df;
        d_bits += decay_counts_[f];
        fd_chk += decay_records_per_frame - decay_counts_[f];
        check((open_decay_pos_[f] >= 0) == (decay_counts_[f] < decay_records_per_frame), "Open decay mismatch");
      } else {
        check(open_decay_pos_[f] == -1, "Empty decay frame still open");
      }
    }
    check(act_df == active_decay_frames_ && d_bits == d_rec && fd_chk == free_decay_records_,
          "Decay accounting mismatch");
    check(owned == num_frames, "Frame conservation violated");
  }

  std::vector<int32_t> allocated_kv_list() const {
    std::vector<int32_t> out;
    out.reserve(kv_count_);
    for (int32_t f = 0; f < num_frames; ++f) if (is_kv_[f]) out.push_back(f);
    return out;
  }
  std::vector<int32_t> allocated_slot_list() const {
    std::vector<int32_t> out;
    out.reserve(allocated_slots_count_);
    for (int32_t i = 0; i < static_cast<int32_t>(slots_.size()); ++i)
      if (slots_[i].kind != SlotKind::kFree) out.push_back(i);
    return out;
  }
  std::vector<int32_t> reclaimable_slot_list() const {
    std::vector<int32_t> out;
    out.reserve(reclaimable_slots_count_);
    for (int32_t i = 0; i < static_cast<int32_t>(slots_.size()); ++i)
      if (slots_[i].kind != SlotKind::kFree && slots_[i].reclaimable) out.push_back(i);
    return out;
  }

 private:
  int32_t words_per_frame_ = 1, kv_count_ = 0, allocated_slots_count_ = 0;
  int32_t active_decay_frames_ = 0, free_decay_records_ = 0;
  int32_t reclaimable_slots_count_ = 0, reclaimable_frames_ = 0;
  int32_t reclaimable_decay_records_ = 0, reclaimable_decay_frames_ = 0;
  int64_t next_retire_id_ = 0;

  std::vector<int32_t> free_frames_, free_slot_ids_, decay_counts_, reclaimable_decay_counts_;
  std::vector<int32_t> open_decay_frames_, open_decay_pos_, dirty_decay_frames_;
  std::vector<uint8_t> is_kv_, dirty_decay_flag_;
  std::vector<SlotRecord> slots_;
  std::vector<uint64_t> decay_masks_, reclaimable_decay_masks_;
  std::unordered_map<int64_t, RetiredResources> retired_;

  void mark_dirty_decay_(int32_t f) {
    if (!dirty_decay_flag_[f]) { dirty_decay_flag_[f] = 1; dirty_decay_frames_.push_back(f); }
  }
  bool get_decay_bit_(const std::vector<uint64_t>& m, int32_t f, int32_t i) const {
    return (m[static_cast<size_t>(f) * words_per_frame_ + (i >> 6)] & (uint64_t{1} << (i & 63))) != 0;
  }
  void set_decay_bit_(std::vector<uint64_t>& m, int32_t f, int32_t i, bool v) {
    auto& w = m[static_cast<size_t>(f) * words_per_frame_ + (i >> 6)];
    uint64_t b = uint64_t{1} << (i & 63);
    if (v) w |= b; else w &= ~b;
  }
  void add_open_decay_(int32_t f) {
    if (open_decay_pos_[f] >= 0) return;
    open_decay_pos_[f] = static_cast<int32_t>(open_decay_frames_.size());
    open_decay_frames_.push_back(f);
  }
  void remove_open_decay_(int32_t f) {
    int32_t pos = open_decay_pos_[f];
    if (pos < 0) return;
    int32_t last = open_decay_frames_.back();
    open_decay_frames_[pos] = last;
    open_decay_pos_[last] = pos;
    open_decay_frames_.pop_back();
    open_decay_pos_[f] = -1;
  }
  void set_decay_reclaimable_(DecayRecord rec, bool r) {
    int32_t f = rec.frame;
    if (get_decay_bit_(reclaimable_decay_masks_, f, rec.index) == r) return;
    bool was_full = (decay_counts_[f] > 0 && reclaimable_decay_counts_[f] == decay_counts_[f]);
    set_decay_bit_(reclaimable_decay_masks_, f, rec.index, r);
    reclaimable_decay_counts_[f] += r ? 1 : -1;
    reclaimable_decay_records_ += r ? 1 : -1;
    bool is_full = (decay_counts_[f] > 0 && reclaimable_decay_counts_[f] == decay_counts_[f]);
    reclaimable_decay_frames_ += static_cast<int32_t>(is_full) - static_cast<int32_t>(was_full);
  }
  int32_t take_frame_() {
    check(!free_frames_.empty(), "Out of free frames");
    int32_t f = free_frames_.back();
    free_frames_.pop_back();
    return f;
  }
  DecayRecord take_decay_() {
    if (!open_decay_frames_.empty()) {
      int32_t f = open_decay_frames_.front(), idx = -1;
      const size_t base = static_cast<size_t>(f) * words_per_frame_;
      for (int32_t w = 0; w < words_per_frame_; ++w) {
        uint64_t avail = ~decay_masks_[base + w];
        if (avail != 0) {
          idx = (w << 6) + std::countr_zero(avail);
          if (idx < decay_records_per_frame) break;
          idx = -1;
        }
      }
      check(idx >= 0 && idx < decay_records_per_frame, "Open decay frame has no free bit");
      bool was_full = (reclaimable_decay_counts_[f] == decay_counts_[f]);
      set_decay_bit_(decay_masks_, f, idx, true);
      ++decay_counts_[f];
      reclaimable_decay_frames_ += static_cast<int32_t>(reclaimable_decay_counts_[f] == decay_counts_[f]) - static_cast<int32_t>(was_full);
      --free_decay_records_;
      if (decay_counts_[f] == decay_records_per_frame) remove_open_decay_(f);
      mark_dirty_decay_(f);
      return {f, idx};
    }
    int32_t f = take_frame_();
    set_decay_bit_(decay_masks_, f, 0, true);
    decay_counts_[f] = 1;
    ++active_decay_frames_;
    free_decay_records_ += decay_records_per_frame - 1;
    if (decay_records_per_frame > 1) add_open_decay_(f);
    mark_dirty_decay_(f);
    return {f, 0};
  }
  void release_decay_(DecayRecord rec) {
    int32_t f = rec.frame;
    check(get_decay_bit_(decay_masks_, f, rec.index), "Releasing unallocated decay bit");
    bool was_full = (decay_counts_[f] > 0 && reclaimable_decay_counts_[f] == decay_counts_[f]);
    set_decay_bit_(decay_masks_, f, rec.index, false);
    --decay_counts_[f];
    bool is_full = (decay_counts_[f] > 0 && reclaimable_decay_counts_[f] == decay_counts_[f]);
    reclaimable_decay_frames_ += static_cast<int32_t>(is_full) - static_cast<int32_t>(was_full);
    mark_dirty_decay_(f);
    if (decay_counts_[f] > 0) { add_open_decay_(f); ++free_decay_records_; }
    else { remove_open_decay_(f); --active_decay_frames_; free_decay_records_ -= decay_records_per_frame - 1; free_frames_.push_back(f); }
  }
  int32_t new_slot_(SlotKind kind) {
    int32_t sid;
    if (!free_slot_ids_.empty()) { sid = free_slot_ids_.back(); free_slot_ids_.pop_back(); }
    else { sid = static_cast<int32_t>(slots_.size()); slots_.emplace_back(); }
    auto& rec = slots_[sid];
    rec.kind = kind;
    rec.reclaimable = false;
    const int32_t pc = (kind == SlotKind::kLog) ? log_payload_frames : payload_frames;
    const int32_t cg = (kind == SlotKind::kLog) ? num_granules : 1;
    rec.payload.resize(pc);
    for (int32_t i = 0; i < pc; ++i) rec.payload[i] = take_frame_();
    rec.conv_granules = cg;
    rec.conv.resize(cg * conv_frames);
    for (int32_t i = 0; i < cg * conv_frames; ++i) rec.conv[i] = take_frame_();
    rec.decay = (kind == SlotKind::kLog) ? take_decay_() : DecayRecord{-1, -1};
    ++allocated_slots_count_;
    return sid;
  }
};

}  // namespace logssms::logstate
