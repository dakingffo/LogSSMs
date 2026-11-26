#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "radix_tree.hpp"

namespace logssms::logstate {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPriorTailAlpha = 1.0 / (kPi * kPi / 6.0);
constexpr double kPriorInternalAlpha = 1.0 - kPriorTailAlpha;
constexpr double kPriorExposure = 32.0;

struct HitChain {
  int32_t id = 0, origin = 0, tail = 0;
  int64_t group_id = 0, exposure_query = 0, hit_query = 0;
  std::array<double, 32> hit_counts{}, exposure{};
  uint32_t hit_mask = 0, exposure_mask = 0;
  double tail_hits = 0.0, tail_exposure = 0.0, exponent = 2.0;
  double log_amplitude = std::numeric_limits<double>::quiet_NaN();
  double tail_probability = kPriorTailAlpha / kPriorExposure;
  bool published = false, alive = true;
};

struct DensityModel {
  int64_t id = 0;
  double origin = 0.0, tail = 0.0, atom = 0.0, g = 1.0, exponent = 2.0, log_amplitude = 0.0;
  const std::vector<double>* shared_prefix = nullptr;
  double shared_scale = 0.0;
  int32_t shared_count = 0;
  std::vector<double> probability{0.0};

  double prefix(double position) const {
    auto i = static_cast<Int>(std::floor((tail - position) / g));
    if (shared_prefix) {
      return (*shared_prefix)[std::clamp<Int>(i, 0, static_cast<Int>(shared_count))] * shared_scale;
    }
    return probability[std::clamp<Int>(i, 0, static_cast<Int>(probability.size()) - 1)];
  }
  double mass(double begin, double end) const {
    begin = std::max(begin, origin + 1.0);
    if (begin >= end) return 0.0;
    const double discrete = (begin <= tail && tail < end) ? atom : 0.0;
    return discrete + std::max(0.0, prefix(begin) - (end <= tail ? prefix(end) : 0.0));
  }
};

struct PendingEventKey {
  std::vector<std::pair<int32_t, int32_t>> targets;
  int32_t position = 0;
  bool operator==(const PendingEventKey& o) const { return position == o.position && targets == o.targets; }
};

struct PendingEventHash {
  size_t operator()(const PendingEventKey& k) const {
    uint64_t h = static_cast<uint64_t>(k.position) * 1099511628211ULL;
    for (const auto& [cid, t] : k.targets) {
      h ^= (static_cast<uint64_t>(cid) << 32) | static_cast<uint32_t>(t);
      h *= 1099511628211ULL;
    }
    return static_cast<size_t>(h);
  }
};

class ReuseModelCore {
 public:
  int32_t g = 16, window = 8192, half_life = 1024;
  double max_exponent = 16.0;
  bool enabled = true;
  int64_t query_count = 0, model_version = 0;

  ReuseModelCore(int32_t granularity, int32_t min_sampling_len, int32_t hit_half_life,
                 double max_density_exponent, bool enable_merge_adjustment)
      : g(granularity), window(min_sampling_len), half_life(hit_half_life),
        max_exponent(max_density_exponent), enabled(enable_merge_adjustment) {
    check(g > 0 && window > 0 && half_life > 0 && max_exponent > 0.0, "Invalid ReuseModelCore config");
    chains_.emplace_back();
    prior_mass_prefix_.push_back(0.0);
    log_dist_table_.push_back(0.0);
  }

  void reset() {
    chains_.assign(1, HitChain{});
    query_count = 0;
    geometry_topology_rev_ = fit_epoch_ = -1;
    descendants_.clear();
    rates_.clear();
    events_.clear();
    density_cache_.clear();
    ++model_version;
  }

  double prior_mass(int32_t bins, double exponent = 2.0) {
    if (exponent == 2.0) {
      while (static_cast<int32_t>(prior_mass_prefix_.size()) <= bins) {
        int32_t j = static_cast<int32_t>(prior_mass_prefix_.size());
        double a = static_cast<double>(j) * g;
        prior_mass_prefix_.push_back(prior_mass_prefix_.back() + g / (a * a));
      }
      return prior_mass_prefix_[bins];
    }
    ensure_log_dist_(bins);
    double total = 0.0;
    for (int32_t j = 1; j <= bins; ++j) total += g * std::exp(-exponent * log_dist_table_[j]);
    return total;
  }

  double bucket_avg_density(int32_t bucket, double exponent) {
    int32_t start = 1 << bucket, end = start << 1;
    if (exponent == 2.0) {
      return (prior_mass(end - 1, 2.0) - prior_mass(start - 1, 2.0)) / (static_cast<double>(g) * start);
    }
    ensure_log_dist_(end - 1);
    double sum = 0.0;
    for (int32_t j = start; j < end; ++j) sum += std::exp(-exponent * log_dist_table_[j]);
    return sum / start;
  }

  int32_t new_chain(int32_t origin, int64_t group_id = -1) {
    fit_epoch_ = -1;
    int32_t id = static_cast<int32_t>(chains_.size());
    HitChain c;
    c.id = id;
    c.group_id = (group_id < 0) ? static_cast<int64_t>(id) : group_id;
    c.origin = c.tail = origin;
    c.exposure_query = c.hit_query = query_count;
    chains_.push_back(c);
    return id;
  }

  void publish(int32_t cid, int32_t tail) {
    if (cid <= 0 || cid >= static_cast<int32_t>(chains_.size())) return;
    auto& c = chains_[cid];
    if (c.tail != tail) { fit_epoch_ = -1; c.tail = tail; }
    if (tail > c.origin) c.published = true;
  }

  const HitChain& chain(int32_t id) const { return chains_[id]; }

  void record_query(const RadixTreeCore& tree, int32_t trace_node_id, int32_t matched_len, int64_t group_id) {
    prepare_geometry_(tree);
    std::vector<std::pair<int32_t, int32_t>> targets;
    if (matched_len > 0 && trace_node_id >= 0) {
      int32_t cur = trace_node_id;
      while (cur > 0 && matched_len <= tree.node(cur).begin()) cur = tree.node(cur).parent;
      if (cur > 0 && cur < static_cast<int32_t>(descendants_.size())) {
        const auto& groups = descendants_[cur];
        if (group_id >= 0) {
          if (auto it = groups.find(group_id); it != groups.end()) targets = it->second;
        } else if (groups.size() == 1) {
          targets = groups.begin()->second;
        }
      }
    }
    record_(std::move(targets), matched_len);
  }

  void refresh(const RadixTreeCore* tree = nullptr, int32_t extra_chain = -1) {
    if (query_count == fit_epoch_ && (extra_chain <= 0 || !std::isnan(chains_[extra_chain].log_amplitude))) return;
    flush_();
    std::vector<uint8_t> live(chains_.size(), 0);
    if (tree) {
      for (int32_t id = 1; id < tree->num_nodes_capacity(); ++id) {
        const auto& n = tree->node(id);
        if (n.alive && n.analysis_owner > 0) live[n.analysis_owner] = 1;
      }
    }
    if (extra_chain > 0) live[extra_chain] = 1;

    std::unordered_map<int64_t, std::vector<int32_t>> groups;
    for (int32_t cid = 1; cid < static_cast<int32_t>(chains_.size()); ++cid) {
      auto& c = chains_[cid];
      if (!c.alive) continue;
      if (tree && !live[cid]) { c.published = false; continue; }
      if (!c.published && !live[cid]) continue;

      if (auto rit = rates_.find(cid); rit != rates_.end()) {
        settle_(c, rit->second.data(), rit->second.size(), 1.0);
      } else if (geometry_topology_rev_ < 0) {
        std::array<double, 32> w{};
        add_widths_(w, g, c.tail - c.origin - g, 1.0);
        settle_(c, w.data(), 32, 1.0);
      } else {
        settle_(c, nullptr, 0, 0.0);
      }

      if (int64_t age = query_count - c.hit_query; age > 0) {
        double wt = weight_(age);
        for (uint32_t m = c.hit_mask; m; m &= m - 1) {
          int b = std::countr_zero(m);
          c.hit_counts[b] *= wt;
          if (c.hit_counts[b] < 1e-8) { c.hit_counts[b] = 0.0; c.hit_mask &= ~(uint32_t{1} << b); }
        }
        c.tail_hits *= wt;
        if (c.tail_hits < 1e-8) c.tail_hits = 0.0;
        c.hit_query = query_count;
      }
      groups[c.group_id].push_back(cid);
    }
    for (const auto& [_, cids] : groups) {
      double k = fit_(cids);
      for (int32_t cid : cids) estimate_(chains_[cid], k);
    }
    fit_epoch_ = query_count;
    density_cache_.clear();
    ++model_version;
  }

  const DensityModel& density_for_node(const RadixTreeCore& tree, int32_t node_id) {
    const auto& n = tree.node(node_id);
    int32_t cid = enabled ? n.analysis_owner : -1;
    if (cid <= 0) return default_density_(std::max(g, n.depth));
    const auto& c = chains_[cid];
    int32_t tail = std::max(c.tail, n.depth);
    return cached_chain_density_(cid, c, tail);
  }

  const DensityModel& active_density(int32_t cid, int32_t tail) {
    tail = std::max(g, tail);
    if (!enabled || cid <= 0) return default_density_(tail);
    refresh(nullptr, cid);
    return cached_chain_density_(cid, chains_[cid], tail);
  }

 private:
  std::vector<HitChain> chains_;
  std::vector<double> prior_mass_prefix_, log_dist_table_;
  int64_t geometry_topology_rev_ = -1, fit_epoch_ = -1;
  std::vector<std::unordered_map<int64_t, std::vector<std::pair<int32_t, int32_t>>>> descendants_;
  std::unordered_map<int32_t, std::array<double, 32>> rates_;
  std::unordered_map<PendingEventKey, std::pair<double, int64_t>, PendingEventHash> events_;
  std::unordered_map<int64_t, DensityModel> density_cache_;

  void ensure_log_dist_(int32_t bins) {
    while (static_cast<int32_t>(log_dist_table_.size()) <= bins) {
      log_dist_table_.push_back(std::log(static_cast<double>(log_dist_table_.size()) * g));
    }
  }
  const DensityModel& default_density_(int32_t tail) {
    int64_t key = -static_cast<int64_t>(tail) - 1;
    if (auto it = density_cache_.find(key); it != density_cache_.end()) return it->second;
    double log_amp = std::log((kPriorInternalAlpha / kPriorExposure) / prior_mass(std::max(1, tail / g), 2.0));
    return density_cache_.emplace(
        key, build_density_(key, 0, tail, 2.0, log_amp, kPriorTailAlpha / kPriorExposure)).first->second;
  }
  const DensityModel& cached_chain_density_(int32_t cid, const HitChain& c, int32_t tail) {
    int64_t key = (static_cast<int64_t>(cid) << 32) | static_cast<uint32_t>(tail);
    if (auto it = density_cache_.find(key); it != density_cache_.end()) return it->second;
    return density_cache_.emplace(
        key, build_density_(cid, c.origin, tail, c.exponent, c.log_amplitude, c.tail_probability)).first->second;
  }

  double weight_(int64_t age) const { return std::exp2(-static_cast<double>(age) / half_life); }
  double exposure_factor_(int64_t begin, int64_t end) const {
    const double ln2 = std::log(2.0);
    return -std::expm1(-ln2 * static_cast<double>(end - begin) / half_life) / (-std::expm1(-ln2 / half_life));
  }

  void add_widths_(std::array<double, 32>& widths, int32_t low, int32_t high, double share) const {
    low = std::max(g, low);
    high = (std::min(window, high) / g) * g;
    while (low <= high) {
      int32_t b = static_cast<int32_t>(std::bit_width(static_cast<uint32_t>(low / g))) - 1;
      int32_t end = std::min(high + g, g << (b + 1));
      if (b >= 0 && b < 32) widths[b] += static_cast<double>(end - low) * share;
      low = end;
    }
  }

  void settle_(HitChain& c, const double* widths, size_t nwidths, double tail_rate) const {
    int64_t elapsed = query_count - c.exposure_query;
    if (elapsed <= 0) return;
    double wt = weight_(elapsed), obs = exposure_factor_(c.exposure_query, query_count);
    for (uint32_t m = c.exposure_mask; m; m &= m - 1) c.exposure[std::countr_zero(m)] *= wt;
    if (widths) {
      for (size_t b = 0; b < nwidths; ++b) {
        if (widths[b] > 0.0) { c.exposure[b] += widths[b] * obs; c.exposure_mask |= (uint32_t{1} << b); }
      }
    }
    c.tail_exposure = c.tail_exposure * wt + tail_rate * obs;
    c.exposure_query = query_count;
  }

  void prepare_geometry_(const RadixTreeCore& tree) {
    if (tree.topology_revision == geometry_topology_rev_) return;
    for (auto& [cid, widths] : rates_) settle_(chains_[cid], widths.data(), 32, 1.0);
    std::vector<int32_t> order, stack = {0};
    order.reserve(tree.num_nodes_capacity());
    while (!stack.empty()) {
      int32_t id = stack.back();
      stack.pop_back();
      order.push_back(id);
      for (int32_t c : tree.node(id).children) stack.push_back(c);
    }
    descendants_.assign(tree.num_nodes_capacity(), {});
    rates_.clear();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      int32_t nid = *it;
      if (nid == 0) continue;
      const auto& n = tree.node(nid);
      auto& groups = descendants_[nid];
      if (n.is_leaf()) {
        int32_t cid = n.analysis_owner;
        if (cid <= 0) continue;
        publish(cid, n.depth);
        settle_(chains_[cid], nullptr, 0, 0.0);
        rates_.try_emplace(cid, std::array<double, 32>{});
        groups[chains_[cid].group_id].emplace_back(cid, n.depth);
      } else {
        for (int32_t cid : n.children) {
          for (const auto& [gid, targets] : descendants_[cid]) {
            auto& dst = groups[gid];
            dst.insert(dst.end(), targets.begin(), targets.end());
          }
        }
      }
      for (const auto& [_, targets] : groups) {
        double share = 1.0 / static_cast<double>(targets.size());
        for (const auto& [cid, tail] : targets) add_widths_(rates_[cid], tail - n.depth, tail - n.begin() - g, share);
      }
    }
    geometry_topology_rev_ = tree.topology_revision;
  }

  void record_(std::vector<std::pair<int32_t, int32_t>> targets, int32_t position) {
    ++query_count;
    if (targets.empty()) return;
    PendingEventKey key{std::move(targets), position};
    if (auto it = events_.find(key); it == events_.end()) {
      events_.emplace(std::move(key), std::make_pair(1.0, query_count));
    } else {
      it->second.first = it->second.first * weight_(query_count - it->second.second) + 1.0;
      it->second.second = query_count;
    }
    if (events_.size() >= 1024) flush_();
  }

  void flush_() {
    for (const auto& [key, val] : events_) {
      double share = val.first * weight_(query_count - val.second) / static_cast<double>(key.targets.size());
      for (const auto& [cid, tail] : key.targets) {
        auto& c = chains_[cid];
        if (int64_t age = query_count - c.hit_query; age > 0) {
          double wt = weight_(age);
          for (uint32_t m = c.hit_mask; m; m &= m - 1) c.hit_counts[std::countr_zero(m)] *= wt;
          c.tail_hits *= wt;
          c.hit_query = query_count;
        }
        int32_t dist = tail - key.position;
        if (dist == 0) c.tail_hits += share;
        else if (dist >= g && dist <= window) {
          int32_t b = static_cast<int32_t>(std::bit_width(static_cast<uint32_t>(dist / g))) - 1;
          if (b >= 0 && b < 32) { c.hit_counts[b] += share; c.hit_mask |= (uint32_t{1} << b); }
        }
      }
    }
    events_.clear();
  }

  double fit_(const std::vector<int32_t>& cids) const {
    struct FitItem { std::vector<std::pair<double, double>> exposures; double count = 0.0, observed = 0.0; };
    std::vector<FitItem> data;
    for (int32_t cid : cids) {
      const auto& c = chains_[cid];
      double count = 0.0, observed = 0.0;
      for (uint32_t hm = c.hit_mask; hm; hm &= hm - 1) {
        int b = std::countr_zero(hm);
        count += c.hit_counts[b];
        observed += c.hit_counts[b] * std::log(static_cast<double>(g << b));
      }
      if (count <= 0.0) continue;
      FitItem item{{}, count, observed};
      for (uint32_t em = c.exposure_mask; em; em &= em - 1) {
        int b = std::countr_zero(em);
        if (c.exposure[b] > 0.0) item.exposures.emplace_back(std::log(static_cast<double>(g << b)), c.exposure[b]);
      }
      if (item.exposures.size() > 1) data.push_back(std::move(item));
    }
    if (data.empty()) return std::min(2.0, max_exponent);
    double low = 0.0, high = max_exponent;
    for (int iter = 0; iter < 24; ++iter) {
      double k = 0.5 * (low + high), deriv = 2.0 - k;
      for (const auto& item : data) {
        double wsum = 0.0, xwsum = 0.0;
        for (const auto& [x, width] : item.exposures) {
          double w = width * std::exp(-k * x);
          wsum += w;
          xwsum += x * w;
        }
        deriv += item.count * (xwsum / wsum) - item.observed;
      }
      if (deriv > 0.0) low = k; else high = k;
    }
    return 0.5 * (low + high);
  }

  void estimate_(HitChain& c, double exponent) {
    double pm = prior_mass(std::max(1, (c.tail - c.origin) / g), exponent);
    double exp_total = 0.0, internal_hits = 0.0;
    for (uint32_t em = c.exposure_mask; em; em &= em - 1) {
      int b = std::countr_zero(em);
      exp_total += c.exposure[b] * bucket_avg_density(b, exponent);
    }
    for (uint32_t hm = c.hit_mask; hm; hm &= hm - 1) internal_hits += c.hit_counts[std::countr_zero(hm)];
    c.exponent = exponent;
    c.log_amplitude = std::log((internal_hits + kPriorInternalAlpha) / (exp_total + kPriorExposure * pm));
    c.tail_probability = (c.tail_hits + kPriorTailAlpha) / (c.tail_exposure + kPriorExposure);
  }

  DensityModel build_density_(int64_t id, int32_t origin, int32_t tail, double exponent,
                              double log_amplitude, double tail_prob) {
    DensityModel d;
    d.id = id;
    d.origin = static_cast<double>(origin);
    d.tail = static_cast<double>(tail);
    d.atom = std::clamp(tail_prob, 0.0, 1.0);
    d.g = static_cast<double>(g);
    d.exponent = exponent;
    d.log_amplitude = log_amplitude;
    const int32_t count = std::max(1, (tail - origin) / g);
    if (exponent == 2.0) {
      double pm = prior_mass(count, 2.0), amp = std::exp(log_amplitude);
      double scale = std::min(1.0, std::max(0.0, 1.0 - d.atom) / std::max(1e-300, amp * pm));
      d.shared_prefix = &prior_mass_prefix_;
      d.shared_scale = amp * scale;
      d.shared_count = count;
      d.probability.clear();
      return d;
    }
    ensure_log_dist_(count);
    d.probability.reserve(count + 1);
    d.probability.push_back(0.0);
    const double amp_g = std::exp(log_amplitude) * d.g;
    for (int32_t j = 1; j <= count; ++j) {
      d.probability.push_back(d.probability.back() + amp_g * std::exp(-exponent * log_dist_table_[j]));
    }
    double scale = std::min(1.0, std::max(0.0, 1.0 - d.atom) / std::max(1e-300, d.probability.back()));
    for (double& v : d.probability) v *= scale;
    return d;
  }
};

}  // namespace logssms::logstate
