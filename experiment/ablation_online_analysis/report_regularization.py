import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
OUT = args.results.resolve()
rows = []
lines = [
    '# Exponent regularization ablation', '',
    'Paired production CPU Scheduler/LogState replay using the same recorded prompts '
    'and actual generated token IDs. Both variants use online eviction. The off variant '
    'removes only -(k-2)^2/2 from the fitted objective, implemented by removing -(k-2) '
    'from its derivative. Amplitude and tail priors, exponent bounds, zero-internal '
    'fallback, weighted exposures and cache capacity are unchanged. The production '
    'implementation retains the penalty. GPU numerical correctness and service latency '
    'are not measured.', '',
    '| Trees | Exponent penalty | Prefix hit | Historical hit | Historical recomputation (tokens) | KV gap | State gap | Final Checkpoints | Final Logs |',
    '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
]
for trees in (32, 64, 128):
    pair = []
    for mode in ('regularized', 'unregularized'):
        d = json.loads((OUT / f'{mode}-trees{trees}.json').read_text())
        pair.append(d)
        final = d['batches'][-1]
        snapshots = d['analysis']['updates']
        sparse = [s['exponent'] for s in snapshots if 0 < s['weighted_internal_hits'] <= 4]
        row = dict(trees=trees, mode=mode, prefix_hit=d['hit_rate'],
                   historical_hit=d['historical_hit_rate'],
                   historical_recompute=d['totals']['historical_recompute_tokens'],
                   kv_gap=d['totals']['historical_kv_gap'], state_gap=d['totals']['historical_state_gap'],
                   checkpoints=final['checkpoint'], logs=final['log'], frames=final['frames_after'],
                   sparse_snapshot_count=len(sparse),
                   sparse_mean_exponent=sum(sparse)/len(sparse) if sparse else None,
                   sparse_mean_deviation=sum(abs(k-2) for k in sparse)/len(sparse) if sparse else None,
                   sparse_min_exponent=min(sparse) if sparse else None,
                   sparse_max_exponent=max(sparse) if sparse else None,
                   dropped_model_updates=d['analysis']['dropped_updates'])
        rows.append(row)
        lines.append(f"| {trees} | {'on' if mode == 'regularized' else 'off'} | {row['prefix_hit']:.4%} | {row['historical_hit']:.4%} | {row['historical_recompute']} | {row['kv_gap']} | {row['state_gap']} | {row['checkpoints']} | {row['logs']} |")
    a,b = pair
    for field in ('execution', 'hardware', 'trace_sha256', 'requests',
                  'arrival_batch_size', 'prefill_budget', 'mixed_prefill_tokens'):
        assert a[field] == b[field], (trees, field)
    for field in ('input_tokens', 'history_tokens', 'verified_parent_prefixes'):
        assert a['totals'][field] == b['totals'][field]
    for field in ('min_sampling_len', 'hit_half_life', 'max_density_exponent'):
        assert a['analysis'][field] == b['analysis'][field]
    before = {r['uid']:r for r in a['requests_detail']}
    assert before.keys() == {r['uid'] for r in b['requests_detail']}
    differences = [r['historical_recompute'] - before[r['uid']]['historical_recompute']
                   for r in b['requests_detail']]
    rows[-1]['change_vs_penalty_on'] = (
        b['totals']['historical_recompute_tokens']/a['totals']['historical_recompute_tokens'] - 1)
    rows[-1]['request_changes'] = dict(
        better=sum(v < 0 for v in differences), worse=sum(v > 0 for v in differences),
        unchanged=sum(v == 0 for v in differences),
        saved_tokens=-sum(v for v in differences if v < 0),
        extra_tokens=sum(v for v in differences if v > 0))
lines += ['', '## Historical recomputation changes with penalty removed', '']
for row in rows:
    if row['mode'] == 'unregularized':
        lines.append(f"- {row['trees']} trees: {row['change_vs_penalty_on']:+.2%}; `{json.dumps(row['request_changes'], sort_keys=True)}`")
lines += ['', '## Sparse model snapshots', '',
          'Statistics below describe saved model-update snapshots with weighted internal '
          'hit count in (0,4]. These are not independent exponent fits. Production '
          'profiling retains at most the latest 8192 model updates; dropped counts '
          'are preserved in comparison.json.', '',
          '| Trees | Exponent penalty | Snapshots | Mean k | Mean abs(k-2) | Minimum k | Maximum k |',
          '|---:|---|---:|---:|---:|---:|---:|']
for row in rows:
    if row['sparse_snapshot_count']:
        lines.append(f"| {row['trees']} | {'on' if row['mode'] == 'regularized' else 'off'} | {row['sparse_snapshot_count']} | {row['sparse_mean_exponent']:.4f} | {row['sparse_mean_deviation']:.4f} | {row['sparse_min_exponent']:.4f} | {row['sparse_max_exponent']:.4f} |")
(OUT / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
(OUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
