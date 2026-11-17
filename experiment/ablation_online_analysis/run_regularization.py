import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiment/ablation_online_analysis/results/aligned-prefill-distribution'
parser = argparse.ArgumentParser(description='Ablate the exponent quadratic penalty in CPU replay.')
parser.add_argument('--out', type=Path, default=ROOT / 'experiment/ablation_online_analysis/results/exponent-regularization')
parser.add_argument('--workers', type=int, default=6)
args = parser.parse_args()
if args.workers < 1:
    parser.error('workers must be positive')
OUT = args.out.resolve()
sys.path[:0] = [str(ROOT / 'python'), str(ROOT)]
from logssms.cache.logstate.online_analysis_scheduler import OnlineAnalysisScheduler

OUT.mkdir(parents=True, exist_ok=True)
source = textwrap.dedent(inspect.getsource(OnlineAnalysisScheduler._fit_hit_bins))
assert source.count(' - (mid - 2.0)') == 1
(OUT / 'regularized_fit.py').write_text(source)
(OUT / 'unregularized_fit.py').write_text(source.replace(' - (mid - 2.0)', ''))
patch = subprocess.check_output(['git', 'diff', '--binary', '--', 'python', 'csrc'], cwd=ROOT)
(OUT / 'production.patch').write_bytes(patch)
manifest = json.loads((BASE / 'manifest.json').read_text())
manifest.update(
    ablation='exponent quadratic penalty only', modes=['regularized', 'unregularized'], workers=args.workers,
    production_patch_sha256=hashlib.sha256(patch).hexdigest(),
    experiment_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                       for name in ('run_regularization.py', 'replay_regularization.py', 'report_regularization.py')},
    preserved=['amplitude prior', 'tail probability prior', 'zero-internal fallback',
               'exponent bounds', 'forgetting', 'exposure accounting', 'eviction cost'],
)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(mode, n):
    case = BASE / f'case-trees{n}.json'
    config = json.loads(case.read_text())
    destination = OUT / f'{mode}-trees{n}.json'
    env = os.environ | {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': f'{ROOT}/python:{ROOT}'}
    with destination.with_suffix('.log').open('w') as log:
        subprocess.run([
            sys.executable, '-m', 'experiment.ablation_online_analysis.replay_regularization', mode,
            '--case', str(case), '--history', config['history'], '--out', str(destination),
            '--mode', 'online', '--batch-size', '8', '--min-sampling-len', '8192',
            '--hit-half-life', '1024',
        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    d = json.loads(destination.read_text())
    print(json.dumps(dict(variant=mode, trees=n, hit=d['hit_rate'],
                          historical=d['historical_hit_rate'],
                          recompute=d['totals']['historical_recompute_tokens'])), flush=True)


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = [pool.submit(run, mode, n) for mode in ('regularized', 'unregularized')
               for n in (32, 64, 128)]
    for future in as_completed(futures):
        future.result()
subprocess.run([sys.executable, str(ROOT / 'experiment/ablation_online_analysis/report_regularization.py'),
                '--results', str(OUT)], cwd=ROOT, check=True)
print(OUT / 'comparison.md', flush=True)
