import inspect
import sys
import textwrap

from experiment.ablation_online_analysis import replay_common as replay
from logssms.cache.logstate.online_analysis_scheduler import OnlineAnalysisScheduler

mode = sys.argv.pop(1)
assert mode in ('regularized', 'unregularized')
if mode == 'unregularized':
    original = OnlineAnalysisScheduler._fit_hit_bins
    source = textwrap.dedent(inspect.getsource(original))
    penalty = ' - (mid - 2.0)'
    assert source.count(penalty) == 1
    namespace = {}
    exec(compile(source.replace(penalty, ''), '<exponent-regularization-ablation>', 'exec'),
         original.__globals__, namespace)
    OnlineAnalysisScheduler._fit_hit_bins = namespace['_fit_hit_bins']
replay.main()
