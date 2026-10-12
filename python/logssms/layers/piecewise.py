from __future__ import annotations

from functools import wraps

from logssms.core import get_global_ctx


def prefill_piece(func):
    @wraps(func)
    def wrapper(self, *args):
        ctx = get_global_ctx()
        runner = ctx.prefill_graph_runner
        if runner is not None and runner.can_use_cuda_graph(ctx.batch):
            return runner.run(self, func, args)
        return func(self, *args)

    wrapper.prefill_function = func
    return wrapper
