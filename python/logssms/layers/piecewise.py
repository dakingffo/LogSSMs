from __future__ import annotations

from functools import wraps

from logssms.core import get_global_ctx


def attention_output_buffer(example, shape, name="attention_output"):
    runner = get_global_ctx().piecewise_graph_runner
    if runner is not None and runner.can_use_cuda_graph(get_global_ctx().batch):
        key = (name, tuple(shape[1:]), example.dtype, example.device)
        if key in runner.buffers:
            return runner.buffers[key][:shape[0]]
    return example.new_empty(shape)


def graph_piece(func):
    @wraps(func)
    def wrapper(self, *args):
        ctx = get_global_ctx()
        runner = ctx.piecewise_graph_runner
        if runner is not None and runner.can_use_cuda_graph(ctx.batch):
            return runner.run(self, func, args)
        return func(self, *args)

    wrapper.piecewise_function = func
    return wrapper
