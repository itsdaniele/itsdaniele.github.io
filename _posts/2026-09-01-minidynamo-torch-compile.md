---
layout: post
title: "MiniDynamo: Building a Small TorchDynamo"
description: Follow one Python function through bytecode tracing, graph capture, guards, and an optional Inductor backend.
date: 2026-09-01 01:00:00+0200
_styles: >
  #markdown-content {
    line-height: 1.75;
  }

  #markdown-content .l-body {
    margin: 1.75rem 0 2.75rem;
  }

  #markdown-content h2 {
    margin-top: 3.75rem;
    margin-bottom: 1.25rem;
    padding-top: 0.25rem;
  }

  #markdown-content h3 {
    margin-top: 2.5rem;
    margin-bottom: 1rem;
  }

  #markdown-content p,
  #markdown-content ul,
  #markdown-content ol {
    margin-bottom: 1.2rem;
  }

  #markdown-content hr {
    margin: 3.5rem 0;
  }

  #markdown-content img {
    display: block;
    max-width: min(100%, 860px);
    margin: 2.75rem auto 3rem;
  }

  #markdown-content table,
  #markdown-content .table-responsive {
    margin-top: 2rem;
    margin-bottom: 2.5rem;
  }

  #markdown-content pre {
    margin-top: 1.75rem;
    margin-bottom: 2rem;
  }

  #markdown-content aside {
    margin: 2rem 0 2.5rem;
    padding: 1.25rem 1.5rem;
    border-left: 3px solid var(--global-theme-color);
    background: rgba(181, 9, 172, 0.06);
  }
---

The part of `torch.compile` I wanted to understand was the step before any optimized kernel is generated: how does a Python function become a graph?

Consider this function:

```python
import torch


def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()
```

We can see three tensor operations: add, multiply, sum. PyTorch needs a way to discover those operations, pass them to a compiler, and decide whether the resulting code is still valid on the next call. That is the part we will build.

[MiniDynamo](https://github.com/itsdaniele/torchdynamo-mini) is a small bytecode interpreter inspired by TorchDynamo, the graph-capture frontend of `torch.compile`. It records tensor operations, turns the graph back into a callable, and caches that callable behind checks called **guards**. We will follow the function above through each step, then connect its graph to the real Inductor backend.

You need some familiarity with Python and PyTorch; no compiler background is assumed. The code targets **Python 3.10 and PyTorch 2.10.0**. The Python version matters because bytecode changes between releases.

## What are we trying to capture?

In eager PyTorch, Python dispatches tensor operations as it reaches them. A compiler benefits from seeing several operations together. For example, it may be able to combine an addition and a multiplication into one kernel, avoiding an intermediate tensor and some launch overhead.

In the usual `torch.compile` pipeline, **Dynamo captures the computation; Inductor optimizes it**. Dynamo also produces guards and replacement Python bytecode to arrange execution around the compiled graph. A function can contain multiple compiled regions with ordinary Python between them. [PyTorch's overview of Dynamo](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html) describes that broader model.

MiniDynamo supports a much smaller program: fixed positional tensor inputs, a sequence of supported operations, and one tensor output. There are no loops, branches, nested helper calls, keyword calls, or graph breaks. Unsupported operations raise an error. In-place operations and random operations are also excluded, for a reason we will see during tracing.

This is enough to make the main steps visible:

```text
first call:  read bytecode → build graph → generate callable → cache → run
later call: check guards → reuse a matching callable, or trace again
```

The default backend generates Python. That lets us inspect exactly what was captured. It does not fuse kernels.

## Reading the function one instruction at a time

CPython compiles a function into **bytecode**, a sequence of instructions. We can inspect it with `dis.get_instructions(fn)`.

Python 3.10 represents `z = x + y` using four instructions:

| Instruction    | Stack afterwards | What happened                      |
| :------------- | :--------------- | :--------------------------------- |
| `LOAD_FAST x`  | `[x]`            | Put `x` on the stack               |
| `LOAD_FAST y`  | `[x, y]`         | Put `y` on the stack               |
| `BINARY_ADD`   | `[x + y]`        | Pop both values and push their sum |
| `STORE_FAST z` | `[]`             | Pop the result and store it as `z` |

A stack is just a list where we add and remove values at the end. Our interpreter keeps one, plus a dictionary of local variables and the graph under construction.

Here is its dispatch loop, with the error message shortened:

```python
def run(self):
    instructions = list(dis.get_instructions(self.fn))
    for inst in instructions:
        handler = getattr(self, f"op_{inst.opname}", None)
        if handler is None:
            raise NotImplementedError(f"Unsupported bytecode: {inst.opname}")
        handler(inst)
    return self.graph
```

The handlers for loading and storing a local are small:

```python
def op_LOAD_FAST(self, inst):
    self.push(self.locals[inst.argval])


def op_STORE_FAST(self, inst):
    self.locals[inst.argval] = self.pop()
```

The interesting change is what goes on the stack. Instead of holding only ordinary Python values, it holds wrappers that tell us how a value should be treated during tracing.

## Tracking values and recording operations

MiniDynamo uses four kinds of wrapper:

| Wrapper            | Holds                                             |
| :----------------- | :------------------------------------------------ |
| `TensorVariable`   | A graph node and an example tensor                |
| `ConstantVariable` | A known value, such as the `2` in `x * 2`         |
| `TorchVariable`    | The `torch` module or a supported function        |
| `MethodVariable`   | A tensor wrapper and a method name, such as `sum` |

These are the toy version of Dynamo's `VariableTracker` classes. They let a bytecode handler distinguish tensor arithmetic from arithmetic on constants.

A `TensorVariable` has only two fields:

```python
class TensorVariable(VariableTracker):
    def __init__(self, node, example_value):
        self.node = node
        self.example_value = example_value
```

Before reading the bytecode, we create a **placeholder node** for each input tensor. A placeholder means “the value that will be passed to this argument.” The local names `x` and `y` initially refer to wrappers around those nodes.

When `BINARY_ADD` pops two tensor wrappers, it records a new node:

```python
node = graph.call_function(torch.add, (x_var.node, y_var.node))
example = torch.add(x_var.example_value, y_var.example_value)
result = TensorVariable(node, example)
```

This is the essential tracing step. The node records the operation and its inputs. The example tensor tells us the shape and dtype of the result. We push the new wrapper, so later instructions can refer to it.

**MiniDynamo really executes the example tensor operations during tracing.** It does not merely inspect their shapes. After tracing, the generated callable executes those operations again to produce the user's result. This keeps the implementation simple, but adds work to the first call.

It also explains why we restrict the supported operations. Tracing `x.add_(1)` on the user's tensor and then running it again would modify `x` twice. Calling an arbitrary helper could print, update a counter, or consume random numbers during tracing. MiniDynamo rejects those operations before calling them. Its allowlists live in `symbolic_interpreter.py`.

Real Dynamo generally propagates metadata using [FakeTensors](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_subclasses/fake_tensor.py), which model tensor properties without running ordinary tensor kernels on the input data. Handling mutation and Python side effects correctly takes additional machinery that this toy leaves out.

Back in our example, `STORE_FAST z` saves the new wrapper in `locals["z"]`. The multiplication then records another node with the literal `2` as an argument. Looking up `w.sum` creates a `MethodVariable`; calling that method records the reduction.

The distinction is useful: **wrappers help us interpret the program; graph nodes describe the computation we have captured.** The stack and local-variable dictionary are temporary. The graph is what the backend receives.

## The graph we get

We can inspect the graph without using the decorator:

```python
from mini_dynamo.symbolic_interpreter import SymbolicInterpreter

x = torch.randn(3, 4)
y = torch.randn(3, 4)
graph = SymbolicInterpreter(fn, (x, y)).run()
print(graph)
```

The output is:

```text
Graph:
  x = placeholder
  y = placeholder
  add_0 = torch.add(x, y)
  mul_0 = torch.mul(add_0, 2)
  sum_0 = mul_0.sum()
  return sum_0
```

There are four node types: `placeholder`, `call_function`, `call_method`, and `output`. Each node has a name, a target, and arguments. Arguments can refer to earlier nodes or contain constants.

![The example function and its captured graph. Inputs feed an addition, then a multiplication by 2, then a sum.](/assets/img/mini-dynamo/graph-ir.svg)

The assignments to `z` and `w` no longer need their own graph nodes. They only gave names to intermediate values. The graph preserves the dependencies: multiplication needs the addition's result, and the reduction needs the multiplication's result.

This is an **intermediate representation**, or IR: a format between the original Python function and the code that will execute it. Our graph is a small custom class, not an actual `torch.fx.Graph`. We will convert it to FX when connecting it to Inductor.

## Turning the graph into a callable

The Python backend walks the nodes and emits source code:

```python
from mini_dynamo.compiler import compile_graph

replay, source = compile_graph(graph)
print(source)
torch.testing.assert_close(replay(x, y), fn(x, y))
```

For this graph, the generated source is:

```python
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)
    mul_0 = __fn_mul_0(add_0, 2)
    sum_0 = mul_0.sum()
    return sum_0
```

`__fn_add_0` and `__fn_mul_0` are names in the globals dictionary supplied to `exec()`. They refer to `torch.add` and `torch.mul`. The backend also puts constants that are not ordinary Python literals, such as a `torch.dtype`, into that dictionary.

We now have an executable version of the graph. It still calls PyTorch operations one by one, using the normal dispatcher. This is useful for checking capture and inspecting generated code, but there is no reason to expect a substantial speedup over the original function. The caching wrapper will add guard-checking overhead too.

The repository also includes a `backend="jit"` option, which applies `torch.jit.trace` to this generated function. It is an optional comparison with TorchScript; it is not part of Dynamo's normal pipeline. TorchScript can reduce repeated Python calls and has its own optimization machinery, so its performance and kernel structure depend on the workload and configuration. We should measure them rather than assume “one op, one kernel.” PyTorch 2.10 [marks this tracing API deprecated](https://docs.pytorch.org/docs/2.10/generated/torch.jit.trace.html).

## When can we reuse it?

Tracing and compiling on every call would defeat the purpose. We need to save the callable and know when it is valid.

A **guard** is a check on an assumption made during tracing. MiniDynamo specializes on exact input metadata. For our example, some of the checks are:

```text
args[0].shape == (3, 4)
args[0].dtype == torch.float32
args[0].device == cpu
```

The implementation also checks strides and other layout details, gradient requirements, the default dtype, and grad/inference/autocast modes. It records identity checks for global bindings and `torch` attributes read during tracing, plus the function's code object. These checks are deliberately conservative: they may trigger tracing even when the old Python callable would still have worked.

Why check globals? Suppose a function multiplies by a module-level `scale`. Tracing reads that value and embeds it in the graph. If `scale` changes from `2` to `3`, the input tensor shapes have not changed, but the cached computation is stale. A guard needs to catch that. Mutable global objects and global tensors are outside MiniDynamo's supported subset.

Each decorated function owns a list of `(guards, callable)` pairs. On a call, the wrapper scans that list and runs the first entry whose guards pass. If none match, it traces again and appends an entry.

Here is the cache loop for the Python backend, omitting argument validation and the extra code-object guard:

```python
for guards, cached_fn in cache:
    if guards.check_all(*args):
        return cached_fn(*args)

interpreter = SymbolicInterpreter(fn, args)
graph = interpreter.run()
compiled_fn, _ = compile_graph(graph)
cache.append((interpreter.guards, compiled_fn))
return compiled_fn(*args)
```

We can watch this happen:

```python
import mini_dynamo

compiled = mini_dynamo.compile(fn)
compiled(torch.randn(3, 4), torch.randn(3, 4))
assert len(compiled._cache) == 1

# Different values, same metadata: reuse the first callable.
compiled(torch.randn(3, 4), torch.randn(3, 4))
assert len(compiled._cache) == 1

# A new shape: trace and save another callable.
compiled(torch.randn(5, 6), torch.randn(5, 6))
assert len(compiled._cache) == 2
```

The cache does not store previous answers. Every call computes a result from the current tensors. It stores ways to perform the computation.

![The wrapper scans cached entries and runs the first whose guards pass. Shape, dtype, and device are three examples of those guards.](/assets/img/mini-dynamo/cache-and-guards.svg)

Our list has no eviction policy or compilation limit. A workload with a new shape on every call will keep adding entries. Real Dynamo has richer guards, symbolic shape support, and recompilation limits. A shape change may reuse a dynamic graph, select another cached version, or trigger compilation; it does not necessarily mean starting over. [PyTorch's guard documentation](https://docs.pytorch.org/devlogs/dynamo/2025-06-04-inside-torch-compile-guards/) explains that trade-off.

## Giving the graph to Inductor

Graph capture becomes more useful when a backend can change how the operations execute. The optional `mini_dynamo/inductor.py` bridge does this:

```text
MiniDynamo graph → FX GraphModule → ATen graph → Inductor callable
```

The first conversion translates our nodes into FX nodes. `make_fx` then traces that module at the PyTorch dispatcher level to expose ATen operations, such as `aten.add.Tensor`. Finally, `compile_fx_inner` compiles the ATen graph.

This is a deliberately narrow route through **private PyTorch 2.10 APIs**. It bypasses the usual AOTAutograd pipeline, including its backward-graph construction and functionalization work. It is not the complete `torch.compile` pipeline with Dynamo swapped out.

To try it with our function:

```python
from mini_dynamo.inductor import mini_dynamo_to_inductor

fused = mini_dynamo_to_inductor(fn, x, y)
torch.testing.assert_close(fused(x, y), fn(x, y))
```

This helper supports forward inference with inputs that do not require gradients. Its wrapper checks the tracing assumptions and raises if they change; it does not manage a recompilation cache. Create a new callable for a new input specification or execution context. The main `@mini_dynamo.compile` decorator still exposes only the Python and JIT backends.

The tests compare MiniDynamo's path with `torch._dynamo.export` followed by the same ATen lowering and private Inductor entry point. For four small CPU programs, they check graph targets, dependencies, constants, generated C++ kernel bodies, and numerical results. That is evidence about those programs and that compilation path. It is not a claim that our toy reproduces `torch.compile` for arbitrary models or devices.

What might make the optimized version faster? Fusion can avoid materializing intermediate tensors and reduce kernel launches. Inductor can also plan buffers and choose implementations for operations. Eligible CUDA workloads may use CUDA graphs to reduce host launch overhead. The gain depends on the workload, hardware, and compilation settings; even a correctly captured graph can be slower than eager execution.

The repository's benchmark scripts let you inspect that trade-off. Measure after warmup, keep compilation time separate from execution time, and synchronize device work when timing a GPU. Also distinguish a raw backend callable from a callable that includes guard checks. Those are different things to time.

## A CUDA run on an H200

I ran the tests and benchmark on one NVIDIA H200, using Python 3.10.20, PyTorch 2.10.0+cu128, and CUDA 12.8. The workload is eleven elementwise operations followed by a sum, on two square float32 tensors. This is a small fusion experiment, not a model benchmark.

The table reports **microseconds per call**: the median of seven trials of 500 calls, after 50 warmup calls per variant. All variants use inference mode. Compilation is excluded; CUDA synchronization brackets each trial, so these are amortized wall times including host dispatch. CUDA graphs are disabled in both Inductor paths.

| Callable                           | 32 × 32 | 2048 × 2048 |
| :--------------------------------- | ------: | ----------: |
| Eager PyTorch                      |    57.1 |       129.6 |
| Generated Python, no guards        |    56.7 |       129.7 |
| MiniDynamo Python, with guards     |    73.6 |       129.8 |
| TorchScript, no guards             |    12.7 |        29.4 |
| MiniDynamo JIT, with guards        |    28.4 |        36.5 |
| MiniDynamo + Inductor, with guards |    28.0 |        41.4 |
| `torch.compile`, static shapes     |    37.4 |        53.3 |

A separate profiler pass makes these numbers more informative. At 2048 × 2048, eager and generated Python each launched **12 compute kernels**. TorchScript launched **2**: a fused elementwise kernel and a reduction. Each Inductor path also launched **2** kernels. These counts exclude memory-set operations and profiler annotation ranges.

So the Python backend preserved the eager execution pattern, while **TorchScript fused this workload too**. It would be wrong to explain the JIT result purely as saved Python overhead, or to say only Inductor can fuse operations. Fewer kernels also did not establish a universal ranking: JIT was faster here, and the wrappers have different costs and capabilities. On the larger input, the Python guard cost was hidden in the amortized timing even though the checks still ran.

The full suite on this CUDA node passed **227 tests**, with **4 MPS-only tests skipped**. CUDA checks cover float32, float16, bfloat16, gradients in the Python/JIT backends, and non-contiguous inputs. Numerical comparisons use tolerances because fusion and reductions can change floating-point rounding.

The [benchmark script](https://github.com/itsdaniele/torchdynamo-mini/blob/main/examples/benchmark_publication.py) and [raw trials, configuration, and profiler events](https://github.com/itsdaniele/torchdynamo-mini/blob/main/benchmarks/h200-publication.json) are in the repository. They also include the intermediate 256 × 256 case and the spread across trials. Treat these as measurements of this workload and configuration, not a prediction for a training run.

## What real Dynamo adds

The toy makes the mechanics visible by choosing a small subset of Python. Three additions explain much of the distance to real Dynamo.

**Entering and resuming Python execution.** Dynamo uses CPython's [PEP 523 frame-evaluation hook](https://peps.python.org/pep-0523/) to intercept eligible frames while compilation is active. It can emit replacement bytecode that calls compiled graphs and handles surrounding Python work. MiniDynamo has a decorator that explicitly invokes our interpreter.

The hook and the bytecode interpreter do different jobs. Dynamo also uses `dis.get_instructions` in its [bytecode transformation code](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_dynamo/bytecode_transformation.py). When it inlines a supported Python helper during tracing, it uses an [inlining interpreter](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_dynamo/symbolic_convert.py) to walk the helper's bytecode; it does not need the helper to execute normally in a new frame first.

**Control flow and graph breaks.** Dynamo can specialize on Python values and tensor metadata, guard the assumptions, and trace the chosen path. A branch on tensor data is a different problem because its outcome may change with the contents of each input. Unsupported regions can cause a graph break: execute a compiled prefix, run some ordinary Python, and resume tracing. With `fullgraph=True`, graph breaks cause an error instead. MiniDynamo rejects unsupported bytecode and has no resume mechanism.

**Training and general Python behavior.** Dynamo tracks far more kinds of values, mutations, aliases, and state. The usual Inductor path uses AOTAutograd to prepare graphs for training. MiniDynamo's Python backend still calls ordinary PyTorch operations, so ordinary autograd can differentiate supported computations; it does not compile a backward pass. The direct Inductor example is inference-only.

These restrictions are part of the exercise. They keep the trace small enough to inspect while showing why a production compiler needs more than a list of tensor operations.

## Read and run the code

From a checkout of [the repository](https://github.com/itsdaniele/torchdynamo-mini):

```bash
uv sync --frozen --group dev
uv run pytest
uv run python examples/inductor_integration.py
```

Start with `mini_dynamo/__init__.py` for the cache loop, then `symbolic_interpreter.py` for the bytecode handlers. `variable_tracker.py` and `graph.py` define the objects those handlers use. `compiler.py` turns the graph into Python; `guards.py` decides when it can be reused. The optional Inductor bridge lives in `inductor.py`.

A useful next experiment is to change our example, print its graph, and predict which calls will reuse the cache. The connection to `torch.compile` becomes easier to follow once you can point to the recorded operation, the assumption that protects it, and the backend that will execute it.

_Coding agents generated much of the initial implementation. I then read the code, added tests, and used the toy system to build intuition._
