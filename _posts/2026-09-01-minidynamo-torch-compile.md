---
layout: post
title: "MiniDynamo: Building a Small TorchDynamo"
description: Build a bytecode tracer from first principles, derive its guards and graph representation, and connect it to Inductor with measured CUDA results.
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

I assume you use PyTorch and are comfortable reading Python. We will derive the compiler machinery as we need it: how to interpret an instruction, what information survives in the graph, and which assumptions make reuse correct. The code targets **Python 3.10 and PyTorch 2.10.0**. The Python version matters because bytecode changes between releases.

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

### Why interpret Python bytecode?

Suppose we recorded tensor operations while running a function normally. We would learn what happened on that execution. We would not automatically learn why it happened, or when the recording could safely be replayed. If Python selected a branch using a global flag, that flag would be part of the recording's validity conditions even though it was never a tensor operation.

Bytecode gives us access to those decisions. We can see a global being loaded, an attribute being read, or a conditional jump being reached. Our interpreter controls what happens next: evaluate something now, record it for later, or reject it. It also knows where values came from, so it can attach checks to the assumptions it makes.

This helps distinguish three ways to obtain a graph. `torch.jit.trace` runs example inputs and records the tensor computation it observes. `torch.fx.symbolic_trace` runs Python with proxy objects that record operations; ordinary Python control flow still has to accept those objects. Dynamo interprets the bytecode itself and can reason about the Python surrounding the tensor operations. All three involve tracing, but they observe execution at different levels. The [FX documentation](https://docs.pytorch.org/docs/2.10/fx.html) gives concrete examples of where proxy-based tracing reaches its limits.

We do not need to hook into CPython to explore this idea. A function object already gives us its code object, argument names, constants, and global namespace. Our decorator passes those to a Python implementation of a small interpreter. The PEP 523 hook used by real Dynamo solves a separate problem: entering that machinery from normal Python frame execution. We will return to it after the interpreter works.

There are two different programs in play throughout the post. The **Python program** loads names, handles constants, and calls methods. The **tensor graph** describes the computation left after we have dealt with that Python. Understanding what disappears between those representations is as important as understanding what becomes a node.

## Reading the function one instruction at a time

CPython compiles a function into **bytecode**, a sequence of instructions. We can inspect it with `dis.get_instructions(fn)`.

Python 3.10 represents `z = x + y` using four instructions:

| Instruction    | Stack afterwards | What happened                      |
| :------------- | :--------------- | :--------------------------------- |
| `LOAD_FAST x`  | `[x]`            | Put `x` on the stack               |
| `LOAD_FAST y`  | `[x, y]`         | Put `y` on the stack               |
| `BINARY_ADD`   | `[x + y]`        | Pop both values and push their sum |
| `STORE_FAST z` | `[]`             | Pop the result and store it as `z` |

A stack is just a list where we add and remove values at the end. Our interpreter keeps one, plus a dictionary of local variables and the graph under construction. `fn.__code__.co_varnames` supplies the argument names; we initialize the corresponding locals from the example inputs. A load copies a reference onto the stack. A store removes a reference from the stack and binds a local name to it.

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

The loop is intentionally less general than CPython's evaluator. It advances through the disassembled instructions in order, with no mechanism for changing the instruction pointer. A loop or conditional therefore needs more than an extra arithmetic handler: it needs jump semantics and a decision about which paths to trace. Here, an unsupported opcode raises immediately rather than being skipped.

The method-call convention is simplified too. CPython 3.10 uses `LOAD_METHOD` and `CALL_METHOD` to avoid constructing some bound method objects. Our interpreter represents the receiver and method name as one wrapper. It preserves the behavior of the supported calls without reproducing CPython's exact internal stack layout for them.

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

These two fields answer different questions. `node` answers “how will this value be computed on a future call?” `example_value` answers “what properties does this value have in the execution we are tracing?” Recording only the example would freeze the current tensor data. Recording only the node would leave our small interpreter unable to answer a later query such as `z.size(0)` without implementing separate metadata rules.

The binary-operation handler first pops the right operand, then the left. If either is a tensor wrapper, it takes the recording path above; if both are constants, it applies the Python operator immediately. Operand order matters for subtraction and division. For `2 - x`, the implementation preserves Python's reflected-operator behavior instead of assuming that `torch.sub(2, x)` is interchangeable with it.

**MiniDynamo really executes the example tensor operations during tracing.** It does not merely inspect their shapes. After tracing, the generated callable executes those operations again to produce the user's result. This keeps the implementation simple, but adds work to the first call.

It also explains why we restrict the supported operations. Tracing `x.add_(1)` on the user's tensor and then running it again would modify `x` twice. Calling an arbitrary helper could print, update a counter, or consume random numbers during tracing. MiniDynamo rejects those operations before calling them. Its allowlists live in `symbolic_interpreter.py`.

Real Dynamo generally propagates metadata using [FakeTensors](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_subclasses/fake_tensor.py), which model tensor properties without running ordinary tensor kernels on the input data. Handling mutation and Python side effects correctly takes additional machinery that this toy leaves out.

Back in our example, `STORE_FAST z` saves the new wrapper in `locals["z"]`. The multiplication then records another node with the literal `2` as an argument. Looking up `w.sum` creates a `MethodVariable`; calling that method records the reduction.

The distinction is useful: **wrappers help us interpret the program; graph nodes describe the computation we have captured.** The stack and local-variable dictionary are temporary. The graph is what the backend receives.

### What happens to Python values?

Consider a variation that divides by the leading dimension:

```python
def normalize_rows(x):
    rows = x.size(0)
    return x / rows
```

With an example of shape `(3, 4)`, `size(0)` produces a `ConstantVariable(3)`. The graph contains a division by `3`; it contains no `size` node. We have evaluated part of the program during tracing and kept the rest for future calls. This is **partial evaluation**.

That transformation creates an obligation. If we replayed the graph on a `(5, 4)` tensor, it would still divide by `3`. The exact-shape guard is what makes specializing `size(0)` legal. Constants in the function body are protected by the function's code; values read from tensor metadata or globals need the corresponding guards. We cannot decide what to omit from the graph independently of how we validate the resulting program.

The same distinction applies to calls. In `torch.relu(x)`, `LOAD_GLOBAL` resolves `torch`, `LOAD_ATTR` resolves `relu`, and `CALL_FUNCTION` consumes the function wrapper and argument. The interpreter records a tensor operation. In an allowed call such as `float(2)`, every argument is a known constant, so it evaluates the call and keeps the result as a constant. An arbitrary Python helper gets neither treatment: the interpreter rejects it because it does not know its effects.

Converting tensor _data_ to a Python value is different from reading metadata. `x.item()` or `bool(x)` can depend on the contents of every new input. Our guards do not inspect those contents. Treating such a result as a constant would silently specialize on data, so these conversions are unsupported. Supporting them would require a representation or runtime strategy beyond the constant wrapper used here.

### Following the complete trace

For the original function, this is the full sequence. `T(n)` denotes a tensor wrapper around node `n`, `C(v)` a constant wrapper, and `M(n, sum)` a method wrapper. The rightmost stack entry is the top.

| Instruction       | Stack afterwards   | Change outside the stack                      |
| :---------------- | :----------------- | :-------------------------------------------- |
| `LOAD_FAST x`     | `[T(x)]`           |                                               |
| `LOAD_FAST y`     | `[T(x), T(y)]`     |                                               |
| `BINARY_ADD`      | `[T(add_0)]`       | Record addition; compute example result       |
| `STORE_FAST z`    | `[]`               | Bind `z` to `T(add_0)`                        |
| `LOAD_FAST z`     | `[T(add_0)]`       |                                               |
| `LOAD_CONST 2`    | `[T(add_0), C(2)]` |                                               |
| `BINARY_MULTIPLY` | `[T(mul_0)]`       | Record multiplication; compute example result |
| `STORE_FAST w`    | `[]`               | Bind `w` to `T(mul_0)`                        |
| `LOAD_FAST w`     | `[T(mul_0)]`       |                                               |
| `LOAD_METHOD sum` | `[M(mul_0, sum)]`  |                                               |
| `CALL_METHOD 0`   | `[T(sum_0)]`       | Record reduction; compute example result      |
| `RETURN_VALUE`    | `[]`               | Mark `sum_0` as the graph output              |

The placeholder nodes already exist before this sequence starts. Notice that loads, stores, and method lookup do not themselves add tensor operations. Each handler only needs to preserve enough information for the next instruction; the graph grows when we encounter work that must happen again on future inputs.

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

### Dependencies, names, and aliasing

The essential node fields are `op`, `target`, `args`, and `kwargs`. For a function call, `target` is the callable itself, such as `torch.add`. For a method call, it is a method name, with the receiver stored as the first argument. A reference to another `Node` is a dependency; a Python scalar in `args` is a literal operand. The printable name is just how we refer to the result in generated code.

This distinction becomes clearer if Python reuses a local name:

```python
def rebind(x):
    old = x
    x = x + 1
    return old * x
```

`old` still refers to the input placeholder. Assigning a new wrapper to `locals["x"]` does not modify that placeholder. The multiplication therefore consumes the original input and the addition's output. Each node defines one value even though Python names can be rebound. This resembles static single assignment, without requiring a separate SSA conversion pass for our straight-line subset.

The list of nodes is already in dependency order: a handler can only consume values that previous instructions produced. Code generation can walk it once. Names still need to be unique; if a function argument happens to be called `add_0`, generating an intermediate with the same name could overwrite that input before a later use. The graph reserves placeholder names when allocating temporary names.

None of this means tensors cannot share storage. A supported view operation may return an alias, and replaying that operation preserves its view behavior. What the toy avoids is _mutation through_ those aliases. Once mutation is allowed, recording only value dependencies is insufficient: a write through one tensor can change a later read through another. That is one reason an interpreter for arbitrary PyTorch needs much more state than this graph.

The IR can also represent nested arguments and keyword arguments, and the backend preserves them. The bytecode frontend does not yet implement keyword-call instructions. These are separate boundaries: a graph format being able to express something does not mean the tracer can capture every Python spelling of it.

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

The translation is almost mechanical. Placeholders become function parameters. A node reference in an argument becomes the corresponding local name. A constant becomes a literal or a binding in the execution namespace. `call_function` emits an ordinary call; `call_method` emits a receiver followed by a method call; `output` emits `return`.

Binding the actual callable matters. A display string such as `torch.add` is not enough to recover every possible target, especially for imported aliases or Python operators. Similarly, `repr(torch.float32)` is not a self-contained literal unless the generated namespace has the right bindings. Keeping these objects in the namespace avoids relying on their printed form. Those binding names must also avoid collisions with graph names and parameters.

Generating Python is not essential to the design. We could interpret the graph on every call using a dictionary from nodes to current values. Emitting a function removes that graph-walking loop and makes the captured computation easy to inspect. It does not by itself combine tensor operations or change their kernels.

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

We can state the requirement more precisely. Let `P` be the original program, `C` a compiled specialization, and `G` its guards. Within the supported subset, we want `G(inputs, state)` to imply that `C(inputs)` has the behavior of `P(inputs, state)`, allowing for the backend's floating-point rounding differences. Here, `state` includes relevant globals and execution modes. Checking just that the compiled code accepts the new tensor shapes is not enough.

This is a correctness condition, not a proof supplied by the toy. Its implementation has to account for every way that supported programs can depend on the information discarded during tracing. Regression tests help check that accounting; the restrictions on Python features make it tractable.

The implementation also checks strides and other layout details, gradient requirements, the default dtype, and grad/inference/autocast modes. It records identity checks for global bindings and `torch` attributes read during tracing, plus the function's code object. These checks are deliberately conservative: they may trigger tracing even when the old Python callable would still have worked.

For example, a tensor and its transpose can have the same shape but different strides. An optimizing backend may have specialized address calculations for one layout. Grad mode and autocast can change how operations execute without changing any explicit function argument. Guarding only shape, dtype, and device would miss those distinctions.

Why check globals? Suppose a function multiplies by a module-level `scale`. Tracing reads that value and embeds it in the graph. If `scale` changes from `2` to `3`, the input tensor shapes have not changed, but the cached computation is stale. A guard needs to catch that. Mutable global objects and global tensors are outside MiniDynamo's supported subset.

The toy checks the identity of the binding it read. For an immutable value this is conservative: rebinding to an equal but distinct object may cause an unnecessary miss. It also guards accessed `torch` attributes, because checking that the global name still refers to the same module would not detect replacing `torch.relu` inside that module. Guarding a container's identity is not the same as guarding everything reachable through it.

### Snapshotting an assumption

This is the shape-checking part of `GuardSet.from_example_inputs`, with the surrounding loop omitted:

```python
expected_shape = tuple(arg.shape)
guard_set.add(
    Guard(
        lambda *args, idx=i, s=expected_shape: (
            type(args[idx]) is torch.Tensor
            and tuple(args[idx].shape) == s
        ),
        f"args[{i}].shape == {expected_shape}",
    )
)
```

The default arguments capture the current index and expected shape when the lambda is created. Closing over the loop variables themselves would make all the checks use their final values. Capturing a tuple also avoids retaining the example tensor just to ask for its shape later. A cache should keep the facts it needs, without accidentally keeping the entire tracing execution alive.

### Matching a cached specialization

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

Returning to `(3, 4)` reuses the earlier specialization; a guard failure on one entry does not invalidate the whole cache. To inspect a miss, ask the old entry which assumptions changed:

```python
new_x, new_y = torch.randn(5, 6), torch.randn(5, 6)
guards, _ = compiled._cache[0]
print(guards.failing_guards(new_x, new_y))
```

In this case both shape and stride checks fail: contiguous tensors of these two shapes have different row strides. This is more informative than treating every recompilation as an opaque compiler event.

![The wrapper scans cached entries and runs the first whose guards pass. Shape, dtype, and device are three examples of those guards.](/assets/img/mini-dynamo/cache-and-guards.svg)

Our list has no eviction policy or compilation limit. A workload with a new shape on every call will keep adding entries. Real Dynamo has richer guards, symbolic shape support, and recompilation limits. A shape change may reuse a dynamic graph, select another cached version, or trigger compilation; it does not necessarily mean starting over. [PyTorch's guard documentation](https://docs.pytorch.org/devlogs/dynamo/2025-06-04-inside-torch-compile-guards/) explains that trade-off.

The economics follow from that design. For one specialization reused `N` times, compilation costs roughly `trace + compile + N × (guard + execute)`. It pays off only if the saved execution time outweighs both the initial work and the recurring checks. A Python list of many specializations makes lookup more expensive too. Guard strength, lookup cost, and specialization frequency are therefore part of performance, even before we consider kernel quality.

## Giving the graph to Inductor

Graph capture becomes more useful when a backend can change how the operations execute. The optional `mini_dynamo/inductor.py` bridge does this:

```text
MiniDynamo graph → FX GraphModule → ATen graph → Inductor callable
```

### From our nodes to FX

The first conversion is structural. We walk the graph in dependency order and maintain a dictionary from our `Node` objects to their new `fx.Node` counterparts. When converting an operation's arguments, we replace references through that dictionary and leave constants unchanged. The implementation does this recursively for tuples, lists, and dictionaries, including keyword arguments.

The result is an `fx.GraphModule`: an `nn.Module` with a `forward` method generated from the FX graph. Constructing it does not discover new operations or optimize them. It repackages the computation into the format the next tool accepts. Our bridge also wraps the output in a one-element tuple to satisfy the downstream calling convention.

### From Python-facing operations to ATen

An FX graph is a representation, not a fixed operator vocabulary. A `call_function` node can target `torch.add`, a Python operator, or a particular ATen overload. That distinction matters to a backend: it needs operators with semantics it knows how to lower.

`make_fx` runs our graph module while recording operations at the PyTorch dispatcher level. For the running example, we can inspect the result directly:

```python
from mini_dynamo.inductor import to_fx_graph_module, lower_to_aten

gm = to_fx_graph_module(graph)
aten_gm = lower_to_aten(gm, x, y)
print(aten_gm.code)
```

Here is the generated forward method, with assignments that release temporary references omitted:

```python
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
    mul = torch.ops.aten.mul.Tensor(add, 2)
    sum_1 = torch.ops.aten.sum.default(mul)
    return (sum_1,)
```

The method call `w.sum()` is now an ATen function call, and the add and multiply targets identify specific overloads. No kernel has been fused at this point. `make_fx` can expose several lower-level operations for a higher-level call; it is not just a pass that renames targets. Our helper uses its default real tracing mode, so this stage also executes example operations.

### The backend contract

Finally, `compile_fx_inner` compiles the ATen graph. The small wrapper in `inductor.py` supplies a list of example inputs; the compiled object subsequently takes a single list of runtime inputs and returns a sequence of outputs. The public helper adapts that to `fused(x, y) -> tensor`.

This is a deliberately narrow route through **private PyTorch 2.10 APIs**. It bypasses the usual AOTAutograd pipeline, including its backward-graph construction and functionalization work. It is not the complete `torch.compile` pipeline with Dynamo swapped out.

To try it with our function:

```python
from mini_dynamo.inductor import mini_dynamo_to_inductor

fused = mini_dynamo_to_inductor(fn, x, y)
torch.testing.assert_close(fused(x, y), fn(x, y))
```

This helper supports forward inference with inputs that do not require gradients. Its wrapper checks the tracing assumptions and raises if they change; it does not manage a recompilation cache. Create a new callable for a new input specification or execution context. The main `@mini_dynamo.compile` decorator still exposes only the Python and JIT backends.

The tests compare MiniDynamo's path with `torch._dynamo.export` followed by the same ATen lowering and private Inductor entry point. For four small CPU programs, they check graph targets, dependencies, constants, generated C++ kernel bodies, and numerical results. That is evidence about those programs and that compilation path. It is not a claim that our toy reproduces `torch.compile` for arbitrary models or devices.

Comparing only operator names would be weak evidence: `x - y` and `y - x` contain the same operator but compute different functions. The graph comparison therefore encodes which earlier nodes each operation consumes, as well as its constants and keywords. The generated-kernel comparison also requires nonempty extracted C++ bodies before comparing them. Otherwise, failing to find either kernel could produce an empty-list equality and a meaningless pass.

### What fusion can save

What might make the optimized version faster? Fusion can avoid materializing intermediate tensors and reduce kernel launches. Inductor can also plan buffers and choose implementations for operations. Eligible CUDA workloads may use CUDA graphs to reduce host launch overhead. The gain depends on the workload, hardware, and compilation settings; even a correctly captured graph can be slower than eager execution.

For a concrete memory-traffic model, take `z = (x + y) * 2` on `N` float32 elements, before the reduction. Two separate elementwise kernels logically read `x` and `y`, write the addition result, read that result, and write `z`: `20N` bytes. A fused kernel can keep the intermediate in registers and reduce that to `12N` bytes. These are idealized tensor reads and writes; cache behavior means they are not a measurement of HBM traffic. They nevertheless explain what exposing both operations to a compiler makes possible.

A reduction changes the scheduling problem. Parallel blocks may need to produce partial sums and combine them in another kernel. A graph containing only supported operations therefore does not imply one kernel, and the fewest kernels need not give the lowest runtime. Register pressure, parallelism, and memory access patterns still matter.

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

The timing method helps explain why guard overhead need not appear as a fixed increment. CUDA launches are asynchronous: while the device processes earlier work, the host can check guards and enqueue the next call. A synchronized block of 500 calls measures the throughput of those overlapping activities. We should not subtract the raw and guarded rows and interpret the difference as an isolated measurement of guard latency. The small inputs make host overhead more visible because the device has less work to overlap with it.

The full suite on this CUDA node passed **227 tests**, with **4 MPS-only tests skipped**. CUDA checks cover float32, float16, bfloat16, gradients in the Python/JIT backends, and non-contiguous inputs. Numerical comparisons use tolerances because fusion and reductions can change floating-point rounding.

The [benchmark script](https://github.com/itsdaniele/torchdynamo-mini/blob/main/examples/benchmark_publication.py) and [raw trials, configuration, and profiler events](https://github.com/itsdaniele/torchdynamo-mini/blob/main/benchmarks/h200-publication.json) are in the repository. They also include the intermediate 256 × 256 case and the spread across trials. Treat these as measurements of this workload and configuration, not a prediction for a training run.

## What real Dynamo adds

The toy makes the mechanics visible by choosing a small subset of Python. The missing pieces follow from relaxing those restrictions.

### Frames and function inlining

Dynamo uses CPython's [PEP 523 frame-evaluation hook](https://peps.python.org/pep-0523/) to intercept eligible frames while compilation is active. A code object describes a function's instructions; a frame supplies the state of a particular invocation, including its locals and execution position. Intercepting frame evaluation gives Dynamo a place to check cached assumptions or begin tracing. It can then emit replacement bytecode that calls compiled graphs and handles surrounding Python work. MiniDynamo has a decorator that explicitly invokes our interpreter.

The hook and the bytecode interpreter do different jobs. Dynamo also uses `dis.get_instructions` in its [bytecode transformation code](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_dynamo/bytecode_transformation.py). When it inlines a supported Python helper during tracing, it uses an [inlining interpreter](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_dynamo/symbolic_convert.py) to walk the helper's bytecode; it does not need the helper to execute normally in a new frame first.

Inlining needs a fresh set of local bindings for the callee, while contributing tensor operations to the ongoing capture. The returned symbolic value then becomes a value in the caller. That is how a Python helper can disappear as a call boundary in the captured computation. Our toy rejects such helpers; putting an arbitrary Python callable into a graph would not amount to implementing this machinery.

### Branches and symbolic dimensions

Consider `if x.shape[0] > 5`. An exact-shape specialization can evaluate the condition during tracing and record only the selected branch. The shape guard ensures later calls make the same decision. A symbolic shape system can instead represent the leading dimension as a symbol, retain arithmetic involving that symbol in the graph, and guard a condition such as `s0 > 5` along with the other required constraints.

This does not put both branches into the graph automatically. It makes one specialization valid over a larger set of shapes. Nor does “dynamic” mean “unguarded”: broadcasting relationships, layout assumptions, and branch decisions can still constrain the sizes. PyTorch's [dynamic-shape documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html) describes how real Dynamo generalizes dimensions. MiniDynamo's constant treatment of `size(0)` cannot express this; adding symbolic integers would change both tracing and guard generation.

A branch on `x.sum() > 0` raises a different issue. Two tensors with identical metadata can choose different branches. The shape-based reasoning above cannot justify dropping one of them. MiniDynamo rejects the relevant bytecode; handling such a program requires an explicit strategy for data-dependent control flow, not merely a more permissive shape guard.

### Resuming after a graph break

Unsupported regions can cause real Dynamo to break the graph: execute a compiled prefix, run some ordinary Python, and resume tracing. With `fullgraph=True`, graph breaks cause an error instead. MiniDynamo has no resume mechanism.

Suppose a tensor produced by the prefix is still needed after an unsupported call. That tensor must become an output of the compiled region, even if it was only an intermediate in the original function. Dynamo must reconstruct the live Python locals and stack entries at the boundary, execute the intervening work with the correct values, and arrange the continuation. Splitting a graph therefore requires reasoning about Python execution state as well as tensor dependencies. The [Dynamo deep dive](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html) walks through the resulting bytecode.

This also explains a performance cost beyond the extra Python work: an intermediate that could have stayed inside a fused region may now have to be materialized so the surrounding Python can access it.

### Training, mutation, and backward graphs

MiniDynamo's Python backend calls ordinary PyTorch operations, so ordinary autograd can differentiate supported computations. It builds the autograd history while executing the generated forward function; it does not compile a backward pass. A forward graph containing differentiable operations is not itself a graph of their derivatives.

The usual Inductor training path uses AOTAutograd to trace forward and backward computations and partition the joint graph into callable pieces. Choosing what the forward saves for backward, and what backward recomputes, creates optimization choices that our bridge never sees. The [AOTAutograd implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_functorch/aot_autograd.py) also coordinates functionalization and metadata about mutations and aliases. Replacing an in-place operation with an out-of-place one must still preserve externally observable effects; simply removing the underscore would not be correct.

Our direct `compile_fx_inner` path bypasses that preparation. Restricting it to forward inference and rejecting inputs that require gradients keeps the example's contract explicit. Supporting training would mean integrating the missing stages, not just removing that check.

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
