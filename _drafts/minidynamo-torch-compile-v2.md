---
layout: post
title: "MiniDynamo: A Tiny torch.compile from Scratch"
description: Rebuilding the core ideas behind torch.compile with a small TorchDynamo-style tracer. (v2, Christensen BALANCED revision)
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

<!-- v2: prose revised with Christensen generative rhetoric, BALANCED preset. Content identical to v1. -->

<div class="l-body" markdown="1">

_Run code under `@torch.compile` and a lot happens beneath you: bytecode interception, graph capture, an optimizing compiler, a cached result. This article rebuilds those pieces from scratch, with a small implementation that exposes the moving parts._

Note: _Coding agents generated a good part of the code. I then read the implementation line by line, adding tests and benchmark scripts along the way. The toy system itself became the tool for building intuition._

</div>

<aside markdown="1">

**TL;DR**

- `torch.compile` is a pipeline: **trace → compile → guard → cache → execute**. This post rebuilds it in ~1,000 lines of readable Python ([mini-dynamo](https://github.com/itsdaniele/torchdynamo-mini)).
- TorchDynamo captures graphs at the **bytecode level**: it re-implements CPython's interpreter over symbolic values and records tensor operations into a graph (Sections 3–6).
- **Guards**, the checks on shape, dtype, and device, are the contract deciding when cached compiled code can be reused. A high recompilation rate is the usual reason `torch.compile` disappoints (Section 8).
- Graph capture alone produces **no speedup** (we measure 1.00x). The wins come from what an optimizing backend like Inductor does with the graph. Our mini graphs can drive the real Inductor to prove it (Section 10).

</aside>

## 1. The Big Picture

PyTorch makes modeling code convenient. Inside a module's `forward()` method, ordinary Python still applies: `if` statements, helper functions, debug prints. PyTorch calls this **eager mode**, each tensor operation dispatched to the device the moment Python reaches it. For a chain of eleven elementwise operations, the shape of our later benchmark, the cost is eleven kernel launches and eleven memory round-trips. A CPU-side dispatcher stands between each pair. That flexibility probably won PyTorch the framework war, though it leaves performance on the table.

PyTorch 2.0 introduced `torch.compile` to keep the flexible programming model while clawing back that performance. The first call to a wrapped function triggers capture: TorchDynamo records your tensor operations into a graph. The graph then goes to Inductor, an optimizing compiler able to fuse many operations into fewer kernels. On every later call, PyTorch tries to reuse the optimized result, as long as the assumptions made during tracing still hold.

> **Why this is faster on GPUs:** eager PyTorch launches one GPU kernel per operation. A fusible chain like `add → relu → mul` repeatedly reads tensors from GPU memory, writes intermediate tensors back, and pays launch overhead each time. Once Dynamo captures the whole chain as a graph, Inductor can fuse those operations into fewer kernels. In the best case, the GPU reads the data once, keeps intermediates in registers, does more work per launch, and writes the final result back once.

All of this starts with the hard part, the symbolic execution of arbitrary Python and PyTorch code.

### The graph capture problem

Capturing a graph of tensor operations from a Python function is hard. PyTorch went through years of earlier approaches before landing on TorchDynamo, every one of them carrying painful trade-offs:

- **`torch.jit.trace`** (2018): Run the function with example inputs, recording the tensor operations that execute. Problem: it observes one execution path, not the whole Python program. If an `if` statement takes one branch for the example inputs, the trace contains only that branch, which later calls replay blindly. Code that depends on tensor values, Python-side control flow, or input-dependent shapes can produce wrong or over-specialized graphs.

- **TorchScript** (`torch.jit.script`, 2018): Parse Python source into a statically analyzable, typed subset of Python. This could preserve control flow in a way tracing could not. Problem: users had to write code the TorchScript compiler understood, and real PyTorch programs often used features outside that subset. Many models needed code changes before they could be scripted.

- **FX Tracing** (`torch.fx.symbolic_trace`, 2021): Execute Python with `Proxy` objects standing in for tensors, their operations recorded into an FX graph. Problem: the tracer still runs ordinary Python. When Python demands a concrete answer from a proxy, a branch condition for instance, tracing fails. At best it specializes to the example-time behavior.

- **Lazy Tensors**: Record tensor operations at the backend level, deferring execution until a result is needed. This hands the backend a graph it can optimize. Problem: Python has already run by the time those ops are recorded. Lazy tensors can optimize tensor execution. They do nothing about Python itself: the frames still run, the control flow stays opaque, and later calls repeat all the Python work.

**TorchDynamo**, the engine behind `torch.compile`, took a different approach: it works at the **bytecode level**, below Python source and above the C++ dispatcher. Using PEP 523's Frame Evaluation API, Dynamo installs a C-level hook that intercepts every Python frame _before_ CPython's interpreter runs it. It then walks the bytecode instructions, symbolically evaluating each one to find the tensor operations bound for an FX graph.

### Calling `torch.compile`

![The torch.compile pipeline: first call runs every stage; subsequent calls with matching guards skip straight to EXECUTE.](/assets/img/mini-dynamo/pipeline.svg)

1. **Trace**: Dynamo intercepts the Python frame via PEP 523. Walking the bytecode, it records tensor operations into an FX graph. Much of the surrounding Python stays outside the graph: some values are evaluated concretely, some assumptions become _guards_, and unsupported regions can trigger _graph breaks_. We will see in detail what this means.

2. **Compile**: The FX graph goes to a compiler backend. The default backend is Inductor, which generates Triton kernels on CUDA and C++ kernels on CPU.

3. **Guard**: Dynamo records the assumptions made during tracing: tensor shapes, dtypes, devices, and the values of Python variables used in control flow.

4. **Cache**: The compiled function and its guards are stored together. On later calls, if all guards pass, the compiled function is reused without re-tracing.

5. **Execute**: If guards pass, the compiled function runs. If they fail, a shape change for example, the pipeline runs again, adding a new cache entry.

**Trace once, execute many times.** The first call pays for bytecode analysis plus compilation. Later calls with matching inputs skip tracing, reusing the compiled result.

### What we'll build

We will build this pipeline as a small, readable Python codebase. Our implementation, [_mini-dynamo_](https://github.com/itsdaniele/torchdynamo-mini), is a deliberately tiny TorchDynamo-style tracer. It captures the same core ideas while leaving out the machinery needed for arbitrary real-world PyTorch programs.

<aside markdown="1">

**Scope.** Real TorchDynamo handles control flow, nested calls, user-defined classes, graph breaks, dynamic shapes, and hundreds of Python opcodes. Mini-dynamo handles straight-line tensor computations over positional tensor arguments. Inside the function body it supports constants, basic arithmetic, an explicit list of common `torch` functions, and positionally-called tensor methods. The tests focus on methods such as `.sum()` and `.mean()`. We skip PEP 523 and everything built on it: real Dynamo hooks CPython frame evaluation, re-implementing a large chunk of Python execution logic in Python. We also skip AOTAutograd and training-graph lowering; the examples focus on forward computations.

</aside>

<aside markdown="1">

**Runtime requirements.** The repository pins Python `3.10.x` and PyTorch `2.10.0`. The narrow range is intentional: CPython bytecode changes across releases, and this educational interpreter implements only the Python 3.10 opcodes used in the examples. The public `@mini_dynamo.compile` decorator likewise accepts only positional tensor arguments. Runtime non-tensor arguments and keyword arguments are deliberately out of scope.

</aside>

---

## 2. Architecture: The Five Components

Mini-dynamo has five pipeline stages spread across six small modules:

```
              fn(x, y) + example args
                      │
                      ▼
          ┌───────────────────────┐
          │   compile() decorator │ ← __init__.py (orchestrator)
          │   Manages the cache,  │   Checks guards, dispatches
          │   wires everything    │   to trace/compile/guard
          └───┬─────────┬─────┬───┘
              │         │     │
              ▼         │     ▼
  ┌──────────────────┐  │  ┌─────────────┐
  │ Symbolic          │  │  │   Guards     │ ← guards.py
  │ Interpreter       │  │  │   Shape,     │   Boolean checks on
  │                   │  │  │   dtype,     │   input metadata
  │ Walks bytecodes,  │  │  │   device     │
  │ manipulates       │  │  └─────────────┘
  │ VariableTrackers  │  │
  │ on a stack,       │  │
  │ builds the Graph  │  │
  └────────┬──────────┘  │
           │             │
     ┌─────┘             │
     ▼                   ▼
  ┌──────────┐    ┌──────────────┐
  │  Graph   │───▶│  Compiler    │ ← compiler.py
  │  (IR)    │    │  Backend     │   Graph → Python source
  │          │    │              │   → exec() → callable
  └──────────┘    └──────────────┘
   graph.py         examples/ → Inductor
```

The package backend in `mini_dynamo/compiler.py` implements only the Python and JIT paths exposed by `@mini_dynamo.compile`. The Inductor path appears later as an example script: mini graph to FX, ATen lowering, then a private Inductor entry point.

The table maps each component to its TorchDynamo counterpart:

| Mini-dynamo               | Real TorchDynamo                                                | Role                                                                                                                                                                                                                                                                                                           |
| :------------------------ | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `symbolic_interpreter.py` | `InstructionTranslator` (~5,000 lines in `symbolic_convert.py`) | Walk bytecodes, build the graph. The heart of the system.                                                                                                                                                                                                                                                      |
| `variable_tracker.py`     | `VariableTracker` hierarchy (200+ subclasses across ~20 files)  | Symbolic values on the interpreter's stack. Tell the interpreter what kind of thing each value is (tensor? constant? torch function?) so it can decide whether to record a graph node or evaluate concretely.                                                                                                  |
| `graph.py`                | `torch.fx.Graph` + `torch.fx.Node`                              | The computation graph IR. A flat list of nodes, each describing one operation. This is the output of tracing and the input to compilation.                                                                                                                                                                     |
| `compiler.py`             | Compiler backends (Inductor, etc.)                              | Takes a finished graph and produces a callable. Our simple backend generates Python source with pre-resolved names in an `exec()` namespace. Real Inductor generates Triton GPU kernels and C++ CPU kernels. The later Inductor integration example adds a small converter from our graph format to FX Graphs. |
| `guards.py`               | `torch._dynamo.guards` (C-accelerated)                          | Boolean predicates that encode the assumptions made during tracing. If guards pass on new inputs, the cached compiled function can be reused.                                                                                                                                                                  |
| `__init__.py`             | `torch._dynamo.convert_frame`                                   | The orchestrator. Manages the guard-cache loop: check guards → hit? run cached fn. Miss? trace → compile → guard → cache.                                                                                                                                                                                      |

Two components deserve special attention because their roles are easy to confuse.

**The Graph is the output.** It is a pure data structure, a list of nodes describing which tensor operations to perform. It has no logic and no execution semantics. After tracing, the compiler receives it as a recipe.

**VariableTrackers are the process.** They are the symbolic values living on the interpreter's stack during tracing. They tell the interpreter what kind of thing each value is, the basis for every per-instruction decision. When tracing finishes, the interpreter throws them away, scaffolding for the graph rather than part of the final product.

We need both because CPython's bytecodes are untyped. When the interpreter sees `BINARY_ADD`, it cannot tell tensors from integers: record `torch.add` in the graph, or just compute the result? VariableTrackers carry the type information behind that decision. The Graph records the decisions once made.

---

## 3. CPython Is a Stack Machine

To understand our symbolic interpreter, you need one fact about CPython: **it is a stack-based virtual machine.** Every Python function compiles to bytecode instructions that manipulate a _value stack_ and a _locals array_.

For `z = x + y`, CPython emits:

| Instruction    | Stack (after) | Effect                     |
| :------------- | :------------ | :------------------------- |
| `LOAD_FAST x`  | `[x]`         | Push local variable `x`    |
| `LOAD_FAST y`  | `[x, y]`      | Push local variable `y`    |
| `BINARY_ADD`   | `[x+y]`       | Pop two, push their sum    |
| `STORE_FAST z` | `[]`          | Pop and store in local `z` |

Our symbolic interpreter mirrors this exactly, with the same stack and the same dispatch loop. The only difference is what the stack holds, symbolic wrappers recording operations into a graph rather than real Python values.

---

## 4. VariableTrackers: The Symbolic Values

Every value in our interpreter is a `VariableTracker`: _"I'm not a real value. I'm a description of a value that will exist at runtime."_

We need exactly four types.

### TensorVariable

The most important type. It holds a _graph node_, its identity in the computation graph, and an _example value_, a real tensor with matching shape, dtype, and device.

```python
class TensorVariable(VariableTracker):
    def __init__(self, node, example_value):
        self.node = node              # Graph Node that produces this tensor
        self.example_value = example_value  # Real tensor for shape tracking
```

When the interpreter sees `x + y` on two `TensorVariable`s, it does not compute the result the user asked for. Instead, it:

1. Creates a new `Node` in the graph: `call_function(torch.add, (x.node, y.node))`
2. Computes an example output for metadata propagation: `torch.add(x.example_value, y.example_value)`
3. Returns `TensorVariable(new_node, example_output)`

The example value flows forward through every operation. At any point during tracing, we therefore know the exact shape, dtype, and device of every intermediate tensor. Example tensor ops do run during tracing, for metadata propagation, but the output of tracing is the graph rather than the eager result.

### ConstantVariable

A value fully known at trace time: the `2` in `x * 2`, a dtype like `torch.float32`, a shape tuple. Constants never become graph nodes; they are inlined directly into the operations that use them.

```python
class ConstantVariable(VariableTracker):
    def __init__(self, value):
        self.value = value  # The actual Python value
```

### TorchVariable

A reference to the `torch` module or one of its functions. When the interpreter encounters `LOAD_GLOBAL torch`, it pushes `TorchVariable(torch)`. A following `LOAD_ATTR relu` resolves `torch.relu`, pushing `TorchVariable(torch.relu)`.

### MethodVariable

A bound tensor method like `x.sum`, created when the interpreter accesses a method on a `TensorVariable`. It remembers which tensor and which method, enough to record the correct graph node when called.

<aside markdown="1">

**Real Dynamo has over 200 VariableTracker subclasses**, covering lists, dicts, iterators, ranges, user-defined classes, `nn.Module`s, and more. Our four types suffice for straight-line tensor code.

</aside>

---

## 5. The Graph IR

As the interpreter runs, it records operations into a `Graph`, an ordered list of `Node` objects forming a DAG of the computation. This is a simplified version of `torch.fx.Graph`.

Each `Node` has four key fields:

```python
class Node:
    name: str       # Unique identifier, e.g. "add_0", "x"
    op: str         # One of: "placeholder", "call_function", "call_method", "output"
    target: Any     # What to call (e.g., torch.add) or method name (e.g., "sum")
    args: tuple     # Positional arguments -- can reference other Nodes
```

Nodes come in four flavors:

| `op`            | Meaning                    | Example                   |
| :-------------- | :------------------------- | :------------------------ |
| `placeholder`   | Function input             | `x = placeholder`         |
| `call_function` | A function call on tensors | `add_0 = torch.add(x, y)` |
| `call_method`   | A method call on a tensor  | `sum_0 = add_0.sum()`     |
| `output`        | The return value           | `return sum_0`            |

For the function:

```python
def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()
```

The captured graph is:

```
Graph:
  x = placeholder
  y = placeholder
  add_0 = torch.add(x, y)
  mul_0 = torch.mul(add_0, 2)
  sum_0 = mul_0.sum()
  return sum_0
```

![The same function before and after tracing. Function name, local variables, and Python operators dissolve into a pure data-flow DAG over tensor ops.](/assets/img/mini-dynamo/graph-ir.svg)

Notice the `2` in `torch.mul(add_0, 2)`: a plain Python integer, not a `Node`. Constants are inlined into the args of the operations that consume them.

---

## 6. The Symbolic Interpreter

The `SymbolicInterpreter` ties the previous pieces together. It walks the bytecode with `VariableTracker`s on its stack, and the `Graph` grows as a side effect. Everything else in mini-dynamo serves this loop.

When you run a Python function normally, CPython executes each instruction on real objects: integers get added, methods get invoked. We use the same bytecode and stack discipline, but we care about the _shape_ of the computation rather than the return value. We re-implement enough of CPython's interpreter to produce a **graph**.

### Two Interpreters in Parallel

Picture two interpreters running side by side on the same bytecode, one real and one symbolic:

|                                 | CPython's interpreter                    | Our symbolic interpreter                                                                 |
| :------------------------------ | :--------------------------------------- | :--------------------------------------------------------------------------------------- |
| **Stack holds**                 | Real Python objects                      | `VariableTracker`s                                                                       |
| **Locals hold**                 | Real values                              | `VariableTracker`s                                                                       |
| **`BINARY_ADD` on two tensors** | Calls `torch.add`, produces a new tensor | Records `torch.add(x, y)` in the graph, pushes a new `TensorVariable` wrapping that node |
| **`BINARY_ADD` on two ints**    | Computes `a + b`                         | Computes `a + b`; constants are evaluated concretely                                     |
| **`CALL_METHOD x.sum()`**       | Invokes the bound method                 | Records `x.sum()` in the graph                                                           |
| **Unsupported opcode**          | Executes it                              | Raises `NotImplementedError`                                                             |
| **Final output**                | A return value                           | A finished `Graph`                                                                       |

CPython operates on values; mini-dynamo operates on _descriptions_ of values. At every bytecode step, the symbolic interpreter makes one decision: **record** the operation into the graph, or **evaluate** it concretely from what it already knows. Repeated across instructions, that decision produces the captured graph.

### The Three Pieces of State

Just like CPython, our interpreter carries three pieces of state through its run:

- **`self.stack`**: a list of `VariableTracker`s. `LOAD_*` opcodes push to it; `BINARY_*`, `CALL_*`, `STORE_*`, and the rest consume it.
- **`self.locals`**: a dict mapping variable names to `VariableTracker`s. It starts with the function arguments and changes on `STORE_FAST`.
- **`self.graph`**: the `Graph` being built. It grows each time a tensor operation gets recorded.

Everything the interpreter does is a transformation of these three. A snapshot after every instruction would give you a complete movie of the trace.

### Correspondence to Real Dynamo

Our `SymbolicInterpreter` is the direct analogue of TorchDynamo's `InstructionTranslator` (in `torch/_dynamo/symbolic_convert.py`). The two share the same skeleton: a value stack, a locals dict, an FX-style graph under mutation, one handler per opcode. The differences are in scope, not in kind:

- Real Dynamo handles ~160 opcodes including jumps, comparisons, exceptions, closures, and generator machinery. We handle a small straight-line subset.
- Real Dynamo _inline-traces_ into called functions. When `fn()` calls `helper()`, the tracer recursively walks the callee's bytecode too; during tracing, `helper`'s frame never actually runs. The result is a single unified graph. Our walker sees only top-level bytecode. If it records a helper call at all, it records it as an opaque callable.
- Real Dynamo emits **guards** on the fly, one for each assumption made during tracing, such as treating `x.shape[0]` as exactly `32`. We emit guards after tracing, from the example inputs.
- Real Dynamo can **break the graph** at something unsupported. It compiles what it has so far, letting the hard part run in plain Python before tracing resumes. Our interpreter simply halts with `NotImplementedError`.

From here, the path runs from initialization through the dispatch loop to one traced example. The call-dispatch logic, which decides between graph node and concrete call, closes the section.

### Initialization

When we begin tracing `fn(x, y)`, we create a `SymbolicInterpreter` with:

- A fresh `Graph`
- An empty `stack`
- `locals` populated with `TensorVariable` placeholders for each tensor argument

```python
def __init__(self, fn, example_args):
    self.fn = fn
    self.graph = Graph()
    self.stack = []     # Mirrors CPython's value stack, but holds VariableTrackers
    self.locals = {}    # Mirrors CPython's locals: name -> VariableTracker
    self.globals = fn.__globals__  # Needed later for LOAD_GLOBAL (e.g. `torch`)

    # fn.__code__ is the compiled CPython code object behind a function.
    # co_varnames is the tuple of *all* local names; the first co_argcount of
    # them are the declared parameters, in order. So this slice gives us just
    # the parameter names without pulling in interior locals.
    code = fn.__code__
    arg_names = code.co_varnames[:code.co_argcount]

    # Seed the locals dict with one tracker per argument:
    #   - tensors enter the graph as `placeholder` nodes (they're the inputs
    #     downstream nodes will reference);
    #   - non-tensors stay as concrete ConstantVariables, so the interpreter
    #     can use their actual Python value during tracing (e.g. literal
    #     arithmetic, shape tuples, dtype objects). The public decorator
    #     intentionally rejects non-tensor runtime arguments.
    for name, example in zip(arg_names, example_args):
        if isinstance(example, torch.Tensor):
            node = self.graph.placeholder(name)
            self.locals[name] = TensorVariable(node, example)
        else:
            self.locals[name] = ConstantVariable(example)
```

### The Main Loop

The interpreter fetches instructions one by one, dispatching each to a handler method:

```python
def run(self):
    # dis.get_instructions decodes the function's bytecode into a flat list
    # of Instruction records: the same data CPython would dispatch on
    # internally. Each record knows its opname (e.g. "LOAD_FAST"), its
    # argument value, and where it sits in the bytecode.
    instructions = list(dis.get_instructions(self.fn))
    for inst in instructions:
        # One handler method per opcode, conventionally named op_<OPNAME>
        # (e.g. op_LOAD_FAST, op_BINARY_ADD). This is the same trick CPython's
        # ceval.c uses, spelled in Python via attribute lookup. Anything
        # we haven't implemented falls through to NotImplementedError rather
        # than silently producing a wrong graph.
        handler = getattr(self, f"op_{inst.opname}", None)
        if handler is None:
            raise NotImplementedError(f"Unsupported bytecode: {inst.opname}")
        handler(inst)
    return self.graph
```

### Walking Through an Example

Trace `fn(x, y)`, where `fn` computes `z = x + y; w = z * 2; return w.sum()`. The first four instructions show bytecode, stack, and graph evolving in lockstep:

![Tracing in motion: only the BINARY_ADD step actually touches the graph. Every other instruction is plumbing.](/assets/img/mini-dynamo/tracing.svg)

The full trace, including the multiplication and the method call:

| Step | Instruction       | Stack                             | Graph (new node)              |
| :--: | :---------------- | :-------------------------------- | :---------------------------- |
|  1   | `LOAD_FAST x`     | `[TensorVar(x)]`                  | --                            |
|  2   | `LOAD_FAST y`     | `[TensorVar(x), TensorVar(y)]`    | --                            |
|  3   | `BINARY_ADD`      | `[TensorVar(add_0)]`              | `add_0 = torch.add(x, y)`     |
|  4   | `STORE_FAST z`    | `[]`                              | --                            |
|  5   | `LOAD_FAST z`     | `[TensorVar(add_0)]`              | --                            |
|  6   | `LOAD_CONST 2`    | `[TensorVar(add_0), ConstVar(2)]` | --                            |
|  7   | `BINARY_MULTIPLY` | `[TensorVar(mul_0)]`              | `mul_0 = torch.mul(add_0, 2)` |
|  8   | `STORE_FAST w`    | `[]`                              | --                            |
|  9   | `LOAD_FAST w`     | `[TensorVar(mul_0)]`              | --                            |
|  10  | `LOAD_METHOD sum` | `[MethodVar(mul_0, "sum")]`       | --                            |
|  11  | `CALL_METHOD 0`   | `[TensorVar(sum_0)]`              | `sum_0 = mul_0.sum()`         |
|  12  | `RETURN_VALUE`    | `[]`                              | `return sum_0`                |

Notice three details.

**Steps 3 and 7**: when a binary operation involves a `TensorVariable`, the interpreter records a `torch.add` or `torch.mul` node, pushing a fresh `TensorVariable` around it. The constant `2` passes directly into the node's args.

**Step 10**: `LOAD_METHOD sum` on a `TensorVariable` produces a `MethodVariable`, not a graph node. The method has been _looked up_, not _called_. Step 11 creates the graph node when `CALL_METHOD` executes it.

**Step 12**: `RETURN_VALUE` marks the output. The graph is now complete.

### The Call Dispatch Logic

The most interesting handler is `_handle_call`, which decides what happens when a function or method gets called:

```python
def _handle_call(self, fn, args):
    # Known torch function with tensor args? → Record graph node.
    if isinstance(fn, TorchVariable) and fn.value in SUPPORTED_TORCH_FUNCTIONS:
        return self._call_torch_function(fn.value, args)

    # Tensor method (x.sum, x.reshape)? → Record graph node.
    if isinstance(fn, MethodVariable):
        return self._call_tensor_method(fn, args)

    # Unknown callable with tensor args? → Try tracing it anyway.
    if isinstance(fn, TorchVariable) and callable(fn.value):
        has_tensors = any(isinstance(a, TensorVariable) for a in args)
        if has_tensors:
            return self._call_torch_function(fn.value, args)
        else:
            concrete_args = [self._to_concrete(a) for a in args]
            return self._wrap_result(fn.value(*concrete_args))

    # Pure Python on constants (int, len, etc.)? → Evaluate directly.
    if isinstance(fn, ConstantVariable) and callable(fn.value):
        concrete_args = [self._to_concrete(a) for a in args]
        return self._wrap_result(fn.value(*concrete_args))

    raise RuntimeError(f"Don't know how to call {type(fn).__name__}")
```

The design has three paths. Known tensor operations are traced, and opaque global callables with tensor inputs can also become `call_function` nodes. Pure-Python work on supported constants is simply evaluated. Outside that narrow subset mini-dynamo still raises, especially for unsupported bytecodes, keyword calls, and non-tensor returns. Real TorchDynamo goes much further, guarding on Python values and even resuming after graph breaks. A _graph break_ is Dynamo's escape hatch for the "don't know how to handle this" case. The graph so far gets compiled, the unsupported bit runs in plain Python, and a fresh trace starts from the next instruction. A single Python function can become several compiled graphs, stitched together with plain eager code in between.

<aside markdown="1">

**A concrete graph-break example.** Imagine a `forward` that calls a custom CUDA kernel through a `ctypes` binding, bypassing the PyTorch dispatcher. Dynamo can see the Python call site. It cannot introspect the C code across the FFI boundary, so the call cannot become an FX node. Rather than failing the whole compile, Dynamo cuts the graph at that instruction. Everything before the call becomes graph #1, compiled with Inductor. The opaque CUDA call runs in eager Python against materialized tensors, and whatever comes after starts graph #2. The kernel fuser cannot see across that boundary, which is why teams work hard to eliminate graph breaks in hot paths. A properly registered PyTorch custom op is a different story. It participates in the dispatcher, so Dynamo may keep it as a graph operator even when Inductor treats it as opaque.

</aside>

---

## 7. The Compiler Backend

The graph is now a clean IR of tensor operations. The compiler's job is to turn it into a callable. In `torch.compile`, this is where Inductor takes over, the kernel-fusion step that produces the speedup. **Our educational backend does something smaller.** It generates a plain Python function re-dispatching to the same `torch` ops one at a time, lookups pre-resolved into the generated namespace. It exists to prove correct capture, yielding a standalone callable for later Inductor experiments rather than any speedup of its own.

### What the compiler generates

Our compiler walks the graph, generating a Python function whose function references are **pre-resolved in the `exec()` namespace**:

```python
# Generated code for many_ops(x, y):
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)       # __fn_add_0 = torch.add (in exec namespace)
    mul_0 = __fn_mul_0(add_0, 2)   # __fn_mul_0 = torch.mul (in exec namespace)
    sub_0 = __fn_sub_0(mul_0, x)   # __fn_sub_0 = torch.sub (in exec namespace)
    ...
    return sum_0
```

Each `__fn_*` name resolves from the custom namespace we pass to `exec()`, sparing the `LOAD_GLOBAL torch` + `LOAD_ATTR add` pair of the original. That sounds like an optimization. It saves almost nothing. CPython's bytecode dispatch costs tens of nanoseconds per instruction. A single eager GPU op spends microseconds in the C++ dispatcher, more microseconds launching the kernel, and then whatever the kernel itself takes. Skipping one lookup pair per op trims a tiny fraction of the smallest of those costs. The step exists to produce a clean, self-contained callable reproducing the graph. An optimizing backend like Inductor expects exactly that kind of graph-shaped callable, and the comparison against the original doubles as a sanity check.

### Code Generation

The compiler walks the graph, emitting one line of Python per node:

```python
def compile_graph(graph):
    # Graph placeholders become the parameters of the generated function,
    # in the same order they appeared in the original `fn`.
    param_names = [n.name for n in graph.inputs]
    signature = ", ".join(param_names)

    body_lines = []
    # `closure_vars` ends up as the globals dict for the exec()'d function.
    # Stashing the actual callables (torch.add, torch.mul, …) in here lets
    # generated code refer to them as plain names — no LOAD_GLOBAL + LOAD_ATTR
    # pair on every call.
    closure_vars = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            continue  # Already covered by the function signature above.
        elif node.op == "call_function":
            # Give this op a unique closure key, stash its target callable,
            # and emit a single line that invokes it.
            closure_key = f"__fn_{node.name}"
            closure_vars[closure_key] = node.target
            args_str = _format_call_args(node.args)
            body_lines.append(f"    {node.name} = {closure_key}({args_str})")
        elif node.op == "call_method":
            # Methods are dispatched on the receiver, so there's no callable
            # to stash in the exec namespace. We write `<self>.<method>(...)`.
            self_name = _arg_to_str(node.args[0])
            rest_args = _format_call_args(node.args[1:])
            body_lines.append(f"    {node.name} = {self_name}.{node.target}({rest_args})")
        elif node.op == "output":
            body_lines.append(f"    return {_arg_to_str(node.args[0])}")

    source = f"def compiled_fn({signature}):\n" + "\n".join(body_lines)
    # Two-step materialization. `compile()` (Python builtin, not ours) turns
    # the source string into a code object; `exec()` runs that code with
    # `closure_vars` as its globals. The side effect is that `compiled_fn`
    # is now defined inside `closure_vars`, ready to be pulled back out.
    code = compile(source, "<mini-dynamo-compiled>", "exec")
    exec(code, closure_vars)
    return closure_vars["compiled_fn"], source
```

The `exec()` call creates the function in a namespace holding the pre-resolved torch functions. This string-codegen trick belongs to our educational backend. Real TorchDynamo's default path produces FX graphs for backends such as Inductor; it does not lean on a tiny Python source generator for performance.

### The JIT Backend

For an additional step, we can trace the generated Python function with `torch.jit.trace`, obtaining a TorchScript function:

```python
def compile_graph_jit(graph, example_inputs):
    # First, produce our usual Python source via compile_graph(). Then hand
    # that callable to torch.jit.trace, which re-records it as a single
    # TorchScript graph by running it once with the example inputs. The
    # result is a function where Python drops out of the per-op loop,
    # but each op still launches its own kernel; nothing is fused.
    compiled_fn, source = compile_graph(graph)
    traced_fn = torch.jit.trace(compiled_fn, example_inputs)
    return traced_fn, source
```

The result executes the entire graph as a single C++ call, with no Python interpreter between operations. Each operation still launches a separate kernel. Nothing is fused, so this is not Inductor's optimization path; in small GPU microbenchmarks it can still help by trimming Python/C++ boundary overhead. It demonstrates graph lowering to another runtime, the same broad move Inductor makes with kernel fusion. One caveat: TorchScript is in maintenance mode, and `torch.jit.trace` is deprecated in recent PyTorch releases. Treat this backend as a demonstration rather than a path to build on.

---

## 8. Guards: When Can We Reuse Compiled Code?

A compiled function makes assumptions about its inputs. The graph traced for `fn(x, y)` at `x.shape = (3, 4)` might not be valid at `(5, 6)`. Different shapes can change broadcasting, output sizes, or even which operations are legal.

**Guards** encode these assumptions as boolean checks:

```python
@classmethod
def from_example_inputs(cls, example_args):
    guard_set = cls()
    guard_set.n_args = len(example_args)  # arity is checked before any per-arg guard
    for i, arg in enumerate(example_args):
        if isinstance(arg, torch.Tensor):
            # Snapshot the shape *now*, while we still have the example tensor.
            expected_shape = tuple(arg.shape)
            guard_set.add(Guard(
                # The `idx=i, s=expected_shape` default arguments are the
                # standard Python trick for capturing loop variables *by value*
                # into a closure. Without them, every lambda would close over
                # the same `i` and `expected_shape` bindings and all end up
                # checking whatever those names held at the end of the loop.
                lambda *args, idx=i, s=expected_shape: tuple(args[idx].shape) == s,
                f"args[{i}].shape == {expected_shape}",
            ))
            # ... similarly for dtype and device
    return guard_set
```

On each call, every guard runs. If all pass, the cached compiled function is valid and tracing is skipped entirely. If any fails, the pipeline retraces for the new input signature, adding a fresh cache entry.

### Checking and Debugging Guards

`GuardSet.from_example_inputs` produces three guards per tensor argument, one each for shape, dtype, and device. On every call, the wrapper runs the cached guard sets against the new arguments. Each guard is a tiny lambda, `tuple(args[0].shape) == (3, 4)` for instance, so a full check costs a handful of Python comparisons. When a call misses the cache, you can ask the guard set why; the wrapper exposes its cache as `fn._cache`, a list of `(guard_set, compiled_fn)` pairs:

```python
# fn was traced with (3, 4) tensors; now call it with a new shape:
a3 = torch.randn(5, 6)
b3 = torch.randn(5, 6)
print(fn._cache[0][0].failing_guards(a3, b3))
# [Guard(args[0].shape == (3, 4)), Guard(args[1].shape == (3, 4))]
```

The mechanism is small: the first call pays the compile tax, identical calls pay only guard checks, and the cache grows one entry per signature. Section 9 walks one function through exactly this life cycle. A function seeing a different shape on every call misses every time, turning `@compile` into pure overhead. That is why **recompilation rate** is the first thing to check when `torch.compile` fails to speed up a workload.

![One cache scan: the wrapper walks entries top-to-bottom and runs the first whose guards all pass. Entries further down never get checked on a hit.](/assets/img/mini-dynamo/cache-and-guards.svg)

Real Dynamo makes the same trade-off:

|                  | First call           | Subsequent calls (cache hit) | Shape change (cache miss) |
| :--------------- | :------------------- | :--------------------------- | :------------------------ |
| **Cost**         | Full trace + compile | Guard checks only            | Full retrace + compile    |
| **Typical time** | Milliseconds         | Microseconds                 | Milliseconds              |

<aside markdown="1">

**Real Dynamo's guards are far more extensive**, checking type IDs, object identity, dict version tags, global values, tensor strides, and more. They are also implemented in C for speed. Our Python lambdas demonstrate the concept.

</aside>

---

## 9. The `compile()` Decorator

The top-level API ties together all five pipeline stages. The snippet below strips the decorator down to its cache loop. The real `mini_dynamo/__init__.py` adds backend selection, a backend-name check, and rejection of non-tensor runtime arguments:

```python
def compile(fn=None, *, backend="python"):
    # The cache lives in this closure, so each @compile'd function gets its
    # own. Entries are appended in the order they were compiled; we scan
    # from the front on every call.
    cache = []

    @functools.wraps(fn)
    def wrapper(*args):
        if not all(isinstance(arg, torch.Tensor) for arg in args):
            raise TypeError("mini_dynamo.compile only supports tensor arguments")

        # Fast path: walk the cache and run the first entry whose guards
        # all pass on the current args. This is the path every steady-state
        # call takes.
        for guard_set, compiled_fn in cache:
            if guard_set.check_all(*args):
                return compiled_fn(*args)       # Cache hit → fast path

        # Slow path: nothing in the cache matches, so run the full pipeline
        # and append a new entry. The next call with the same signature
        # will hit it in the loop above.
        graph = SymbolicInterpreter(fn, args).run()         # STEP 1: Trace
        compiled_fn, _ = compile_graph(graph)                # STEP 2: Compile
        guard_set = GuardSet.from_example_inputs(args)       # STEP 3: Guard
        cache.append((guard_set, compiled_fn))               # STEP 4: Cache
        return compiled_fn(*args)                            # STEP 5: Execute

    # Exposed only so the examples below can inspect the cache and compare
    # against the original eager function.
    wrapper._cache = cache
    wrapper._original = fn
    return wrapper
```

The previous sections built each component in isolation: the interpreter, the graph, the compiler, the guards. Now one call can run through the wrapper, each artifact inspected as it appears.

```python
import torch
import mini_dynamo

@mini_dynamo.compile
def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()

a = torch.randn(3, 4)
b = torch.randn(3, 4)
```

One reminder from Section 1: the wrapper accepts only positional tensor arguments, and the traced body cannot use keyword calls. Constants like the `2` in `x * 2` are fine; they live in the function body, visible during tracing.

### Step 1: Trace

The wrapper's cache is empty, so we fall into the slow path. `SymbolicInterpreter(fn, (a, b)).run()` walks `fn`'s bytecode, recording every tensor operation as a `Node`. It returns a `Graph`:

```
Graph:
  x = placeholder
  y = placeholder
  add_0 = torch.add(x, y)
  mul_0 = torch.mul(add_0, 2)
  sum_0 = mul_0.sum()
  return sum_0
```

Notice what disappeared: the intermediate locals `z` and `w`, and every `LOAD_GLOBAL torch` lookup. The Python source has been flattened into a straight-line DAG of tensor operations. The constant `2` is inlined into `torch.mul`'s args rather than becoming a node. The graph works as an IR because it describes only the tensor ops and their wiring, stripped of everything the compiler does not need.

### Step 2: Compile

`compile_graph(graph)` walks those nodes, emitting one line of Python per operation. It returns a callable plus the source string, small enough to read in full:

```python
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)
    mul_0 = __fn_mul_0(add_0, 2)
    sum_0 = mul_0.sum()
    return sum_0
```

The `__fn_add_0` and `__fn_mul_0` names are keys into the namespace handed to `exec()`. That dict looks like `{"__fn_add_0": torch.add, "__fn_mul_0": torch.mul}`; it becomes the globals of the materialized function. Each op still travels through `torch.add` and the full PyTorch dispatcher. Each op still launches its own kernel. Nothing has been fused, and no dispatcher work has been avoided.

The Python backend produces a faithful, standalone callable doing what the captured graph says. Kernel fusion and dispatcher elimination arrive when the _same_ graph goes to Inductor instead, the subject of Section 10.

### Step 3: Guard

`GuardSet.from_example_inputs((a, b))` inspects each tensor argument, building three lambda guards per tensor: shape, dtype, and device.

```
GuardSet([
  args[0].shape == (3, 4)
  args[0].dtype == torch.float32
  args[0].device == cpu
  args[1].shape == (3, 4)
  args[1].dtype == torch.float32
  args[1].device == cpu
])
```

These six predicates form the contract: the fresh `compiled_fn` stays valid exactly as long as they hold. Nothing ties the guard set to the tensors `a` and `b` themselves; future arguments simply must satisfy its checks.

### Step 4: Cache

The pair `(guard_set, compiled_fn)` gets appended to the cache list. After this first call:

```python
print(len(fn._cache))    # → 1
```

The cache now has one entry. It lives in the wrapper's closure, one cache per decorated function, and its order matters. Every later call scans from index 0, running the first entry whose guards all pass.

### Step 5: Execute

Finally the wrapper calls `compiled_fn(a, b)`. The result matches the original eager function; dispatch was reorganized, the computation untouched:

```python
compiled_fn(a, b) == fn._original(a, b)   # → tensor(True)
```

One first call ran all five steps. The first-call latency, a few milliseconds here, is mostly spent in steps 1–3; step 5 takes microseconds.

### Second and Third Calls

The structure pays off on later calls. On the **second call** with the same shape, dtype, and device, the wrapper finds `check_all` returning `True` on the first entry. It jumps directly to `compiled_fn(*args)`, steps 1–4 skipped entirely. The cache is still length 1.

On the **third call** with `(5, 6)` tensors, `check_all` returns `False` on every existing entry; the shape guards fail. The wrapper falls through to the slow path again, appending a freshly traced and compiled entry. Now:

```python
print(len(fn._cache))    # → 2
```

Future calls scan both entries in order. A `(3, 4)` call hits entry 0 and a `(5, 6)` call hits entry 1; any brand-new shape falls through to a third compile.

That is the full pipeline in motion. Five stages produce five artifacts: a `Graph`, a `compiled_fn`, a `GuardSet`, a cache list, and a tensor result. Real `torch.compile` adds machinery at every stage, from keyword arguments and dynamic shapes to C-level guards and graph breaks. The spine keeps the same shape: **trace → compile → guard → cache → execute**.

---

## 10. Where Does Speedup Come From?

With the full system built, we can ask: **how much faster is it?**

Graph capture on its own produces no meaningful speedup; the win comes from what an optimizing backend does with the graph. A common description says `torch.compile` "removes Python overhead." That phrase bundles several distinct costs, everything between a user's `x + y` and the kernel running on the GPU.

### Where the Time Goes

A rough per-op cost decomposition for an elementwise op in eager PyTorch, on a modern GPU, looks like this. Exact numbers vary with device, driver, PyTorch version, and tensor size; the ordering is what matters:

| Cost                                                 | Typical scale per op        | Who pays it                         |
| :--------------------------------------------------- | :-------------------------- | :---------------------------------- |
| CPython bytecode dispatch                            | tens of nanoseconds         | The interpreter                     |
| Python-level method resolution, `__torch_function__` | hundreds of nanoseconds     | CPython + PyTorch's Python bindings |
| PyTorch C++ dispatcher (device, autograd, vmap, …)   | a few microseconds          | libtorch                            |
| Kernel launch onto the CUDA / MPS stream             | 5–20 microseconds           | The GPU driver                      |
| The kernel itself                                    | nanoseconds to milliseconds | The GPU                             |

The first two rows are what most people mean by "Python overhead." They are also the _smallest_ rows. Our Python backend touches only those, pre-resolving lookups so each op skips one `LOAD_GLOBAL` + `LOAD_ATTR` pair. Nothing below that line changes. Every op still boxes its arguments into PyObjects and still traverses libtorch's dispatch-key logic. Each still waits on its own kernel launch.

The numbers follow that pattern, though the exact outcome depends on backend and shape. Running the CPU benchmark `examples/benchmark.py` on one Slurm node (Python 3.10.12, PyTorch 2.10.0+cu128) produced this 256×256 no-guard result for the raw generated Python function. These are CPU tensors; Inductor is generating fused C++ kernels here, not GPU code:

```
Eager (original):             268.9 us
mini_dynamo (python):         267.9 us   (1.00x -- mostly noise)
torch.compile (inductor):      68.0 us   (3.95x)
```

The CUDA benchmark needs one subtle precaution: reset Dynamo between per-shape `torch.compile` runs. A shape sweep of one Python function can otherwise push Dynamo into a generalized dynamic-shape path, contaminating later timings. With that reset in place, `examples/benchmark_mps.py` selected `cuda`, producing this result on a single NVIDIA H100 80GB:

```
Tensor size: 2048x2048
Eager:                       160.4 us
mini_dynamo (python):        160.5 us   (1.00x)
mini_dynamo (jit):            33.6 us   (4.78x)
torch.compile (inductor):     30.6 us   (5.25x)
```

Treat these as measurements of this repository's toy benchmarks, not universal results. The stable lesson is narrower and more useful. The Python backend barely moves the needle. The JIT backend can remove Python↔C++ boundary overhead without fusing kernels. Inductor is the only path here that can change the kernel structure. Whether that wins depends on operation mix, tensor size, device backend, and PyTorch build.

![Where the time goes for 11 chained elementwise ops. When Inductor wins, the win comes from collapsing multiple launches into fewer fused kernels.](/assets/img/mini-dynamo/cost-decomposition.svg)

The Python backend's result sits within noise. We saved a handful of bytecodes per op, savings the lower layers dwarf.

The JIT backend's wins, when they appear, do _not_ come from bytecode dispatch. `torch.jit.trace` wraps the generated function into a single TorchScript graph call; from Python's point of view, the whole chain becomes one call into C++. Python drops out of the loop between ops, and some per-op dispatcher work gets amortized. We are nibbling at rows 2–3 of the table, not row 1.

### Where Inductor Can Win

When Inductor wins, the speedup comes from a different layer. It operates _below_ the dispatcher rather than shaving interpreter instructions above it, and it relies on a captured graph as input:

- **Kernel fusion.** Inductor can generate a single Triton (GPU) or C++ (CPU) kernel for a whole chain of memory-bound ops. Eager runs one kernel per operation; for the `many_ops` benchmark above, that means 11 elementwise kernels plus the final reduction. Each kernel reads from HBM, computing one op before writing back. Fusion reduces those round-trips, often a large share of the speedup on favorable elementwise chains.
- **Launch overhead collapse.** Even after fusion, each launch still costs microseconds. When the same shapes recur, as in a training loop's steady state, CUDA graph integration records the launches once for replay as one stream op.
- **Memory planning.** With a full graph in hand, Inductor plans intermediate buffers once, avoiding the per-op allocator churn of eager.

None of these live in our mini-dynamo backend. They require the graph as input. They attack the biggest rows of the cost table: the dispatcher, the launch, and the kernel itself, where the microseconds live.

### Dynamo Captures, Inductor Optimizes

**Dynamo and Inductor solve different problems.** Dynamo captures the graph, bringing almost no performance gain on its own. Inductor optimizes the graph, the source of the meaningful speedup in many deep-learning workloads. The bytecode tracer exists to hand an optimizing backend a graph it can fuse and lower. Our mini-dynamo replaces only the Dynamo part. Because we produce a compatible graph, the real Inductor backend can plug in, letting us measure the backend's behavior directly.

The path runs from our graph to an `fx.GraphModule`, then through ATen lowering, and finally into `compile_fx_inner`, Inductor's internal entry point. This is a private PyTorch API; the repository therefore pins PyTorch `2.10.0`, treating the integration as educational rather than stable surface area. For the straight-line programs covered by the parity tests, this produces the same ATen graph as real Dynamo's export path. The generated Inductor kernels match too, on the tested backend. The tests validate that narrow claim, nothing broader about arbitrary programs, devices, or configurations.

```python
def mini_dynamo_to_inductor(fn, *example_inputs):
    # 1. Trace with our symbolic interpreter, producing a mini-dynamo Graph
    #    whose nodes call torch.add, torch.mul, etc.
    graph = SymbolicInterpreter(fn, example_inputs).run()

    # 2. Repackage our graph as a torch.fx.GraphModule, which is the format
    #    Inductor's pipeline accepts.
    gm = to_fx_graph_module(graph)

    # 3. make_fx re-traces gm one more time, this time under PyTorch's ATen
    #    dispatch layer. Surface-level ops (torch.add) get rewritten to their
    #    canonical ATen counterparts (torch.ops.aten.add.Tensor). Inductor
    #    works on ATen, not on the Python-facing torch API.
    aten_gm = make_fx(gm)(*example_inputs)

    # 4. Hand the ATen graph to Inductor's private entry point, which does
    #    the actual kernel fusion and code generation.
    compiled = compile_fx_inner(aten_gm, list(example_inputs))

    # Inductor's callable uses an internal calling convention: it receives
    # one list of tensor inputs and returns a tuple of outputs. Wrap it so the
    # result behaves like the original Python function.
    def wrapper(*args):
        return compiled(list(args))[0]

    return wrapper
```

---

## 11. What We Left Out

Mini-dynamo demonstrates the architecture of TorchDynamo. Real Dynamo is a vastly more complex system. Here are the most important gaps.

### PEP 523 Frame Evaluation

Real Dynamo doesn't use `dis.get_instructions()`. It installs a **C-level frame evaluator** via PEP 523, which intercepts every Python frame before CPython's interpreter runs it. Nothing changes about how you call your functions. On top of that interception, the tracer supports:

- **Function inlining:** When `fn()` calls `helper()`, Dynamo traces _into_ the callee by recursively walking its bytecode; during tracing, `helper`'s frame never actually runs. The capture stays a single unified graph. Our bytecode walker sees only the top-level function.

- **Graph breaks:** When Dynamo hits an unsupported operation, a `print()` or an exotic data structure, it can break the graph. It compiles what it has, resuming a fresh trace after the unsupported bit runs in normal Python. Our interpreter has no such machinery: unsupported bytecodes raise `NotImplementedError`, and opaque helper calls become plain call nodes at best.

### Control Flow

We skip all jump instructions (`JUMP_IF_TRUE`, `FOR_ITER`, and the rest). Real Dynamo handles control flow by specializing. If a branch condition is knowable at trace time, `x.shape[0] > 5` for instance, it evaluates the condition, tracing only the taken branch under a guard.

### Dynamic Shapes

Our guards require exact shape matches. Real Dynamo supports **dynamic shapes**, symbolic integers standing in for unknown dimensions. This avoids recompilation when batch size changes, at the cost of heavier guard logic and symbolic reasoning.

### 200+ VariableTracker Subclasses

Our four types cover tensors, constants, torch functions, and tensor methods. Real Dynamo has trackers for lists, dicts, ranges, slices, iterators, `nn.Module` instances, user-defined classes, closures, generators, and more.

---

## 12. Summary

`torch.compile` is a well-structured pipeline:

1. **Intercept** Python execution at the bytecode level
2. **Replay** each instruction symbolically, recording tensor operations into a graph
3. **Compile** the graph with an optimizing backend
4. **Guard** against changes in input metadata
5. **Cache** the result for fast reuse

Each component reduces to something familiar: a CPython emulator, an IR, a code generator, and a handful of boolean predicates. Each is small enough to understand in isolation. Together they explain how `torch.compile` speeds up PyTorch programs. In this mini implementation the graph-capture machinery is the educational focus; the large speedups arrive once the captured graph meets an optimizing backend like Inductor.

<div class="l-body" markdown="1">

_The full source code for mini-dynamo is at [github.com/itsdaniele/torchdynamo-mini](https://github.com/itsdaniele/torchdynamo-mini). Every module is heavily commented, designed to be read linearly._

</div>

---

<div class="appendix" markdown="1">

## Appendix: File Map

| File                                  | Purpose                                                     |
| :------------------------------------ | :---------------------------------------------------------- |
| `mini_dynamo/__init__.py`             | The `compile()` decorator -- ties together all five stages  |
| `mini_dynamo/symbolic_interpreter.py` | The bytecode walker -- CPython emulator on VariableTrackers |
| `mini_dynamo/variable_tracker.py`     | Four symbolic value types                                   |
| `mini_dynamo/graph.py`                | The computation graph IR (`Node` + `Graph`)                 |
| `mini_dynamo/compiler.py`             | Code generation backends (Python + JIT)                     |
| `mini_dynamo/guards.py`               | Guard creation and checking                                 |
| `examples/benchmark.py`               | Performance analysis: where speedup comes from              |
| `examples/benchmark_mps.py`           | GPU benchmark: Python vs JIT vs Inductor                    |
| `examples/benchmark_transformers.py`  | Transformer-style fusion pattern benchmark                  |
| `examples/inductor_integration.py`    | Plugging into the real Inductor backend                     |

</div>
