---
layout: post
title: "minidynamo: A Tiny torch.compile from Scratch"
description: Rebuilding the core ideas behind torch.compile with a small TorchDynamo-style tracer.
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

  #markdown-content .minidynamo-figure {
    margin: 2.5rem 0;
  }

  #markdown-content .minidynamo-figure picture {
    display: block;
  }

  #markdown-content .minidynamo-figure picture,
  #markdown-content .minidynamo-figure figcaption {
    max-width: 760px;
    margin-left: auto;
    margin-right: auto;
  }

  #markdown-content .compile-animation {
    margin: 2rem 0;
  }

  #markdown-content .compile-animation iframe {
    display: block;
    width: 100%;
    height: 710px;
    border: 1px solid var(--global-divider-color);
    border-radius: 14px;
  }

  #markdown-content .compile-animation figcaption {
    margin-top: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  @media (max-width: 700px) {
    #markdown-content .compile-animation iframe {
      height: 930px;
    }
  }

  #markdown-content .minidynamo-figure img {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
    margin: 0;
    border-radius: 10px;
  }

  #markdown-content .minidynamo-figure figcaption {
    margin-top: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.5;
    color: var(--global-text-color);
  }

  @media (max-width: 700px) {
    #markdown-content .minidynamo-figure picture,
    #markdown-content .minidynamo-figure figcaption {
      max-width: 340px;
      margin-left: auto;
      margin-right: auto;
    }
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

<div class="l-body" markdown="1">

_When you run code under `@torch.compile`, a lot happens under the hood: PyTorch intercepts Python bytecode, captures a graph of tensor operations, hands that graph to an optimizing compiler, and caches the result. This article rebuilds the pieces of that system from scratch with [minidynamo](https://github.com/itsdaniele/minidynamo), a small implementation that exposes the moving parts._

Note: _Coding agents generated a good part of the code. I then read the implementation line by line, added tests, wrote benchmark scripts, and used the toy system to build intuition._

</div>

<aside markdown="1">

**TL;DR**

- `torch.compile` is a pipeline: **trace → compile → guard → cache → execute**. This post rebuilds it in ~1,500 lines of readable Python ([minidynamo](https://github.com/itsdaniele/minidynamo)).
- TorchDynamo captures graphs at the **bytecode level**: it re-implements CPython's interpreter over symbolic values and records tensor operations into a graph (Sections 3–6).
- **Guards** — checks on input metadata and tracing assumptions — are the contract that decides when cached compiled code can be reused. A high recompilation rate is one reason `torch.compile` disappoints (Section 8).
- Our Python graph-replay backend produces **no meaningful speedup** in the benchmark below. The wins come from what an optimizing backend like Inductor does with the captured graph — and our mini graphs can drive the real Inductor to prove it (Section 10).

</aside>

## 1. The Big Picture

PyTorch makes modeling code convenient: in a module `forward()` method you can use `if` statements, call helper functions, print for debugging, and rely on ordinary Python control flow. PyTorch calls this **eager mode** — each tensor operation runs the moment Python reaches it, dispatched to the device one at a time. A chain of eleven elementwise operations (like the benchmark function we will use later) can mean eleven kernel launches, repeated intermediate memory reads and writes, and a CPU-side dispatcher between each operation. This flexibility was probably the main reason PyTorch won over TensorFlow, but it leaves performance on the table.

PyTorch 2.0 introduced `torch.compile` to keep the flexible Python programming model while clawing back that performance. When you wrap a function in `torch.compile()` and call it, PyTorch captures a graph of your tensor operations (TorchDynamo's job), then hands that graph to an optimizing compiler (Inductor) that can fuse multiple operations into fewer kernels. Whenever the function is called again, PyTorch tries to reuse the same optimized graph, as long as the assumptions made during tracing still hold.

> **Why this is faster on GPUs:** eager PyTorch typically launches separate GPU kernels for successive elementwise operations; an operation can also require zero or multiple kernels. A fusible chain like `add → relu → mul` repeatedly reads tensors from GPU memory, writes intermediate tensors back, and pays launch overhead each time. Once Dynamo captures the whole chain as a graph, Inductor can fuse those operations into fewer kernels. In the best case, the GPU reads the data once, keeps intermediates in registers, does more work per launch, and writes the final result back once.

All of this starts with the hard part: symbolically executing arbitrary Python and PyTorch code.

### The graph capture problem

Capturing a graph of tensor operations from a Python function is hard. PyTorch went through years of earlier approaches, each useful and each with painful trade-offs, before landing on TorchDynamo:

- **`torch.jit.trace`** (2018): Run the function with example inputs and record the tensor operations that execute. Problem: it observes one execution path, not the Python program. If an `if` statement takes one branch for the example inputs, the trace contains only that branch and reuses it later. Code that depends on tensor values, Python-side control flow, or input-dependent shapes can produce wrong or over-specialized graphs.

- **TorchScript** (`torch.jit.script`, 2018): Parse Python source into a statically analyzable, typed subset of Python. This could preserve control flow in a way tracing could not. Problem: users had to write code that the TorchScript compiler understood, and real PyTorch programs often used Python features outside that subset. Many models needed code changes before they could be scripted.

- **FX Tracing** (`torch.fx.symbolic_trace`, 2021): Execute Python with `Proxy` objects standing in for tensors, and record operations performed on those proxies into an FX graph. Problem: the tracer still runs ordinary Python. If Python tries to branch on a proxy value, iterate over it, or use it where a concrete value is required, tracing fails or specializes to the example-time behavior.

- **Lazy Tensors**: Record tensor operations at the tensor/backend level and defer execution until the result is needed. This gives the backend a graph it can optimize. Problem: Python has already run by the time those tensor ops are recorded. Lazy tensors can optimize tensor execution, but they do not solve the problem of intercepting arbitrary Python frames, understanding Python control flow, or skipping Python work on later calls.

**TorchDynamo** (the thing that powers torch.compile) took a different approach: it works at the **bytecode level**, below Python source and above the C++ dispatcher. Using PEP 523's Frame Evaluation API, Dynamo installs a C-level hook that intercepts eligible Python frames while compilation is active, _before_ CPython's interpreter runs them. It then walks the bytecode instructions, symbolically evaluating them to identify tensor operations and record them into an FX graph.

### Calling `torch.compile`

<figure class="compile-animation" id="compile-walkthrough">
  <iframe id="minidynamo-flow" src="/assets/minidynamo-flow/index.html?v=3" title="Inside minidynamo: first call, cache hit, and recompilation" loading="lazy"></iframe>
  <figcaption>This illustrates minidynamo’s Python backend; animation speed does not represent execution time. <a href="/assets/minidynamo-flow/index.html?v=3" target="_blank" rel="noopener">Open the animation</a>.</figcaption>
</figure>
<script src="/assets/minidynamo-flow/embed.js" defer></script>

1. **Trace**: Dynamo intercepts the Python frame via PEP 523 and walks the bytecode. Tensor operations are recorded into an FX graph. Much of the surrounding Python logic is handled outside the graph: some values are evaluated concretely, some assumptions become _guards_, and unsupported regions can trigger _graph breaks_. We will see in detail what this means.

2. **Compile**: The FX graph is passed to a compiler backend. The default backend is Inductor, which can generate Triton kernels on CUDA and C++ kernels on CPU, or call existing libraries.

3. **Guard**: Dynamo records the assumptions made during tracing: tensor shapes, dtypes, devices, and values of Python variables used in control flow.

4. **Cache**: The compiled function and its guards are stored together. On subsequent calls, if all guards pass, the compiled function is reused without re-tracing.

5. **Execute**: If guards pass, run the compiled function. If no cached entry matches (e.g., tensor shape changed), re-trace and compile, subject to recompilation limits. Dynamic-shape graphs can also accept some shape changes without recompiling.

**Trace once, execute many times.** The first call is slow (bytecode analysis + compilation). Later calls with matching inputs skip tracing and reuse the compiled result.

### What we'll build

We will build this pipeline as a small, readable Python codebase. Our implementation, [_minidynamo_](https://github.com/itsdaniele/minidynamo), is a deliberately tiny TorchDynamo-style tracer. It captures the same core ideas while leaving out the machinery needed for arbitrary real-world PyTorch programs.

<aside markdown="1">

**Scope.** Real TorchDynamo handles control flow, nested function calls, user-defined classes, graph breaks, dynamic shapes, and many Python opcodes. minidynamo handles straight-line tensor computations over plain, strided tensors passed as positional arguments, constants in the function body, basic arithmetic, an explicit set of common `torch` functions, and tensor methods that can be called with positional arguments during tracing. The tests cover methods such as `.sum()` and `.mean()`. Custom tensor attributes, arbitrary helpers, in-place operations, random operations, and tensor-to-Python data conversions are rejected. We skip PEP 523 and the machinery built on top of it: real Dynamo hooks into CPython frame evaluation and re-implements a large chunk of Python execution logic in Python. We also skip AOTAutograd and training-graph lowering; the examples focus on forward computations.

</aside>

<aside markdown="1">

**Runtime requirements.** The repository pins Python `3.10.x` and PyTorch `2.10.0`. That narrow version range is intentional: CPython bytecode changes across Python releases, and this educational interpreter only implements the Python 3.10 opcodes used in the examples. The public `@minidynamo.compile` decorator also supports only positional tensor arguments; runtime non-tensor arguments and keyword arguments are deliberately out of scope.

</aside>

---

## 2. Architecture: The Five Components

minidynamo has five pipeline stages spread across six core modules, plus an optional Inductor bridge:

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
   graph.py         inductor.py → Inductor
```

The package backend in `minidynamo/compiler.py` only implements the Python and JIT paths exposed by `@minidynamo.compile`. The Inductor path appears later through `minidynamo/inductor.py`, which the example scripts use to convert the mini graph to FX, lower it to ATen, and call a private Inductor entry point.

The table maps each component to its TorchDynamo counterpart:

| minidynamo                | Real TorchDynamo                                                | Role                                                                                                                                                                                                                                                                                                           |
| :------------------------ | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `symbolic_interpreter.py` | `InstructionTranslator` (~5,000 lines in `symbolic_convert.py`) | Walk bytecodes, build the graph. The heart of the system.                                                                                                                                                                                                                                                      |
| `variable_tracker.py`     | `VariableTracker` hierarchy (200+ subclasses across ~20 files)  | Symbolic values on the interpreter's stack. Tell the interpreter what kind of thing each value is (tensor? constant? torch function?) so it can decide whether to record a graph node or evaluate concretely.                                                                                                  |
| `graph.py`                | `torch.fx.Graph` + `torch.fx.Node`                              | The computation graph IR. A flat list of nodes, each describing one operation. This is the output of tracing and the input to compilation.                                                                                                                                                                     |
| `compiler.py`             | Compiler backends (Inductor, etc.)                              | Takes a finished graph and produces a callable. Our simple backend generates Python source with pre-resolved names in an `exec()` namespace. Real Inductor generates Triton GPU kernels and C++ CPU kernels. The later Inductor integration example adds a small converter from our graph format to FX Graphs. |
| `guards.py`               | `torch._dynamo.guards` (C-accelerated)                          | Boolean predicates that encode the assumptions made during tracing. If guards pass on new inputs, the cached compiled function can be reused.                                                                                                                                                                  |
| `__init__.py`             | `torch._dynamo.convert_frame`                                   | The orchestrator. Manages the guard-cache loop: check guards → hit? run cached fn. Miss? trace → compile → guard → cache.                                                                                                                                                                                      |

Two components deserve special attention because their roles are easy to confuse:

**The Graph is the output.** It is a pure data structure: a list of nodes describing which tensor operations to perform. It does not execute operations itself; the node types and their dependencies specify the computation. After tracing, the compiler receives it as a recipe.

**VariableTrackers are the process.** They are the symbolic values that live on the interpreter's stack _during_ tracing. They tell the interpreter what type of thing each value is, so it can decide what to do with each bytecode instruction. The backend consumes the graph those wrappers helped build. They are scaffolding for the graph, not part of the final product.

We need both because CPython's bytecodes are untyped. When the interpreter sees `BINARY_ADD`, it doesn't know if it's adding two tensors (→ record `torch.add` in the graph) or two integers (→ just compute the result). VariableTrackers carry the type information that lets it make this decision. The Graph records the decisions that were made.

---

## 3. CPython Is a Stack Machine

To understand our symbolic interpreter, you need one fact about CPython: **it's a stack-based virtual machine.** Every Python function compiles to a sequence of bytecode instructions that manipulate a _value stack_ and a _locals array_.

For `z = x + y`, CPython emits:

| Instruction    | Stack (after) | Effect                     |
| :------------- | :------------ | :------------------------- |
| `LOAD_FAST x`  | `[x]`         | Push local variable `x`    |
| `LOAD_FAST y`  | `[x, y]`      | Push local variable `y`    |
| `BINARY_ADD`   | `[x+y]`       | Pop two, push their sum    |
| `STORE_FAST z` | `[]`          | Pop and store in local `z` |

Our symbolic interpreter mirrors this stack discipline with its own stack, locals, and dispatch loop; it simplifies CPython's method-call convention. The central difference: instead of real Python values, the stack holds _symbolic wrappers_ that record operations into a graph.

---

## 4. VariableTrackers: The Symbolic Values

Every value in our interpreter is a `VariableTracker`: _"I'm not a real value. I'm a description of a value that will exist at runtime."_

We need exactly four types:

### TensorVariable

The most important type. It holds a _graph node_ (its identity in the computation graph) and an _example value_ (a real tensor with the same shape/dtype/device, used for metadata propagation).

When the interpreter sees `x + y` where both are `TensorVariable`s, it doesn't compute the runtime result that the user asked for. Instead, it:

1. Creates a new `Node` in the graph: `call_function(torch.add, (x.node, y.node))`
2. Computes an example output for metadata propagation: `torch.add(x.example_value, y.example_value)`
3. Returns `TensorVariable(new_node, example_output)`

The example value flows forward through every operation, so at any point during tracing, we know the exact shape, dtype, and device of every intermediate tensor. We are still running individual example tensor ops during tracing to propagate metadata, but the output of tracing is the graph, not the eager result of the original function. The generated callable then runs those operations again. This is why we reject mutation and random operations before executing them. Real Dynamo generally uses FakeTensors to propagate metadata without running ordinary tensor kernels on the input data.

### ConstantVariable

A value fully known at trace time: the `2` in `x * 2`, a dtype like `torch.float32`, a shape tuple. Constants don't become graph nodes -- they're inlined directly into the operations that use them.

### TorchVariable

A reference to the `torch` module or one of its functions. When the interpreter encounters `LOAD_GLOBAL torch`, it pushes `TorchVariable(torch)`. When it then encounters `LOAD_ATTR relu`, it resolves `torch.relu` and pushes `TorchVariable(torch.relu)`.

### MethodVariable

A bound tensor method like `x.sum`. Created when the interpreter accesses a method on a `TensorVariable`. It remembers _which tensor_ and _which method_, so when called, it can record the correct graph node.

<aside markdown="1">

**Real Dynamo has over 200 VariableTracker subclasses**, covering lists, dicts, iterators, ranges, user-defined classes, `nn.Module`s, and more. Our four types suffice for the supported subset of straight-line tensor code.

</aside>

---

## 5. The Graph IR

As the interpreter runs, it records operations into a `Graph` -- an ordered list of `Node` objects that form a DAG of the computation. This is a simplified version of `torch.fx.Graph`.

Each `Node` has five key fields (schema excerpt):

```python
class Node:
    name: str       # Unique identifier, e.g. "add_0", "x"
    op: str         # One of: "placeholder", "call_function", "call_method", "output"
    target: Any     # What to call (e.g., torch.add) or method name (e.g., "sum")
    args: tuple     # Positional arguments -- can reference other Nodes
    kwargs: dict    # Keyword arguments (the IR supports these; the tracer does not capture keyword calls)
```

Nodes come in four flavors:

| `op`            | Meaning                    | Example                   |
| :-------------- | :------------------------- | :------------------------ |
| `placeholder`   | Function input             | `x = placeholder`         |
| `call_function` | A function call on tensors | `add_0 = torch.add(x, y)` |
| `call_method`   | A method call on a tensor  | `sum_0 = mul_0.sum()`     |
| `output`        | The return value           | `return sum_0`            |

We will use this function throughout the walkthrough:

```python
def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()
```

Its captured graph is:

```
Graph:
  x = placeholder
  y = placeholder
  add_0 = torch.add(x, y)
  mul_0 = torch.mul(add_0, 2)
  sum_0 = mul_0.sum()
  return sum_0
```

<figure class="minidynamo-figure">
  <picture>
    <source media="(max-width: 700px)" srcset="/assets/img/minidynamo/graph-ir-mobile.svg">
    <img src="/assets/img/minidynamo/graph-ir.svg" alt="Python source and its captured graph: x and y feed torch.add, then torch.mul with constant 2, then .sum(), then the output.">
  </picture>
</figure>

Notice the `2` in `torch.mul(add_0, 2)` -- it's a plain Python integer, not a `Node`. Constants are inlined into the args of the operations that consume them.

---

## 6. The Symbolic Interpreter

The `SymbolicInterpreter` ties the previous pieces together. It reads bytecode, manipulates `VariableTracker`s on a stack, and writes nodes into the `Graph`. Everything else feeds into this loop or consumes its output.

When you run a Python function normally, CPython walks the bytecode and _executes_ each instruction on real objects: integers get added, tensors get multiplied, methods get invoked. We use the same bytecode and stack discipline, but we care about the _shape_ of the computation rather than the user-facing return value. We re-implement enough of CPython's interpreter to produce a **graph**.

### Two Interpreters in Parallel

Picture two interpreters running side by side on the same bytecode, one real and one symbolic:

|                                 | CPython's interpreter                             | Our symbolic interpreter                                                                 |
| :------------------------------ | :------------------------------------------------ | :--------------------------------------------------------------------------------------- |
| **Stack holds**                 | Real Python objects                               | `VariableTracker`s                                                                       |
| **Locals hold**                 | Real values                                       | `VariableTracker`s                                                                       |
| **`BINARY_ADD` on two tensors** | Dispatches tensor addition, produces a new tensor | Records `torch.add(x, y)`, computes an example result, and pushes a new `TensorVariable` |
| **`BINARY_ADD` on two ints**    | Computes `a + b`                                  | Computes `a + b`; constants are evaluated concretely                                     |
| **`CALL_METHOD x.sum()`**       | Invokes the bound method                          | Records `x.sum()` and computes an example result                                         |
| **Unsupported opcode**          | Executes it                                       | Raises `NotImplementedError`                                                             |
| **Final output**                | A return value                                    | A finished `Graph`                                                                       |

CPython operates on values; minidynamo operates on _descriptions_ of values. At every bytecode step, the symbolic interpreter makes one decision: **record** this operation into the graph, or **evaluate** it concretely on constants and metadata we already know. Repeating that decision across instructions produces the captured graph.

### The Three Pieces of State

Our interpreter carries three central pieces of state through its run:

- **`self.stack`**: a list of `VariableTracker`s. `LOAD_*` opcodes push to it; `BINARY_*`, `CALL_*`, `STORE_*`, and the rest consume it.
- **`self.locals`**: a dict mapping variable names to `VariableTracker`s. It starts with the function arguments and changes on `STORE_FAST`.
- **`self.graph`**: the `Graph` being built. It grows each time a tensor operation gets recorded.

If you snapshotted these after every instruction, you'd see the stack, locals, and graph evolve through the trace. The interpreter also tracks the global namespace and guards on its assumptions.

### Correspondence to Real Dynamo

Our `SymbolicInterpreter` is the direct analogue of TorchDynamo's `InstructionTranslator` (in `torch/_dynamo/symbolic_convert.py`). The two share the same skeleton: a value stack, a locals dict, an FX-style graph being mutated, and one handler per opcode. The differences are in scope, not in kind:

- Real Dynamo handles ~160 opcodes including jumps, comparisons, exceptions, closures, and generator machinery. We handle a small straight-line subset.
- Real Dynamo _inline-traces_ into called functions. When `fn()` calls `helper()`, the tracer recursively walks the callee's bytecode too — during tracing, `helper`'s frame never actually runs — producing a single unified graph. Our walker only sees top-level bytecode and rejects arbitrary helper calls.
- Real Dynamo emits **guards** on-the-fly as it makes assumptions (e.g. "I looked at `x.shape[0]` and treated it as `32`, so guard on that"). We create input and execution-context guards before tracing, then add guards for global bindings and accessed `torch` attributes during tracing.
- Real Dynamo can **break the graph** when it hits something unsupported: compile what it has so far, let the hard part run in plain Python, and resume tracing after. Our interpreter halts with `NotImplementedError`.

From here, we follow the walker from initialization through dispatch, then trace one example and finish with the call-dispatch logic that decides whether a call becomes a graph node or a concrete Python call.

### Initialization

Tracing starts with an empty value stack and a graph containing one placeholder per tensor argument. Each local parameter, such as `x`, points to a `TensorVariable` holding that placeholder and its example tensor. The interpreter also records input and execution-context guards and keeps access to the function's global and builtin namespaces.

The state is ready before the first bytecode instruction runs: `LOAD_FAST x` can retrieve its symbolic value, and tensor operations can append nodes to the graph. See [the initializer](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L174) for the parameter binding and validation.

### The Main Loop

The interpreter fetches instructions one by one and dispatches each to its opcode handler. In **pseudocode**:

```text
for instruction in bytecode(fn):
    handler = lookup_handler(instruction.opname)
    if handler is missing:
        reject this function
    handler(instruction)
return graph
```

Each handler updates the stack, locals, or graph. For example, the tensor–tensor case of `BINARY_ADD` is just these steps (**pseudocode**):

```text
right = stack.pop()
left = stack.pop()
node = graph.call_function(torch.add, (left.node, right.node))
example = torch.add(left.example_value, right.example_value)
stack.push(TensorVariable(node, example))
```

The node records the operation; the example result carries its metadata forward. If both operands are constants, the handler instead computes their sum and pushes a `ConstantVariable`, without adding a graph node. The implementation splits this work between [the dispatch loop](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L199), [the binary-op handler](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L504), and [tensor-call recording](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L417).

### Walking Through an Example

Trace `fn(x, y)`, where `fn` computes `z = x + y; w = z * 2; return w.sum()`. The first four instructions show bytecode, stack, and graph evolving in lockstep:

<figure class="minidynamo-figure">
  <picture>
    <source media="(max-width: 700px)" srcset="/assets/img/minidynamo/tracing-mobile.svg">
    <img src="/assets/img/minidynamo/tracing.svg" alt="Four bytecode instructions and their stack effects. Only BINARY_ADD adds a graph node: torch.add(x, y).">
  </picture>
  <figcaption>Stack entries are symbolic values, shown from bottom to top.</figcaption>
</figure>

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

Notice three details:

**Steps 3 and 7** -- when a binary operation involves a `TensorVariable`, the interpreter records a `torch.add` or `torch.mul` node in the graph and pushes a new `TensorVariable` wrapping that node. The constant `2` is passed directly into the node's args.

**Step 10** -- `LOAD_METHOD sum` on a `TensorVariable` produces a `MethodVariable`, not a graph node. The method has been _looked up_, not _called_. Step 11 creates the graph node when `CALL_METHOD` executes it.

**Step 12** -- `RETURN_VALUE` marks the output. The graph is now complete.

### The Call Dispatch Logic

Function and method calls use the same decision: record an operation, evaluate a value known during tracing, or reject an unsupported call. In **pseudocode**:

```text
if this is a supported tensor operation:
    record a node and compute an example result
elif this is an allowed metadata query or pure call on constants:
    evaluate it now and wrap the constant result
else:
    reject this call
```

[Call dispatch](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L391) and [tensor-method handling](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/symbolic_interpreter.py#L450) enforce which operations belong in each case.

minidynamo still raises outside that narrow subset, especially for unsupported bytecodes, keyword calls, and non-tensor returns. Real TorchDynamo can guard on Python values, rewrite bytecode, and resume after graph breaks. A _graph break_ is Dynamo's escape hatch for the "don't know how to handle this" case: it compiles the graph it has built so far, hands control back to the regular Python interpreter to run the unsupported bit (a `print`, an unusual data structure, a call into a C extension), and then starts a fresh trace from the next instruction. A single Python function can become several compiled graphs stitched together with plain eager code in between.

<aside markdown="1">

**A concrete graph-break example.** Imagine your `forward` calls into a custom CUDA kernel through a `ctypes` binding or a third-party library that bypasses the PyTorch dispatcher. Without a supported tracing or custom-operator integration, Dynamo cannot safely capture the opaque call. In the default mode it can break the graph around it; `fullgraph=True` instead raises an error. In the graph-break case: everything before the call becomes graph #1 (compiled with Inductor), the opaque CUDA call runs in eager Python against the materialized tensors, and whatever comes after starts graph #2. The kernel fuser cannot see across that boundary, so teams work hard to eliminate graph breaks in hot code paths. A properly registered PyTorch custom op is a different story: because it participates in the dispatcher, Dynamo may be able to keep it as an operator in the graph even if Inductor treats it as an opaque call.

</aside>

---

## 7. The Compiler Backend

The graph is now a clean IR of tensor operations. The compiler's job is to turn it into a callable. In `torch.compile`, this is where the graph gets handed to Inductor for kernel fusion, the step that produces the speedup. **Our educational backend does something smaller.** It generates a plain Python function that re-dispatches to the same `torch` ops as the original, one at a time, with function lookups pre-resolved into the generated function's namespace. It lets us check whether we captured the graph correctly and produces a standalone callable you could feed to real Inductor later. It does not fuse kernels or bypass the dispatcher.

### What the compiler generates

For our running example, the generated Python source is short enough to inspect in full:

```python
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)
    mul_0 = __fn_mul_0(add_0, 2)
    sum_0 = mul_0.sum()
    return sum_0
```

The names `__fn_add_0` and `__fn_mul_0` refer to `torch.add` and `torch.mul` in the function's global namespace. They are resolved when the compiler creates the callable.

Resolving the callables ahead of time avoids the `LOAD_GLOBAL torch` + `LOAD_ATTR add` pair for explicitly spelled `torch.add` calls. Python arithmetic such as `x + y` uses a binary opcode instead. That sounds like an optimization, but it saves very little. The relative costs of Python dispatch, the C++ dispatcher, kernel submission, and device execution depend on the workload and runtime; CPU submission can also overlap GPU execution. Saving some attribute lookups does not remove the dispatcher or kernel launches. This step exists to produce a clean, self-contained callable that reproduces the graph. An optimizing backend like Inductor consumes an FX graph, which we construct separately from our IR, and it gives us a useful sanity check that our tracer matched the original function.

### Code Generation

The emitter walks the graph in dependency order. Placeholders become function parameters; each operation becomes an assignment; the output node becomes a `return` statement. References to earlier nodes become local variable names, which is why `mul_0` can use `add_0` directly in the source above.

The compiler assembles those lines into a function definition and runs `exec()` with a namespace containing the pre-resolved callables. The resulting `compiled_fn` executes the captured graph. The [full emitter](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/compiler.py#L26) handles nested arguments, binds constants that cannot be written as Python literals, and avoids name collisions.

This string-codegen approach belongs to our educational backend. Real TorchDynamo's default path produces FX graphs and hands them to backends such as Inductor; it does not rely on this tiny Python source generator for performance.

### The JIT Backend

The [JIT backend](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/compiler.py#L143) adds one call after Python code generation:

```python
compiled_fn, source = compile_graph(graph)
traced_fn = torch.jit.trace(compiled_fn, example_inputs)
```

The result is a TorchScript function where the entire graph executes as a single C++ call, with no Python interpreter between operations. TorchScript has its own optimization passes and can fuse operations, so its speedups can include both reduced Python/C++ boundary overhead and changed kernel structure. This is a different optimization path from Inductor. It demonstrates graph lowering to another runtime, the same broad move that Inductor makes with kernel fusion. One caveat: TorchScript is in maintenance mode and `torch.jit.trace` is deprecated in recent PyTorch releases, so treat this backend as a demonstration of graph lowering, not a path to build on.

---

## 8. Guards: When Can We Reuse Compiled Code?

A compiled function makes assumptions about its inputs. The graph we traced for `fn(x, y)` with `x.shape = (3, 4)` might not be valid for `x.shape = (5, 6)` -- different shapes could change broadcasting behavior, output sizes, or even which operations are valid.

**Guards** encode these assumptions as boolean checks. A shape guard remembers the example tensor's shape during tracing and compares it against the new tensor on each later call. In **pseudocode**, for one input:

```text
during tracing: remember expected_shape = (3, 4)
on a later call: require shape(new_x) == expected_shape
```

The remembered shape is a snapshot; it must not change when a new argument arrives. The [guard builder](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/guards.py#L109) creates checks for the other tensor metadata and execution modes in the same way.

On each call, a cache entry's guards are checked until one fails or all pass. If all pass, we use its compiled function and skip tracing. If an entry fails, we try the next; only when none match do we retrace and add a new entry.

### Checking and Debugging Guards

`GuardSet.from_example_inputs` checks shape, dtype, device, strides, storage offset, gradient requirements, and conjugate/negative view flags, plus grad/inference/autocast modes, the default dtype, and CUDA’s current device. The interpreter adds guards for globals and accessed `torch` attributes; the wrapper also guards the function's code object. On every call, the wrapper runs the cached guard sets against the new arguments. Each guard is a tiny lambda (e.g. `tuple(args[0].shape) == (3, 4)`), so checking a specialization incurs Python overhead. And when a call misses the cache, you can ask the guard set _why_ — the wrapper exposes its cache as `fn._cache`, a list of `(guard_set, compiled_fn)` pairs.

For example, after tracing with `(3, 4)` tensors, calling `fn._cache[0][0].failing_guards(...)` with contiguous `(5, 6)` tensors reports shape and stride mismatches for both inputs: the row stride changes along with the shape. The [guard inspection methods](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/guards.py#L88) make these failures explicit.

The mechanism is small: the first call pays the compile tax, identical calls pay guard checks, and the cache grows by one entry for each call that matches no existing guard set. (Section 9 walks one function through exactly this life cycle.) If a function sees a different shape on every call, every call misses and pays tracing and compilation overhead again. That is why **recompilation rate** is one of the first things to check when `torch.compile` does not speed up a workload.

<figure class="minidynamo-figure">
  <picture>
    <source media="(max-width: 700px)" srcset="/assets/img/minidynamo/cache-and-guards-mobile.svg">
    <img src="/assets/img/minidynamo/cache-and-guards.svg" alt="The first cache entry misses. The second matches and executes. The third is not checked.">
  </picture>
</figure>

Real Dynamo makes the same trade-off, though symbolic shapes can accept some shape changes without a miss. Ignoring execution itself, the costs are:

|          | First call           | Subsequent calls (cache hit) | Shape change (cache miss) |
| :------- | :------------------- | :--------------------------- | :------------------------ |
| **Cost** | Full trace + compile | Guard checks only            | Full retrace + compile    |

<aside markdown="1">

**Real Dynamo's guards are far more extensive.** They check type IDs, object identity, dict version tags, global variable values, tensor strides, and more. Many checks are implemented in C++ for speed. Our Python lambda guards demonstrate the concept.

</aside>

---

## 9. The `compile()` Decorator

The top-level API ties together all five pipeline stages. Each decorated function owns a cache. For the Python backend, the wrapper follows this loop (**pseudocode**):

```text
on each call with inputs:
    validate inputs
    for (guards, compiled_fn) in this function's cache:
        if all guards pass for this call:
            return compiled_fn(inputs)

    graph, guards = trace(fn, inputs)
    compiled_fn = backend(graph)
    guards.add(function code is unchanged)
    cache.append((guards, compiled_fn))
    return compiled_fn(inputs)
```

Here, `trace` is the symbolic interpreter from Section 6, including the guards it collects. Before caching, the wrapper adds a check that the function's code object is unchanged. The [full decorator](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/__init__.py#L45) implements this loop, the optional JIT backend, and the cache inspection attributes used below.

The previous sections built each component in isolation: the interpreter, the graph, the compiler, and the guards. Now we can run one call through the wrapper and inspect each artifact it produces.

With `torch` and `minidynamo` imported, wrap the function from Section 5 and call it:

```python
fn = minidynamo.compile(fn)
a, b = torch.randn(3, 4), torch.randn(3, 4)
result = fn(a, b)
```

One reminder from Section 1: the wrapper accepts only positional tensor arguments, and the traced body cannot use keyword calls. Constants like the `2` in `x * 2` are fine — they live in the function body and are seen during tracing.

### Step 1: Trace

The wrapper's cache is empty, so it runs the symbolic interpreter. Walking the original function's bytecode produces the graph from Section 5: two input placeholders feeding `torch.add`, then `torch.mul`, then `.sum()`, whose result is the output.

Notice what disappeared: no `z = ...`, no `w = ...`, no `STORE_FAST` instructions. The intermediate local variables from the Python source have been flattened into a straight-line DAG of tensor operations. The constant `2` is inlined directly into `torch.mul`'s args rather than becoming a node. The graph works as an IR because it is a pure description of _"what tensor ops, in what order, wired how"_, stripped of everything the compiler does not need.

### Step 2: Compile

`compile_graph(graph)` turns those nodes into the `compiled_fn` shown in Section 7. The bound `torch.add` and `torch.mul` calls still pass through the PyTorch dispatcher. On CUDA, these operations still launch separate kernels.

The Python backend produces a faithful, standalone callable that does what the captured graph says. Kernel fusion and reduced per-op dispatch can happen when you hand the _same_ graph to Inductor instead, which we get to in Section 10.

### Step 3: Guard

`GuardSet.from_example_inputs((a, b))` supplies the input and execution-context checks. The interpreter augments these during tracing, and the wrapper adds a code-object guard. Here are just the shape, dtype, and device checks:

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

These predicates are part of the contract (the additional checks above must also pass): "the `compiled_fn` we just produced is valid as long as these hold". The guard set is not attached to the tensors `a` and `b`; it is a set of _checks_ that future arguments must satisfy.

### Step 4: Cache

The pair `(guard_set, compiled_fn)` gets appended to the cache list. After this first call, the cache has one entry. The cache is per-`@compile`d function (it lives in the wrapper's closure), and its order matters. On every later call, we scan it from index 0 upward and return the first entry whose guards all pass.

### Step 5: Execute

Finally, we call `compiled_fn(a, b)` and return the result. The result matches the original eager function. We have reorganized dispatch, not changed the computation:

```python
compiled_fn = fn._cache[0][1]
torch.testing.assert_close(compiled_fn(a, b), fn._original(a, b))
```

One first call ran all five steps. First-call latency includes tracing and compilation as well as execution; its size depends on the backend and workload. The measurements below report warmed execution separately.

### Second and Third Calls

The structure pays off on later calls. On the **second call** with all guarded metadata and state unchanged, the wrapper iterates `cache`, finds that `guard_set.check_all(*args)` returns `True` on the first entry, and jumps directly to `compiled_fn(*args)`. Steps 1–4 are skipped entirely. The cache is still length 1.

On the **third call**, `fn(torch.randn(5, 6), torch.randn(5, 6))`, `check_all` returns `False` on every existing entry (the shape guards fail). The wrapper falls through to the slow path again, traces a fresh graph, compiles a new function, builds a new guard set, and appends. The cache now has two entries.

Future calls scan both entries in order. A `(3, 4)` call hits entry 0, a `(5, 6)` call hits entry 1, and any brand-new shape falls through to a new compile and a third entry.

That is the full pipeline in motion. Five stages produce five concrete artifacts: a `Graph`, a `compiled_fn`, a `GuardSet`, a cache list, and a tensor result. Real `torch.compile` handles more machinery in every stage (keyword arguments, nested calls via an inlining interpreter, dynamic shapes, C-level guard evaluation, per-code-object caches, graph breaks), but the spine has the same shape: **trace → compile → guard → cache → execute**.

---

## 10. Where Does Speedup Come From?

With the full system built, we can ask: **how much faster is it?**

**Our Python graph-replay backend produces no meaningful speedup on this workload.** The win comes from what an optimizing backend does with the graph. A common description says `torch.compile` "removes Python overhead." That phrase bundles together several costs between a user's `x + y` and the kernel running on the GPU.

### Where the Time Goes

The work behind an elementwise op in eager PyTorch on a GPU includes the following. Its relative costs vary by device, driver, PyTorch version, tensor size, and whether you're on CUDA or MPS; the host and device work can overlap:

| Cost                                               | Who pays it                     |
| :------------------------------------------------- | :------------------------------ |
| CPython bytecode dispatch                          | The interpreter                 |
| Python-level method resolution and bindings        | CPython + PyTorch               |
| PyTorch C++ dispatcher (device, autograd, vmap, …) | libtorch                        |
| Kernel submission onto the CUDA / MPS stream       | The host runtime and GPU driver |
| The kernel itself                                  | The GPU                         |

The first two rows are what most people mean when they say "Python overhead." Their share of total time depends on the workload. Our Python backend only touches those: it pre-resolves function lookups into the generated function's namespace so explicit `torch` function calls can avoid attribute lookup. Nothing below that line changes. Every op still enters a Python binding, traverses libtorch's dispatch key logic, and submits the operation's kernels. CUDA submission is asynchronous; the host need not wait for each kernel to finish.

The benchmark numbers follow that pattern, but the exact outcome is backend- and shape-dependent. The verified CUDA run used one NVIDIA H200, Python 3.10.20, PyTorch 2.10.0+cu128, and CUDA 12.8. The workload is eleven elementwise operations followed by a sum on two square float32 tensors.

These are microseconds per call: the median of fifteen randomized-order trials of 1,000 calls, after 500 warmup calls. All variants use inference mode. Compilation is excluded, synchronization brackets each trial, and CUDA graphs are disabled in both Inductor paths. This measures amortized wall time including host dispatch, not isolated kernel latency. The `torch.compile` path uses `fullgraph=True, dynamic=False`.

| Callable                           | 32 × 32 | 2048 × 2048 |
| :--------------------------------- | ------: | ----------: |
| Eager PyTorch                      |    59.4 |       129.7 |
| Generated Python, no guards        |    59.7 |       129.6 |
| minidynamo Python, with guards     |    77.5 |       129.7 |
| TorchScript, no guards             |    13.4 |        29.4 |
| minidynamo JIT, with guards        |    30.8 |        39.8 |
| minidynamo + Inductor, with guards |    31.3 |        47.4 |
| `torch.compile`, static shapes     |    40.4 |        57.4 |

A separate profiler pass, checked across two captured calls after warming up the collector, found 12 compute kernels for eager and generated Python at 2048 × 2048, and 2 each for TorchScript and the two Inductor paths. **TorchScript fused the elementwise chain too**; it is not an unfused baseline. The counts exclude memory sets and profiler annotation ranges.

Treat these as measurements of this repository's toy benchmark, not universal benchmark results. The raw JIT callable was fastest in both columns. The wrappers also have different costs and capabilities; subtracting raw and guarded timings does not isolate guard latency, because host checks can overlap earlier GPU work. The [benchmark script](https://github.com/itsdaniele/minidynamo/blob/main/examples/benchmark_publication.py) and [raw trials and profiler events](https://github.com/itsdaniele/minidynamo/blob/main/benchmarks/h200-publication.json) record the configuration and source hashes. Small-input trials showed host-side variation; the raw record includes every trial and its range. The same CUDA allocation passed 244 tests, with 4 MPS-only tests skipped.

<figure class="minidynamo-figure">
  <picture>
    <source media="(max-width: 700px)" srcset="/assets/img/minidynamo/cost-decomposition-mobile.svg">
    <img src="/assets/img/minidynamo/cost-decomposition.svg" alt="The 2048 × 2048 H200 benchmark launches 12 compute kernels eagerly and 2 with Inductor.">
  </picture>
  <figcaption>Each box is one compute kernel; widths do not represent time.</figcaption>
</figure>

The raw Python backend's result is within noise; the guarded wrapper also adds overhead that is visible on the small input.

The JIT backend's wins, when they appear, do _not_ come from bytecode dispatch savings. `torch.jit.trace` wraps the generated function into a single TorchScript graph call, so from Python's point of view the whole chain becomes one `call into C++`. Python drops out of the loop between ops, and some of the per-op dispatcher and Python↔C++ boundary-crossing work gets amortized. TorchScript can also fuse kernels, as the profiler shows here, so the measured gain cannot be attributed solely to host overhead.

### Where Inductor Can Win

When Inductor wins, the speedup comes from a different layer. It operates _below_ the dispatcher rather than saving a few interpreter instructions above it, and it relies on having a captured graph as input:

- **Kernel fusion.** Inductor can generate a single Triton (GPU) or C++ (CPU) kernel for a whole chain of memory-bound ops. In the `many_ops` benchmark above, eager launched 11 elementwise kernels plus the final reduction. The separate kernels read and write intermediate tensors; caches can serve some of that traffic instead of HBM. Fusion reduces those round-trips. For favorable elementwise chains and activations, this often accounts for large speedups in PyTorch benchmarks.
- **Launch overhead collapse.** Even after fusion, each kernel launch still costs microseconds. When the same shapes recur (e.g. the steady-state of a training loop), eligible CUDA workloads can use CUDA graph integration to record work once and replay it, reducing repeated host dispatch and launch overhead. This does not eliminate all launch costs, and it was disabled in the measurements above.
- **Memory planning.** With a full graph in hand, Inductor can plan intermediate buffers once and reuse them, avoiding the per-op allocator churn eager incurs.

Our Python backend implements none of these; the optional JIT backend can apply its own fusion optimizations. They require the graph as input, and they address costs in the lower rows of the table: the dispatcher, the launch, and the kernel itself. The microseconds live there.

### Dynamo Captures, Inductor Optimizes

**Dynamo and Inductor solve different problems.** Dynamo captures the graph; our Python replay backend shows why capture alone need not bring a performance gain. Inductor optimizes the graph; in many deep-learning workloads, that is where the meaningful speedup comes from. The bytecode tracer exists to hand an optimizing backend a graph it can fuse, schedule, and lower. minidynamo supplies the graph-capture part. Because we produce a compatible graph, we can plug in the _real_ Inductor backend and measure the backend's behavior directly:

We can convert our minidynamo graph into an `fx.GraphModule`, lower it to ATen ops, and pass it directly to `compile_fx_inner`, Inductor's internal entry point. This is a private PyTorch API, so the repository pins PyTorch `2.10.0` and treats the integration as educational rather than stable public surface area.

For four small CPU programs covered by the parity tests, this produces matching ATen graphs and generated C++ kernel bodies when both our capture and real Dynamo's export path use the same subsequent `make_fx` and `compile_fx_inner` calls. The CUDA tests also compare complete ATen graphs and Triton source for those four programs at 32 × 32 and the longer chain at 2048 × 2048. The [saved CUDA graph and kernels](https://github.com/itsdaniele/minidynamo/blob/main/benchmarks/h200-inductor-evidence.json) show the two reduction kernels generated for the large chain. These tests validate the shared lowering path, not general equivalence across arbitrary PyTorch programs, devices, or Inductor configurations.

This route bypasses AOTAutograd, including its backward-graph construction, functionalization, and decomposition pipeline; it is an inference-only bridge, not the complete `torch.compile` pipeline. Some operations our tracer accepts, such as a device-changing `.to()` call, can still fail in this direct backend. The wrapper checks its assumptions and raises on a mismatch rather than recompiling.

Using the original function (`fn._original`) and the sample tensors from Section 9, the essential capture and lowering calls are:

```python
interpreter = SymbolicInterpreter(fn._original, (a, b))
graph = interpreter.run()
gm = to_fx_graph_module(graph)
aten_gm = make_fx(gm)(a, b)
compiled = compile_fx_inner(aten_gm, [a, b])
```

`make_fx` traces the FX callable at the ATen level, turning operations such as `torch.add` into `torch.ops.aten.add.Tensor` nodes. Inductor then receives that ATen graph for optimization and code generation.

This excerpt shows the lowering path. The [complete bridge](https://github.com/itsdaniele/minidynamo/blob/7d09d8953f58e93fd4c825219895c891d32a3474/minidynamo/inductor.py#L114) also records and checks guards, rejects unsupported inputs, and adapts Inductor's list-of-inputs and tuple-of-outputs calling convention to the original function.

---

## 11. What We Left Out

minidynamo demonstrates the architecture of TorchDynamo. But real Dynamo is a vastly more complex system. Here are the most important gaps:

### PEP 523 Frame Evaluation

Real Dynamo also uses `dis.get_instructions()` in its [bytecode transformation machinery](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_dynamo/bytecode_transformation.py). It additionally installs a **C-level frame-evaluation hook** via [PEP 523](https://peps.python.org/pep-0523/) that intercepts eligible frames while compilation is active. The hook and the symbolic bytecode interpreter do different jobs. On top of that interception, real Dynamo's tracer supports:

- **Function inlining:** When `fn()` calls `helper()`, Dynamo traces _into_ the callee by recursively walking its bytecode (during tracing, `helper`'s frame never actually runs), capturing a single unified graph. Our bytecode walker only sees the top-level function.

- **Graph breaks:** When Dynamo hits an unsupported operation (a `print()`, an unsupported data structure), it can _break the graph_ by compiling what it has so far, executing the unsupported operation in normal Python, and resuming tracing after. Our interpreter has no graph-break machinery: unsupported bytecodes raise `NotImplementedError`, and arbitrary helper calls are rejected. With real Dynamo's `fullgraph=True`, graph breaks also cause an error.

### Control Flow

We reject jump and iteration instructions (`POP_JUMP_IF_FALSE`, `FOR_ITER`, etc.). Real Dynamo handles control flow by specializing: if the branch condition is a tensor property known at trace time (like `x.shape[0] > 5`), it evaluates it and traces only the taken branch, guarding on the condition.

### Dynamic Shapes

Our guards require exact shape matches. Real Dynamo supports **dynamic shapes** -- symbolic integers that represent unknown dimensions. This can avoid recompilation when batch size changes, at the cost of more complex guard logic and symbolic reasoning.

### 200+ VariableTracker Subclasses

Our four types cover tensors, constants, torch functions, and tensor methods. Real Dynamo has trackers for lists, dicts, ranges, slices, iterators, `nn.Module` instances, user-defined classes, closures, generators, and more.

---

## 12. Summary

`torch.compile` is a well-structured pipeline:

1. **Intercept** Python execution at the bytecode level
2. **Replay** each instruction symbolically, recording tensor operations into a graph
3. **Compile** the graph with an optimizing backend
4. **Guard** against changes in input metadata and other tracing assumptions
5. **Cache** the result for fast reuse

The symbolic interpreter is a CPython emulator. The graph is an IR. The compiler is a code generator. The guards are boolean predicates. Each component is small enough to understand in isolation. Together, they explain how `torch.compile` speeds up PyTorch programs. In this mini implementation, the graph-capture machinery is the educational focus; the large speedups arrive once you pair that captured graph with an optimizing backend like Inductor.

<div class="l-body" markdown="1">

_The full source code for minidynamo is at [github.com/itsdaniele/minidynamo](https://github.com/itsdaniele/minidynamo). Every module is heavily commented and designed to be read linearly._

</div>

---

<div class="appendix" markdown="1">

## Appendix: File Map

| File                                 | Purpose                                                     |
| :----------------------------------- | :---------------------------------------------------------- |
| `minidynamo/__init__.py`             | The `compile()` decorator -- ties together all five stages  |
| `minidynamo/symbolic_interpreter.py` | The bytecode walker -- CPython emulator on VariableTrackers |
| `minidynamo/variable_tracker.py`     | Four symbolic value types                                   |
| `minidynamo/graph.py`                | The computation graph IR (`Node` + `Graph`)                 |
| `minidynamo/compiler.py`             | Code generation backends (Python + JIT)                     |
| `minidynamo/guards.py`               | Guard creation and checking                                 |
| `minidynamo/inductor.py`             | Shared guarded, inference-only FX/ATen/Inductor bridge      |
| `examples/benchmark_publication.py`  | Reproducible H200 measurements and profiler records         |
| `examples/benchmark.py`              | Performance analysis: where speedup comes from              |
| `examples/benchmark_mps.py`          | GPU benchmark: Python vs JIT vs Inductor                    |
| `examples/benchmark_transformers.py` | Transformer-style fusion pattern benchmark                  |
| `examples/inductor_integration.py`   | Plugging into the real Inductor backend                     |

</div>

**[Code on GitHub: itsdaniele/minidynamo](https://github.com/itsdaniele/minidynamo)**
