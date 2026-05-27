---
layout: post
title: "MiniDynamo: torch.compile in 500 lines of Python"
description: Rebuilding the core ideas behind torch.compile with a tiny TorchDynamo-style tracer.
date: 2026-05-27 11:12:00-0400
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

<div class="l-body" markdown="1">



*When you run code wrapped in `@torch.compile`, PyTorch intercepts Python bytecode execution, captures a graph of tensor operations, hands that graph to an optimizing compiler, and caches the result -- all transparently. This article rebuilds the basic pieces of that system from scratch, with a small implementation designed to make the moving parts visible.*


Note: *A good part of the code was generated with coding agents. To understand it properly, I then went line by line through the implementation, added tests, wrote small benchmark scripts, and used the resulting toy system to build intuition.*

</div>


## 1. The Big Picture

Deep learning frameworks all face the same tension at some point: Python is easy to write, but slow to execute.
When writing modeling code in PyTorch, everything is convenient for the programmer. In a `forward()` method you can use `if` statements, call helper functions, print for debugging, and rely on ordinary Python control flow. In eager mode, though, GPU tensor programs are usually dispatched one operation at a time. Eleven elementwise operations can mean eleven separate kernel launches, eleven memory round-trips, and a CPU-side dispatcher sitting between each operation.

This mode of execution is called **eager mode**, and it was probably the main reason PyTorch won over TensorFlow.

If you remember the early static-graph TensorFlow days, this trade-off was obvious: eager Python is vastly more productive than building static computation graphs by hand. But it's bad for performance.

`torch.compile`, introduced in PyTorch 2.0, attempts to resolve this tension.
When you wrap a function in `torch.compile()` and call it, PyTorch does something remarkable: it captures a graph of your tensor operations (TorchDynamo takes care of this), then hands that graph to an optimizing compiler (Inductor) that can fuse multiple operations into a single kernel. One read from memory, all operations computed in registers, one write. Whenever the function is called again, PyTorch tries to reuse the same optimized graph, as long as the assumptions made during tracing still hold.

But to do any of this, you first need to symbolically execute arbitrary Python and PyTorch code, and that's the hard part.

### The graph capture problem

Getting a graph of tensor operations from a Python function is harder than it sounds. PyTorch went through years of earlier approaches, each useful but each with painful trade-offs, before landing on TorchDynamo:

- **`torch.jit.trace`** (2018): Run the function with real inputs and record which C++ operations fire. Problem: completely invisible to Python control flow. An `if` statement gets silently baked in as whichever branch happened to execute during tracing. Dynamic shapes break everything.

- **TorchScript** (`torch.jit.script`, 2018): Parse the Python source code into a restricted typed language that can be compiled. Problem: required users to rewrite their code to fit a restricted Python subset. Most real-world code couldn't be scripted without significant refactoring. This was ultimately a non-starter.

- **FX Tracing** (`torch.fx.symbolic_trace`, 2021): Python-level symbolic tracing with proxy objects. Better, but still breaks on dynamic Python constructs, data-dependent control flow, and code whose behavior depends on concrete Python values.

- **Lazy Tensors** (2021): Operate below the dispatcher — record operations lazily and flush them as a batch. Can optimize operations but can't eliminate Python overhead because Python still runs eagerly.

**TorchDynamo** (the thing that powers torch.compile) took a fundamentally different approach: instead of working at the Python level or the C++ dispatcher level, it works at the **bytecode level**. Using PEP 523's Frame Evaluation API, Dynamo installs a C-level hook that intercepts every Python frame *before* CPython's interpreter runs it. It then walks the bytecode instructions, symbolically evaluating them to identify tensor operations and record them into an FX graph.


### What happens when you call `torch.compile`

![The torch.compile pipeline: first call runs every stage; subsequent calls with matching guards skip straight to EXECUTE.](/assets/img/mini-dynamo/pipeline.svg)

1. **Trace**: Dynamo intercepts the Python frame via PEP 523 and walks the bytecode. Tensor operations are recorded into an FX graph. Much of the surrounding Python logic is handled outside the graph: some values are evaluated concretely, some assumptions become *guards*, and unsupported regions can trigger *graph breaks*. We will see in detail what this means.

2. **Compile**: The FX graph is passed to a compiler backend. The default backend is Inductor, which generates fused Triton kernels.

3. **Guard**: Dynamo records the assumptions made during tracing — tensor shapes, dtypes, devices, values of Python variables used in control flow — as boolean predicates.

4. **Cache**: The compiled function and its guards are stored together. On subsequent calls, if all guards pass, the compiled function is reused without re-tracing.

5. **Execute**: If guards pass, run the compiled function. If they fail (e.g., tensor shape changed), re-trace and compile, adding a new cache entry.

The key insight: **tracing happens once, execution happens many times.** The first call is slow (bytecode analysis + compilation), but subsequent calls with matching inputs can skip tracing and reuse the compiled result.

### What we build

Now to the fun part. We will build this pipeline in ~500 lines of Python. Our implementation, *mini-dynamo*, is a deliberately tiny TorchDynamo-style tracer. It captures the same core ideas while leaving out the machinery needed for arbitrary real-world PyTorch programs.

<aside markdown="1">

**What we build vs. what we skip.** Real TorchDynamo handles control flow, nested function calls, user-defined classes, graph breaks, dynamic shapes, and hundreds of Python opcodes. Mini-dynamo handles straight-line tensor computations — enough to understand the architecture without drowning in edge cases. We also skip PEP 523 and the machinery built on top of it: real Dynamo hooks into CPython frame evaluation and effectively re-implements a large chunk of Python execution logic in Python. We also skip AOTAutograd, so we only trace forward computations.

</aside>

---

<!-- ## 2. Why Bytecode-Level Tracing?

A natural question: PyTorch already had `torch.jit.trace`, which records tensor operations by running the function with real inputs. Why build something as complex as a bytecode interpreter?

`torch.jit.trace` operates at the **dispatcher level** — it hooks into PyTorch's C++ dispatcher to observe which operations execute. This means it can only see tensor operations. Everything else (Python control flow, integer arithmetic, attribute access) is invisible to it and gets *baked in* as constants.

TorchDynamo operates at the **bytecode level**. It sees *every instruction* the Python interpreter would execute:

```python
def fn(x, y):
    z = x + y          # LOAD_FAST x, LOAD_FAST y, BINARY_ADD, STORE_FAST z
    w = z * 2           # LOAD_FAST z, LOAD_CONST 2, BINARY_MULTIPLY, STORE_FAST w
    return w.sum()      # LOAD_FAST w, LOAD_METHOD sum, CALL_METHOD 0, RETURN_VALUE
```

Because it sees everything, it can make informed decisions: trace tensor ops into the graph, evaluate Python operations concretely, and emit guards for values that might change across calls. And crucially, when it encounters something it *can't* handle (a `print()`, an unsupported data structure), it can **break the graph** — compile what it has so far, let the unsupported code run normally, and resume tracing afterward. `torch.jit.trace` had no such escape hatch; it either traced everything or gave you a silently wrong result. -->

---

## 2. Architecture: The Five Components

Before diving into each component, here's the map. Mini-dynamo has five modules, each corresponding to a distinct concern in the TorchDynamo architecture:

```
              fn(x, y) + example args
                      │
                      ▼
          ┌───────────────────────┐
          │    compile() decorator │ ← __init__.py (orchestrator)
          │    Manages the cache,  │   Checks guards, dispatches
          │    wires everything    │   to trace/compile/guard
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
   graph.py         (or → Inductor)
```

Here's what each component does, and what it corresponds to in real TorchDynamo:

| Mini-dynamo | Real TorchDynamo | Role |
|:--|:--|:--|
| `symbolic_interpreter.py` | `InstructionTranslator` (~15,000 lines in `symbolic_convert.py`) | Walk bytecodes, build the graph. The heart of the system. |
| `variable_tracker.py` | `VariableTracker` hierarchy (~50 subclasses across 30+ files) | Symbolic values on the interpreter's stack. Tell the interpreter what kind of thing each value is (tensor? constant? torch function?) so it can decide whether to record a graph node or evaluate concretely. |
| `graph.py` | `torch.fx.Graph` + `torch.fx.Node` | The computation graph IR. A flat list of nodes, each describing one operation. This is the output of tracing and the input to compilation. |
| `compiler.py` | Compiler backends (Inductor, etc.) | Takes a finished graph and produces a callable. Our simple backend generates Python source with pre-resolved closures. Real Inductor generates fused Triton/C++ kernels. However, we will build a converter from our graph format to FX Graphs, so that Inductor can be used as a backend as well |
| `guards.py` | `torch._dynamo.guards` (C-accelerated) | Boolean predicates that encode the assumptions made during tracing. If guards pass on new inputs, the cached compiled function can be reused. |
| `__init__.py` | `torch._dynamo.convert_frame` | The orchestrator. Manages the guard-cache loop: check guards → hit? run cached fn. Miss? trace → compile → guard → cache. |

Two components deserve special attention because their roles are easy to confuse:

**The Graph is the output.** It's a pure data structure — a list of nodes describing "what tensor operations to perform." It has no logic, no execution semantics. After tracing is done, it gets handed to the compiler. Think of it as a recipe.

**VariableTrackers are the process.** They're the symbolic values that live on the interpreter's stack *during* tracing. They tell the interpreter what type of thing each value is, so it can decide what to do with each bytecode instruction. When tracing finishes, they're thrown away. Think of them as scaffolding — essential for constructing the graph, but not part of the final product.

We need both because CPython's bytecodes are untyped. When the interpreter sees `BINARY_ADD`, it doesn't know if it's adding two tensors (→ record `torch.add` in the graph) or two integers (→ just compute the result). VariableTrackers carry the type information that lets it make this decision. The Graph records the decisions that were made.

---

## 3. CPython Is a Stack Machine

To understand our symbolic interpreter, you need one fact about CPython: **it's a stack-based virtual machine.** Every Python function compiles to a sequence of bytecode instructions that manipulate a *value stack* and a *locals array*.

<!--
[DIAGRAM: CPython's evaluation loop]

  Bytecodes              Stack (grows right →)        Locals
  ─────────              ─────────────────────        ──────
  LOAD_FAST x            [x_val]                      {x: x_val, y: y_val}
  LOAD_FAST y            [x_val, y_val]
  BINARY_ADD             [result]                      result = x_val + y_val
  STORE_FAST z           []                            {x: ..., y: ..., z: result}
-->

For `z = x + y`, CPython emits:

| Instruction | Stack (after) | Effect |
|:--|:--|:--|
| `LOAD_FAST x` | `[x]` | Push local variable `x` |
| `LOAD_FAST y` | `[x, y]` | Push local variable `y` |
| `BINARY_ADD` | `[x+y]` | Pop two, push their sum |
| `STORE_FAST z` | `[]` | Pop and store in local `z` |

Our symbolic interpreter mirrors this exactly -- same stack, same locals, same dispatch loop. The only difference: instead of real Python values, the stack holds *symbolic wrappers*  that record operations into a graph.

---

## 4. VariableTrackers: The Symbolic Values

Every value in our interpreter is a `VariableTracker`. Think of it as: *"I'm not a real value -- I'm a description of a value that will exist at runtime."*

We need exactly four types:

<!--
[DIAGRAM: VariableTracker hierarchy as a tree]

  VariableTracker
  ├── TensorVariable     -- "I'm a tensor. Here's my graph node."
  ├── ConstantVariable   -- "I'm a known constant. Here's my value."
  ├── TorchVariable      -- "I'm a reference to torch or a torch function."
  └── MethodVariable     -- "I'm a bound tensor method, like x.sum."
-->

### TensorVariable

The most important type. It holds a *graph node* (its identity in the computation graph) and an *example value* (a real tensor with the same shape/dtype/device, used for metadata propagation).

```python
class TensorVariable(VariableTracker):
    def __init__(self, node, example_value):
        self.node = node              # Graph Node that produces this tensor
        self.example_value = example_value  # Real tensor for shape tracking
```

When the interpreter sees `x + y` where both are `TensorVariable`s, it doesn't compute the runtime result that the user asked for. Instead, it:
1. Creates a new `Node` in the graph: `call_function(torch.add, (x.node, y.node))`
2. Computes an example output for metadata propagation: `torch.add(x.example_value, y.example_value)`
3. Returns `TensorVariable(new_node, example_output)`

The example value flows forward through every operation, so at any point during tracing, we know the exact shape, dtype, and device of every intermediate tensor. We are still running individual example tensor ops during tracing to propagate metadata, but the output of tracing is the graph, not the eager result of the original function.

### ConstantVariable

A value fully known at trace time: the `2` in `x * 2`, a dtype like `torch.float32`, a shape tuple. Constants don't become graph nodes -- they're inlined directly into the operations that use them.

```python
class ConstantVariable(VariableTracker):
    def __init__(self, value):
        self.value = value  # The actual Python value
```

### TorchVariable

A reference to the `torch` module or one of its functions. When the interpreter encounters `LOAD_GLOBAL torch`, it pushes `TorchVariable(torch)`. When it then encounters `LOAD_ATTR relu`, it resolves `torch.relu` and pushes `TorchVariable(torch.relu)`.

### MethodVariable

A bound tensor method like `x.sum`. Created when the interpreter accesses a method on a `TensorVariable`. It remembers *which tensor* and *which method*, so when called, it can record the correct graph node.

<aside markdown="1">

**Real Dynamo has ~50 VariableTracker subclasses**, covering lists, dicts, iterators, ranges, user-defined classes, `nn.Module`s, and more. Our four types suffice for straight-line tensor code.

</aside>

---

## 5. The Graph IR

As the interpreter runs, it records operations into a `Graph` -- an ordered list of `Node` objects that form a DAG of the computation. This is a simplified version of `torch.fx.Graph`.

Each `Node` has four key fields:

```python
class Node:
    name: str       # Unique identifier, e.g. "add_0", "x"
    op: str         # One of: "placeholder", "call_function", "call_method", "output"
    target: Any     # What to call (e.g., torch.add) or method name (e.g., "sum")
    args: tuple     # Positional arguments -- can reference other Nodes
```

Nodes come in four flavors:

| `op` | Meaning | Example |
|:--|:--|:--|
| `placeholder` | Function input | `x = placeholder` |
| `call_function` | A function call on tensors | `add_0 = torch.add(x, y)` |
| `call_method` | A method call on a tensor | `sum_0 = add_0.sum()` |
| `output` | The return value | `return sum_0` |

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

Notice the `2` in `torch.mul(add_0, 2)` -- it's a plain Python integer, not a `Node`. Constants are inlined into the args of the operations that consume them.

---

## 6. The Symbolic Interpreter

This is the heart of mini-dynamo. The `SymbolicInterpreter` is what ties the previous pieces together: it reads bytecode, manipulates `VariableTracker`s on a stack, and writes nodes into the `Graph`. Everything else in the system either feeds into this loop or consumes its output.

The intuition behind it is simple. When you run a Python function normally, CPython walks the bytecode and *actually executes* each instruction on real objects: integers get added, tensors get multiplied, methods get invoked. We want to do almost the same thing, except we don't care about the result — we care about the *shape* of the computation. So instead of re-implementing CPython's interpreter to produce a value, we re-implement just enough of it to produce a **graph**. Same bytecode, same stack discipline, same dispatch loop — different domain.

### Two Interpreters in Parallel

The cleanest way to think about this is to picture two interpreters running side by side on the same bytecode, one real and one symbolic:

| | CPython's interpreter | Our symbolic interpreter |
|:--|:--|:--|
| **Stack holds** | Real Python objects | `VariableTracker`s |
| **Locals hold** | Real values | `VariableTracker`s |
| **`BINARY_ADD` on two tensors** | Calls `torch.add`, produces a new tensor | Records `torch.add(x, y)` in the graph, pushes a new `TensorVariable` wrapping that node |
| **`BINARY_ADD` on two ints** | Computes `a + b` | Computes `a + b` — constants are evaluated concretely |
| **`CALL_METHOD x.sum()`** | Invokes the bound method | Records `x.sum()` in the graph |
| **Unsupported opcode** | Executes it | Raises `NotImplementedError` |
| **Final output** | A return value | A finished `Graph` |

Same shape, different domain. CPython operates on values; we operate on *descriptions* of values. At every bytecode step, the symbolic interpreter faces one recurring decision: **record** this operation into the graph, or **evaluate** it concretely on whatever constants and metadata we already know. That single choice, repeated across every instruction, is what produces the captured graph.

### The Three Pieces of State

Just like CPython, our interpreter carries three pieces of state through its run:

- **`self.stack`** — a list of `VariableTracker`s. Pushed to by `LOAD_*` opcodes, consumed by `BINARY_*`, `CALL_*`, `STORE_*`, and the rest.
- **`self.locals`** — a dict mapping variable names to `VariableTracker`s. Initialized from the function's arguments and updated on `STORE_FAST`.
- **`self.graph`** — the `Graph` being built. Grows each time a tensor operation gets recorded.

Everything the interpreter does is a transformation of these three. If you snapshotted them after every instruction, you'd have a complete movie of the trace.

### Correspondence to Real Dynamo

Our `SymbolicInterpreter` is the direct analogue of TorchDynamo's `InstructionTranslator` (in `torch/_dynamo/symbolic_convert.py`). The two share the same skeleton: a value stack, a locals dict, an FX-style graph being mutated, and one handler per opcode. The differences are in scope, not in kind:

- Real Dynamo handles ~200 opcodes including jumps, comparisons, exceptions, closures, and generator machinery. We handle ~15.
- Real Dynamo *inline-traces* into called functions via PEP 523 frame hooks — when `fn()` calls `helper()`, Dynamo intercepts the new frame and keeps tracing into it, producing a single unified graph. Our walker only sees top-level bytecode.
- Real Dynamo emits **guards** on-the-fly as it makes assumptions (e.g. "I looked at `x.shape[0]` and treated it as `32`, so guard on that"). We emit guards after tracing, from the example inputs.
- Real Dynamo can **break the graph** when it hits something unsupported — compile what it has so far, let the hard part run in plain Python, and resume tracing after. Our interpreter just halts with `NotImplementedError`.

Everything past this point is a bottom-up tour of the walker itself: how it initializes, how the dispatch loop fetches and executes instructions, a worked example showing the stack and graph evolving side by side, and finally the call-dispatch logic that decides whether a given call becomes a graph node or a concrete Python call.

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
    #     can use their actual Python value during tracing (e.g. for `if`
    #     conditions on ints, shape tuples, dtype objects).
    for name, example in zip(arg_names, example_args):
        if isinstance(example, torch.Tensor):
            node = self.graph.placeholder(name)
            self.locals[name] = TensorVariable(node, example)
        else:
            self.locals[name] = ConstantVariable(example)
```

### The Main Loop

The interpreter fetches instructions one by one and dispatches to handler methods:

```python
def run(self):
    # dis.get_instructions decodes the function's bytecode into a flat list
    # of Instruction records — the same data CPython would dispatch on
    # internally. Each record knows its opname (e.g. "LOAD_FAST"), its
    # argument value, and where it sits in the bytecode.
    instructions = list(dis.get_instructions(self.fn))
    for inst in instructions:
        # One handler method per opcode, conventionally named op_<OPNAME>
        # (e.g. op_LOAD_FAST, op_BINARY_ADD). This is the same trick CPython's
        # ceval.c uses, just spelled in Python via attribute lookup. Anything
        # we haven't implemented falls through to NotImplementedError rather
        # than silently producing a wrong graph.
        handler = getattr(self, f"op_{inst.opname}", None)
        if handler is None:
            raise NotImplementedError(f"Unsupported bytecode: {inst.opname}")
        handler(inst)
    return self.graph
```

### Walking Through an Example

Let's trace `fn(x, y)` where `fn` computes `z = x + y; w = z * 2; return w.sum()`. Before walking the full table, here's the picture for the first four instructions — bytecode, stack, and graph all evolving in lockstep:

![Tracing in motion: only the BINARY_ADD step actually touches the graph. Every other instruction is plumbing.](/assets/img/mini-dynamo/tracing.svg)

The full trace, including the multiplication and the method call:

| Step | Instruction | Stack | Graph (new node) |
|:--:|:--|:--|:--|
| 1 | `LOAD_FAST x` | `[TensorVar(x)]` | -- |
| 2 | `LOAD_FAST y` | `[TensorVar(x), TensorVar(y)]` | -- |
| 3 | `BINARY_ADD` | `[TensorVar(add_0)]` | `add_0 = torch.add(x, y)` |
| 4 | `STORE_FAST z` | `[]` | -- |
| 5 | `LOAD_FAST z` | `[TensorVar(add_0)]` | -- |
| 6 | `LOAD_CONST 2` | `[TensorVar(add_0), ConstVar(2)]` | -- |
| 7 | `BINARY_MULTIPLY` | `[TensorVar(mul_0)]` | `mul_0 = torch.mul(add_0, 2)` |
| 8 | `STORE_FAST w` | `[]` | -- |
| 9 | `LOAD_FAST w` | `[TensorVar(mul_0)]` | -- |
| 10 | `LOAD_METHOD sum` | `[MethodVar(mul_0, "sum")]` | -- |
| 11 | `CALL_METHOD 0` | `[TensorVar(sum_0)]` | `sum_0 = mul_0.sum()` |
| 12 | `RETURN_VALUE` | `[]` | `return sum_0` |

Three things to notice:

**Steps 3 and 7** -- when a binary operation involves a `TensorVariable`, the interpreter records a `torch.add` or `torch.mul` node in the graph and pushes a new `TensorVariable` wrapping that node. The constant `2` is passed directly into the node's args.

**Step 10** -- `LOAD_METHOD sum` on a `TensorVariable` produces a `MethodVariable` -- not a graph node. The method hasn't been *called* yet, just *looked up*. The graph node is created in step 11 when `CALL_METHOD` executes it.

**Step 12** -- `RETURN_VALUE` marks the output. The graph is now complete.

### The Call Dispatch Logic

The most interesting handler is `_handle_call`, which decides what to do when a function or method is called:

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

    # Pure Python on constants (int, len, etc.)? → Evaluate directly.
    if isinstance(fn, ConstantVariable) and callable(fn.value):
        concrete_args = [self._to_concrete(a) for a in args]
        return ConstantVariable(fn.value(*concrete_args))

    raise RuntimeError(f"Don't know how to call {type(fn).__name__}")
```

This two-path dispatch is the core of the design: **supported tensor operations are traced; supported pure-Python work on constants is evaluated.** Anything outside that narrow subset is where mini-dynamo stops and raises an error. Real TorchDynamo is much more sophisticated: it can guard on Python values, rewrite bytecode, and resume after graph breaks. A *graph break* is Dynamo's escape hatch for the "don't know how to handle this" case: instead of giving up on the whole function, it compiles the graph it has built so far, hands control back to the regular Python interpreter to run the unsupported bit (a `print`, an unusual data structure, a call into a C extension), and then starts a fresh trace from the next instruction — so a single Python function can end up as several compiled graphs stitched together with plain eager code in between.

For a concrete example, imagine your `forward` calls into a custom CUDA kernel via a `ctypes` binding or a third-party library that bypasses the PyTorch dispatcher. Dynamo can see the Python call site but has no way to introspect the C code on the other side of the FFI boundary, so it can't represent that call as an FX node. Rather than failing the whole compile, it cuts the graph at that instruction: everything before the call becomes graph #1 (compiled with Inductor), the opaque CUDA call runs in eager Python against the materialized tensors, and whatever comes after starts graph #2. The two compiled graphs never meet the kernel fuser across that boundary — which is exactly why, in practice, people work hard to eliminate graph breaks in hot code paths. A properly registered PyTorch custom op is a different story: because it participates in the dispatcher, Dynamo may be able to keep it as an operator in the graph even if Inductor treats it as an opaque call.

---

## 7. The Compiler Backend

The graph is now a clean IR of tensor operations. The compiler's job is to turn it into a callable. In a real system like `torch.compile`, this is where the graph gets handed to Inductor for kernel fusion — the step that actually produces the speedup. **Our educational backend is not that.** It generates a plain Python function that re-dispatches to the same `torch` ops as the original, one at a time, with function lookups pre-resolved into a closure. Think of it as the minimum viable backend: it proves we captured the graph correctly and produces a standalone callable you could feed to real Inductor later. It is not where speed comes from.

### What the compiler generates

Our compiler walks the graph and generates a Python function with all function references **pre-resolved in the closure**:

```python
# Generated code for many_ops(x, y):
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)       # __fn_add_0 = torch.add (in closure)
    mul_0 = __fn_mul_0(add_0, 2)   # __fn_mul_0 = torch.mul (in closure)
    sub_0 = __fn_sub_0(mul_0, x)   # __fn_sub_0 = torch.sub (in closure)
    ...
    return sum_0
```

Each `__fn_*` variable is resolved from the function's `__globals__` dict (the closure namespace), avoiding the `LOAD_GLOBAL torch` + `LOAD_ATTR add` pair in the original. It's worth being explicit about what this does and doesn't save, because it's tempting to read the previous paragraph as if we've *optimized* something. We haven't, really. CPython's bytecode dispatch runs on the order of tens of nanoseconds per instruction, while a single eager PyTorch op on the GPU spends microseconds in the C++ dispatcher, more microseconds launching the kernel, and then whatever the kernel itself takes. Skipping one `LOAD_GLOBAL` + `LOAD_ATTR` pair per op saves at most a tiny fraction of the smallest of those costs — it's immeasurable on real workloads. The real purpose of this step is not performance; it's to produce a clean, self-contained callable that faithfully reproduces the graph. That's what an optimizing backend like Inductor expects as input, and it's also a useful sanity check that our tracer produced something equivalent to the original function.

### Code Generation

The compiler walks the graph and emits one line of Python per node:

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
            # Methods are dispatched on the receiver, so there's nothing to
            # stash in the closure — we just write `<self>.<method>(...)`.
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

The `exec()` call creates the function in a namespace that contains the closure variables -- the pre-resolved torch functions. This string-codegen trick is just our educational backend. Real TorchDynamo's default path produces FX graphs and hands them to backends such as Inductor; it does not rely on this tiny Python source generator for performance.

### The JIT Backend

For an additional step, we can trace the generated Python function with `torch.jit.trace` to get a TorchScript function:

```python
def compile_graph_jit(graph, example_inputs):
    # First, produce our usual Python source via compile_graph(). Then hand
    # that callable to torch.jit.trace, which re-records it as a single
    # TorchScript graph by running it once with the example inputs. The
    # result is a function where Python drops out of the per-op loop —
    # but each op still launches its own kernel; nothing is fused.
    compiled_fn, source = compile_graph(graph)
    traced_fn = torch.jit.trace(compiled_fn, example_inputs)
    return traced_fn, source
```

This produces a TorchScript function where the entire graph executes as a single C++ call — no Python interpreter between operations. However, each operation is still a separate kernel launch. There is no kernel fusion, so this does not produce a meaningful speedup. It's included here because it demonstrates the concept of lowering a graph to a different runtime, which is what real backends like Inductor do (but with actual kernel fusion).

---

## 8. Guards: When Can We Reuse Compiled Code?

A compiled function makes assumptions about its inputs. The graph we traced for `fn(x, y)` with `x.shape = (3, 4)` might not be valid for `x.shape = (5, 6)` -- different shapes could change broadcasting behavior, output sizes, or even which operations are valid.

**Guards** encode these assumptions as boolean checks:

```python
@classmethod
def from_example_inputs(cls, example_args):
    guard_set = cls()
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

On each call, every guard is checked. If all pass, the cached compiled function is valid and we skip tracing entirely. If any guard fails, we retrace and compile for the new input signature, adding a new entry to the cache.

### Seeing Guards in Action

The easiest way to build intuition is to actually call a compiled function and watch the cache evolve. Take the same function we used earlier:

```python
import torch
import mini_dynamo

@mini_dynamo.compile
def fn(x, y):
    z = x + y
    w = z * 2
    return w.sum()
```

On the **first call**, there's no cache entry yet, so we fall through to the slow path: trace → compile → build guards. The guard set produced by `GuardSet.from_example_inputs` gets three guards per tensor argument — one for shape, one for dtype, one for device:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)
fn(a, b)     # ≈ milliseconds — full trace + compile

# Inspect what got stored in the cache:
guard_set, compiled_fn = fn._cache[0]
print(guard_set)
# GuardSet([
#   args[0].shape == (3, 4)
#   args[0].dtype == torch.float32
#   args[0].device == cpu
#   args[1].shape == (3, 4)
#   args[1].dtype == torch.float32
#   args[1].device == cpu
# ])
```

On the **second call**, the wrapper iterates the cache and calls `guard_set.check_all(a2, b2)`. Every guard is a tiny lambda (e.g. `tuple(args[0].shape) == (3, 4)`), so the whole check is a handful of Python comparisons — microseconds — and we jump straight to the compiled function without re-tracing:

```python
a2 = torch.randn(3, 4)   # same shape/dtype/device → guards pass
b2 = torch.randn(3, 4)
fn(a2, b2)               # ≈ microseconds of guard check + the compiled fn
```

On the **third call**, we pass in tensors with a new shape. The shape guards on `args[0]` and `args[1]` both fail, `check_all` returns `False`, so the wrapper falls through to the slow path again: retrace, recompile, build a new guard set, append it to the cache. Now the cache has *two* entries, and future calls will check both in order:

```python
a3 = torch.randn(5, 6)   # different shape → guards fail
b3 = torch.randn(5, 6)
fn(a3, b3)               # ≈ milliseconds — cache miss, retrace

# If you want to know *why* a call missed, ask the guard set:
print(fn._cache[0][0].failing_guards(a3, b3))
# [Guard(args[0].shape == (3, 4)), Guard(args[1].shape == (3, 4))]
# And now len(fn._cache) == 2 — one entry per input signature we've seen.
```

This is the mechanism in a nutshell: the first call pays the compile tax, identical calls are nearly free, and the cache grows by one entry every time Dynamo encounters a genuinely new input signature. In the worst case — a function called with a different shape every time — every call misses and `@compile` is pure overhead, which is why **recompilation rate** is one of the first things to look at when `torch.compile` isn't giving you the speedup you expected.

![One cache scan: the wrapper walks entries top-to-bottom and runs the first whose guards all pass. Entries further down never get checked on a hit.](/assets/img/mini-dynamo/cache-and-guards.svg)

<!--
[DIAGRAM: Guard checking flow]

                     fn(x, y)
                        │
               ┌────────▼────────┐
               │  Check guard #1  │──── shape match? ───→ yes ──┐
               └────────┬────────┘                              │
                        │ no                                    │
                ┌───────▼────────┐                      ┌───────▼───────┐
                │ Check guard #2  │                     │ Use cached fn  │
                └───────┬────────┘                      └───────────────┘
                        │ no
                ┌───────▼────────┐
                │  Retrace &      │
                │  compile        │
                └────────────────┘
-->

This is the same trade-off real Dynamo makes:

| | First call | Subsequent calls (cache hit) | Shape change (cache miss) |
|:--|:--|:--|:--|
| **Cost** | Full trace + compile | Guard checks only | Full retrace + compile |
| **Typical time** | Milliseconds | Microseconds | Milliseconds |

<aside markdown="1">

**Real Dynamo's guards are far more extensive.** They check type IDs, object identity, dict version tags, global variable values, tensor strides, and more. They're also implemented in C for speed. Our Python lambda guards demonstrate the concept.

</aside>

---

## 9. The `compile()` Decorator: Putting It All Together

The top-level API ties together all five pipeline stages:

```python
def compile(fn=None, *, backend="python"):
    # The cache lives in this closure, so each @compile'd function gets its
    # own. Entries are appended in the order they were compiled; we scan
    # from the front on every call.
    cache = []

    @functools.wraps(fn)
    def wrapper(*args):
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

    return wrapper
```

The previous sections built each component in isolation — the interpreter, the graph, the compiler, the guards. It's worth seeing them compose end-to-end on a real call, with each intermediate artifact laid out explicitly. Take the same toy function as before, and imagine we're running the very first call through the wrapper, stage by stage. Each stage produces something concrete you can print, so we'll print it.

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

### Step 1: Trace

The wrapper's cache is empty, so we fall into the slow path. `SymbolicInterpreter(fn, (a, b)).run()` walks `fn`'s bytecode, pushing `VariableTracker`s on its stack, and records every tensor operation as a `Node`. It returns a `Graph`:

```
Graph:
  x = placeholder
  y = placeholder
  add_0 = torch.add(x, y)
  mul_0 = torch.mul(add_0, 2)
  sum_0 = mul_0.sum()
  return sum_0
```

Notice what's *not* there: no `z = ...`, no `w = ...`, no `STORE_FAST` noise, no `LOAD_GLOBAL torch` lookups. The intermediate local variables from the Python source have been flattened out into a straight-line DAG of tensor operations. The constant `2` is inlined directly into `torch.mul`'s args rather than becoming a node. This is precisely what makes the graph useful as an IR: it's a pure description of *"what tensor ops, in what order, wired how"*, stripped of everything the compiler doesn't care about.

### Step 2: Compile

`compile_graph(graph)` walks those nodes and emits one line of Python per operation. It returns a callable plus the source string, which is worth looking at because it's small enough to read in full:

```python
def compiled_fn(x, y):
    add_0 = __fn_add_0(x, y)
    mul_0 = __fn_mul_0(add_0, 2)
    sum_0 = mul_0.sum()
    return sum_0
```

The `__fn_add_0` and `__fn_mul_0` names aren't magic — they're just keys into the closure namespace the compiler builds alongside the source. That dict looks like `{"__fn_add_0": torch.add, "__fn_mul_0": torch.mul}`, and it becomes the globals for the `exec()` call that materializes the function. Each op still goes through `torch.add` and the full PyTorch dispatcher, and each still launches its own kernel — we haven't fused anything, haven't skipped the C++ dispatcher, haven't avoided a single kernel launch.

This backend's role is purely to produce a faithful, standalone callable that does exactly what the captured graph says. Kernel fusion and dispatcher elimination are what happens when you hand the *same* graph to Inductor instead, which we get to in Section 10.

### Step 3: Guard

`GuardSet.from_example_inputs((a, b))` inspects each tensor argument and builds three lambda guards per tensor — one each for shape, dtype, and device:

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

These six predicates are the contract: "the `compiled_fn` we just produced is valid as long as these hold". The guard set isn't attached to the tensors `a` and `b`; it's a set of *checks* that any future arguments must satisfy.

### Step 4: Cache

The pair `(guard_set, compiled_fn)` gets appended to the cache list. After this first call:

```python
print(len(fn._cache))    # → 1
```

That's the entire cache: one entry. The cache is per-`@compile`d function (it lives in the wrapper's closure), and its order matters — on every subsequent call, we'll scan it from index 0 upward, returning the first entry whose guards all pass.

### Step 5: Execute

Finally, we actually call `compiled_fn(a, b)` and return the result. The result is identical to what eager `fn(a, b)` would produce — we haven't *changed* the computation, we've just reorganized how it's dispatched:

```python
compiled_fn(a, b) == fn(a, b)   # → tensor(True)
```

All five steps have run in service of this one call. The first-call latency (a few milliseconds on our example) is almost entirely spent in steps 1–3; step 5 is microseconds.

### What Happens on the Second and Third Calls

Now the structure earns its keep. On the **second call** with the same shape/dtype/device, the wrapper iterates `cache`, finds that `guard_set.check_all(*args)` returns `True` on the first entry, and jumps directly to `compiled_fn(*args)`. Steps 1–4 are skipped entirely. The cache is still length 1.

On the **third call** with `(5, 6)` tensors, `check_all` returns `False` on every existing entry (the shape guards fail). The wrapper falls through to the slow path again, traces a fresh graph, compiles a new function, builds a new guard set, and appends. Now:

```python
print(len(fn._cache))    # → 2
```

Future calls will scan both entries in order — a `(3, 4)` call hits entry 0, a `(5, 6)` call hits entry 1, and any brand-new shape falls through to a new compile and a third entry.

This is the full pipeline in motion. Five stages, five concrete artifacts — a `Graph`, a `compiled_fn`, a `GuardSet`, a cache list, and a tensor result — each one handed to the next, cached so that steady-state calls skip all the expensive work. Real `torch.compile` is dramatically more sophisticated in every stage (keyword arguments, nested calls via PEP 523, dynamic shapes, C-level guard evaluation, per-code-object caches, graph breaks), but the spine is the same shape: **trace → compile → guard → cache → execute**.

---

## 10. Back to the Big Picture: Where Does Speedup Actually Come From?

Now that we've built the full system, we can ask honestly: **how much faster is it?**

The short version: **graph capture on its own produces no meaningful speedup.** All of the win comes from what an optimizing backend does with the graph. It's worth unpacking why, because a common and slightly misleading way to describe `torch.compile` is to say it "removes Python overhead." That phrasing glosses over several very different costs that live between a user's `x + y` and the kernel running on the GPU.

### Where the Time Actually Goes

Here's the rough per-op cost decomposition for an elementwise op in eager PyTorch on a modern CUDA setup:

| Cost | Typical scale per op | Who pays it |
|:--|:--|:--|
| CPython bytecode dispatch | tens of nanoseconds | The interpreter |
| Python-level method resolution, `__torch_function__` | hundreds of nanoseconds | CPython + PyTorch's Python bindings |
| PyTorch C++ dispatcher (device, autograd, vmap, …) | a few microseconds | libtorch |
| Kernel launch onto the CUDA / MPS stream | 5–20 microseconds | The GPU driver |
| The kernel itself | nanoseconds to milliseconds | The GPU |

The first two rows are what most people mean when they say "Python overhead." They are also the *smallest* rows. Our Python backend only touches those: it pre-resolves function lookups into a closure so each op skips one `LOAD_GLOBAL` + `LOAD_ATTR` pair. Nothing below that line changes — every op still boxes arguments into PyObjects, still traverses libtorch's dispatch key logic, still waits on its own kernel launch.

The numbers reflect this. For a function with 11 chained elementwise ops on 256×256 tensors:

```
Eager (original):            ~150 us
mini_dynamo (python):        ~140 us   (~1.07x — mostly noise)
mini_dynamo (jit):           ~120 us   (~1.25x)
torch.compile (inductor):     ~40 us   (~3.75x)
```

![Where the time goes for 11 chained elementwise ops. Inductor's win is not from making each op faster — it's from collapsing 11 launches into a single fused kernel.](/assets/img/mini-dynamo/cost-decomposition.svg)

The Python backend's ~1.07× is essentially within noise. We saved a handful of bytecodes per op, and that's dwarfed by everything underneath.

The JIT backend's ~1.25× is small but real, and it's worth understanding where it comes from — because it is *not* bytecode dispatch savings. `torch.jit.trace` wraps the generated function into a single TorchScript graph call, so from Python's point of view the whole chain becomes one `call into C++`. Python drops out of the loop between ops, and some of the per-op dispatcher and Python↔C++ boundary-crossing work gets amortized. We're nibbling at rows 2–3 of the table, not row 1.

### Where the Real Speedup Comes From

Inductor's ~3.75× is a different beast entirely. It comes from operating *below* the dispatcher rather than saving a few interpreter instructions above it, and it relies on having a captured graph as input:

- **Kernel fusion.** Inductor generates a single Triton (GPU) or C++ (CPU) kernel for a whole chain of memory-bound ops. Eager does 11 kernels, each reading from HBM, computing one op, writing back. Fusion does one read, all the arithmetic in registers, one write. For elementwise chains, softmax, layer norm, activations — almost everything that isn't a matmul — this alone is the 3–5× number you see in PyTorch benchmarks.
- **Launch overhead collapse.** Even after fusion, each kernel launch still costs microseconds. When the same shapes recur (e.g. the steady-state of a training loop), CUDA graph integration lets you record the launches once and replay them as a single stream op, eliminating the per-step dispatcher and launch costs.
- **Memory planning.** With a full graph in hand, Inductor can plan intermediate buffers once and reuse them, avoiding the per-op allocator churn eager incurs.

None of these live in our mini-dynamo's backend, or could. They require the graph as input, and they operate on the *biggest* rows of the cost table — the dispatcher, the launch, and the kernel itself. That is the part of the stack where the microseconds actually live.

### The Key Insight

**Dynamo and Inductor are separate concerns, and they are not symmetric.** Dynamo captures the graph; on its own, that gets you essentially nothing for performance. Inductor optimizes the graph; in normal deep-learning workloads, that is where the meaningful speedup comes from. The entire point of going through the elaborate machinery of a bytecode-level tracer is to hand an optimizing backend something it can fuse, schedule, and lower. Our mini-dynamo replaces only the Dynamo part — and because we produce a compatible graph, we can plug in the *real* Inductor backend and recover the actual speedup:

<!--
[DIAGRAM: The separation of concerns]

  ┌─────────────────────┐     ┌─────────────────────┐
  │    Graph Capture     │     │   Graph Optimization  │
  │                      │     │                       │
  │  Real: TorchDynamo   │     │  Real: Inductor       │
  │  Ours: mini-dynamo   │     │  (fused GPU kernels)  │
  │                      │     │                       │
  │  PEP 523 hooks vs    │ ──→ │  Same backend,        │
  │  bytecode walking    │     │  same kernels,        │
  │                      │     │  same speedup         │
  └─────────────────────┘     └─────────────────────────┘
```
-->

We can convert our mini-dynamo graph into an `fx.GraphModule`, lower it to ATen ops, and pass it directly to `compile_fx_inner` -- Inductor's entry point. For the straight-line tensor programs that mini-dynamo supports, this can produce the same ATen graph and therefore the same fused kernels as real `torch.compile`. That's what the parity tests in this repository validate. It is not a claim of general equivalence across arbitrary PyTorch programs.

```python
def mini_dynamo_to_inductor(fn, *example_inputs):
    # 1. Trace with our symbolic interpreter — produces a mini-dynamo Graph
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

    # 4. Hand the ATen graph to Inductor's entry point, which does the actual
    #    kernel fusion and code generation. The returned `compiled` is the
    #    fused-kernel callable.
    compiled = compile_fx_inner(aten_gm, list(example_inputs))
    return compiled
```

---

## 11. What We Left Out

Mini-dynamo demonstrates the architecture of TorchDynamo. But real Dynamo is a vastly more complex system. Here are the most important gaps:

### PEP 523 Frame Evaluation

Real Dynamo doesn't use `dis.get_instructions()`. It installs a **C-level frame evaluator** via PEP 523 that intercepts every Python function call before CPython's interpreter runs. This is transparent -- no special calling convention needed -- and enables:

- **Function inlining:** When `fn()` calls `helper()`, Dynamo intercepts the new frame and traces *into* it, capturing a single unified graph. Our bytecode walker only sees the top-level function.

- **Graph breaks:** When Dynamo hits an unsupported operation (a `print()`, an unsupported data structure), it can *break the graph* -- compiling what it has so far, executing the unsupported operation in normal Python, and resuming tracing after. Our interpreter simply raises `NotImplementedError`.

### Control Flow

We skip all jump instructions (`JUMP_IF_TRUE`, `FOR_ITER`, etc.). Real Dynamo handles control flow by specializing: if the branch condition is a tensor property known at trace time (like `x.shape[0] > 5`), it evaluates it and traces only the taken branch, guarding on the condition.

### Dynamic Shapes

Our guards require exact shape matches. Real Dynamo supports **dynamic shapes** -- symbolic integers that represent unknown dimensions. This avoids recompilation when batch size changes, at the cost of more complex guard logic and symbolic reasoning.

### 50+ VariableTracker Subclasses

Our four types cover tensors, constants, torch functions, and tensor methods. Real Dynamo has trackers for lists, dicts, ranges, slices, iterators, `nn.Module` instances, user-defined classes, closures, generators, and more.

---

## 12. Summary

`torch.compile` is not magic. It's a well-structured pipeline:

1. **Intercept** Python execution at the bytecode level
2. **Replay** each instruction symbolically, recording tensor operations into a graph
3. **Compile** the graph with an optimizing backend
4. **Guard** against changes in input metadata
5. **Cache** the result for fast reuse

The symbolic interpreter is a CPython emulator. The graph is an IR. The compiler is a code generator. The guards are boolean predicates. Each component is understandable in isolation, and together they explain how `torch.compile` can speed up PyTorch programs. In this mini implementation, the graph-capture machinery is the educational focus; the large speedups only arrive once you pair that captured graph with an optimizing backend like Inductor.

<div class="l-body" markdown="1">

*The full source code for mini-dynamo is in this repository. Every module is heavily commented and designed to be read linearly.*

</div>

---

<div class="appendix" markdown="1">

## Appendix: File Map

| File | Purpose |
|:--|:--|
| `mini_dynamo/__init__.py` | The `compile()` decorator -- ties together all five stages |
| `mini_dynamo/symbolic_interpreter.py` | The bytecode walker -- CPython emulator on VariableTrackers |
| `mini_dynamo/variable_tracker.py` | Four symbolic value types |
| `mini_dynamo/graph.py` | The computation graph IR (`Node` + `Graph`) |
| `mini_dynamo/compiler.py` | Code generation backends (Python + JIT) |
| `mini_dynamo/guards.py` | Guard creation and checking |
| `examples/benchmark.py` | Performance analysis: where speedup comes from |
| `examples/benchmark_mps.py` | GPU benchmark: Python vs JIT vs Inductor |
| `examples/inductor_integration.py` | Plugging into the real Inductor backend |

</div>
