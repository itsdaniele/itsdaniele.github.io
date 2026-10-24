/* A deterministic illustration of minidynamo's default Python backend.
 * Each frame is a snapshot: stepping backward or selecting a scenario never
 * changes the real Python program or runs a compiler in the browser.
 */
(() => {
  "use strict";
  const root = document.getElementById("interactive");
  const byId = (id) => document.getElementById(id);
  const stages = ["lookup", "trace", "compile", "guards", "store", "execute"];
  const scenarios = {
    first: { shape: "(2, 2)", version: 1, result: 220, startCache: 0 },
    repeat: { shape: "(2, 2)", version: 1, result: 252, startCache: 1 },
    changed: { shape: "(2, 3)", version: 2, result: 462, startCache: 1 },
  };
  const frame = (stage, view, title, detail, options = {}) => ({
    stage,
    view,
    title,
    detail,
    line: -1,
    depth: 0,
    ...options,
  });
  function captureFrames(scenario) {
    const version = scenarios[scenario].version;
    return [
      frame("trace", "graph", "Start with the inputs", "Create x and y placeholders and collect input and execution-context guards.", { line: 1 }),
      frame("trace", "graph", "Record x + y", "Record torch.add and run the example operation to propagate metadata.", { line: 2, depth: 1 }),
      frame("trace", "graph", "Record z * 2", "Record torch.mul; the 2 is a constant in the graph.", { line: 3, depth: 2 }),
      frame("trace", "graph", "Record w.sum()", "Record the reduction and mark its result as the graph output.", { line: 4, depth: 3 }),
      frame("compile", "compiled", "Compile the graph", "Generate a Python callable that replays the operations, without kernel fusion."),
      frame("guards", "guards", "Finish the guard set", "Add a function-code guard to the input and context guards collected during tracing."),
      frame("store", "stored", "Cache the callable", "Store the compiled function with its guards for later calls.", {
        cache: version,
        match: version,
      }),
      frame("execute", "result", "Run the compiled function", "Execute the callable on the current tensors.", { cache: version, match: version }),
    ];
  }
  const timelines = {
    first: [
      frame("lookup", "wrapper", "Compilation waits for a call", "The decorator creates a wrapper and an empty cache.", { line: 0 }),
      frame("lookup", "empty", "Nothing cached yet", "The first call supplies tensors, so tracing can begin."),
      ...captureFrames("first"),
    ],
    repeat: [
      frame("lookup", "incoming", "New values, same guarded properties", "Only x’s values change; all guarded properties still match."),
      frame("lookup", "guards", "Check the guards", "Input metadata, execution context, and function code still match.", { match: 1 }),
      frame("lookup", "hit", "Reuse the cached function", "Skip tracing and compilation; run v1.", { match: 1 }),
      frame("execute", "result", "Same callable, new result", "The new inputs produce 252. The cached function still performs the computation.", {
        match: 1,
      }),
    ],
    changed: [
      frame("lookup", "incoming", "New shape", "Both inputs now have shape (2, 3); v1 was compiled for (2, 2)."),
      frame("lookup", "guards", "The old shape guard fails", "No cached entry matches, so trace and compile a new version.", { miss: 1 }),
      ...captureFrames("changed"),
    ],
  };
  let scenario = "first";
  let index = 0;
  let playing = false;
  let timer = null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function graph(depth) {
    const visibility = (level) => (level > depth ? "pending" : level === depth ? "reveal" : "");
    const node = (level, x, y, width, label, input = false) =>
      '<g class="node ' +
      (input ? "input" : "") +
      " " +
      visibility(level) +
      (level === depth && level > 0 ? " active" : "") +
      '">' +
      '<rect x="' +
      x +
      '" y="' +
      y +
      '" width="' +
      width +
      '" height="32" rx="7"/>' +
      '<text x="' +
      (x + width / 2) +
      '" y="' +
      (y + 22) +
      '" text-anchor="middle">' +
      label +
      "</text></g>";
    const edge = (level, path) => '<path class="edge ' + visibility(level) + '" d="' + path + '" marker-end="url(#arrow)"/>';
    return (
      '<svg class="graph" viewBox="0 0 340 238" role="img" aria-label="Captured graph: ' +
      depth +
      ' operations recorded">' +
      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L9 5 L0 10Z" fill="currentColor"/></marker></defs>' +
      edge(1, "M104 40 L137 67") +
      edge(1, "M204 40 L171 67") +
      edge(2, "M154 100 V125") +
      edge(2, "M272 142 H229") +
      edge(3, "M154 158 V183") +
      node(0, 80, 8, 48, "x", true) +
      node(0, 180, 8, 48, "y", true) +
      node(1, 80, 68, 148, "torch.add") +
      node(2, 80, 126, 148, "torch.mul") +
      '<g class="' +
      visibility(2) +
      '"><text x="289" y="148" text-anchor="middle">2</text></g>' +
      node(3, 80, 184, 148, ".sum()") +
      '<text class="' +
      visibility(3) +
      '" x="244" y="207" font-size="13">→ output</text></svg>'
    );
  }
  const compiledSource = [
    "def compiled_fn(x, y):",
    "    add_0 = __fn_add_0(x, y)",
    "    mul_0 = __fn_mul_0(add_0, 2)",
    "    sum_0 = mul_0.sum()",
    "    return sum_0",
  ].join("\n");
  function center(symbol, title, detail = "", className = "") {
    return (
      '<div class="stage-center scene-enter ' +
      className +
      '"><span class="symbol">' +
      symbol +
      '</span><p class="big">' +
      title +
      "</p>" +
      (detail ? '<p class="sub">' + detail + "</p>" : "") +
      "</div>"
    );
  }
  function renderPanel(current, spec) {
    const labels = {
      wrapper: "Before any tensor call",
      empty: "Cache lookup",
      graph: "The captured graph",
      compiled: "Generated Python",
      guards: "Selected guards",
      stored: "A new cache entry",
      incoming: "Current inputs",
      hit: "Reuse the matching callable",
      result: "Return value",
    };
    byId("panel-label").textContent = labels[current.view];
    switch (current.view) {
      case "graph":
        return graph(current.depth);
      case "wrapper":
        return center("fn → wrapper", "Ready for a call");
      case "empty":
        return center("cache = []", "No entries");
      case "compiled":
        return (
          '<div class="scene-enter"><pre class="compiled">' +
          compiledSource +
          '</pre><p class="bindings">__fn_add_0 → torch.add<br>__fn_mul_0 → torch.mul</p></div>'
        );
      case "guards": {
        const failed = Boolean(current.miss);
        const isCheck = current.stage === "lookup";
        const shape = failed ? "(2, 2) ≠ (2, 3)" : spec.shape;
        const mark = failed ? "×" : isCheck ? "✓" : "•";
        return (
          '<div class="scene-enter"><ul class="guard-list">' +
          '<li class="' +
          (failed ? "fail" : "") +
          '"><span>shape ' +
          shape +
          '</span><span class="mark ' +
          (failed ? "fail" : "") +
          '">' +
          mark +
          "</span></li>" +
          '<li><span>dtype float32</span><span class="mark">' +
          (failed ? "—" : mark) +
          "</span></li>" +
          '<li><span>device cpu</span><span class="mark">' +
          (failed ? "—" : mark) +
          "</span></li></ul>" +
          '<p class="small-note">' +
          (failed ? "Stops at the first mismatch." : "Other tensor, context, and code guards also apply.") +
          "</p></div>"
        );
      }
      case "stored":
        return center("guards + compiled_fn", "Saved as v" + spec.version, "shape " + spec.shape, "hit");
      case "hit":
        return center("guards ✓", "Use v1", "", "hit");
      case "incoming":
        return center(
          "shape " + spec.shape,
          scenario === "repeat" ? "New x values" : "A new shape",
          scenario === "repeat" ? "x = [[5, 6], [7, 8]]<br>y = [[10, 20], [30, 40]]" : "x = [[1, 2, 3], [4, 5, 6]]<br>y = 10 * x"
        );
      case "result":
        return center(
          "compiled_fn_v" + spec.version + "(x, y)",
          "tensor(" + spec.result + ".)",
          scenario === "repeat" ? "" : scenario === "first" ? "x = [[1, 2], [3, 4]] · y = 10 * x" : "",
          "result"
        );
      default:
        throw new Error("Unknown animation view");
    }
  }
  function render() {
    const timeline = timelines[scenario];
    const current = timeline[index];
    const spec = scenarios[scenario];
    const atEnd = index === timeline.length - 1;
    root.dataset.scenario = scenario;
    root.dataset.step = String(index);
    root.dataset.stage = current.stage;
    root.dataset.playing = String(playing);
    document.querySelectorAll(".scenarios [data-scenario]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.scenario === scenario));
    });
    const position = stages.indexOf(current.stage);
    document.querySelector(".route").style.setProperty("--position", String(position));
    document.querySelectorAll(".route [data-stage]").forEach((step) => {
      step.classList.toggle("active", step.dataset.stage === current.stage);
      step.classList.toggle("skipped", scenario === "repeat" && !["lookup", "execute"].includes(step.dataset.stage));
      if (step.dataset.stage === current.stage) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
    document.querySelectorAll("[data-line]").forEach((line) => line.classList.toggle("active", Number(line.dataset.line) === current.line));
    byId("shape").textContent = spec.shape;
    byId("input-note").textContent = scenario === "repeat" ? "new x values · float32 · CPU" : "float32 · CPU";
    byId("state-view").innerHTML = renderPanel(current, spec);
    const cacheSize = current.cache ?? spec.startCache;
    byId("cache-entries").innerHTML =
      cacheSize === 0
        ? '<span class="cache-empty">empty</span>'
        : Array.from({ length: cacheSize }, (_, i) => {
            const version = i + 1;
            const state = current.match === version ? "matched" : current.miss === version ? "missed" : "";
            return '<span class="cache-entry ' + state + '">v' + version + " · " + (version === 1 ? "(2, 2)" : "(2, 3)") + "</span>";
          }).join("");
    byId("step-title").textContent = current.title;
    byId("step-detail").textContent = current.detail;
    byId("step-count").textContent = index + 1 + " / " + timeline.length;
    byId("previous").disabled = index === 0;
    byId("next").disabled = atEnd;
    byId("play").textContent = playing
      ? "Pause"
      : atEnd
        ? { first: "Run again →", repeat: "Change shape →", changed: "Replay ↻" }[scenario]
        : reducedMotion.matches
          ? "Next step →"
          : index === 0
            ? { first: "Play first call", repeat: "Play rerun", changed: "Play new shape" }[scenario]
            : "Continue";
  }
  function pause() {
    playing = false;
    window.clearTimeout(timer);
    timer = null;
  }
  function schedule() {
    if (!playing) return;
    timer = window.setTimeout(
      () => {
        timer = null;
        index += 1;
        if (index === timelines[scenario].length - 1) pause();
        render();
        if (playing) schedule();
      },
      index === 0 ? 1800 : 2400
    );
  }
  function select(name) {
    pause();
    scenario = name;
    index = 0;
    render();
  }
  byId("play").addEventListener("click", () => {
    if (playing) {
      pause();
      render();
      return;
    }
    if (index === timelines[scenario].length - 1) {
      select({ first: "repeat", repeat: "changed", changed: "first" }[scenario]);
    } else if (reducedMotion.matches) {
      index += 1;
      render();
      return;
    }
    if (!reducedMotion.matches) {
      playing = true;
      render();
      schedule();
    }
  });
  byId("previous").addEventListener("click", () => {
    pause();
    index = Math.max(0, index - 1);
    render();
  });
  byId("next").addEventListener("click", () => {
    pause();
    index = Math.min(timelines[scenario].length - 1, index + 1);
    render();
  });
  document
    .querySelectorAll(".scenarios [data-scenario]")
    .forEach((button) => button.addEventListener("click", () => select(button.dataset.scenario)));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && playing) {
      pause();
      render();
    }
  });
  reducedMotion.addEventListener("change", () => {
    pause();
    render();
  });
  // Stop playback once the embedded illustration has left the visible article.
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting && playing) {
        pause();
        render();
      }
    }).observe(document.querySelector(".walkthrough"));
  }
  render();
  root.hidden = false;
  if (window.parent !== window && "ResizeObserver" in window) {
    const reportHeight = () => {
      window.parent.postMessage(
        { type: "minidynamo-flow:resize", height: Math.ceil(document.querySelector(".walkthrough").getBoundingClientRect().height) },
        location.origin
      );
    };
    new ResizeObserver(reportHeight).observe(document.querySelector(".walkthrough"));
    window.addEventListener("message", (event) => {
      if (event.origin === location.origin && event.source === window.parent && event.data?.type === "minidynamo-flow:measure") reportHeight();
    });
  }
})();
