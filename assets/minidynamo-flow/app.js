// An illustration of minidynamo's Python backend; no compiler runs in the browser.
(() => {
  "use strict";
  const root = document.getElementById("interactive");
  const scene = document.getElementById("scene");
  const play = document.getElementById("play");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const arrow = '<svg class="arrow" viewBox="0 0 34 12"><path class="shaft" d="M0 6H32"/><path d="m27 1 5 5-5 5"/></svg>';
  const token = (label, accent = false) => `<span class="token${accent ? " accent" : ""}">${label}</span>`;
  const flow = (from, to) => `<div class="flow scene-enter">${from}${arrow}${to}</div>`;
  const stack = (content) => `<div class="stack scene-enter">${content}</div>`;
  const execute = (input, output) => stack(`<span>${input}</span>${arrow}${token("compiled_fn", true)}${arrow}<span>${output}</span>`);
  const graph = `<svg class="graph scene-enter" viewBox="0 0 330 174">
    <path d="M45 57V87H95M45 117V87M145 87H180M230 87H265"/>
    <g><rect x="15" y="27" width="60" height="30" rx="6"/><text x="45" y="43">x</text></g>
    <g><rect x="15" y="117" width="60" height="30" rx="6"/><text x="45" y="133">y</text></g>
    <g class="op" style="--delay: .15s"><rect x="95" y="67" width="50" height="40" rx="7"/><text x="120" y="88">+</text></g>
    <g class="op" style="--delay: .45s"><rect x="180" y="67" width="50" height="40" rx="7"/><text x="205" y="88">× 2</text></g>
    <g class="op" style="--delay: .75s"><rect x="265" y="67" width="60" height="40" rx="7"/><text x="295" y="88">sum</text></g>
  </svg>`;
  const calls = {
    first: [
      {
        title: "Start with a function",
        description: "A function decorated with minidynamo.compile returns the sum of (x + y) times 2.",
        detail: "@compile creates a wrapper with an empty cache. Tracing waits until the first call.",
        visual:
          '<pre class="source scene-enter"><span class="decorator">@minidynamo.compile</span>\ndef fn(x, y):\n    z = x + y\n    w = z * 2\n    return w.sum()</pre>',
      },
      {
        title: "The first call misses the cache",
        description: "fn(x, y) reaches an empty cache.",
        detail: "The wrapper receives the tensors. There is no compiled function to reuse yet.",
        visual: flow("<span>fn(x, y)</span>", token("empty cache")),
      },
      {
        title: "Trace the operations",
        description: "Graph: inputs x and y feed addition, then multiplication by 2, then sum.",
        detail: "Walk the bytecode, record tensor operations, and collect guards for the assumptions made.",
        visual: graph,
      },
      {
        title: "Compile the graph",
        description: "The graph becomes compiled_fn.",
        detail: "The Python backend generates a callable that replays these torch operations.",
        visual: flow("<span>graph</span>", token("compiled_fn", true)),
      },
      {
        title: "Save it for later",
        description: "A pair of guards and compiled_fn is saved in the cache.",
        detail: "Keep the callable and its guards together in the cache.",
        visual: stack(`${token("guards + compiled_fn")}<span class="note">saved in the cache</span>`),
      },
      {
        title: "Run the compiled function",
        description: "x and y flow through compiled_fn to produce a result.",
        detail: "Pass in the current tensors and return the computed result.",
        visual: execute("x, y", "result"),
      },
    ],
    repeat: [
      {
        title: "Call it with fresh tensors",
        description: "fn(x₂, y₂) looks up the cache.",
        detail: "The wrapper checks the cache again. This time, it has a compiled function.",
        visual: flow("<span>fn(x₂, y₂)</span>", token("cache")),
      },
      {
        title: "The guards pass",
        description: "The guard checks succeed. Reuse compiled_fn.",
        detail: "The new inputs and execution context satisfy the cached guards, so tracing and compilation are skipped.",
        visual: stack(`${token("guards ✓", true)}<span class="note">reuse compiled_fn</span>`),
      },
      {
        title: "Same function, new result",
        description: "New tensors x₂ and y₂ flow through the same compiled_fn to compute a new result.",
        detail: "Run the cached callable on the new tensors. The function is cached; the answer is computed anew.",
        visual: execute("x₂, y₂", "new result"),
      },
    ],
  };
  let call = "first";
  let step = 0;
  let playing = false;
  let timer;
  const atEnd = () => step === calls[call].length - 1;

  function renderScene() {
    const frame = calls[call][step];
    root.dataset.call = call;
    root.dataset.step = step;
    document.getElementById("phase").textContent = call === "repeat" ? "Next call" : step === 0 ? "Before the first call" : "First call";
    document.getElementById("step-count").textContent = `${step + 1} / ${calls[call].length}`;
    scene.innerHTML = frame.visual;
    scene.setAttribute("aria-label", frame.description);
    document.getElementById("step-title").textContent = frame.title;
    document.getElementById("step-detail").textContent = frame.detail;
    renderControl();
  }
  function renderControl() {
    root.dataset.playing = playing;
    const label = playing
      ? "Pause"
      : atEnd()
        ? call === "first"
          ? "Run again"
          : "Replay"
        : reducedMotion.matches
          ? "Next step"
          : step === 0
            ? call === "first"
              ? "Play first call"
              : "Play next call"
            : "Continue";
    play.innerHTML = `${label} <span aria-hidden="true">${playing ? "Ⅱ" : "→"}</span>`;
  }
  function pause() {
    clearTimeout(timer);
    playing = false;
    renderControl();
  }
  function schedule() {
    timer = setTimeout(() => {
      step += 1;
      if (atEnd()) playing = false;
      renderScene();
      if (playing) schedule();
    }, 3400);
  }
  play.addEventListener("click", () => {
    if (playing) {
      pause();
      return;
    }
    if (atEnd()) {
      call = call === "first" ? "repeat" : "first";
      step = 0;
      renderScene();
    } else if (reducedMotion.matches) {
      step += 1;
      renderScene();
      return;
    }
    if (!reducedMotion.matches) {
      playing = true;
      renderControl();
      schedule();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  reducedMotion.addEventListener("change", pause);
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) pause();
    }).observe(document.querySelector(".walkthrough"));
  }
  renderScene();
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
