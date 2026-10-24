// Keep this article's frame at its content height, including on narrow screens.
(() => {
  const frame = document.getElementById("minidynamo-flow");
  if (!frame) return;
  window.addEventListener("message", (event) => {
    if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
    if (event.data?.type !== "minidynamo-flow:resize") return;
    const height = event.data.height;
    if (Number.isFinite(height) && height >= 200 && height <= 1600) frame.style.height = height + 2 + "px";
  });
  const measure = () => frame.contentWindow.postMessage({ type: "minidynamo-flow:measure" }, location.origin);
  frame.addEventListener("load", measure);
  measure();
})();
