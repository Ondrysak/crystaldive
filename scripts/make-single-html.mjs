import { readFile, writeFile } from "node:fs/promises";

const js = await readFile("web/pkg/crystal-viz.js", "utf8");
if (js.includes("</script>")) {
  throw new Error("Generated JS unexpectedly contains a closing script tag.");
}

const wasm = await readFile("web/pkg/crystal-viz_bg.wasm");
const wasmBase64 = wasm.toString("base64");

const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>crystaldive</title>
    <link rel="icon" href="data:," />
    <style>
      html, body {
        width: 100%;
        height: 100%;
        margin: 0;
        overflow: hidden;
        background: #02030a;
      }
      canvas {
        display: block;
        width: 100vw;
        height: 100vh;
        outline: none;
      }
    </style>
  </head>
  <body>
    <script type="module">
      if (!("gpu" in navigator)) {
        document.body.innerHTML = \`
          <main style="box-sizing:border-box;min-height:100vh;display:grid;place-items:center;padding:24px;color:#e8ecff;font:15px/1.45 system-ui,sans-serif;background:#02030a">
            <section style="max-width:520px">
              <h1 style="margin:0 0 12px;font-size:22px">WebGPU is not available on this browser</h1>
              <p style="margin:0 0 10px;color:#b8bfd6">crystaldive needs WebGPU. Try current Chrome or Edge on Android, or current Safari on iOS/iPadOS. Some older phones, embedded browsers, and in-app browsers do not expose WebGPU yet.</p>
              <p style="margin:0;color:#7f879d">Desktop Chrome/Edge/Safari/Firefox is the most reliable path right now.</p>
            </section>
          </main>\`;
        throw new Error("WebGPU is not available in this browser.");
      }

      if ("GPUAdapter" in globalThis) {
        const requestDevice = GPUAdapter.prototype.requestDevice;
        GPUAdapter.prototype.requestDevice = function (descriptor = {}) {
          if (descriptor.requiredLimits) {
            delete descriptor.requiredLimits.maxInterStageShaderComponents;
          }
          return requestDevice.call(this, descriptor);
        };
      }

${js}

      const wasmBase64 = "${wasmBase64}";
      function decodeWasm(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
      }

      __wbg_init(decodeWasm(wasmBase64)).catch((error) => {
        console.error(error);
        document.body.textContent =
          "Failed to start crystaldive. Make sure this page is served over http:// or https:// in a WebGPU-capable browser.";
      });
    </script>
  </body>
</html>
`;

await writeFile("web/crystaldive-single.html", html, "utf8");
