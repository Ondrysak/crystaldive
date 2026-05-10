# crystaldive
diving into crystals in various ways

## Browser/WebGPU build

Install the WASM target and `wasm-bindgen-cli` once:

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
```

Build and generate the browser bindings:

```powershell
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/crystal-viz.wasm
```

Serve the `web` folder from a local HTTP server, then open it in a WebGPU-capable browser:

```powershell
python web/serve.py
```

Then visit `http://localhost:8080`.
