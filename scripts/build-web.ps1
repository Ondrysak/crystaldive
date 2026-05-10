$ErrorActionPreference = "Stop"

cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/crystal-viz.wasm
node scripts/make-single-html.mjs

Write-Host "Built web/pkg and web/crystaldive-single.html"
