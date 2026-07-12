# Visual debugging with screenshots

Use the native screenshot path to verify visual changes. It copies the final
swapchain image after the scene, post-processing, feedback, and egui UI have
all rendered, so the PNG is the correct artifact for judging the composed UI.

## Capture methods

- **Interactive:** click **CAPTURE** in the command bar. The native build writes
  `crystal-viz-<unix-seconds>.png` to the process working directory.
- **Automated:** run the native app with both variables set:

  ```sh
  CRYSTALVIZ_SCREENSHOT=/tmp/crystaldive-visual/<case>.png \
  CRYSTALVIZ_SCREENSHOT_DELAY=2 \
  cargo run -- --timeline --play
  ```

  `CRYSTALVIZ_SCREENSHOT_DELAY` is seconds since app startup. It triggers exactly
  one capture after the requested state has settled. The application stays open;
  stop it after the output file exists (and, when logging is configured, the
  `Saved screenshot` confirmation appears). Without a delay, setting
  `CRYSTALVIZ_SCREENSHOT` captures the first rendered frame.

Screenshot capture is native-only. The browser build logs that screenshots are
not wired. A backend must advertise `COPY_SRC`; otherwise the app logs that
surface screenshots are unsupported rather than silently creating an invalid
image.

## Visual QA loop

1. Exercise the narrow scenario that changed: open the relevant panel, select
   its mode or preset, and wait for animation/transitions to settle.
2. Save a deterministic named image in a scratch location such as
   `/tmp/crystaldive-visual/`. Do not commit generated PNGs unless the task
   explicitly asks for visual baselines.
3. Open the PNG with the image reader. Check the entire composed frame: clipping
   at window edges, panel overlap, typography contrast, control states, and
   whether the field/feedback remains visible behind the UI.
4. Compare like-for-like screenshots when fixing a regression: same window size,
   preset, animation delay, and panel state. Do not infer a visual result from a
   compile or unit-test success.
5. Pair screenshot QA with the smallest matching automated test. Shader or
   renderer edits additionally require the render-validation commands in
   `skill://testing`.

## Capture failures

- No file and no `Saved screenshot` log: wait beyond the configured delay and
  ensure both environment variables are set for delayed capture.
- `Surface screenshots are unsupported`: this WGPU backend does not expose
  `COPY_SRC`; use an interactive/manual screenshot for the visual check or a
  backend that supports surface copy.
- Missing UI in the output would be a defect: capture is intentionally scheduled
  after egui renders, so report it with the PNG and the runtime log.
