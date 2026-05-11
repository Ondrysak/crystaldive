---
active: true
iteration: 1
session_id: 
max_iterations: 0
completion_promise: null
started_at: "2026-05-11T20:00:14Z"
---

Optimize shader modes in crystaldive at C:UsersOndraDocumentsGitHubcrystaldive. Slowest modes from benchmark: WANNIER 1.145ms, 3D ISO 0.739ms, LINKS 0.731ms, BERRY 0.641ms, NEMATIC 0.557ms, STM 0.380ms. Median 0.176ms. For each mode: read its WGSL shader in src/modes/, apply GPU optimizations inspired by scientific packages like GPAW and Wannier90, run benchmark via cargo test mode_perf -- --nocapture, commit if improved, repeat for next mode.
