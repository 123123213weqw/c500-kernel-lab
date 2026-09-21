# Repository working rules

- Match the user's language; current README and experimental findings use Chinese.
- GPU experiments require explicit authorization, a fresh isolated run directory,
  one allocated C500 only, a common exclusive lock and a future hard deadline.
- Do not resume historical controllers with expired deadlines. Do not modify
  system drivers, clock/power policy, authentication or original checkpoints.
- Do not compile Rust locally. This repository does not require Rust.
- Retain strict full-vocabulary logits, all four logical caches, complete IDs,
  fresh-request replay and real128-step correctness checks before any speed claim.
- Keep n>=3 paired repeats with identical model/dtype/lengths/token accounting.
  Preserve regressions, OOM, rejected and unattempted cases; no cross-model TPS wins.
- Kernel-only correctness or speed is never a substitute for whole-model results.
  Host dispatch counters are capture counters, not CUDA graph replay counters.
- Version graph strategies separately and run them in fresh Python processes.
  Do not transplant a Llama math patch into Qwen3.5 hybrid GDN.
- CPU contracts and syntax/publication audit must pass before committing:
  `python3 -m unittest discover -s tests -v`
  `python3 tools/check_repository.py`
- No checkpoints, SDK/wheel binaries, credentials, private endpoints or personal
  paths in tracked files/history. No public push without explicit authorization.
