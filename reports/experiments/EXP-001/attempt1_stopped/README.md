# EXP-001 — attempt 1 stopped (archived 2026-11-03)

Attempt 1 stopped at configuration 14 when ordered-logit BFGS returned precision
loss. Only 13/1280 configurations completed. No model was selected, and no stage
two, LOSO or refinement ran. This attempt remains an incomplete historical run.

The B3a-fix task, approved by Amit, now permits status 2 only when the recomputed
normalized-objective gradient is at most 1e-6. This archive retains the evidence
from before that change. It does not contain a rerun of the full grid.

All original payloads and manifests are preserved byte-for-byte. The original
README is `README.original.md`; this new README explains the relocation. Historical
manifests intentionally retain their original paths and source hashes.

| Original path (relative to repository root) | Current path |
| --- | --- |
| `reports/experiments/EXP-001/README.md` | `reports/experiments/EXP-001/attempt1_stopped/README.original.md` |
| Other contents of `reports/experiments/EXP-001/` | Same relative suffix under `reports/experiments/EXP-001/attempt1_stopped/` |
| `data/derived/elo/exp001_partial_snapshots.parquet` | `data/derived/elo/attempt1_stopped/exp001_partial_snapshots.parquet` |
| `data/reports/nb03_B3b_results.md` | `data/reports/nb03_B3b_attempt1_stopped.md` |
| `data/reports/nb03_B3b_stopped.json` | `data/reports/nb03_B3b_attempt1_stopped.json` |

To verify an original manifest, first interpret its relative file paths from its
**original** directory, then apply the mapping above. Do not resolve its historical
relative paths from the new archive directory and do not rewrite the manifest.
The archived `partial_stage1/metadata.json` field `completed: true` means that the
partial save completed; it does not mean that EXP-001 completed.

Verification details and per-file checksums are recorded in
`data/reports/nb03_B3a_fix_archive_mapping.json`. The fix report is
`data/reports/nb03_B3a_fix.md`. Attempt 2 requires a separate authorized prompt.
