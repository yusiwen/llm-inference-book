# Behaviour records

One JSON file per probe run, written by `scripts/contract-probes.sh --record`. They are committed on
purpose: a difference between two records is the only mechanical evidence that minfer's *behaviour*
moved while its names stayed the same, and a record that lives only on one machine cannot be compared.

```bash
bash scripts/contract-probes.sh --list                                # what each probe records
MINFER_MODEL=~/models/qwen3-0.6b-q8_0.gguf \
  bash scripts/contract-probes.sh --record runs/2026-12-01-<sha>.json
bash scripts/contract-probes.sh --diff runs/<old>.json runs/<new>.json
```

Naming: `runs/<date>-<minfer short sha>.json`. A probe whose input is missing records `skipped` with
the reason, so an incomplete record is honest rather than empty - `flags` needs only the binary, and
every other probe needs a model.

What a difference means: nothing, until a person reads it. `--diff` prints the fields that moved and
says so; deciding whether Chapter 13's determinism claim or Chapter 29's metric keys are still true is
the reading the book's own rules demand.
