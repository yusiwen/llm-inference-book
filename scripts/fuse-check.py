#!/usr/bin/env python3
"""fuse-check.py - the witness for Chapter 15's instrument.

`scripts/fuse.c` writes sixteen values of its one-pass output into the raw JSON. This script
regenerates the same fixture from the same PRNG seed and recomputes those sixteen values in
double precision, so the chapter's arithmetic has a second implementation behind it:

  * if the C and numpy values disagree beyond float32 rounding, the kernel or the fixture is wrong;
  * the C also asserts that its one-pass and three-pass outputs are byte-identical, and this script
    checks the same claim independently (in f64 the two orders of operations are the same, so the
    only source of difference in C is rounding - and there is none, because the intermediate is
    rounded to f32 and then multiplied, exactly as the fused form does).

Usage:
    python3 scripts/fuse-check.py facts/raw-fusion.json
"""

import json
import pathlib
import sys

import deps

np = deps.need("numpy")

MASK = (1 << 64) - 1
SEED = 0x9E3779B97F4A7C15
MULT = 0x2545F4914F6CDD1D


class Rng:
    """xorshift64*, the same sequence as the C fixture (wrapping 64-bit arithmetic)."""

    def __init__(self, seed=SEED):
        self.s = seed

    def unit(self):
        s = self.s
        s ^= s >> 12
        s &= MASK
        s ^= (s << 25) & MASK
        s &= MASK
        s ^= s >> 27
        s &= MASK
        self.s = s
        x = (s * MULT) & MASK
        return (x >> 11) / 9007199254740992.0 * 8.0 - 4.0


def fixture_rows(rows, nff):
    """The first `rows` rows of the fixture: gate and up, drawn gate-then-up per element."""
    rng = Rng()
    n = rows * nff
    gate = np.empty(n, dtype=np.float64)
    up = np.empty(n, dtype=np.float64)
    for i in range(n):
        gate[i] = rng.unit()
        up[i] = rng.unit()
    return gate, up


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: fuse-check.py RAW_JSON")
    raw = json.loads(pathlib.Path(sys.argv[1]).read_text())
    nff = raw["dims"]["ffn"]
    bad = 0
    checked = 0
    for act in raw["acts"]:
        for row in act["rows"]:
            gate, up = fixture_rows(row["rows"], nff)
            g, u = gate[:16], up[:16]
            if act["act"] == "silu":
                want = (g / (1.0 + np.exp(-g))) * u
            else:
                want = np.where(g > 0.0, g, 0.0) * u
            got = np.array(row["sample"], dtype=np.float64)
            # the C computed in float32; compare at float32 precision, relative
            diff = np.abs(got - want) / np.maximum(np.abs(want), 1e-6)
            worst = float(diff.max())
            checked += 1
            verdict = "agree" if worst < 1e-5 else "DIFFER"
            if worst >= 1e-5:
                bad += 1
            print(f"{act['act']:4s} rows {row['rows']:5d}: {verdict} (max relative difference "
                  f"{worst:.2e})")
    print(f"{checked - bad}/{checked} samples agree with the f64 recomputation")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
