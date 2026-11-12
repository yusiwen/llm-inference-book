#!/usr/bin/env bash
# Behaviour probes for the minfer baseline.
#
# The static checker (scripts/minfer-drift.py) catches addresses and counts. It cannot catch a
# mechanism that changed while every name stayed the same - a flag that now defaults the other way, a
# graph that gained a node, a cache format that moved. These probes record what the engine actually
# does, and a difference between two records is a question for a human, never an automatic edit.
#
#   bash scripts/contract-probes.sh --list                 # what this checks, and what each one needs
#   bash scripts/contract-probes.sh --record               # writes runs/<date>-<sha>.json
#   bash scripts/contract-probes.sh --record /tmp/p.json   # an explicit path
#   bash scripts/contract-probes.sh --diff runs/a.json runs/b.json
#
# Environment:
#   MINFER_BIN    the engine binary            (default: ../minfer/target/release/minfer)
#   MINFER_MODEL  any GGUF file                (unset: every probe that needs one is skipped)
#   MINFER_DIR    the minfer check-out         (default: ../minfer)
#
# Prerequisites are never assumed: a probe whose input is missing records `skipped` with the reason,
# so a record with fewer probes is honest rather than empty.
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${MINFER_BIN:-../minfer/target/release/minfer}"
DIR="${MINFER_DIR:-../minfer}"
MODEL="${MINFER_MODEL:-}"
OUT="runs/$(date -u +%Y-%m-%d)-$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || echo unknown).json"

# Flags the prose of this book names. A removed or renamed flag is a behaviour change even though
# every path in the chapter still resolves.
BOOK_FLAGS="--meta --no-template --chat-template --cnv --session --greedy --top-k --top-p --min-p
--mirostat --xtc-probability --dry-multiplier --grammar --json-schema --stop --n-predict --seed
--backend --gpu-layers --spec-draft --spec-draft-adaptive --n-ctx --threads --n-slots --slots-file
--dump-graph --dump-graph-json"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }

list_probes() {
  cat <<'TXT'
probe            needs            what it records
flags            binary           which of the flags the book names still exist in --help
meta             binary + model   tensor count and dtype set from `--meta`
graph-json       binary + model   node and edge count from `--dump-graph-json`
greedy-ids       binary + model   a hash of 16 greedy tokens, twice (determinism, not the tokens)
backend-cpu      binary + model   the same hash with MINFER_DISABLE_MPS=1 (device-free path)
serve-keys       binary + model   /health status and the key set of /metrics
cuda-gate        CUDA build       the MMQ gate line the log prints at load
TXT
}

json_str() { printf '"%s"' "$(printf '%s' "${1:-}" | tr -d '\n' | sed 's/\\/\\\\/g; s/"/\\"/g')"; }
json_num() { printf '%s' "${1:-null}"; }

record() {
  local dest="${1:-$OUT}"
  mkdir -p "$(dirname "$dest")"
  local sha date
  sha=$(git -C "$DIR" rev-parse HEAD 2>/dev/null || echo unknown)
  date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local tmp; tmp=$(mktemp)

  # --- flags: no model needed
  if [ -x "$BIN" ]; then
    local help; help=$("$BIN" --help 2>&1)
    {
      printf '"flags":{"status":"ran","bin":%s,"present":{' "$(json_str "$(basename "$BIN")")"
      local first=1
      for f in $BOOK_FLAGS; do
        if printf '%s' "$help" | grep -q -- "$f"; then
          [ $first -eq 1 ] || printf ','; first=0
          printf '%s:true' "$(json_str "$f")"
        else
          [ $first -eq 1 ] || printf ','; first=0
          printf '%s:false' "$(json_str "$f")"
        fi
      done
      printf '}},'
    } >> "$tmp"
  else
    printf '"flags":{"status":"skipped","reason":"no binary at %s"},' "$(json_str "$BIN")" >> "$tmp"
  fi

  local have_model=false
  [ -n "$MODEL" ] && [ -f "$MODEL" ] && [ -x "$BIN" ] && have_model=true

  # --- meta
  if $have_model; then
    local meta; meta=$("$BIN" "$MODEL" --meta 2>/dev/null)
    local tensors; tensors=$(printf '%s' "$meta" | grep -cE '^\s' || true)
    { printf '"meta":{"status":"ran","tensors_lines":%s,"bytes":%s},' "$(json_num "$tensors")" "$(json_num "$(printf '%s' "$meta" | wc -c | tr -d ' ')")"; } >> "$tmp"
  else
    printf '"meta":{"status":"skipped","reason":"set MINFER_MODEL to a .gguf file"},' >> "$tmp"
  fi

  # --- graph json
  if $have_model; then
    local gj; gj=$(mktemp)
    "$BIN" "$MODEL" --dump-graph-json "$gj" >/dev/null 2>&1
    local nodes edges; nodes=$(grep -o '"op"' "$gj" 2>/dev/null | wc -l | tr -d ' ')
    edges=$(grep -o '"inputs"' "$gj" 2>/dev/null | wc -l | tr -d ' ')
    { printf '"graph-json":{"status":"ran","nodes":%s,"nodes_with_inputs":%s,"bytes":%s},' \
        "$(json_num "$nodes")" "$(json_num "$edges")" "$(json_num "$(wc -c < "$gj" | tr -d ' ')")"; } >> "$tmp"
    rm -f "$gj"
  else
    printf '"graph-json":{"status":"skipped","reason":"needs MINFER_MODEL"},' >> "$tmp"
  fi

  # --- greedy ids, twice (a determinism claim the book makes in Chapter 13)
  if $have_model; then
    local h1 h2
    h1=$("$BIN" "$MODEL" "The capital of France is" -n 16 --greedy --seed 42 2>/dev/null | shasum -a 256 | cut -c1-16)
    h2=$("$BIN" "$MODEL" "The capital of France is" -n 16 --greedy --seed 42 2>/dev/null | shasum -a 256 | cut -c1-16)
    { printf '"greedy-ids":{"status":"ran","hash_1":%s,"hash_2":%s,"identical":%s},' \
        "$(json_str "$h1")" "$(json_str "$h2")" "$([ "$h1" = "$h2" ] && echo true || echo false)"; } >> "$tmp"
  else
    printf '"greedy-ids":{"status":"skipped","reason":"needs MINFER_MODEL"},' >> "$tmp"
  fi

  # --- the device-free path (Chapter 21/27: a backend must not change the answer)
  if $have_model; then
    local hc
    hc=$(MINFER_DISABLE_MPS=1 "$BIN" "$MODEL" "The capital of France is" -n 16 --greedy --seed 42 2>/dev/null \
         | shasum -a 256 | cut -c1-16)
    { printf '"backend-cpu":{"status":"ran","hash":%s},' "$(json_str "$hc")"; } >> "$tmp"
  else
    printf '"backend-cpu":{"status":"skipped","reason":"needs MINFER_MODEL"},' >> "$tmp"
  fi

  # --- server surface (Chapter 29)
  if $have_model && command -v curl >/dev/null; then
    "$BIN" serve --port 8099 "$MODEL" >/dev/null 2>&1 &
    local pid=$!; sleep 4
    local health keys
    health=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8099/health || echo 000)
    keys=$(curl -s http://127.0.0.1:8099/metrics 2>/dev/null | grep -oE '^[a-z_]+' | sort -u | tr '\n' ',' | sed 's/,$//')
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    { printf '"serve-keys":{"status":"ran","health":%s,"metric_prefixes":%s},' "$(json_str "$health")" "$(json_str "$keys")"; } >> "$tmp"
  else
    printf '"serve-keys":{"status":"skipped","reason":"needs MINFER_MODEL and curl"},' >> "$tmp"
  fi

  # --- the CUDA gate line (Chapters 24-26)
  if $have_model && [ -x "$BIN" ] && "$BIN" --help 2>&1 | grep -q cuda; then
    local line
    line=$("$BIN" "$MODEL" "hi" -n 1 --backend cuda 2>&1 | grep -iE 'mmq|gate|device tier' | head -3 | tr '\n' ';')
    { printf '"cuda-gate":{"status":"ran","lines":%s}' "$(json_str "$line")"; } >> "$tmp"
  else
    printf '"cuda-gate":{"status":"skipped","reason":"needs a CUDA build, a device and MINFER_MODEL"}' >> "$tmp"
  fi

  {
    printf '{\n "recorded": %s,\n "baseline": {"commit": %s, "describe": %s},\n "probes": {' \
      "$(json_str "$date")" "$(json_str "$sha")" "$(json_str "$(git -C "$DIR" describe --tags --always 2>/dev/null || echo unknown)")"
    cat "$tmp"; printf '}\n}\n'
  } > "$dest"
  rm -f "$tmp"
  echo "wrote $dest"
}

diff_records() {
  local a="$1" b="$2"
  python3 - "$a" "$b" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["probes"]
b = json.load(open(sys.argv[2]))["probes"]
keys = sorted(set(a) | set(b))
changed = 0
for k in keys:
    x, y = a.get(k, {}), b.get(k, {})
    if x == y:
        continue
    changed += 1
    print(f"== {k}")
    for field in sorted(set(x) | set(y)):
        if x.get(field) != y.get(field):
            print(f"   {field}: {x.get(field)!r} -> {y.get(field)!r}")
print(f"{changed} probe(s) differ; a difference is a question, not a defect")
PY
}

case "${1:---list}" in
  --list) list_probes ;;
  --record) shift; record "${1:-}" ;;
  --diff) shift; diff_records "${1:?two records}" "${2:?two records}" ;;
  -h|--help) usage ;;
  *) usage; exit 2 ;;
esac
