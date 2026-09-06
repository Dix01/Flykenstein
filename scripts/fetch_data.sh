#!/usr/bin/env bash
# Fetch the public FlyWire FAFB v783 connectome exports (no login required)
# and the third-party model repos this project builds on.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=https://storage.googleapis.com/flywire-data/codex/data/fafb/783
OUT="$ROOT/data/flywire_783"
mkdir -p "$OUT" "$ROOT/third_party"

for f in connections.csv.gz classification.csv.gz consolidated_cell_types.csv.gz \
         cell_stats.csv.gz coordinates.csv.gz labels.csv.gz names.csv.gz \
         connectivity_tags.csv.gz neurons.csv.gz column_assignment.csv.gz \
         classification_with_fw_and_hemibrain_types.csv.gz; do
  echo "fetching $f"
  curl -sSfL --retry 4 --retry-delay 2 -o "$OUT/$f" "$BASE/$f"
done

for r in philshiu/Drosophila_brain_model flyconnectome/flywire_annotations TuragaLab/flyvis; do
  d="$ROOT/third_party/$(basename "$r")"
  [ -d "$d" ] || git clone --depth 1 -q "https://github.com/$r.git" "$d"
done

echo "compiling connectome"
python "$ROOT/scripts/build_connectome.py"
