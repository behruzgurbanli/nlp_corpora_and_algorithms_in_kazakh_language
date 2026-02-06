#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------
# Preprocessing pipeline runner
# --------------------------------------
# Steps:
# 1) Audit raw JSONL
# 2) Clean text -> clean_text
# 3) Normalize metadata (doc_id, published_at_iso)
# 4) Generate corpus summary report
#
# All parameters come from YAML configs.
# --------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"

echo "=== [1/4] QC: audit raw dataset ==="
python -m nlp_project.cli qc audit-raw \
  --config configs/qc_audit_raw.yaml
echo

echo "=== [2/4] Preprocess: clean text ==="
python -m nlp_project.cli preprocess clean \
  --config configs/preprocess_clean.yaml
echo

echo "=== [3/4] Preprocess: normalize metadata ==="
python -m nlp_project.cli preprocess metadata \
  --config configs/preprocess_metadata.yaml
echo

echo "=== [4/4] QC: corpus summary ==="
python -m nlp_project.cli qc corpus-summary \
  --config configs/qc_corpus_summary.yaml
echo

echo "✅ Preprocessing pipeline completed successfully"
