#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------
# Scraping pipeline runner
# --------------------------------------
# Uses YAML config to scrape qz.inform.kz
# --------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src"

SCRAPE_CONFIG="${1:-configs/scrape_qz_inform.yaml}"

echo "=== Scraping: qz.inform.kz ==="
python -m nlp_project.cli scrape \
  --config "$SCRAPE_CONFIG"
echo

echo "✅ Scraping completed successfully"
