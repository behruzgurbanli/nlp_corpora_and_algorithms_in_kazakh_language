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

echo "=== Scraping: qz.inform.kz ==="
python -m nlp_project.cli scrape \
  --config configs/scrape_qz_inform.yaml
echo

echo "✅ Scraping completed successfully"
