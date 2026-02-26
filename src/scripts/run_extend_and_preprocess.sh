#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Uses append mode and larger crawl budget to grow existing dataset.
SCRAPE_CONFIG="${1:-configs/scrape_qz_inform_extend.yaml}"

echo "Project root: $PROJECT_ROOT"
echo

echo "=== [1/2] Extend dataset (scrape append) ==="
"$PROJECT_ROOT/src/scripts/run_scrape.sh" "$SCRAPE_CONFIG"
echo

echo "=== [2/2] Rebuild processed artifacts ==="
"$PROJECT_ROOT/src/scripts/run_preprocess.sh"
echo

echo "✅ Dataset extension + preprocessing completed"
