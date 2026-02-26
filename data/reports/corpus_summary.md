# Corpus Summary (Phase 1 QC)
## Overview
- Documents: **2473**
- Language/script: **Kazakh (kk-Latn)**
- Genre: **News articles (edited text)**
- Source: **qz.inform.kz (Kazinform)**
- Raw file: `data/raw/qz_kazakh_latn.jsonl`
- Processed file: `data/processed/qz_kazakh_latn_clean_norm.jsonl`
- Published date range (from `published_at_iso`): **2024-06-25T18:38:00+05:00 → 2026-02-26T12:35:00+05:00**
- Scraped date range (from `scraped_at`): **2026-02-02T06:15:10.820189Z → 2026-02-26T08:06:28.306421Z**

## Category distribution
- 523  category=`world`  subcategory=`Ortalyq Aziıa`
- 450  category=`kazakhstan`  subcategory=`Mádenıet`
- 450  category=`kazakhstan`  subcategory=`Sport`
- 450  category=`world`  subcategory=`Amerıka`
- 450  category=`world`  subcategory=`Eýrazııa`
-  50  category=`economics`  subcategory=`Ekonomika`
-  50  category=`politics`  subcategory=`Halyqaralyq qatynastar`
-  50  category=`politics`  subcategory=`Prezıdent`

## Text length statistics (characters)
- Raw `text`: {'min': 417, 'p50': 1410, 'mean': 1793, 'p95': 3710, 'max': 66137}
- Cleaned `clean_text`: {'min': 417, 'p50': 1410, 'mean': 1791, 'p95': 3710, 'max': 66118}

## Notes on preprocessing
- `clean_text` was created from `text` by removing common web/news artifacts (e.g., agency header line `ASTANA. KAZINFORM –`), URLs, and normalizing whitespace.
- Original raw content is preserved in `text`.
- A small number of long-form documents (e.g., speeches) are present; these are kept (no truncation) and will be noted in the datasheet and report.

## Longest cleaned documents (for awareness)
1. **66118 chars** — Prezıdenttiń Ulttyq quryltaıdyń besinshi otyrysynda sóılegen sózi
   - https://qz.inform.kz/news/prezidenttn-ulttik-kuriltaydin-besnsh-otirisinda-soylegen-soz-1241ee/
2. **17523 chars** — 2025 qorytyndysy: Mádenıet salasyndaǵy aıtýly oqıǵalar
   - https://qz.inform.kz/news/2025-koritindisi-madeniet-salasindagi-aytuli-okigalar-febdae/
3. **14952 chars** — Qasym-Jomart Toqaevtyń Volonterler forýmynda sóılegen sózi
   - https://qz.inform.kz/news/kasim-zhomart-tokaevtin-volonterler-foruminda-soylegen-soz-52f14f/
4. **13538 chars** — Konstıtýtsııalyq reforma jónindegi komıssııanyń quramynda kimder bar
   - https://qz.inform.kz/news/konstitutsiyalik-reforma-zhonndeg-komissiyanin-kuraminda-kmder-bar-fbcd0e/
5. **12940 chars** — TMD elderinde qandaı qoljetimdi turǵyn úı baǵdarlamalary bar
   - https://qz.inform.kz/news/tmd-eldernde-kanday-kolzhetmd-turgin-uy-bagdarlamalari-bar-0d714c/
