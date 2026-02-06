# Corpus Summary (Phase 1 QC)
## Overview
- Documents: **400**
- Language/script: **Kazakh (kk-Latn)**
- Genre: **News articles (edited text)**
- Source: **qz.inform.kz (Kazinform)**
- Raw file: `data/raw/qz_kazakh_latn.jsonl`
- Processed file: `data/processed/qz_kazakh_latn_clean_norm.jsonl`
- Published date range (from `published_at_iso`): **2025-11-30T18:45:00+05:00 → 2026-02-02T10:03:00+05:00**
- Scraped date range (from `scraped_at`): **2026-02-02T06:15:10.820189Z → 2026-02-02T06:29:20.192548Z**

## Category distribution
-  50  category=`economics`  subcategory=`Ekonomika`
-  50  category=`kazakhstan`  subcategory=`Mádenıet`
-  50  category=`kazakhstan`  subcategory=`Sport`
-  50  category=`politics`  subcategory=`Halyqaralyq qatynastar`
-  50  category=`politics`  subcategory=`Prezıdent`
-  50  category=`world`  subcategory=`Amerıka`
-  50  category=`world`  subcategory=`Eýrazııa`
-  50  category=`world`  subcategory=`Ortalyq Aziıa`

## Text length statistics (characters)
- Raw `text`: {'min': 486, 'p50': 1601, 'mean': 2411, 'p95': 7206, 'max': 66137}
- Cleaned `clean_text`: {'min': 466, 'p50': 1594, 'mean': 2400, 'p95': 7186, 'max': 66118}

## Notes on preprocessing
- `clean_text` was created from `text` by removing common web/news artifacts (e.g., agency header line `ASTANA. KAZINFORM –`), URLs, and normalizing whitespace.
- Original raw content is preserved in `text`.
- A small number of long-form documents (e.g., speeches) are present; these are kept (no truncation) and will be noted in the datasheet and report.

## Longest cleaned documents (for awareness)
- 66118 chars | https://qz.inform.kz/news/prezidenttn-ulttik-kuriltaydin-besnsh-otirisinda-soylegen-soz-1241ee/ | Prezıdenttiń Ulttyq quryltaıdyń besinshi otyrysynda sóılegen sózi
- 14952 chars | https://qz.inform.kz/news/kasim-zhomart-tokaevtin-volonterler-foruminda-soylegen-soz-52f14f/ | Qasym-Jomart Toqaevtyń Volonterler forýmynda sóılegen sózi
- 13538 chars | https://qz.inform.kz/news/konstitutsiyalik-reforma-zhonndeg-komissiyanin-kuraminda-kmder-bar-fbcd0e/ | Konstıtýtsııalyq reforma jónindegi komıssııanyń quramynda kimder bar
- 12940 chars | https://qz.inform.kz/news/tmd-eldernde-kanday-kolzhetmd-turgin-uy-bagdarlamalari-bar-0d714c/ | TMD elderinde qandaı qoljetimdi turǵyn úı baǵdarlamalary bar
- 12533 chars | https://qz.inform.kz/news/tramp-vs-latin-amerikasi-batis-zhartisharda-sogis-bastala-ma-fa8189/ | Tramp vs Latyn Amerıkasy: Batys jartysharda soǵys bastala ma
