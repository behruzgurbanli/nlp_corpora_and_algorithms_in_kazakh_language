#!/usr/bin/env python3
"""
Normalize metadata:
- doc_id = sha1(url)[:16]
- published_at_iso parsed from datetime_raw, else None

Keeps original regex + month mapping + GMT offset logic.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


DT_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*,\s*(?P<d>\d{1,2})\s+(?P<mon>[A-Za-zÁáÓóÚúǴǵŃńÝýİıŊŋŞşÇçÖöÜüÄäḠḡ\-]+)\s+(?P<y>\d{4})",
    flags=re.UNICODE,
)
TZ_RE = re.compile(r"GMT\s*([+-])\s*(\d{1,2})", flags=re.IGNORECASE)

MONTHS = {
    "Qañtar": 1, "Qańtar": 1, "Qantar": 1,
    "Aqpan": 2,
    "Naýryz": 3, "Nauryz": 3,
    "Sáýir": 4, "Sáýir": 4, "Sáuir": 4,
    "Mamyr": 5,
    "Maýsym": 6, "Mausym": 6,
    "Shilde": 7, "Şilde": 7,
    "Tamyz": 8,
    "Qyrkúıek": 9, "Qyrkuyek": 9, "Qyrkүйek": 9,
    "Qazan": 10, "Qazán": 10,
    "Qarasha": 11,
    "Jeltoqsan": 12,
}


@dataclass(frozen=True)
class MetadataConfig:
    inp_jsonl: Path
    out_jsonl: Path
    default_gmt_offset_hours: int = 5  # keep original assumption


def make_doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def parse_published_iso(datetime_raw: Any, *, default_gmt_offset_hours: int) -> Optional[str]:
    if not isinstance(datetime_raw, str) or not datetime_raw.strip():
        return None

    m = DT_RE.search(datetime_raw)
    if not m:
        return None

    h = int(m.group("h"))
    mi = int(m.group("m"))
    d = int(m.group("d"))
    y = int(m.group("y"))
    mon_str = m.group("mon")

    # Normalize some hyphen/dot variants (same as your original)
    mon_str = mon_str.replace(".", "").strip()

    month = MONTHS.get(mon_str)
    if not month:
        return None

    tz_m = TZ_RE.search(datetime_raw)
    if tz_m:
        sign = 1 if tz_m.group(1) == "+" else -1
        hours = int(tz_m.group(2))
        tz = timezone(sign * timedelta(hours=hours))
    else:
        tz = timezone(timedelta(hours=default_gmt_offset_hours))

    dt = datetime(y, month, d, h, mi, tzinfo=tz)
    return dt.isoformat()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_metadata(cfg: MetadataConfig) -> Dict[str, int]:
    if not cfg.inp_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {cfg.inp_jsonl}")

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    parsed = 0

    with cfg.out_jsonl.open("w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(cfg.inp_jsonl):
            url = obj.get("url", "")
            obj["doc_id"] = make_doc_id(url)

            iso = parse_published_iso(obj.get("datetime_raw", ""), default_gmt_offset_hours=cfg.default_gmt_offset_hours)
            if iso:
                obj["published_at_iso"] = iso
                parsed += 1
            else:
                obj["published_at_iso"] = None

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    return {
        "docs_processed": n,
        "published_parsed": parsed,
        "published_missing": n - parsed,
    }
