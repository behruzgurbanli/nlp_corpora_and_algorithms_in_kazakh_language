#!/usr/bin/env python3
"""
Kazinform (qz.inform.kz) scraper for Kazakh Latin news articles.

Design goals:
- Reusable from CLI/UI
- Parameterized via a config object (later YAML)
- Keep scraping logic close to your original implementation
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, Dict, List, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional: better extraction if installed
try:
    import trafilatura  # type: ignore
    HAS_TRAF = True
except Exception:
    HAS_TRAF = False


# -------------------------
# Config + result structs
# -------------------------

@dataclass(frozen=True)
class ScrapeConfig:
    base_url: str = "https://qz.inform.kz"

    # Categories and subcategories (category -> {subcategory_name: url})
    categories: Dict[str, Dict[str, str]] = None  # set default below safely

    pages_per_subcategory: int = 5
    articles_per_subcategory: int = 50
    max_articles_total: int = 600

    # Politeness/network
    sleep_sec: float = 1.0
    timeout_sec: int = 25
    user_agent: str = "Mozilla/5.0"
    respect_robots: bool = True

    # Content filtering
    min_text_len: int = 250
    use_trafilatura: str = "auto"  # "auto" | "true" | "false"

    # Output
    out_jsonl: str = "qz_kazakh_latn.jsonl"
    out_txt: str = "qz_kazakh_latn.txt"
    out_stats: str = "qz_category_stats.txt"

    def __post_init__(self):
        # dataclass(frozen=True) -> need object.__setattr__
        if self.categories is None:
            object.__setattr__(self, "categories", default_categories())


@dataclass
class ScrapeResult:
    planned_urls: int
    saved: int
    used_trafilatura: bool
    counts_by_bucket: Counter
    skipped_robots: int
    skipped_fetch: int
    skipped_too_short: int


def default_categories() -> Dict[str, Dict[str, str]]:
    # Same structure as your original script (kept as default).
    return {
        "world": {
            "Ortalyq Aziıa": "https://qz.inform.kz/category/ortalyk-aziya_c211/",
            "Eýrazııa": "https://qz.inform.kz/category/euraziya_c212/",
            "Amerıka": "https://qz.inform.kz/category/amerika_c214/",
        },
        "kazakhstan": {
            "Mádenıet": "https://qz.inform.kz/category/madeniet_s10030/",
            "Sport": "https://qz.inform.kz/category/sport_s10031/",
        },
        "politics": {
            "Prezıdent": "https://qz.inform.kz/category/prezident_c241/",
            "Halyqaralyq qatynastar": "https://qz.inform.kz/category/halykaralyk-katynastar_c244/",
        },
        "economics": {
            "Ekonomika": "https://qz.inform.kz/category/ekonomika_s10556/",
        },
    }


# -------------------------
# Robots cache
# -------------------------

class RobotsCache:
    def __init__(self):
        self.cache: Dict[str, RobotFileParser] = {}

    def allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{base}/robots.txt"

        if base not in self.cache:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # If robots.txt can't be read, be permissive (same as your original)
                pass
            self.cache[base] = rp

        rp = self.cache[base]
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True


# -------------------------
# Core helpers (mostly same logic)
# -------------------------

def fetch_html(url: str, *, timeout: int, user_agent: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def category_page_url(category_url: str, page: int) -> str:
    return category_url if page <= 1 else f"{category_url}?page={page}"


def extract_article_links_from_listing(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/news/"):
            links.append(urljoin(base_url, href))
    # Remove duplicates while preserving order
    return list(dict.fromkeys(links))


def clean_text_basic(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        return t or None
    if soup.title:
        return soup.title.get_text(" ", strip=True) or None
    return None


def extract_datetime_raw(soup: BeautifulSoup) -> Optional[str]:
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\b\d{1,2}:\d{2},\s*\d{2}\s+\S+\s+\d{4}\b", text)
    return m.group(0) if m else None


def extract_with_trafilatura(url: str) -> Optional[str]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text:
            return None
        return clean_text_basic(text)
    except Exception:
        return None


def extract_with_bs4_fallback(html: str, min_text_len: int) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")

    art = soup.find("article")
    if art:
        t = art.get_text("\n", strip=True)
        if t and len(t) > 300:
            return clean_text_basic(t)

    ps = soup.find_all("p")
    blob = "\n".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))
    blob = re.sub(r"\bKAZINFORM\b", "", blob, flags=re.IGNORECASE)
    blob = clean_text_basic(blob)
    return blob if blob and len(blob) >= min_text_len else None


def scrape_article(url: str, *, cfg: ScrapeConfig) -> Optional[dict]:
    html = fetch_html(url, timeout=cfg.timeout_sec, user_agent=cfg.user_agent)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")
    title = extract_title(soup)
    dt_raw = extract_datetime_raw(soup)

    use_traf = (cfg.use_trafilatura == "true") or (cfg.use_trafilatura == "auto" and HAS_TRAF)
    text = None

    if use_traf and HAS_TRAF:
        text = extract_with_trafilatura(url)

    if not text:
        text = extract_with_bs4_fallback(html, cfg.min_text_len)

    if not text or len(text) < cfg.min_text_len:
        return None

    return {
        "url": url,
        "domain": urlparse(url).netloc,
        "title": title,
        "datetime_raw": dt_raw,
        "text": text,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "lang_script": "kk-Latn",
        "source": "qz.inform.kz",
    }


def write_stats(stats_path: str, total: int, counts: Counter) -> None:
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Total articles: {total}\n")
        f.write("Articles per category:\n")
        for cat, n in counts.most_common():
            f.write(f"- {cat}: {n}\n")


# -------------------------
# Public entrypoint
# -------------------------

def scrape(cfg: ScrapeConfig) -> ScrapeResult:
    """
    Full scrape run:
    1) Collect article links from listings
    2) Scrape each article
    3) Write JSONL + TXT + STATS
    """
    robots = RobotsCache()

    jobs: List[Tuple[str, str, str]] = []
    seen = set()

    skipped_robots = 0
    skipped_fetch = 0
    skipped_too_short = 0

    # 1) Collect article links
    for cat, subcats in cfg.categories.items():
        for subcat_name, subcat_url in subcats.items():
            collected = 0

            for page in range(1, cfg.pages_per_subcategory + 1):
                if len(jobs) >= cfg.max_articles_total:
                    break

                listing_url = category_page_url(subcat_url, page)

                if cfg.respect_robots and (not robots.allowed(listing_url, cfg.user_agent)):
                    skipped_robots += 1
                    continue

                html = fetch_html(listing_url, timeout=cfg.timeout_sec, user_agent=cfg.user_agent)
                time.sleep(cfg.sleep_sec)

                if not html:
                    skipped_fetch += 1
                    continue

                links = extract_article_links_from_listing(html, cfg.base_url)
                for link in links:
                    if len(jobs) >= cfg.max_articles_total:
                        break
                    if link not in seen and collected < cfg.articles_per_subcategory:
                        seen.add(link)
                        jobs.append((cat, subcat_name, link))
                        collected += 1

                if collected >= cfg.articles_per_subcategory:
                    break

            # next subcat

    # 2) Scrape articles
    counts = Counter()
    saved = 0

    with open(cfg.out_jsonl, "w", encoding="utf-8") as f_json, open(cfg.out_txt, "w", encoding="utf-8") as f_txt:
        for cat, subcat_name, url in tqdm(jobs, desc="Scraping articles"):
            if cfg.respect_robots and (not robots.allowed(url, cfg.user_agent)):
                skipped_robots += 1
                continue

            item = scrape_article(url, cfg=cfg)
            time.sleep(cfg.sleep_sec)

            if not item:
                # Could be fetch fail or too short; we can't perfectly separate without deeper logging,
                # but we can approximate:
                skipped_too_short += 1
                continue

            item["category"] = cat
            item["subcategory"] = subcat_name

            f_json.write(json.dumps(item, ensure_ascii=False) + "\n")
            one_line = " ".join(item["text"].split())
            f_txt.write(one_line + "\n")

            counts[f"{cat} - {subcat_name}"] += 1
            saved += 1

    # 3) Write stats
    write_stats(cfg.out_stats, saved, counts)

    used_traf = (cfg.use_trafilatura == "true") or (cfg.use_trafilatura == "auto" and HAS_TRAF)

    return ScrapeResult(
        planned_urls=len(jobs),
        saved=saved,
        used_trafilatura=bool(used_traf and HAS_TRAF),
        counts_by_bucket=counts,
        skipped_robots=skipped_robots,
        skipped_fetch=skipped_fetch,
        skipped_too_short=skipped_too_short,
    )
