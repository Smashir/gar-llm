#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAR Retriever（cleaner統合版）

【役割】
- DuckDuckGo で人物名などを検索し、上位のURLを取得する。
- 各URLから HTML をダウンロードし、記事本文を抽出する。
- cleaner.py の正規化処理（clean_text生成）を内部に統合している。
- 最終的に、semantic_condenser が読める JSON を生成する。

【出力形式】
[
  {
    "title": "記事タイトル",
    "url": "元URL",
    "clean_text": "正規化した本文テキスト"
  },
  ...
]

【使用例（CLI）】
python retriever.py --query "織田信長" --limit 5 --output /path/to/retrieved_織田信長.json

--query : 検索語（人物名）
--limit : 取得するURL件数（デフォルト5）
--output: 保存先パス（省略時は GAR の data ディレクトリに自動保存）

【備考】
- cleaner.py は廃止し、本文の正規化は retriever 内で完結している。
- semantic_condenser.py 以降は、このファイルが生成する JSON をそのまま使用する。
"""


import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse, parse_qs, unquote


from garllm.utils.env_utils import get_data_path

from garllm.utils.logger import get_logger

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
}

TIMEOUT = 15


# ロガー初期化
logger = get_logger("response_modulator", level="INFO", to_console=False)

# ============================================================
# TEXT NORMALIZATION（元 cleaner.py の処理を完全統合）
# 余計な改行・空白を除去し、後続処理が扱いやすい clean_text を生成する。
# ============================================================

def normalize_text(text: str) -> str:
    """cleaner.py の normalize_text を完全移植"""
    if not text:
        return ""

    # 改行と空白の正規化
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


# ============================================================
# DuckDuckGo SEARCH
# ============================================================

def _decode_ddg_redirect(href: str, base_url: str) -> str | None:
    if not href:
        return None

    abs_url = urljoin(base_url, href)

    try:
        u = urlparse(abs_url)
        qs = parse_qs(u.query)

        # DDGの /l/?uddg=... を実URLに復元
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])

        # 直リンクが混じる場合はそれを返す
        if u.scheme in ("http", "https") and "duckduckgo.com" not in u.netloc:
            return abs_url
    except Exception:
        return None

    return None


def ddg_search(query: str, limit: int = 20) -> List[str]:
    base_url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "jp-jp", "ia": "web"}

    try:
        res = requests.post(base_url, headers=HEADERS, data=params, timeout=TIMEOUT)
        res.raise_for_status()
    except Exception as e:
        print(f"[Retriever] DuckDuckGo 検索失敗: {e}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")

    urls: List[str] = []
    seen = set()

    candidates = soup.select("a.result__title, a.result__a")
    if not candidates:
        candidates = soup.find_all("a", href=True)

    for a in candidates:
        href = a.get("href") if hasattr(a, "get") else None
        real = _decode_ddg_redirect(href, base_url)
        if not real or real in seen:
            continue
        seen.add(real)
        urls.append(real)
        if len(urls) >= limit:
            break

    print(f"[Retriever] DuckDuckGo から {len(urls)} 件の URL を取得 (query='{query}')")
    return urls



# ============================================================
# ARTICLE BODY EXTRACTION
# BeautifulSoup を用いて、HTMLから本文らしい部分（main / article / <p>群）を抽出する。
# ここで生の本文（raw_text）を取り出し、後で normalize_text にかける。
# ============================================================


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")
    if main:
        text = main.get_text("\n", strip=True)
        if len(text) > 200:
            return text

    article = soup.find("article")
    if article:
        text = article.get_text("\n", strip=True)
        if len(text) > 200:
            return text

    ps = soup.find_all("p")
    if ps:
        return "\n".join([p.get_text(" ", strip=True) for p in ps])

    body = soup.find("body")
    if body:
        return body.get_text("\n", strip=True)

    return ""

# ============================================================
# fetch_article(url)
# URLから記事を取得し、本文抽出 → 正規化した clean_text を作成する。
# cleaner.py の出力形式と完全に互換の JSON dict を返す。
# ============================================================


def fetch_article(url: str) -> Dict[str, str]:
    logger.info(f"[Retriever] Fetching: {url}")

    try:
        res = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        res.raise_for_status()
        html = res.text
    except Exception as e:
        logger.info(f"[Retriever] URL取得失敗: {e}")
        return {"title": "", "url": url, "description": ""}

    # title
    title = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        title = tag.get_text(strip=True) if tag else ""
    except Exception:
        pass

    # content → cleaner の役割も統合
    raw_text = extract_main_text(html)
    clean_text = normalize_text(raw_text)

    # cleaner 出力構造に合わせる
    return {
        "title": title,
        "url": url,
        "description": clean_text
    }


# ============================================================
# retrieve(query)
# 検索語 query に基づいて DuckDuckGo から URL を収集し、
# 各URLを fetch_article() で処理して記事リストを作る。
# ============================================================


def retrieve(queries: List[str], limit: int = 5) -> List[Dict[str, str]]:
    """
    複数クエリで DuckDuckGo 検索を行い、
    URL 単位で dedupe した上で記事本文を取得する。
    """
    seen_urls: set[str] = set()
    results: List[Dict[str, str]] = []

    for q in queries:
        logger.info(f"[Retriever] 検索クエリ: {q}")
        urls = ddg_search(q, limit=limit)

        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)

            article = fetch_article(url)
            if article.get("description"):
                results.append(article)

            time.sleep(1.0)

    logger.info(f"[Retriever] 合計取得記事数（dedupe後）: {len(results)}")
    return results


# ============================================================
# save_results
# retriever の出力（retrieved_xxx.json）を保存する。
# cleaner.py 互換の場所に保存されるため、後続 semantic_condenser がそのまま使える。
# ============================================================


def save_results(data: List[Dict[str, str]], output_path: str | None = None) -> Path:
    """
    取得結果を JSON として保存する。
    ファイル名の意味づけは呼び出し側が行う。
    """
    if output_path:
        path = Path(output_path)
    else:
        base = Path(get_data_path("retrieved"))
        base.mkdir(parents=True, exist_ok=True)
        path = base / "retrieved.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"[Retriever] 保存: {path}")
    return path


# ============================================================
# CLI Entry Point
# retriever を単体で呼ぶためのインターフェース。
# --query のみ必須。--output を省略すると GAR のデフォルトパスへ保存される。
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="GAR Retriever + Cleaner統合版")
    parser.add_argument(
    "--queries",
    required=True,
    help="JSON配列形式の検索クエリ一覧（例: '[\"轟はじめ\",\"轟はじめ 話し方\"]'）'")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=str)
    parser.add_argument("--debug", action="store_true", help="デバッグ表示（プロンプト出力）")
    parser.add_argument("--log-console", action="store_true", help="ログをコンソールにも出力")     
    args = parser.parse_args()

    # ------------------------------
    # ロガー設定（--debug で制御）
    # ------------------------------
    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("retriever", level=log_level, to_console=args.log_console)
    logger.info(f"Retriever log_level={log_level})")

    queries = json.loads(args.queries)
    if not isinstance(queries, list) or not queries:
        raise ValueError("--queries は JSON 配列で指定してください")

    data = retrieve(queries, limit=args.limit)

    save_results(data, args.output)



if __name__ == "__main__":
    main()
