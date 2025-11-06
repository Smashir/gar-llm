#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retriever.py — GAR Context Layer: Online Retriever
--------------------------------------------------
指定されたクエリ（人物名など）に関する情報を DuckDuckGo で検索し、
記事本文を抽出して JSON に保存する。
relay_server.py から自動呼び出しされることを想定。
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from readability import Document
from ddgs import DDGS
from pathlib import Path

# ============================================================
# 設定
# ============================================================
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ContextRetriever/1.0; +https://localhost)"}
TIMEOUT = 10


# ============================================================
# DuckDuckGo検索
# ============================================================
def search_web(query: str, limit: int = 5):
    print(f"[search] '{query}' を検索中...")
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=limit):
            if "href" in r:
                urls.append({"title": r.get("title", ""), "url": r["href"]})
    return urls


# ============================================================
# 記事本文抽出
# ============================================================
def fetch_article(url: str):
    try:
        res = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        res.raise_for_status()
        doc = Document(res.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        print(f"[error] {url}: {e}")
        return ""


# ============================================================
# 検索・抽出統合
# ============================================================
def retrieve(query: str, limit: int = 5):
    """検索→抽出→整形→結果リスト返却"""
    results = []
    urls = search_web(query, limit)
    for u in urls:
        content = fetch_article(u["url"])
        if content:
            snippet = content[:2000]
            results.append({
                "title": u["title"],
                "url": u["url"],
                "content": snippet
            })
        time.sleep(1.5)
    return results


# ============================================================
# 結果保存関数（共通化ポイント）
# ============================================================
def save_results(data, query, output_path=None):
    """
    結果を保存。
    relay_server から output_path が指定されていればそれを優先。
    なければ env_utils のデフォルトパス ~/data/retrieved を使用。
    """
    try:
        from garllm.utils.env_utils import get_data_path
    except ImportError:
        # スタンドアロン実行時のフォールバック
        get_data_path = lambda subdir: os.path.expanduser(f"~/data/{subdir}")

    if output_path:
        path = Path(output_path)
    else:
        path = Path(get_data_path("retrieved")) / f"retrieved_{query}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(data)} items to {path}")
    return path


# ============================================================
# CLI 実行部
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GAR Context Layer: Online Retriever",
        epilog="例: python3 retriever.py --query 織田信長 --limit 5"
    )
    parser.add_argument("--query", type=str, required=True, help="検索クエリ（例：織田信長）")
    parser.add_argument("--limit", type=int, default=5, help="取得件数（デフォルト:5）")
    parser.add_argument("--output", type=str, default=None, help="出力先ファイルパス")
    args = parser.parse_args()

    print("=== Context Layer: Online Retriever ===")

    data = retrieve(args.query, limit=args.limit)
    save_results(data, args.query, args.output)
