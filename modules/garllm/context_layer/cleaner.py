#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleaner.py — GAR Context Layer: Cleaner
----------------------------------------
retriever の出力 JSON を正規化・整形するモジュール。
relay_server 経由または CLI から利用可能。
"""

import json
import re
import sys
from pathlib import Path


# ============================================================
# テキスト正規化
# ============================================================
def normalize_text(text: str) -> str:
    """改行・空白・広告文の削除など基本整形"""
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)
    text = re.sub(r"\r|\t", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \u3000]+", " ", text)
    for kw in ["この記事をシェア", "関連記事", "続きを読む", "提供："]:
        text = text.replace(kw, "")
    return text.strip()


# ============================================================
# 結果保存（共通）
# ============================================================
def save_results(data, filename, output_path=None):
    """
    結果を保存。
    relay_server から output_path が指定されていればそれを優先。
    なければ env_utils.get_data_path('cleaned') を使用。
    """
    try:
        from garllm.utils.env_utils import get_data_path
    except ImportError:
        get_data_path = lambda subdir: str(Path.home() / f"data/{subdir}")

    if output_path:
        path = Path(output_path)
    else:
        path = Path(get_data_path("cleaned")) / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(data)} entries -> {path}")
    return path


# ============================================================
# メイン処理
# ============================================================
def clean_json(input_path: str, output_path: str = None):
    """retrieved JSONを読み込み、clean_textを生成"""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cleaned = []

    for entry in data:
        raw = entry.get("content", "")
        clean = normalize_text(raw)
        cleaned.append({
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "clean_text": clean
        })

    filename = f"cleaned_{Path(input_path).stem}.json"
    return save_results(cleaned, filename, output_path)


# ============================================================
# CLI 実行部
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GAR Context Layer: Cleaner",
        epilog="例: python3 cleaner.py --input retrieved_織田信長.json"
    )
    parser.add_argument("--input", type=str, required=True, help="入力 JSON パス（retriever 出力）")
    parser.add_argument("--output", type=str, default=None, help="出力先 JSON パス（任意）")
    args = parser.parse_args()

    print("=== Context Layer: Cleaner ===")
    clean_json(args.input, args.output)
