#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_condenser.py — GAR Context Layer: Semantic Condenser (persona-agnostic)
-------------------------------------------------------------------------------
Cleaner/Condenser の出力（記事やテキストの配列 JSON）から本文を抽出し、
LLM（vLLM/OpenAI互換）で要約を生成・収集する“意味抽出”専用スクリプト。

※ persona は不要です。人物固有の抽出は persona_assimilator 側の責務です。

使い方:
  python3 semantic_condenser.py --input /path/to/condensed_*.json \
                                [--mode chat|completions] \
                                [--output /path/to/semantic_*.json]

- --input  : Condenser の出力 JSON（配列）を指定
- --mode   : LLM エンドポイント種別（既定: completions）
- --output : 出力先（未指定なら ~/data/semantic/semantic_<推定名>.json）
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

# GAR ルートをパスに追加（~/modules/garllm 配下想定）
sys.path.append(os.path.expanduser("~/modules/garllm"))

# 内部ユーティリティ
from utils.env_utils import get_data_path
from utils.llm_client import request_llm as request_openai

# ---- 抽出用キー定義 ---------------------------------------------------------
PRIMARY_TEXT_KEYS = [
    "clean_text", "content", "text", "body", "snippet", "description", "article",
    "raw_text", "summary"
]


LIST_TEXT_KEYS = ["texts", "chunks", "paragraphs"]

def extract_text(entry: Dict[str, Any]) -> str:
    """辞書から本文らしき最初の値を抽出。無ければ空文字を返す。"""
    for k in PRIMARY_TEXT_KEYS:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in LIST_TEXT_KEYS:
        v = entry.get(k)
        if isinstance(v, list) and v:
            joined = "\n".join(str(x) for x in v if str(x).strip())
            if joined.strip():
                return joined.strip()
    return ""

def summarize(text: str, title: str = "", mode: str = "completions") -> str:
    """LLM で 100〜200 文字の要約を生成。"""
    if not text or len(text.strip()) < 10:
        return ""
    prompt = (
        "以下の記事内容を100〜200文字で要約してください。"
        "固有名詞・日時・制度名などの重要情報は保持し、冗長な表現は避けてください。\n"
        f"タイトル: {title}\n本文:\n----\n{text[:4000]}\n----\n要約:"
    )
    try:
        if mode == "chat":
            return (request_openai(messages=[{"role": "user", "content": prompt}],
                                   endpoint_type="chat") or "").strip()
        else:
            return (request_openai(prompt=prompt, endpoint_type="completions") or "").strip()
    except Exception as e:
        print(f"[ERROR summarize] {e}")
        return ""

def infer_semantic_path(input_path: Path) -> Path:
    """
    出力パス未指定時の既定パスを推定:
      ~/data/semantic/semantic_<name>.json
    - 入力ファイル名が condensed_<name>.json → <name> を再利用
    - それ以外は stem を使う
    """
    base_dir = Path(get_data_path("semantic"))
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem  # 例) condensed_徳川家康
    m = re.match(r"condensed_(.+)", stem)
    name = m.group(1) if m else stem
    # スラッシュ等を安全化
    safe = re.sub(r"[^\w\-\u3040-\u30FF\u4E00-\u9FFF]+", "_", name)
    return base_dir / f"semantic_{safe}.json"

def save_results(data: List[Dict[str, Any]], output_path: Path) -> Path:
    """結果を JSON 保存。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(data)} entries -> {output_path}")
    return output_path

def run_semantic_condense(input_path: str, output_path: str | None = None, mode: str = "completions"):
    infile = Path(input_path)
    if not infile.exists():
        print(f"[ERROR] Input not found: {infile}")
        sys.exit(1)

    # 出力ファイル名の既定を決定
    out_path = Path(output_path) if output_path else infer_semantic_path(infile)

    # 入力読み込み（配列想定）
    try:
        data: List[Dict[str, Any]] = json.loads(infile.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to load JSON: {e}")
        sys.exit(1)

    out: List[Dict[str, Any]] = []
    total = len(data) if isinstance(data, list) else 0
    ok = 0
    skipped = 0

    for idx, entry in enumerate(data, 1):
        if not isinstance(entry, dict):
            skipped += 1
            continue
        title = (entry.get("title") or "").strip()
        url = entry.get("url") or ""
        source = entry.get("source") or ""

        text = extract_text(entry)
        if not text:
            print(f"⚠️ Skip(no-text) [{idx}/{total}]: {title or url}")
            skipped += 1
            continue

        summary = summarize(text, title, mode)
        if not summary:
            print(f"⚠️ Skip(empty-summary) [{idx}/{total}]: {title or url}")
            skipped += 1
            continue

        out.append({
            "title": title,
            "summary": summary,
            "source": source,
            "url": url
        })
        ok += 1

    save_results(out, out_path)
    print(f"✅ Semantic condensation complete: {ok}/{total} (skipped={skipped})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAR Context Layer: Semantic Condenser (persona-agnostic)",
        epilog=(
            "例:\n"
            "  python3 semantic_condenser.py --input ~/data/condensed/condensed_徳川家康.json\n"
            "  python3 semantic_condenser.py --input ~/data/condensed/condensed_記事.json --mode chat\n"
            "  python3 semantic_condenser.py --input ./condensed_x.json --output ./semantic_x.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Condenser出力のJSONファイル（配列）")
    parser.add_argument("--mode", choices=["chat", "completions"], default="completions",
                        help="使用するエンドポイントモード（既定: completions）")
    parser.add_argument("--output", type=str, default=None, help="出力JSONパス（任意）")
    args = parser.parse_args()

    print("=== Context Layer: Semantic Condenser ===")
    run_semantic_condense(args.input, args.output, args.mode)

