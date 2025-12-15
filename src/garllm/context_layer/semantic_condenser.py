#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAR Semantic Condenser（condenser統合版）

【役割】
- retriever（cleaner統合済）が生成した "description" を入力として受け取る。
- LLM による意味要約（summary）を生成する。
- LLM が失敗した場合は簡易バックアップ要約（元 condenser）を使う。
- condenser.py は廃止され、このファイルのみが意味抽出レイヤとなる。

【入力】
[
  {
    "title": "...",
    "url": "...",
    "description": "本文テキスト..."
  },
  ...
]

【出力】
[
  {
    "title": "...",
    "url": "...",
    "summary": "意味要約（LLM or fallback）"
  },
  ...
]
"""

import json
import argparse
from pathlib import Path
import re

from garllm.utils.llm_client import request_llm
from garllm.utils.env_utils import get_data_path


# ============================================================
# Fallback: 簡易要約（旧 condenser.py より統合）
# ============================================================

def naive_summarize(text: str, ratio: float = 0.2, max_sentences: int = 5):
    """
    LLM が失敗した際に使用する単純なバックアップ要約。
    - 文を句読点で分割
    - 出現単語の頻度でスコアリング
    - 上位 max_sentences 件を連結
    """
    if not text:
        return ""

    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if not sentences:
        return ""

    words = re.findall(r'\w+', text)
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    scored = []
    for s in sentences:
        score = sum(freq.get(w, 0) for w in re.findall(r'\w+', s))
        scored.append((score, s))

    scored.sort(reverse=True, key=lambda x: x[0])
    selected = [s for _, s in scored[:max_sentences]]

    return " ".join(selected)


def llm_summarize(text: str, title: str = "") -> str:
    """
    description を人物中心の意味要約に変換する。

    ねらい:
      - 後段の thought_profiler / persona_generator が、
        「この資料はこの人物について何を言っているか」を
        ひと目で分かるようにする。
    """
    if not text or len(text.strip()) < 10:
        return ""

    clipped = text[:4000]

    prompt = f"""
以下は、ある人物に関するウェブ記事の本文です。
この人物についての情報だけに焦点を当てて、120〜200文字程度の日本語で要約してください。

【要約に必ず含めること】
- その人物が「何者か」（職業・役割・分野など）
- 活躍した時代や期間（本文に記載がある場合のみ）
- 主な功績・活動・出来事（代表的なものを2〜3個）
- 関連する固有名詞（人名・組織名・地名・作品名などの重要なもの）

【省くべきこと】
- ウェブサイトの案内文や注意書き
- 「この記事では〜を紹介します」などメタな説明
- 広告・メニュー・フッターなど本文以外の情報

【スタイル】
- 三人称の説明文で書く（です・ます調でもだ・である調でもよい）
- 箇条書きは使わず、1〜2文の連続した文章にする
- 「この文章では〜」などのメタ発言はしない

【文化的背景として含めるべき事項】
- 出身地（市町村・都道府県）
- 育った地域（分かれば）
- 主な活動地域（国内外）
- 使用言語（日本語・英語など）
- 国際的な背景（移住・外国籍など）

【注意】
- 出身地から具体的な方言を決めつけない。
  （例：大阪府出身だから「大阪弁」と断言しない）
- 文化的背景は、後段の人格生成が参照する “候補情報” としてまとめる。

タイトル: {title}

本文:
----
{clipped}
----
人物についての要約:
"""

    try:
        res = request_llm(
            messages=[
                {"role": "system", "content": "あなたは人物情報に特化した日本語要約アシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            endpoint_type="chat",
            max_tokens=320,
            temperature=0.2,
        )
        summary = (res or "").strip()
        if summary:
            return summary
    except Exception as e:
        print(f"[semantic_condenser] LLM要約失敗: {e}")

    # LLMが落ちた場合のフェイルセーフ（旧 condenser 相当）
    return naive_summarize(text)


# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_items(items):
    processed = []

    for entry in items:
        title = entry.get("title", "")
        url = entry.get("url", "")
        description = entry.get("description", "")

        summary = llm_summarize(description, title=title)

        processed.append({
            "title": title,
            "url": url,
            "summary": summary
        })

    return processed


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(data, name: str, output_path: str | None = None) -> Path:
    if output_path:
        path = Path(output_path)
    else:
        base = Path(get_data_path("semantic"))
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"semantic_{name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[semantic_condenser] 保存: {path}")
    return path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GAR Semantic Condenser（condenser統合版）")
    parser.add_argument("--input", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    print(f"[semantic_condenser] Loading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    processed = process_items(items)
    save_results(processed, args.persona, args.output)


if __name__ == "__main__":
    main()

