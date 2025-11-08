#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
condenser.py — GAR Context Layer: Condenser (SudachiPy版)
-----------------------------------------------------
Cleaner層出力JSONを要約・圧縮し、主要トピックを抽出する。
fugashi(MeCab)を排除し、SudachiPyで完全Python化。
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# ============================================================
# 形態素解析: SudachiPy による安定・高速分かち書き
# ============================================================
try:
    from sudachipy import dictionary, tokenizer
    tokenizer_obj = dictionary.Dictionary().create()
    split_mode = tokenizer.Tokenizer.SplitMode.C  # 最も細かい分割
except Exception as e:
    print(f"[ERROR] SudachiPy initialization failed: {e}")
    sys.exit(1)


# ============================================================
# 要約関数（簡易TFスコア）
# ============================================================
def naive_summarize(text: str, ratio: float = 0.2, max_sentences: int = 5):
    """頻度スコアによる簡易要約"""
    sentences = re.split(r'(?<=[。！？])', text)
    if len(sentences) <= 2:
        return text

    words = re.findall(r'\w+', text)
    freq = Counter(words)
    scored = [(sum(freq.get(w, 0) for w in re.findall(r'\w+', s)), s) for s in sentences]
    top = [s for _, s in sorted(scored, reverse=True)[:max_sentences]]
    return ''.join(top).strip()


# ============================================================
# キーワード抽出 (SudachiPy)
# ============================================================
def extract_tags(text: str, topn: int = 3):
    """名詞を抽出して出現頻度順にソート"""
    try:
        tokens = tokenizer_obj.tokenize(text, split_mode)
        nouns = [m.surface() for m in tokens if m.part_of_speech()[0] == "名詞"]
        freq = Counter([n for n in nouns if len(n) > 1])
        return [w for w, _ in freq.most_common(topn)]
    except Exception as e:
        print(f"[WARN] Tokenization failed: {e}")
        return []


# ============================================================
# 保存共通化
# ============================================================
def save_results(data, filename, output_path=None):
    """出力先を共通管理"""
    try:
        from garllm.utils.env_utils import get_data_path
    except ImportError:
        get_data_path = lambda subdir: str(Path.home() / f"data/{subdir}")

    if output_path:
        path = Path(output_path)
    else:
        path = Path(get_data_path("condensed")) / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {len(data)} entries -> {path}")
    return path


# ============================================================
# メイン処理
# ============================================================
def condense_json(input_path: str, output_path: str = None):
    """Cleaner出力JSONを読み込み、要約とタグを生成"""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    condensed = []

    for entry in data:
        clean = entry.get("clean_text", "")
        summary = naive_summarize(clean)
        tags = extract_tags(clean)
        condensed.append({
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "summary": summary,
            "tags": tags
        })

    filename = f"condensed_{Path(input_path).stem}.json"
    return save_results(condensed, filename, output_path)


# ============================================================
# CLI 実行部
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GAR Context Layer: Condenser (SudachiPy版)",
        epilog="例: python3 condenser.py --input cleaned_織田信長.json"
    )
    parser.add_argument("--input", type=str, required=True, help="入力 JSON パス（cleaner 出力）")
    parser.add_argument("--output", type=str, default=None, help="出力 JSON パス（任意）")
    args = parser.parse_args()

    print("=== Context Layer: Condenser (SudachiPy版) ===")
    condense_json(args.input, args.output)
