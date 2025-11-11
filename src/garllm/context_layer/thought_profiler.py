#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thought_profiler.py — GAR Context Layer: Thought Profiler
------------------------------------------------------------
目的:
  - condensed_*.json（または指定入力）から人物の思想・価値観を抽出
  - vLLM(OpenAI互換API)経由でプロファイル生成
  - relay_server / CLI 両対応
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# sys.path.append(os.path.expanduser("~/modules/"))
from garllm.utils.env_utils import get_data_path
#from garllm.utils.vllm_client import request_openai
from garllm.utils.llm_client import request_llm as request_openai


# ================================================================
# Utility Functions
# ================================================================
def load_condensed_files(input_path=None):
    """condensed_*.jsonをロード。指定があればそれを優先"""
    entries = []
    if input_path and Path(input_path).exists():
        files = [input_path]
    else:
        files = sorted(glob.glob("condensed_*.json"))
    for f in files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            if isinstance(data, list):
                entries.extend(data)
            elif isinstance(data, dict):
                entries.append(data)
            print(f"[Thought Profiler] 読込成功: {f} ({len(entries)}件)")
        except Exception as e:
            print(f"[Thought Profiler] 読込失敗: {f} ({e})")
    return entries


def ask_vllm(prompt, max_tokens=400, temperature=0.3, mode="completions"):
    """vLLM API呼び出し (request_openai)"""
    try:
        if mode == "chat":
            resp = request_openai(messages=[{"role": "user", "content": prompt}], endpoint_type="chat", max_tokens=max_tokens, temperature=temperature)
        else:
            resp = request_openai(prompt=prompt, endpoint_type="completions", max_tokens=max_tokens, temperature=temperature)
        return resp.strip()
    except Exception as e:
        print(f"[Thought Profiler] vLLM呼出エラー: {e}")
        return ""


def calc_score(profile):
    """プロファイル品質スコア"""
    try:
        t = json.dumps(profile, ensure_ascii=False)
        len_score = min(len(t) / 800, 1.0)
        key_score = sum(k in profile for k in ["summary", "values", "reasoning_pattern", "speech_pattern"]) / 4
        uniq = len(set("".join(profile.get("values", [])))) / 50 if isinstance(profile.get("values"), list) else 0
        return round((len_score * 0.4 + key_score * 0.4 + uniq * 0.2), 2)
    except Exception:
        return 0.0


def evaluate_consistency(old, new):
    """前回と今回の類似度比較"""
    try:
        s1 = json.dumps(old, ensure_ascii=False)
        s2 = json.dumps(new, ensure_ascii=False)
        return 1 - SequenceMatcher(None, s1, s2).ratio()
    except Exception:
        return 0.5


# ================================================================
# 思想抽出
# ================================================================
def dialogue_extract(entries, persona, mode="completions"):
    """思想・価値観・思考傾向・文体分析"""
    texts = []
    for e in entries:
        for key in ("summary", "condensed_text", "clean_text"):
            if key in e and e[key]:
                texts.append(e[key])
                break
    if not texts:
        print("[Thought Profiler] 利用可能なテキストなし。")
        return None

    condensed = "\n".join(texts[:50])
    print(f"[Thought Profiler] 読込 summary数: {len(texts)}")

    qa = {
        "summary": f"{persona}の思想と価値観を200〜300字で要約してください。",
        "values": f"{persona}の価値観を3〜5語で具体化し、各々を簡潔に説明してください。",
        "reasoning_pattern": f"{persona}の思考や判断の傾向を述べ、どう行動する人物か説明してください。",
        "speech_pattern": f"{persona}の話し方や語彙の特徴を説明してください。"
    }

    result = {"persona_name": persona}
    for key, question in qa.items():
        print(f"[Thought Profiler] {key} 抽出中...")
        prompt = (
            f"以下は{persona}に関する資料です。次の質問に答えてください。\n"
            f"資料:\n{condensed[:2500]}\n"
            f"質問: {question}\n"
            "出力は日本語で簡潔に答えてください。"
        )
        ans = ask_vllm(prompt, mode=mode)
        result[key] = ans.strip()
    return result


# ================================================================
# 保存処理
# ================================================================
def save_profile(profile, persona, output_path=None):
    """結果を保存"""
    if output_path:
        path = Path(output_path)
    else:
        base_dir = Path(get_data_path("thoughts"))
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"thought_{persona}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"[Thought Profiler] 保存完了: {path}")
    return path


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="GAR Context Layer: Thought Profiler")
    parser.add_argument("--persona", required=True, help="対象人物名")
    parser.add_argument("--input", type=str, default=None, help="入力JSON（condenser出力）")
    parser.add_argument("--output", type=str, default=None, help="出力パス（任意）")
    parser.add_argument("--backup", action="store_true", help="既存プロファイルをバックアップ")
    parser.add_argument("--mode", choices=["chat", "completions"], default="completions", help="vLLM API モード")
    args = parser.parse_args()

    entries = load_condensed_files(args.input)
    if not entries:
        print("[Thought Profiler] 入力データなし。終了。")
        return

    persona = args.persona
    profile = dialogue_extract(entries, persona, args.mode)
    if not profile:
        print("[Thought Profiler] 抽出失敗。")
        return

    new_score = calc_score(profile)
    path = Path(get_data_path("thoughts")) / f"thought_{persona}.json"
    old_score = 0
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_score = calc_score(old)
        diff = evaluate_consistency(old, profile)
        print(f"[Thought Profiler] スコア比較: 旧={old_score}, 新={new_score}, 差分={diff:.2f}")
    else:
        diff = 1.0

    if new_score >= old_score or diff > 0.15:
        save_profile(profile, persona, args.output)
    else:
        print("[Thought Profiler] 改訂不要。")

    print("[Thought Profiler] 完了。")


if __name__ == "__main__":
    main()
