#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
persona_assimilator.py（完全版・一問一答モード・統合構造対応）
------------------------------------------------------------
目的:
  - thought_*.json から思想・文体・背景情報を統合し、 persona_*.json を生成。
  - vLLM(OpenAI互換API)を systemctl 経由で自動検出して利用。
  - LLM出力はすべて一問一答形式のプレーンテキスト。JSONパース依存なし。
  - relay_server / CLI 両対応。

使い方:
  python3 persona_assimilator.py \
    --input ~/data/thoughts/thought_<人物名>.json \
    --persona <人物名> [--debug]

出力:
  ~/data/personas/persona_<人物名>.json
"""

import os
import sys
import json
import re
import argparse
from typing import Any, Dict, List

#sys.path.append(os.path.expanduser("~/modules/gar-llm/src/"))
from garllm.utils.env_utils import get_data_path
from garllm.utils.llm_client import request_llm as request_openai

# ================================================================
# ディレクトリ設定
# ================================================================
PERSONA_DIR = get_data_path("personas")
os.makedirs(PERSONA_DIR, exist_ok=True)


# ================================================================
# vLLM呼び出し（プレーンテキスト一問一答）
# ================================================================
def ask_vllm_text(prompt: str, temperature: float = 0.25, max_tokens: int = 256, debug: bool = False) -> str:
    """LLM呼び出し: プレーンテキスト応答"""
    try:
        response = request_openai(
            messages=[
                {"role": "system", "content": "あなたは正確で簡潔な回答を行う日本語アシスタントです。JSONは禁止。"},
                {"role": "user", "content": prompt}
            ],
            endpoint_type="chat",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if debug:
            print(f"[DEBUG vLLM raw output]\n{response}\n")
        return response.strip()
    except Exception as e:
        print(f"[persona_assimilator] vLLM error: {e}")
        return ""


# ================================================================
# テキスト正規化
# ================================================================
def lines_to_list(s: str, limit: int = 5) -> List[str]:
    """LLM出力を改行・句読点で分割しクリーンアップ"""
    if not s:
        return []
    s = s.replace("・", "\n").replace("—", "-").replace("―", "-")
    s = re.sub(r"^[\s\-\*\d\.\)（）・]+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^発話文末の語尾表現.*", "", s, flags=re.MULTILINE)
    parts = re.split(r"[\n,、。]+", s)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^[\-\*\.\s]+", "", p)
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned[:limit]


# ================================================================
# 抽出関数群
# ================================================================
def extract_anchors(persona_name: str, summary: str, debug=False):
    prompt = (
        f"{persona_name} の思想や行動の背景となる重要な概念・経験・信念を3〜5個挙げてください。"
        f"各行に「概念 — 出典や背景 — 重要性(40〜80文字)」の形式で書いてください。"
        f"思想概要: {summary}"
    )
    raw = ask_vllm_text(prompt, max_tokens=600, debug=debug)
    raw = raw.replace("―", "-").replace("–", "-").replace("—", "-")

    anchors = []
    for line in raw.splitlines():
        line = line.strip(" -*・\t")
        if not line or len(line) < 3:
            continue
        parts = re.split(r"\s*(?:[-–—―]|:|：|–)\s*", line, maxsplit=2)
        if len(parts) == 3:
            label, reference, significance = parts
        elif len(parts) == 2:
            label, significance = parts
            reference = ""
        else:
            label = parts[0]
            reference = significance = ""
        if label.strip():
            anchors.append({
                "type": "concept",
                "label": label.strip(),
                "reference": reference.strip(),
                "significance": significance.strip()
            })
    return anchors[:5]


def extract_style(persona_name: str, summary: str, debug=False):
    """発話スタイル・語尾・キーワード抽出"""
    prompts = {
        "first_person": f"{persona_name} が自分を指す一人称を1つだけ日本語で答えてください。出力は1行のみ。思想概要: {summary}",
        "second_person": f"{persona_name} が他者を呼ぶときの二人称を1つだけ日本語で答えてください。出力は1行のみ。思想概要: {summary}",
        "speech_suffix": (
            f"{persona_name} の発話文末によく現れる語尾や言い回しを3〜5個、各行に1つずつ列挙してください。"
            f"説明や例文は不要です。思想概要:{summary}"
        ),
        "keywords": f"{persona_name} の思想を象徴する語彙を5つ挙げてください。思想概要:{summary}"
    }
    style = {}
    for key, prompt in prompts.items():
        raw = ask_vllm_text(prompt, max_tokens=300, debug=debug)
        style[key] = lines_to_list(raw, limit=5)
    return style


def extract_expression_prompt(persona_name: str, summary: str, style: Dict[str, Any], debug=False) -> str:
    """文体ガイド生成"""
    style_summary = json.dumps(style, ensure_ascii=False)
    prompt = (
        f"{persona_name} の人格・価値観・話し方を反映した文体ガイドを1文で日本語で書いてください。"
        f"例:『断定的で威厳ある口調。歴史的事象を語るように話す。』"
        f"思想概要:{summary} 文体情報:{style_summary}"
    )
    return ask_vllm_text(prompt, temperature=0.3, max_tokens=150, debug=debug)


# ================================================================
# persona統合処理
# ================================================================
def extract_persona_profile(thought_data: Dict[str, Any], persona_name: str, debug=False) -> Dict[str, Any]:
    """思想＋スタイル情報を統合してPersonaデータ生成"""
    summary = thought_data.get("summary", "")
    values = thought_data.get("values", [])
    reasoning_pattern = thought_data.get("reasoning_pattern", "")
    speech_pattern = thought_data.get("speech_pattern", "")

    anchors = extract_anchors(persona_name, summary, debug)
    style = extract_style(persona_name, summary, debug)
    expression_prompt = extract_expression_prompt(persona_name, summary, style, debug)

    return {
        "persona_name": persona_name,
        "core_profile": {
            "summary": summary,
            "values": values if isinstance(values, list) else [values],
            "reasoning_pattern": reasoning_pattern,
            "speech_pattern": speech_pattern,
            "knowledge_anchors": anchors
        },
        "style": style,
        "expression_prompt": expression_prompt
    }


def load_thought(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_persona_profile(persona: Dict[str, Any], persona_name: str):
    """生成したペルソナを保存"""
    path = os.path.join(PERSONA_DIR, f"persona_{persona_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)
    print(f"[persona_assimilator] {path} を生成しました。")


# ================================================================
# main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="GAR Persona Layer: Persona Assimilator")
    parser.add_argument("--input", required=True, help="入力 thought_*.json パス")
    parser.add_argument("--persona", required=True, help="対象人物名")
    parser.add_argument("--debug", action="store_true", help="デバッグ出力を有効化")
    args = parser.parse_args()

    thought_data = load_thought(args.input)
    persona = extract_persona_profile(thought_data, args.persona, debug=args.debug)
    save_persona_profile(persona, args.persona)


if __name__ == "__main__":
    main()
