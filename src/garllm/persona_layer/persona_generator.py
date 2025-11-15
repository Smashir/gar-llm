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
from garllm.utils.logger import get_logger

# ================================================================
# ディレクトリ設定
# ================================================================
PERSONA_DIR = get_data_path("personas")
os.makedirs(PERSONA_DIR, exist_ok=True)

# ============================================================
# ロガー設定（初期値はINFO、mainで上書き）
# ============================================================

logger = get_logger("persona_generator", level="INFO", to_console=False)

# ================================================================
# thought_*.json ローダー（復元版）
# ================================================================
def load_thought(path: str) -> Dict[str, Any]:
    """
    thought_profiler が生成した thought_*.json を読み込んで dict を返す。
    JSON破損時は空 dict を返す。
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        logger.error(f"[persona_generator] Thought file not found: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # "persona_name" がある形式なので、破損確認もかねて最低1キー確認
            if not isinstance(data, dict):
                raise ValueError("loaded thought is not a dict")
            return data
    except Exception as e:
        logger.error(f"[persona_generator] Failed to load thought file '{path}': {e}")
        return {}


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

        logger.debug(f"[DEBUG vLLM raw output]\n{response}\n")

        return response.strip()
    except Exception as e:
        logger.error(f"[persona_assimilator] vLLM error: {e}")
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
# Phase（相）生成ロジック
# ================================================================

def extract_phases(persona_name: str, summary: str, values: list, reasoning: str, speech_pattern: str, debug=False):
    """
    人物の「相（Phase）」を抽出する。
    ここでは LLM に JSON を出力させ、その JSON をパースして phase dict に変換する。

    返り値:
    {
      "戦略相": {
        "description": "...",
        "style_bias": {...},   # 6軸
        "emotion_bias": {...}, # 8軸
        "tone_hint": "..."
      },
      ...
    }
    """

    # -------- プロンプト定義 --------
    prompt = f"""
あなたの役割は、人物の「相（Phase）」を設計する専門家です。

【相（Phase）の定義】
ここでの「相」とは、その人物の
- 話し方や語気
- 感情の出し方
- 相手との距離感（支配的 / 共感的 など）
- 態度・雰囲気

がひとまとまりになった「振る舞いパターン」のことです。
1人の人物の中に複数の相があり、状況によって前に出る相が変化します。

あなたは、この人物の相を 3 つ設計し、それぞれに
- name: 相の名前（3〜8文字程度、日本語）
- description: その相がどのような振る舞いをするかの説明（日本語）
- style_bias: 対人スタンスのバイアス（-1.0〜1.0）
- emotion_bias: 感情のバイアス（-1.0〜1.0）
- tone_hint: その相で話すときの口調・話し方の説明

を与えてください。

【出力形式（厳守）】
次の JSON のみを出力してください。説明文やコメントは一切不要です。

```json
{{
  "phases": [
    {{
      "name": "相の名前1",
      "description": "説明1",
      "style_bias": {{
        "Trust": 0.0, "Familiarity": 0.0, "Hostility": 0.0,
        "Dominance": 0.0, "Empathy": 0.0, "Instrumentality": 0.0
      }},
      "emotion_bias": {{
        "joy": 0.0, "trust": 0.0, "fear": 0.0, "surprise": 0.0,
        "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.0
      }},
      "tone_hint": "口調の説明1"
    }},
    {{
      "name": "相の名前2",
      "description": "説明2",
      "style_bias": {{ ... 同様 ... }},
      "emotion_bias": {{ ... 同様 ... }},
      "tone_hint": "口調の説明2"
    }},
    {{
      "name": "相の名前3",
      "description": "説明3",
      "style_bias": {{ ... 同様 ... }},
      "emotion_bias": {{ ... 同様 ... }},
      "tone_hint": "口調の説明3"
    }}
  ]
}}

数値は必ず -1.0〜1.0 の実数で記述し、
キー名の綴りや大文字小文字は変更しないでください。

対象人物：{persona_name}

【思想概要】
{summary}

【価値観】
{values}

【思考様式】
{reasoning}

【話し方】
{speech_pattern}
    """

    # -------- LLM 呼び出し（JSON 解禁）--------
    try:
        # ここは ask_vllm_text を使わず、JSON を期待する専用呼び出しにする
        raw = request_openai(
            messages=[
                {
                    "role": "system",
                    "content": "あなたは指定されたスキーマに従って厳密な JSON を出力する日本語アシスタントです。"
                },
                {"role": "user", "content": prompt}
            ],
            endpoint_type="chat",
            max_tokens=900,
            temperature=0.3,
        )
        raw = raw.strip()
        logger.debug(f"\n[DEBUG extract_phases raw LLM output]\n{raw}\n")
        
    except Exception as e:
        logger.error(f"[persona_generator] LLM error in extract_phases: {e}")
        return {}

    # -------- JSON 抽出（context_controller と同じ発想）--------
    try:
        # ```json ... ``` を優先
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            # 最初の { ... } ブロックを拾う
            m2 = re.search(r"\{[\s\S]*\}", raw, re.DOTALL)
            json_str = m2.group(0) if m2 else raw

        logger.debug("\n[DEBUG extract_phases candidate JSON]\n", json_str, "\n")

        parsed = json.loads(json_str)
    except Exception as e:
        logger.error(f"[persona_generator] Failed to parse phases JSON: {e}")
        return {}

    # -------- phases 構造に正規化 --------
    phases_out: Dict[str, Dict[str, Any]] = {}
    style_axes = ["Trust","Familiarity","Hostility","Dominance","Empathy","Instrumentality"]
    emo_axes = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]

    phase_list = parsed.get("phases") or []
    for idx, ph in enumerate(phase_list):
        name = str(ph.get("name") or f"相{idx+1}").strip()
        if not name:
            name = f"相{idx+1}"

        desc = str(ph.get("description") or "").strip()
        if not desc:
            # 説明がない相はスキップ
            continue

        sb_raw = ph.get("style_bias") or {}
        eb_raw = ph.get("emotion_bias") or {}

        style_bias = {ax: float(sb_raw.get(ax, 0.0)) for ax in style_axes}
        emotion_bias = {ax: float(eb_raw.get(ax, 0.0)) for ax in emo_axes}
        tone_hint = str(ph.get("tone_hint") or "").strip() or "落ち着いた調子。"

        phases_out[name] = {
            "description": desc,
            "style_bias": style_bias,
            "emotion_bias": emotion_bias,
            "tone_hint": tone_hint,
        }

    logger.debug("\n[DEBUG extract_phases normalized phases]\n",
            json.dumps(phases_out, ensure_ascii=False, indent=2),"\n")

    return phases_out


def default_phase_dynamics():
    return {
        "alpha": 0.3,     # Emotion反映強度
        "beta": 0.2,      # Relation反映強度
        "gamma": 0.05,    # ノイズ揺らぎ
        "temperature": 0.4
    }


# ================================================================
# persona統合処理
# ================================================================
def extract_persona_profile(thought_data: Dict[str, Any], persona_name: str, debug=False) -> Dict[str, Any]:
    """思想＋スタイル＋相（phase）情報を統合してPersonaデータ生成"""
    summary = thought_data.get("summary", "")
    values = thought_data.get("values", [])
    reasoning_pattern = thought_data.get("reasoning_pattern", "")
    speech_pattern = thought_data.get("speech_pattern", "")

    # === 既存処理 ===
    anchors = extract_anchors(persona_name, summary, debug)
    style = extract_style(persona_name, summary, debug)
    expression_prompt = extract_expression_prompt(persona_name, summary, style, debug)

    # === 新規追加：Phase生成 ===
    phases = extract_phases(
        persona_name,
        summary,
        values,
        reasoning_pattern,
        speech_pattern,
        debug=debug
    )

    # === 動的位相パラメータ ===
    phase_dynamics = default_phase_dynamics()

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

        "expression_bank": {},  # 後工程で埋める余地（今は空でOK）

        "phases": phases,                  # ★ 相を統合
        "phase_dynamics": phase_dynamics,  # ★ Dynamicsを統合

        "expression_prompt": expression_prompt
    }

# ================================================================
# main
# ================================================================
def main():
    """
    persona_generator の CLI エントリポイント。
    - 思想（thought_xxx.json）を読み込み
    - Persona を生成し
    - persona_xxx.json として保存する
    """

    parser = argparse.ArgumentParser(
        description="GAR Persona Generator - 思想データから Persona JSON を生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="入力となる thought_*.json のパス"
    )

    parser.add_argument(
        "--persona", "-p",
        type=str,
        required=True,
        help="生成する persona 名（保存ファイル名にも使用）"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="LLM 出力や内部状態を表示する"
    )

    args = parser.parse_args()

    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("persona_generator", level=log_level, to_console=True)

    logger.info(f"Persona Generation started (log_level={log_level})")

    # ------------------------------------
    # thought_xxx.json の読み込み
    # ------------------------------------
    thought_data = load_thought(args.input)
    if not thought_data:
        logger.error(f"[persona_generator] Error: cannot load thought file {args.input}")
        return

    # ------------------------------------
    # persona JSON の生成
    # ------------------------------------
    persona = extract_persona_profile(
        thought_data,
        persona_name=args.persona,
        debug=args.debug
    )

    # ------------------------------------
    # persona データを保存
    # ------------------------------------
    out_dir = os.path.expanduser("~/data/personas")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"persona_{args.persona}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)

    logger.info(f"[persona_generator] Persona saved: {out_path}")


if __name__ == "__main__":
    main()
