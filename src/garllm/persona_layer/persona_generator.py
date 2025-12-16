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

import textwrap

from typing import Any, Dict, List
from pathlib import Path

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
def extract_style(persona_name: str, summary: str, debug=False):
    """発話スタイル・語尾・キーワード抽出（人称は関係性で揺れる前提）"""

    prompts = {
        # 一人称：状況で変わるので候補を複数
        "first_person": f"""
{persona_name} が自分を指す一人称の「候補」を2〜5個、日本語で列挙してください。
以下の観点で“使い分けができるように”選ぶこと：
- 親密（フラット） / 敬意（改まった） / 威圧（上から） / 弱気（下から）の揺らぎ
出力は候補のみ（各行1つ）。説明文は不要。

思想概要: {summary}
""".strip(),

        # 二人称：相手との上下・距離・敵対で変わるので候補を複数
        "second_person": f"""
{persona_name} が他者を呼ぶ二人称・呼称の「候補」を3〜7個、日本語で列挙してください。
以下の観点で“関係性に応じて選べるように”すること：
- 親密（タメ口） / 丁寧（敬語） / 上下（目上・目下） / 敵対（突き放す）
- 敬称（さん/殿/君/お前 等）も含めてよい
出力は候補のみ（各行1つ）。説明文は不要。

思想概要: {summary}
""".strip(),

        # 語尾：従来通り
        "speech_suffix": (
            f"{persona_name} の発話文末によく現れる語尾や言い回しを3〜7個、各行に1つずつ列挙してください。"
            f"説明や例文は不要です。思想概要:{summary}"
        ),

        # キーワード：従来通り（少し増やしてもよい）
        "keywords": f"{persona_name} の思想を象徴する語彙を5〜8個挙げてください。各行に1つ。思想概要:{summary}",
    }

    style = {}
    for key, prompt in prompts.items():
        raw = ask_vllm_text(prompt, max_tokens=350, debug=debug)
        # 人称や語尾は候補が増えるので limit を少し上げる
        limit = 10 if key in ("first_person", "second_person", "speech_suffix", "keywords") else 5
        style[key] = lines_to_list(raw, limit=limit)

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


def generate_expression(persona_name: str, persona: dict, debug: bool = False) -> dict:
    """
    persona 情報から expression_<persona>.json を自動生成する。
    - シンプル固定フォーマット
    - 各配列 2〜3 個
    - JSON のみ出力させる
    """

    core = persona.get("core_profile", {})
    style = persona.get("style", {})

    json_schema = textwrap.dedent("""\
        {
        "talk": {
            "intro": [],
            "agree": [],
            "disagree": []
        },
        "emotion": {
            "joy": [],
            "anger": []
        },
        "battle_cries": []
        }
    """).strip()

    prompt = textwrap.dedent(f"""\
        あなたは{persona_name}の発話表現を設計する専門家です。

        以下の人物情報から、世界観、時代、文化、社会的立場や属性、価値観、話し方の特徴を反映した
        発話の素材となる本人の決まり文句や口癖をそれぞれ2～3個ずつ JSON 形式で生成してください。

        人物概要:
        {json.dumps(core, ensure_ascii=False, indent=2)}

        話し方:
        {json.dumps(style, ensure_ascii=False, indent=2)}

        出力形式:
        出力は JSON のみ。説明は禁止。        
        {json_schema}

    """).strip()


    try:
        raw = request_openai(
            messages=[
                {"role": "system", "content": "出力は必ず JSON のみ。説明や前置きは禁止。"},
                {"role": "user", "content": prompt},
            ],
            endpoint_type="chat",
            max_tokens=700,
            temperature=0.5,
        )
        raw = (raw or "").strip()
        if debug:
            logger.debug("[expression raw]\n" + raw)
    except Exception as e:
        logger.error(f"[expression] LLM error: {e}")
        return {}

    try:
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[expression] JSON parse failed: {e}")
        return {}

    return parsed




# ================================================================
# Phase（相）生成ロジック（改良版）
# ================================================================

def extract_phases(
    persona_name: str,
    summary: str,
    values: list,
    reasoning: str,
    speech_pattern: str,
    background: str = None,
    episodes: list = None,
    anchors: list = None,
    debug=False
):
    """
    人物の相（Phase）を抽出する改良版。
    background / episodes / anchors がある場合のみ使用。
    """

    import json, re

    background_text = background or "(なし)"
   
    episodes_text = ""
    anchors_text = ""

    if episodes:
        episodes_text = "\n【重要な出来事】\n" + json.dumps(
            episodes, ensure_ascii=False, indent=2
        )
    if anchors:
        anchors_text = "\n【信念の核】\n" + json.dumps(
            anchors, ensure_ascii=False, indent=2
        )

    # ======================================================
    # ★ ここが最重要：style_bias / emotion_bias の意味定義を明示
    # ======================================================
    prompt = f"""
あなたの役割は、人物の「相（Phase）」を設計する専門家です。

相（Phase）とは、以下の要素が組み合わさった「人格の振る舞いモード」です：
- 話し方や語気
- 感情の出し方
- 相手との距離感（支配的、共感的など）
- 態度・雰囲気

各相には、以下の **対人スタンス6軸（style_bias）** と **感情傾向8軸（emotion_bias）** を数値で指定してください。

【style_bias（6軸）の意味】
- **Trust**: 高い=相手を信用する / 低い=警戒・疑い
- **Familiarity**: 高い=親しみ、カジュアル / 低い=形式的、距離を置く
- **Hostility**: 高い=批判的・対立的 / 低い=協調的・平和的
- **Dominance**: 高い=主導権を握る / 低い=控えめ・受容的
- **Empathy**: 高い=共感・情緒理解 / 低い=客観・ドライ
- **Instrumentality**: 高い=実務的・目的優先 / 低い=物語的・感情寄り

【emotion_bias（8軸）の意味】
- **joy**: 喜び / 明るさ  
- **trust**: 安心・温かさ  
- **fear**: 不安・臆病さ  
- **surprise**: 驚きやすさ  
- **sadness**: 悲しみやすさ  
- **disgust**: 嫌悪・批判性  
- **anger**: 怒り・苛立ち  
- **anticipation**: 期待・先走り・予測への傾倒

数値はすべて -1.0〜1.0 で表してください。

------------------------------------------------------------
【人物情報】
対象人物：{persona_name}

【思想要約】
{summary}

【背景】
{background_text}

【価値観】
{values}

【思考様式】
{reasoning}

【人生の重要な出来事】
{episodes_text}

【信念の核】
{anchors_text}

【話し方の特徴】
{speech_pattern}



------------------------------------------------------------
あなたの仕事：
この人物にふさわしい 3つの「相（Phase）」を定義してください。

【出力形式（厳守）】
次の JSON のみを返してください。

```json
{{
  "phases": [
    {{
      "name": "相名",
      "description": "振る舞いの説明",
      "style_bias": {{
        "Trust": 0.0, "Familiarity": 0.0, "Hostility": 0.0,
        "Dominance": 0.0, "Empathy": 0.0, "Instrumentality": 0.0
      }},
      "emotion_bias": {{
        "joy": 0.0, "trust": 0.0, "fear": 0.0, "surprise": 0.0,
        "sadness": 0.0, "disgust": 0.0, "anger": 0.0, "anticipation": 0.0
      }},
      "tone_hint": "口調の説明"
    }}
  ]
}}```

    """
    try:
        raw = request_openai(
            messages=[
                {"role": "system", "content": "指定スキーマに従って厳密な JSON を返すアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            endpoint_type="chat",
            max_tokens=900,
            temperature=0.3,
        ).strip()
    except Exception as e:
        logger.error(f"[persona_generator] extract_phases LLM error: {e}")
        return {}

    try:
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, re.DOTALL)
        json_str = m.group(1) if m else re.search(r"\{[\s\S]*\}", raw, re.DOTALL).group(0)
        parsed = json.loads(json_str)
    except Exception as e:
        logger.error(f"[persona_generator] Failed to parse phases JSON: {e}")
        return {}

    phases_out = {}
    style_axes = ["Trust","Familiarity","Hostility","Dominance","Empathy","Instrumentality"]
    emo_axes = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]

    for idx, ph in enumerate(parsed.get("phases") or []):
        name = ph.get("name") or f"相{idx+1}"
        desc = ph.get("description") or ""
        if not desc:
            continue
        sb_raw = ph.get("style_bias") or {}
        eb_raw = ph.get("emotion_bias") or {}

        phases_out[name] = {
            "description": desc,
            "style_bias": {ax: float(sb_raw.get(ax, 0.0)) for ax in style_axes},
            "emotion_bias": {ax: float(eb_raw.get(ax, 0.0)) for ax in emo_axes},
            "tone_hint": ph.get("tone_hint", "落ち着いた調子"),
        }

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
    """
    再設計版:
      - anchors/episodes は thought_profiler の出力をそのまま使用する
      - style は summary + background を参照して抽出
      - phases は anchors/episodes を含めて抽出
      - core_profile に episodes を追加
    """

    summary = thought_data.get("summary", "")
    background = thought_data.get("background", "")
    values = thought_data.get("values", [])
    reasoning_pattern = thought_data.get("reasoning_pattern", "")
    speech_pattern = thought_data.get("speech_pattern", "")

    demographic = thought_data.get("demographic", {}) or {}
    language_profile = thought_data.get("language_profile", {}) or {}


    # thought_profiler が生成した episodes / anchors を採用
    episodes = thought_data.get("episodes", [])
    anchors = thought_data.get("anchors", [])

    # --- style 抽出: summary + background + language_profile を使う ---
    style_input_text = f"{summary}\n\n【背景】{background}"

    # 言語・口調の背景をテキスト化
    lang_lines = []
    if isinstance(language_profile, dict):
        dialect = language_profile.get("dialect")
        if isinstance(dialect, str) and dialect.strip() and dialect.strip() != "不明":
            lang_lines.append(f"方言・なまり: {dialect.strip()}")
        speech_style = language_profile.get("speech_style")
        if isinstance(speech_style, str) and speech_style.strip():
            lang_lines.append(f"話し方のスタイル: {speech_style.strip()}")
        samples = language_profile.get("sample_phrases")
        if isinstance(samples, list) and samples:
            joined = " / ".join(str(s) for s in samples[:6])
            lang_lines.append(f"よく使いそうな表現: {joined}")

    if lang_lines:
        style_input_text += "\n\n【言語・話し方の背景】\n" + "\n".join(lang_lines)

    style = extract_style(persona_name, style_input_text, debug)



    # --- expression_prompt ---
    expression_prompt = extract_expression_prompt(persona_name, summary, style, debug)

    # --- phases: episodes / anchors を使用して LLM 生成 ---
    phases = extract_phases(
        persona_name=persona_name,
        summary=summary,
        values=values,
        reasoning=reasoning_pattern,
        speech_pattern=speech_pattern,
        background=background,
        episodes=episodes,
        anchors=anchors,
        debug=debug
    )



    # phase_dynamics（そのまま使用）
    phase_dynamics = default_phase_dynamics()

    return {
        "persona_name": persona_name,

        "core_profile": {
            "summary": summary,
            "background": background,
            "values": values if isinstance(values, list) else [values],
            "reasoning_pattern": reasoning_pattern,
            "speech_pattern": speech_pattern,
            "episodes": episodes,
            "knowledge_anchors": anchors,
            "demographic": demographic,
            "language_profile": language_profile,
        },


        "style": style,
        "expression_bank": {},

        "phases": phases,
        "phase_dynamics": phase_dynamics,

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

    expr_dir = Path(get_data_path("personas"))
    expr_dir.mkdir(parents=True, exist_ok=True)

    expr_path = expr_dir / f"expression_{args.persona}.json"
    if not expr_path.exists():
        expression = generate_expression(args.persona, persona, debug=args.debug)
        if expression:
            expr_path.write_text(
                json.dumps(expression, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"[persona_generator] expression generated: {expr_path}")


if __name__ == "__main__":
    main()
