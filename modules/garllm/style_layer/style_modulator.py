#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style_modulator.py — Persona Style Layer with Relation + Emotion Axes 3.1（出力整形修正版）

自然な文体変換に加えて、関係性軸（Friendship, Power, Trust, Formality, Dominance）
および感情軸（Plutchik 4軸モデル）を統合的に制御できる。

【更新内容】
- LLM出力後処理を強化：補足説明（--- その他...）を自動除去
- CLIコメントおよび使用例を整理
- 従来の動作互換性を保持

Author: GPT-5 (for Inanna Project)
Version: 5.1
Last updated: 2025-10-20
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import re
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.expanduser("~/modules/"))

from garllm.utils.llm_client import request_llm
from garllm.utils.env_utils import get_data_path

# ============================================================
# 📂 Persona Profile Loader
# ============================================================
def load_persona_profile(persona_name: str) -> Dict[str, Any]:
    """Load persona profile JSON"""
    profile_dir = Path(get_data_path("personas"))
    profile_path = profile_dir / f"persona_{persona_name}.json"

    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"[style_modulator] Persona file not found: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 🔁 Axis Hints
# ============================================================
def axis_hint(name: str, value: float) -> str:
    """Generate natural-language hints for relation axes."""
    if name == "Friendship":
        if value >= 0.6: return "非常に友好的で温かい口調で話す。"
        if value >= 0.2: return "やや親しみを込めて話す。"
        if value > -0.2: return "中立的な口調で話す。"
        if value > -0.6: return "やや冷たく距離を置いた話し方をする。"
        return "敵意を含み挑発的な語気を交える。"
    if name == "Power":
        if value >= 0.6: return "相手を目上として敬語で話す。"
        if value >= 0.2: return "やや敬意を払って話す。"
        if value > -0.2: return "対等な立場で話す。"
        if value > -0.6: return "やや命令的な調子を加える。"
        return "明確に目下として命令調で話す。"
    if name == "Trust":
        if value >= 0.6: return "高い信頼を示す表現を用いる。"
        if value >= 0.2: return "やや信頼を見せる口調にする。"
        if value > -0.2: return "中立的態度を取る。"
        if value > -0.6: return "慎重で疑念を含む話し方をする。"
        return "強い疑念を示す語気で話す。"
    if name == "Formality":
        if value >= 0.6: return "儀礼的で形式的な文体を用いる。"
        if value >= 0.2: return "丁寧な表現を用いる。"
        if value > -0.2: return "自然体で話す。"
        if value > -0.6: return "くだけた口調を混ぜる。"
        return "非常にくだけた口語で話す。"
    if name == "Dominance":
        if value >= 0.6: return "主導的な立場で自信をもって語る。"
        if value >= 0.2: return "やや主導的な態度を取る。"
        if value > -0.2: return "対等な立場を保つ。"
        if value > -0.6: return "やや受け身の姿勢で話す。"
        return "従属的で控えめな話し方をする。"
    return ""

def emotion_hint(name: str, value: float) -> str:
    """Generate descriptive emotional hints from Plutchik’s 4-axis model."""
    if name == "Joy":
        if value > 0.5: return "喜びと幸福感を含めて明るく話す。"
        if value < -0.5: return "悲しみや落ち着きを帯びて話す。"
    if name == "Trust":
        if value > 0.5: return "信頼や安心感を持って語る。"
        if value < -0.5: return "嫌悪や拒絶感をにじませる。"
    if name == "Fear":
        if value > 0.5: return "恐れや慎重さを含んだ表現にする。"
        if value < -0.5: return "怒りや断固たる語調で話す。"
    if name == "Surprise":
        if value > 0.5: return "驚きや混乱を含む調子で語る。"
        if value < -0.5: return "期待と希望を込めて話す。"
    return ""

def axes_to_hints(axes: Dict[str, float], converter) -> str:
    """Combine axis hints into a natural phrase."""
    hints = [converter(k, v) for k, v in axes.items() if isinstance(v, (int, float))]
    return " ".join([h for h in hints if h])

# ============================================================
# 🧠 Prompt Construction
# ============================================================
def build_prompt(input_text: str, persona_name: str, persona_data: Dict[str, Any],
                 intensity: float = 0.7, verbose: bool = False,
                 relation_axes: Dict[str, float] | None = None,
                 emotion_axes: Dict[str, float] | None = None) -> str:
    style = persona_data.get("style", {})
    knowledge = persona_data.get("knowledge_anchors", [])
    tone = persona_data.get("style_guide", "")

    one_pronoun = style.get("first_person", ["私"])[0]
    two_pronoun = style.get("second_person", ["あなた"])[0]

    expressiveness = (
        "原文の意味を保ちつつ、簡潔で自然な表現にしてください。"
        if not verbose else
        "原文の意味を保ちながら、人格・感情・文体の特徴を豊かに反映させてください。"
    )

    relation_hint = f"\n🤝 関係性指針: {axes_to_hints(relation_axes, axis_hint)}" if relation_axes else ""
    emotion_hint_text = f"\n💓 感情指針: {axes_to_hints(emotion_axes, emotion_hint)}" if emotion_axes else ""

    prompt = f"""
あなたは「{persona_name}」として発話してください。与えられた文章を、{persona_name}らしい文体・語彙・口調に書き換えてください。
---
🧭 文体指針: {tone}
🎭 一人称: {one_pronoun}
🎭 二人称: {two_pronoun}
📚 重要な概念: {', '.join(knowledge) if knowledge else '（指定なし）'}
🎚️ スタイル強度: {intensity * 100:.0f}%{relation_hint}{emotion_hint_text}
---
【入力文】
{input_text}

【出力条件】
- {expressiveness}
- {persona_name}の人格・語彙・口調を自然に反映。
- 不自然な人称の挿入は避ける。
- 出力は日本語のみ。説明文は禁止。
- 改行・リズムは自然に保つ。

【出力】
"""
    return prompt.strip()

# ============================================================
# 💬 LLM Interface with Output Cleaner
# ============================================================
def ask_llm(prompt: str) -> str:
    try:
        response = request_llm(prompt=prompt, backend="auto", temperature=0.6, max_tokens=800)
        # --- 補足説明（--- 以降）を削除 ---
        cleaned = re.split(r"---+", response, maxsplit=1)[0].strip()
        return cleaned
    except Exception as e:
        print(f"[style_modulator] LLM error: {e}")
        return ""

# ============================================================
# 🎭 Style Modulation Core
# ============================================================
def modulate_style(text: str, persona_name: str, intensity: float = 0.7,
                   verbose: bool = False, debug: bool = False,
                   relation_axes: Dict[str, float] | None = None,
                   emotion_axes: Dict[str, float] | None = None) -> str:

    persona_data = load_persona_profile(persona_name)
    prompt = build_prompt(text, persona_name, persona_data, intensity, verbose, relation_axes, emotion_axes)

    if debug:
        print("[DEBUG prompt]\n" + prompt + "\n" + "=" * 80)

    response = ask_llm(prompt)
    return response.strip() if response else text

# ============================================================
# 🧰 CLI Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "🧭 パーソナ文体変調ツール（Style Modulator）\n"
            "指定したパーソナの文体でテキストを変換します。\n"
            "関係性（Friendship, Power, Trust, Formality, Dominance）および\n"
            "感情（Joy, Trust, Fear, Surprise）の軸を指定可能です。"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--persona", required=True,
        help="使用する人格（例: 織田信長、紫式部など）")
    parser.add_argument("--text", required=True,
        help="変換対象の文章（例: 『この戦いが終われば酒を飲もう。』）")
    parser.add_argument("--intensity", type=float, default=0.7,
        help="文体の影響度（0.0〜1.0, 高いほどパーソナの個性が強くなる）")
    parser.add_argument("--verbose", action="store_true",
        help="饒舌モード（より豊かな文体で出力）")
    parser.add_argument("--relation_axes", type=str, default=None,
        help=(
            "関係性ベクトル（JSON形式）\n"
            "例: '{\"Friendship\":0.6,\"Power\":-0.4}'\n"
            "軸: Friendship, Power, Trust, Formality, Dominance"
        ))
    parser.add_argument("--emotion_axes", type=str, default=None,
        help=(
            "感情ベクトル（Plutchikの4軸モデル, JSON形式）\n"
            "例: '{\"Joy\":0.8,\"Fear\":-0.3}'\n"
            "軸: Joy(喜び), Trust(信頼), Fear(恐れ), Surprise(驚き)"
        ))
    parser.add_argument("--debug", action="store_true",
        help="プロンプト生成内容を表示（開発・検証用）")

    args = parser.parse_args()

    relation_axes = json.loads(args.relation_axes) if args.relation_axes else None
    emotion_axes = json.loads(args.emotion_axes) if args.emotion_axes else None

    rewritten = modulate_style(
        args.text, args.persona, args.intensity, args.verbose, args.debug,
        relation_axes, emotion_axes
    )

    print("\n==== Rewritten Text ====")
    print(rewritten)
    print("=" * 80)

if __name__ == "__main__":
    main()

# ============================================================
# 💡 Example Usage
# ============================================================
# 1️⃣ デフォルト（従来互換）
#   python3 style_modulator.py --persona 織田信長 --text "この戦いが終われば酒を飲もう。"
#
# 2️⃣ 関係軸付き（友好的・対等）
#   python3 style_modulator.py --persona 織田信長 --text "この戦いが終われば酒を飲もう。" \
#       --relation_axes '{"Friendship":0.5,"Power":0.0}'
#
# 3️⃣ 感情軸付き（Plutchikモデル）
#   python3 style_modulator.py --persona 織田信長 --text "この戦いが終われば酒を飲もう。" \
#       --emotion_axes '{"Joy":0.8,"Trust":0.5}'
#
# 4️⃣ 両方指定（友好かつ喜び）
#   python3 style_modulator.py --persona 織田信長 --text "この戦いが終われば酒を飲もう。" \
#       --relation_axes '{"Friendship":0.5,"Power":0.0}' \
#       --emotion_axes '{"Joy":0.7,"Surprise":-0.4}'
#
# 5️⃣ 雄弁モード（verbose）
#   python3 style_modulator.py --persona 織田信長 --text "この戦いが終われば酒を飲もう。" --verbose
# ============================================================