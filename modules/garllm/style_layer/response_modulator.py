#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
response_modulator.py — Persona Response Layer with Relation + Emotion Axes

目的:
  - 入力テキスト（= ユーザー発話）に対して、
    「ペルソナ（persona_*.json）」と「状態（relation_axes / emotion_axes）」を反映した
    応答文（assistant発話）を **直接生成** する。

重要ポイント（最小変更方針）:
  - 既存 style_modulator と同じ CLI/関数シグネチャを維持
    * def modulate_style(text, persona_name, intensity, verbose, debug, relation_axes, emotion_axes)
    * --persona / --text / --intensity / --verbose / --relation_axes / --emotion_axes
  - 内部のプロンプトのみを「言い換え」→「応答文生成」に変更
  - relation_axes: Respect を正式対応（Power も後方互換）
  - LLM 出力の後処理ロジックは変更なし（--- 区切りの除去など）
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import re
import sys

# プロジェクトルートをパスに追加（~/modules を解決）
sys.path.append(os.path.expanduser("~/modules/"))

from garllm.utils.llm_client import request_llm
from garllm.utils.env_utils import get_data_path

# ============================================================
# 📂 Persona Profile Loader
# ============================================================
def load_persona_profile(persona_name: str) -> Dict[str, Any]:
    """
    ペルソナ定義JSONのロード:
      data/personas/persona_<name>.json を読む
    """
    base_dir = Path(get_data_path("personas"))
    persona_path = base_dir / f"persona_{persona_name}.json"
    with open(persona_path, "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    # expression_bank が外部ファイルに存在する場合は統合
    expr_path = base_dir / f"expression_{persona_name}.json"
    if expr_path.exists():
        with open(expr_path, "r", encoding="utf-8") as f:
            persona_data["expression_bank"] = json.load(f)
    return persona_data


# === これを response_modulator.py の上部ユーティリティ群の近くに追加 ===
def build_pronoun_guidance(persona_data: Dict[str, Any], relations: Dict[str, Dict[str, float]] | None) -> str:
    """候補リストをそのまま提示し、候補外の使用禁止と選択規則を明記する。"""
    style = persona_data.get("style", {})
    fp_list = style.get("first_person", []) or ["私"]
    sp_list = style.get("second_person", []) or ["あなた"]

    # 関係性を見て LLM に選ばせる（ルール明記）
    relation_hint = "関係性に応じて自然に選択すること。親密度が高いほど砕けた候補、低いほど丁寧な候補を選ぶこと。"
    # 候補外禁止・「余/我/拙者」等の勝手な変換抑止
    hard_rules = (
        "一人称と二人称は必ず下記候補から選ぶこと。候補に無い人称は絶対に使わない。"
        " 既存履歴の口調に引きずられないこと。"
    )
    return (
        f"一人称候補: {', '.join(fp_list)} / "
        f"二人称候補: {', '.join(sp_list)}。"
        f" {relation_hint} {hard_rules}"
    )


# ============================================================
# 🔁 Axis Hints（関係性/感情を自然言語の指針に）
# ============================================================
AXIS_DESCRIPTIONS = {
    "Trust": ("安心感・肯定・寛容に話す", "慎重・疑念を持ち距離を取る"),
    "Familiarity": ("砕けた・軽口・親密に話す", "丁寧・説明的・形式的に話す"),
    "Hostility": ("攻撃的・挑発的・批判的に話す", "穏やか・柔らかく・譲歩的に話す"),
    "Dominance": ("主導的・命令的・断定的に話す", "従属的・受容的・傾聴的に話す"),
    "Empathy": ("感情を拾い・共感を示す", "冷静・客観的・感情を省く"),
    "Instrumentality": ("効率重視・取引的に話す", "無償・感情的・純粋に話す")
}

def describe_axis(name: str, value: float) -> str:
    """Relation軸を連続トーンで記述（強度=絶対値、符号で方向選択）"""
    pos_text, neg_text = AXIS_DESCRIPTIONS.get(name, ("正方向", "負方向"))
    strength = abs(value)
    if strength < 0.05:
        return f"{name}: 中立的（影響ほぼなし）"
    if value > 0:
        return f"{name}: {strength:.0%}の強さで「{pos_text}」"
    else:
        return f"{name}: {strength:.0%}の強さで「{neg_text}」"

def synthesize_relation_hint(axes: dict[str, float] | None) -> str:
    """全軸のトーンを結合して1文にまとめる"""
    if not axes:
        return "（指定なし）"
    lines = [describe_axis(k, v) for k, v in axes.items()]
    # 強度0.05未満は除外し、残りを結合
    active = [ln for ln in lines if "影響ほぼなし" not in ln]
    return " / ".join(active) if active else "（指定なし）"


# ============================================================
# 💓 Emotion Layer（8軸 + 滑らか補間モデル）
# ============================================================

EMOTION_TEMPLATES = {
    "joy": {
        "weak": "穏やかで心が安らいでいるように話す。",
        "medium": "明るく軽やかに、自然と声に弾みが出るように話す。",
        "strong": "感情が高ぶり、嬉しさが抑えきれないように話す。"
    },
    "trust": {
        "weak": "落ち着きと安らぎを感じ、静かに穏やかに話す。",
        "medium": "安心と安定を感じながら、自然体でゆったりと話す。",
        "strong": "深い安心と充足感に包まれ、温かく穏やかに話す。"
    },
    "fear": {
        "weak": "慎重で緊張を感じながら、少し抑えた声で話す。",
        "medium": "不安と恐れが混ざり、言葉に張り詰めた緊張がにじむように話す。",
        "strong": "恐怖や焦りが支配し、呼吸が浅く断片的な口調で話す。"
    },
    "surprise": {
        "weak": "小さな驚きと興味を感じて、軽く反応するように話す。",
        "medium": "はっきりと驚きが現れ、テンポが速くなるように話す。",
        "strong": "強い衝撃や驚愕を受け、思わず声や語気が大きくなるように話す。"
    },
    "sadness": {
        "weak": "静かに沈み込み、少し間を置きながら話す。",
        "medium": "切なさや哀しみが声に滲み、ゆっくりとした調子で話す。",
        "strong": "深い悲嘆に包まれ、途切れ途切れにかすれるように話す。"
    },
    "disgust": {
        "weak": "軽い不快感を覚え、やや無関心な調子で話す。",
        "medium": "明確な嫌悪や拒否の感情があり、語気が鋭くなる。",
        "strong": "強烈な不快感や拒絶の感情が溢れ、言葉に荒さが出る。"
    },
    "anger": {
        "weak": "いら立ちを抑えつつ、声の強さにわずかな緊張がこもる。",
        "medium": "明確な怒りが湧き上がり、短く強い言葉で話す。",
        "strong": "激しい怒りに突き動かされ、荒く激しい調子で話す。"
    },
    "anticipation": {
        "weak": "少し先を思い描きながら、期待と集中を感じて話す。",
        "medium": "高揚した期待感があり、語気が前のめりになるように話す。",
        "strong": "確信と興奮に満ち、勢いよく先を語るように話す。"
    }
}

def smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)

def emotion_weights(value):
    w_low  = 1 - smoothstep(0.25, 0.33, value)
    w_mid  = smoothstep(0.20, 0.66, value) - smoothstep(0.33, 0.66, value)
    w_high = smoothstep(0.66, 1.0, value)
    total = w_low + w_mid + w_high
    return {k: v/total for k,v in zip(['weak','medium','strong'], [w_low,w_mid,w_high])}

def generate_emotion_prompt(emotion_vector: dict[str, float]) -> str:
    lines = []
    for emo, val in emotion_vector.items():
        val = max(0.0, min(1.0, val))  # 安全クランプ
        w = emotion_weights(val)
        tmpl = EMOTION_TEMPLATES.get(emo.lower())
        if not tmpl:
            continue
        lines.append(
            f"{emo.capitalize()}({val:.2f}): "
            f"{w['weak']*100:.0f}%→{tmpl['weak']} "
            f"{w['medium']*100:.0f}%→{tmpl['medium']} "
            f"{w['strong']*100:.0f}%→{tmpl['strong']}"
        )
    joined = " / ".join(lines)
    return f"感情指針: {joined if joined else '（指定なし）'}"


def axes_to_hints(axes: Dict[str, float] | None, converter) -> str:
    if not axes:
        return ""
    hints = [converter(k, v) for k, v in axes.items() if isinstance(v, (int, float))]
    return " ".join([h for h in hints if h])

# --- 👇この関数を上部ユーティリティ群の近くに追加 ---
def extract_relation_axes_for_target(relations: dict, target_name: str) -> dict | None:
    """relations から特定の target_name の軸を抽出"""
    if not isinstance(relations, dict):
        return None
    axes = relations.get(target_name)
    if isinstance(axes, dict):
        return axes
    return None

# ============================================================
# 🧠 Prompt Construction（ここを「応答生成」に変更）
# ============================================================
def build_prompt(
    input_text: str,
    persona_name: str,
    persona_data: Dict[str, Any],
    intensity: float = 0.7,
    verbose: bool = False,
    relation_axes: Dict[str, float] | None = None,
    relations: Dict[str, Dict[str, float]] | None = None,
    emotion_axes: Dict[str, float] | None = None
):
    """ユーザー発話に対する『ペルソナとしての応答』を生成するプロンプトを構築。"""
    style = persona_data.get("style", {})
    knowledge = persona_data.get("knowledge_anchors") or persona_data.get("core_profile", {}).get("knowledge_anchors", [])
    tone = persona_data.get("style_guide") or persona_data.get("expression_prompt") or ""

    # 人称ガイダンス（候補提示＋候補外禁止）
    pronoun_guidance = build_pronoun_guidance(persona_data, relations)

    # 冗長さガイド
    expressiveness = (
        "簡潔で要点を押さえた一段落の応答を出力してください。"
        if not verbose else
        "豊かな表現で1〜3段落の応答を出力してください（過度な水増しは避ける）。"
    )

    # 関係性ヒント
    relation_hint = synthesize_relation_hint(relation_axes)
    # 他ペルソナとの関係
    if relations and isinstance(relations, dict):
        others_hint = []
        for target, axes in relations.items():
            if target in ["ユーザ", "ユーザー", "User", "user"]:
                continue
            desc = synthesize_relation_hint(axes)
            if desc and desc != "（指定なし）":
                others_hint.append(f"{target}: {desc}")
        relation_context = " / ".join(others_hint) if others_hint else "（他ペルソナとの関係なし）"
    else:
        relation_context = "（他ペルソナとの関係なし）"

    # 感情ヒント
    emotion_hint_text = generate_emotion_prompt(emotion_axes) if emotion_axes else "（指定なし）"

    # プロンプト
    prompt = f"""
あなたは今から完全に『{persona_name}』として応答します。
口調・語彙・価値観・判断基準は {persona_name} のものを厳守してください。
{pronoun_guidance}
文体指針: {tone if tone else "（特記なし）"}
スタイル強度: {intensity * 100:.0f}%
他者との関係: {relation_hint if relation_hint else "（指定なし）"}
他ペルソナとの関係: {relation_context}
感情指針: {emotion_hint_text if emotion_hint_text else "（指定なし）"}
関係性や感情指針の内容は、応答の語彙・口調・態度・話法に必ず反映させること。

【ユーザー発話】 
{input_text}

【厳守事項】
- 出力は**あなた（{persona_name}）としての応答文のみ**。説明・前置き・メタ記述は禁止。
- 人称は上記候補からのみ選択し、一貫して用いる。候補外の人称は使用禁止。
- 質問返しは避け、まずは**答え**を返す（必要なら最後に1件だけ簡潔な問い返し可）。
- 日本語で書く。古風な文語調や歴史口調は、候補と関係性指針が示す場合のみ許容。

【出力】
""".strip()
    return prompt


# ============================================================
# 💬 LLM Interface with Output Cleaner
# ============================================================
def ask_llm(prompt: str) -> str:
    """
    LLM 呼び出し。--- 以降の補足説明を除去する簡易クリーナ付き。
    """
    try:
        response = request_llm(prompt=prompt, backend="auto", temperature=0.6, max_tokens=800)
        # --- 補足説明（--- 以降）を削除 ---
        cleaned = re.split(r"---+", response, maxsplit=1)[0].strip()
        return cleaned
    except Exception as e:
        print(f"[response_modulator] LLM error: {e}")
        return ""

# ============================================================
# 💬 Chat形式 LLM Interface（新規追加）
# ============================================================
def ask_llm_chat(messages: list[dict[str, str]]) -> str:
    """
    Chat形式 (messages[]) 入力対応版。
    OpenWebUI や relay_server から直接 messages を受け取る場合に使用。
    """
    try:
        response = request_llm(messages=messages, backend="auto", temperature=0.6, max_tokens=800)
        cleaned = re.split(r"---+", response, maxsplit=1)[0].strip()
        return cleaned
    except Exception as e:
        print(f"[response_modulator] Chat LLM error: {e}")
        return ""

# ============================================================
# 🎭 Response Modulation Core（I/Fはそのまま）
# ============================================================
def modulate_response(
    text: str | list[dict[str, str]],
    persona_name: str,
    intensity: float = 0.7,
    verbose: bool = False,
    debug: bool = False,
    relation_axes: dict[str, float] | None = None,
    relations: dict[str, dict[str, float]] | None = None,  # ←この1行を追加
    emotion_axes: dict[str, float] | None = None
):
    """
    text が str なら従来どおり build_prompt() を使う。
    text が list (messages形式) なら Chat形式で LLM を呼び出す。
    """
    persona_data = load_persona_profile(persona_name)


    target_name = "ユーザ"
    if relations and isinstance(relations, dict):
        # 会話履歴に gar.persona: が含まれていれば対象ペルソナを抽出
        for m in reversed(text if isinstance(text, list) else []):
            if m.get("role") == "user" and "gar.persona:" in m.get("content", ""):
                try:
                    target_name = m["content"].split("gar.persona:")[1].split(")")[0].strip()
                    break
                except Exception:
                    pass

    # 対象ペルソナ（ユーザ or 他ペルソナ）の関係軸を抽出
    relation_axes = extract_relation_axes_for_target(relations, target_name)


    # Chat形式の場合（relay_server 経由など）
    if isinstance(text, list):
        if debug:
            print("[DEBUG] Chat-mode messages input detected")
            print(json.dumps(text, ensure_ascii=False, indent=2))

        style = persona_data.get("style", {})
        tone  = persona_data.get("style_guide") or persona_data.get("expression_prompt") or ""
        # 人称候補
        fp_list = style.get("first_person", []) or ["私"]
        sp_list = style.get("second_person", []) or ["あなた"]
        pronoun_guidance = (
            f"一人称候補: {', '.join(fp_list)} / 二人称候補: {', '.join(sp_list)}。"
            " 関係性に応じて自然に選択すること。候補外の人称は絶対に使わない。"
            " 履歴の口調に引きずられず、候補と関係性に基づいて選ぶこと。"
        )

        # 関係性の自然文ヒント
        rel_user_hint = synthesize_relation_hint(relation_axes) if relation_axes else "（指定なし）"
        if relations and isinstance(relations, dict):
            others_hint = []
            for target, axes in relations.items():
                if target in ["ユーザ", "ユーザー", "User", "user"]:
                    continue
                desc = synthesize_relation_hint(axes)
                if desc and desc != "（指定なし）":
                    others_hint.append(f"{target}: {desc}")
            rel_others_hint = " / ".join(others_hint) if others_hint else "（他ペルソナとの関係なし）"
        else:
            rel_others_hint = "（他ペルソナとの関係なし）"

        emo_hint = generate_emotion_prompt(emotion_axes) if emotion_axes else "（指定なし）"

        persona_system_message = {
            "role": "system",
            "content": (
                f"あなたは今から完全に『{persona_name}』として応答します。\n"
                f"{pronoun_guidance}\n"
                f"文体指針: {tone if tone else '（特記なし）'} / スタイル強度: {intensity*100:.0f}%\n"
                f"関係性（ユーザ⇄{persona_name}）: {rel_user_hint}\n"
                f"他ペルソナとの関係: {rel_others_hint}\n"
                f"感情指針: {emo_hint}\n"
                f"出力は応答文のみ。メタ発言禁止。"
            )
        }

        messages_with_persona = [persona_system_message] + text

        if debug:
            print("[DEBUG] persona_system_message]\n" + json.dumps(persona_system_message, ensure_ascii=False, indent=2))
            print("[DEBUG] first_person candidates:", fp_list)
            print("[DEBUG] second_person candidates:", sp_list)

        response = ask_llm_chat(messages_with_persona)
        return response.strip() if response else ""


    # 従来のテキストモード
    prompt = build_prompt(text, persona_name, persona_data, intensity, verbose, relation_axes, relations, emotion_axes)

    if debug:
        print("[DEBUG prompt]\n" + prompt + "\n" + "=" * 80)

    response = ask_llm(prompt)
    return response.strip() if response else text  # フォールバック: 応答失敗時は原文を返す


# ============================================================
# 🧰 CLI Entry（互換）
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "🧭 ペルソナ応答変調ツール（Response Modulator: 互換I/F）\n"
            "ユーザー入力（--text）に対する、ペルソナ＋関係性＋感情を反映した『応答文』を生成します。\n"
            "※ 既存 style_modulator と同じ引数・使い方で動作します。"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--persona", required=True, help="使用するペルソナ名（例: 織田信長、徳川家康）")
    parser.add_argument("--text", required=True, help="ユーザー発話（例: 『よくもやってくれたな』）")
    parser.add_argument("--intensity", type=float, default=0.7, help="文体の影響度（0.0〜1.0）")
    parser.add_argument("--verbose", action="store_true", help="饒舌モード（1〜3段落で豊かに表現）")
    parser.add_argument("--relation_axes", type=str, default=None,
                        help="関係性ベクトル（JSON: {'Friendship':0.5,'Respect':-0.2} など）")
    parser.add_argument("--emotion_axes", type=str, default=None,
                        help="感情ベクトル（JSON: {'Joy':0.8,'Fear':-0.3} など）")
    parser.add_argument("--debug", action="store_true", help="デバッグ表示（プロンプト出力）")
    parser.add_argument("--relations", type=str, default=None, help="関係性構造（JSON: {'ユーザ': {...}, '徳川家康': {...}}）")

    args = parser.parse_args()

    relation_axes = json.loads(args.relation_axes) if args.relation_axes else None
    relations = json.loads(args.relations) if args.relations else None
    emotion_axes = json.loads(args.emotion_axes) if args.emotion_axes else None

    rewritten = modulate_response(
        args.text, args.persona, args.intensity, args.verbose, args.debug, 
        relation_axes, relations, 
        emotion_axes)

    print("\n==== Rewritten Text ====")
    print(rewritten)
    print("=" * 80)

if __name__ == "__main__":
    main()

# ============================================================
# 💡 Usage
# ============================================================
# 1) 互換（最小）：応答生成（簡潔）
#   python3 response_modulator.py --persona 織田信長 --text "よくもやってくれたな"
#
# 2) 関係 + 感情 反映（例: 友好-0.2, 尊敬-0.5, 喜び-0.4）
#   python3 response_modulator.py --persona 織田信長 --text "よくもやってくれたな" \
#       --relation_axes '{"Friendship":-0.2,"Respect":-0.5}' \
#       --emotion_axes  '{"Joy":-0.4}'
#
# 3) 饒舌モード
#   python3 response_modulator.py --persona 織田信長 --text "この戦が終われば酒を飲もう。" --verbose
#
# 備考:
#  - state_*.json 側の軸名が "Respect" の場合も本ファイルは自然に解釈します。
#  - 旧 "Power" 軸も後方互換で同義扱いします。
