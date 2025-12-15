#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context_controller.py
--------------------------------------------------
発話文脈を解析して emotion_axes / relation_axes を更新し、状態を保存するコントローラ。

【重要】
- デフォルト動作では「状態更新のみ」。応答文の生成は行いません。
- 応答文を試したい場合のみ --emit_text を指定してください（CLIテスト用途）。
  本番系（relay_server など）からは --emit_text を渡さないでください。

主な機能:
- LLMベース or ルールベース文脈解析 (--mode)
- ```json ... ``` を含むマルチライン JSON の堅牢抽出
- state(JSON) に感情・関係の連続状態を保存

使用例:
  # 状態更新のみ（推奨：サーバー連携時）
  python3 context_controller.py --persona 織田信長 --input_text "よくもやってくれたな" --mode llm --state_file ./persona_state.json --verbose

  # CLI単体テスト（応答も見たい時だけ）
  python3 context_controller.py --persona 織田信長 --input_text "よくもやってくれたな" --mode llm --emit_text --verbose
"""

import os
import re
import sys
import json
import math
import random
import argparse
import subprocess
from typing import Dict
from pathlib import Path


#sys.path.append(os.path.expanduser("~/modules/gar-llm/src/"))

from garllm.utils.env_utils import get_data_path
from garllm.utils.llm_client import request_llm
from garllm.utils.logger import get_logger


# ==========================================
# Utility
# ==========================================

# ロガー初期化
logger = get_logger("context_controller", level="INFO", to_console=False)


def _get_state_path(persona_name: str) -> str:
    """ペルソナ名に対応する state ファイルの絶対パスを返す"""
    return str(Path(get_data_path("personas")) / f"state_{persona_name}.json")


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

def _cc_sanitize(text: str) -> str:
    # 構造化ブロックは丸ごと除去して“観察ノイズ”を消す（本文はそのまま）
    return _CODE_BLOCK_RE.sub("", text)

def _extract_json_safely(raw: str) -> dict:
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")

        candidate = raw[start:end + 1]
        logger.debug(f"[ContextController] JSON candidate:\n{candidate}")
        return json.loads(candidate)

    except Exception as e:
        logger.error(f"[ContextController] Context JSON parse failed: {e}")
        return {}

def clamp(x: float, lo=-1.0, hi=1.0) -> float:
    """値を -1.0〜1.0 に制限"""
    return max(lo, min(hi, x))


def load_state(state_file: str) -> Dict:
    """stateファイルを読み込む。存在しなければ14軸構造のデフォルトを生成"""
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        # relation_axes が残っていればユーザ関係にマイグレーション
        if "relation_axes" in state:
            user_rel = state.pop("relation_axes")
            state.setdefault("relations", {})["user"] = user_rel
        return state
    # 新規初期状態
    rel_axes = {k: 0.0 for k in ["Trust","Familiarity","Hostility","Dominance","Empathy","Instrumentality"]}
    emo_axes = {k: 0.0 for k in ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]}
    return {"relations":{"user":rel_axes},"emotion_axes":emo_axes,"phase_weights":{}}


def save_state(state_file: str, state: Dict):
    """更新後の状態を保存"""
    os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ==========================================
# ルールベース解析
# ==========================================

def analyze_context_rule(text: str) -> Dict:
    
    """簡易ルールベース解析：6軸Relation + 8軸Emotion"""
    t = text.lower()
    d_rel = {k: 0.0 for k in ["Trust", "Familiarity", "Hostility", "Dominance", "Empathy", "Instrumentality"]}
    d_emo = {k: 0.0 for k in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]}

    # ポジティブ・ネガティブワードによる単純変化
    if re.search(r"(ありがとう|感謝|助かっ|うれしい)", t):
        d_emo["joy"] += 0.6; d_emo["trust"] += 0.3
        d_rel["Trust"] += 0.4; d_rel["Familiarity"] += 0.4; d_rel["Empathy"] += 0.3
    elif re.search(r"(怒|ふざけ|許さない|殺)", t):
        d_emo["anger"] += 0.6; d_emo["disgust"] += 0.4
        d_rel["Hostility"] += 0.6; d_rel["Dominance"] += 0.3; d_rel["Empathy"] -= 0.4
    elif re.search(r"(頼む|お願い|助けて)", t):
        d_emo["trust"] += 0.3; d_emo["anticipation"] += 0.3
        d_rel["Trust"] += 0.3; d_rel["Empathy"] += 0.3; d_rel["Dominance"] -= 0.2
    elif re.search(r"(勝|やった|すごい|最高)", t):
        d_emo["joy"] += 0.5; d_emo["anticipation"] += 0.3
        d_rel["Dominance"] += 0.5; d_rel["Hostility"] -= 0.3
    elif re.search(r"(怖|恐|怯)", t):
        d_emo["fear"] += 0.6; d_emo["sadness"] += 0.2
        d_rel["Dominance"] -= 0.5; d_rel["Trust"] -= 0.3
    elif re.search(r"(驚い|なんと|まさか|えっ)", t):
        d_emo["surprise"] += 0.6; d_emo["anticipation"] += 0.3

    # 軽いランダム揺らぎ
    import random
    for k in d_rel:
        d_rel[k] += random.uniform(-0.05, 0.05)
    for k in d_emo:
        d_emo[k] += random.uniform(-0.03, 0.03)

    return {"emotion_axes": d_emo, "relation_axes": d_rel}



# ==========================================
# LLM文脈解析（堅牢JSON抽出）
# ==========================================

def analyze_context_llm(text: str, persona_name: str = "default", debug=False, show_prompt=False) -> Dict:
    """
    LLMベースの文脈解析（6軸Relation + 8軸Emotion対応版）
    GARのペルソナ（AI側）がユーザー発話を受けてどう感じ、関係をどう変化させたかを推定する。
    対話履歴を含む全文を入力とし、変化量のみを -1.0〜+1.0 で出力。
    """
    text = _cc_sanitize(text)

    prompt = f"""
以下は（{persona_name}）に関係する人との対話履歴です。
対話内容を踏まえて、（{persona_name}）の感情と他の人に対する関係性の変化を推定してください。

出力仕様：
- emotion_axes:{persona_name}の感情の変化量（-1.0〜1.0）
- relations: 対象ごとの関係変化（user＋他persona）

出力形式（厳守）:
{{
  "emotion_axes": {{
    "joy": 値, "trust": 値, "fear": 値, "surprise": 値,
    "sadness": 値, "disgust": 値, "anger": 値, "anticipation": 値
  }},
  "relations": {{
    "user": {{
      "Trust": 値, "Familiarity": 値, "Hostility": 値,
      "Dominance": 値, "Empathy": 値, "Instrumentality": 値
    }},
    "<{persona_name}でない他の人>": {{
      "Trust": 値, "Familiarity": 値, "Hostility": 値,
      "Dominance": 値, "Empathy": 値, "Instrumentality": 値
    }},
    <以下同様に他の人との関係性パラメータが続く場合あり>
  }}
}}
各値は -1.0〜1.0 の範囲で、前回状態との差分として「変化量」を示す実数値にしてください。

【会話履歴】
{text}
"""

    logger.debug("====== [DEBUG PROMPT BEGIN] ======")
    logger.debug(prompt)
    logger.debug("====== [DEBUG PROMPT END] ======")

    try:
        raw = request_llm(
            backend="auto",
            messages=[{"role":"user","content": prompt}],
            temperature=0.25,
            max_tokens=600
        ).strip()

        logger.debug("====== [DEBUG LLM raw Output BEGIN] ======")
        logger.debug(raw)
        logger.debug("====== [DEBUG LLM raw Output END] ======")

        parsed = _extract_json_safely(raw)
        if not parsed:
            raise ValueError("empty parsed json")

        # 正規化処理
        emo = {k: clamp(float(parsed.get("emotion_axes", {}).get(k, 0.0))) for k in
               ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]}
        rels = {}

        rel_block = parsed.get("relations") or {}
        for target, axes in rel_block.items():
            rels[target] = {a: clamp(float(v)) for a, v in axes.items()}
        return {"emotion_axes": emo, "relations": rels}

    except Exception as e:

        logger.error(f"LLM context analysis failed: {e}")

        # ==== フォールバック ====
        # 現在の state の構造を維持しつつ「差分は全部0」で返す
        try:
            current = load_emotion_state(persona_name)
        except:
            current = {"emotion_axes": {}, "relations": {}}

        # emotion_axes（構造を維持して全て 0）
        emo_axes = {
            k: 0.0 for k in current.get("emotion_axes", {
                "joy":0,"trust":0,"fear":0,"surprise":0,
                "sadness":0,"disgust":0,"anger":0,"anticipation":0
            }).keys()
        }

        # relations（構造を維持して全て 0）
        rels = {}
        for target, axes in current.get("relations", {}).items():
            rels[target] = {k: 0.0 for k in axes.keys()}

        return {
            "emotion_axes": emo_axes,
            "relations": rels
        }



# ==========================================
# 状態更新
# ==========================================

def update_axes(old: Dict, delta: Dict, alpha=0.3) -> Dict:
    """前回状態と今回の変化を指数移動平均で更新"""
    new = {"emotion_axes": {}, "relations": old.get("relations", {})}    

    # Emotion層：8軸
    for ax in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
        new["emotion_axes"][ax] = clamp(
            (1 - alpha) * old["emotion_axes"].get(ax, 0.0)
            + alpha * delta["emotion_axes"].get(ax, 0.0)
        )

    # Relation層：対象ごとに6軸
    for target, d_axes in delta.get("relations", {}).items():
        new["relations"].setdefault(target, {k:0.0 for k in ["Trust","Familiarity","Hostility","Dominance","Empathy","Instrumentality"]})
        for ax, dval in d_axes.items():
            old_val = new["relations"][target].get(ax,0.0)
            new["relations"][target][ax] = clamp((1 - alpha) * old_val + alpha * dval)

    return new

# ==========================================
# Phase 更新ロジック (soft-argmaxモデル)
# ==========================================

def softmax(values, temperature=0.5):
    """soft-argmaxに基づく正規化"""
    exps = [math.exp(v / max(temperature, 1e-6)) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def update_phase_weights(persona_file: str, state: Dict, delta: Dict,
                         alpha: float = 0.3, beta: float = 0.2,
                         gamma: float = 0.05, temperature: float = 0.4) -> Dict:
    """
    personaファイル内のphase定義を参照し、
    Emotion/Relationの変化量に基づきsoft-argmaxでphase重みを更新する。
    """
    # ペルソナ定義を読み込む
    with open(persona_file, "r", encoding="utf-8") as f:
        persona = json.load(f)

    phases = persona.get("phases", {}) or {}

    # ペルソナ固有の phase_dynamics があれば、デフォルト値を上書きして使う
    phase_dyn = persona.get("phase_dynamics") or {}
    try:
        alpha = float(phase_dyn.get("alpha", alpha))
        beta = float(phase_dyn.get("beta", beta))
        gamma = float(phase_dyn.get("gamma", gamma))
        temperature = float(phase_dyn.get("temperature", temperature))
    except (TypeError, ValueError):
        # 異常値が混じっていても落とさず、デフォルトパラメータで継続
        pass

    # 既存の重みを取得または初期化
    weights = state.get("phase_weights")
    if not isinstance(weights, dict) or not weights:
        n = len(phases)
        if n > 0:
            w = 1.0 / n
            weights = {name: w for name in phases.keys()}
        else:
            weights = {}

    new_vals: Dict[str, float] = {}
    rel_delta = delta.get("relations", {}) or {}
    emo_delta = delta.get("emotion_axes", {}) or {}

    for name, info in phases.items():
        # 以前の重み（なければ 0）
        w_raw = (weights or {}).get(name, 0.0)
        w = float(w_raw) if isinstance(w_raw, (int, float)) else 0.0

        bias_r = info.get("style_bias") or {}
        bias_e = info.get("emotion_bias") or {}
        if not isinstance(bias_r, dict):
            bias_r = {}
        if not isinstance(bias_e, dict):
            bias_e = {}

        # 関係の変化量から寄与を算出
        dr_sum = 0.0
        for k, coeff in bias_r.items():
            if not isinstance(coeff, (int, float)):
                continue
            acc = 0.0
            for axes in rel_delta.values():
                if isinstance(axes, dict):
                    v = axes.get(k)
                    if isinstance(v, (int, float)):
                        acc += float(v)
            dr_sum += float(coeff) * acc

        # 感情の変化量から寄与を算出
        de_sum = 0.0
        for k, coeff in bias_e.items():
            if not isinstance(coeff, (int, float)):
                continue
            v = emo_delta.get(k)
            if isinstance(v, (int, float)):
                de_sum += float(coeff) * float(v)

        # ランダム揺らぎ
        noise = random.uniform(-1.0, 1.0) * gamma

        new_vals[name] = w + alpha * dr_sum + beta * de_sum + noise

    # soft-argmax正規化
    if new_vals:
        vals = list(new_vals.values())
        normed = softmax(vals, temperature)
        new_weights = {k: v for k, v in zip(new_vals.keys(), normed)}

        # 主相（dominant phase）
        state["phase_weights"] = new_weights
        state["dominant_phase"] = max(new_weights, key=new_weights.get)

    # phases が定義されていない場合は state をそのまま返す
    return state


# ==========================================
# 応答生成（CLI検証用オプション）
# ==========================================

def call_response_modulator(persona: str, text: str, state: Dict, intensity: float = 0.8, verbose=False) -> str:
    """
    ※ 本番ワークフローでは使用しないでください。
      CLI で挙動確認したいときだけ --emit_text と併用します。
    """
    # relations構造（ユーザ＋他ペルソナ）
    relations_str = json.dumps(state.get("relations", {}), ensure_ascii=False)
    emotion_str = json.dumps(state.get("emotion_axes", {}), ensure_ascii=False)

    cmd = [
        "python3", os.path.expanduser("~/modules/garllm/style_layer/response_modulator.py"),
        "--persona", persona,
        "--text", text,
        "--intensity", str(intensity),
        "--relations", relations_str,
        "--emotion_axes", emotion_str
    ]

    if verbose:
        cmd.append("--verbose")

    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip()

# ==========================================
# main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Context Controller: update emotion/relation state (no text emission by default)")
    parser.add_argument("--persona", required=True, help="ペルソナ名（例：織田信長）")
    parser.add_argument("--input_text", required=True, help="入力発話テキスト")
    parser.add_argument("--state_file", default="./persona_state.json", help="状態保存ファイルのパス")
    parser.add_argument("--intensity", type=float, default=0.8, help="（CLI検証用）文体強調度(0.0-1.0)")
    parser.add_argument("--mode", choices=["rule", "llm"], default="llm", help="解析モード（rule / llm）")
    parser.add_argument("--emit_text", action="store_true", help="※CLI検証用: 更新状態を用いて応答文も生成して表示する")
    parser.add_argument("--debug", action="store_true", help="デバッグ出力を有効化")
    parser.add_argument("--relations", type=str, help="JSON structure for relations override")
    parser.add_argument("--emotion_axes", type=str, help="JSON structure for emotion axes override")
    args = parser.parse_args()

    # ------------------------------
    # ロガー設定（--debug で制御）
    # ------------------------------
    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("context_controller", level=log_level, to_console=True)

    logger.info(f"Context Controller started (mode={args.mode}, log_level={log_level})")

    # ---------------------------------
    # state ファイルパスの決定
    #  - デフォルト値("./persona_state.json")のままなら
    #    ペルソナごとの state_<persona>.json を使う
    #  - 明示的に --state_file が指定されていればそれを優先
    # ---------------------------------
    if args.state_file and args.state_file != "./persona_state.json":
        state_path = args.state_file
    else:
        state_path = _get_state_path(args.persona)

    # 現在状態をロード
    state = load_state(state_path)

    # CLI からの直接指定を反映
    if args.relations:
        try:
            state["relations"] = json.loads(args.relations)
            logger.debug(f"Overriding relations from CLI: {json.dumps(state['relations'], ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON for --relations")

    if args.emotion_axes:
        try:
            state["emotion_axes"] = json.loads(args.emotion_axes)
            logger.debug(f"Overriding emotion_axes from CLI: {json.dumps(state['emotion_axes'], ensure_ascii=False, indent=2)}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON for --emotion_axes")

    # 文脈解析
    if args.mode == "llm":
        # persona 名を渡すように修正（フォールバック時などの一貫性のため）
        delta = analyze_context_llm(args.input_text, persona_name=args.persona, debug=args.debug, show_prompt=args.debug)
    else:
        delta = analyze_context_rule(args.input_text)

    # 状態更新 & 保存
    new_state = update_axes(state, delta)

    # ペルソナ定義のパス（env_utils.get_data_path に揃える）
    persona_path = str(Path(get_data_path("personas")) / f"persona_{args.persona}.json")

    updated_state = update_phase_weights(persona_path, new_state, delta)
    save_state(state_path, updated_state)

    logger.debug(f"Δ Emotion/Relation: {json.dumps(delta, ensure_ascii=False, indent=2)}")
    logger.debug(f"Updated State: {json.dumps(updated_state, ensure_ascii=False, indent=2)}")

    # CLI検証用
    if args.emit_text:
        print(call_response_modulator(args.persona, args.input_text, updated_state, args.intensity, args.debug))


# After state update and file write
if __name__ == "__main__":
    main()

