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
import time
import hashlib


# sys.path.append(os.path.expanduser("~/modules/gar-llm/src/"))

from garllm.utils.llm_client import request_llm
from garllm.utils.env_utils import get_data_path
from garllm.utils.logger import get_logger

# ==========================================
# Utility
# ==========================================

# ロガー初期化
logger = get_logger("response_modulator", level="INFO", to_console=False)


_STYLE_PROFILE_STATS = {"hit": 0, "miss": 0}

# ============================================================
# ⚡ Speed-up caches (in-process)
# ============================================================
_PERSONA_CACHE: dict[str, dict] = {}
_STYLE_PROFILE_CACHE: dict[str, dict[str, object]] = {}  # key -> {"profile": str, "ts": float}
# style_profile cache GC (TTL + max entries)
def _gc_style_profile_cache(ttl_sec: float, max_entries: int) -> int:
    """
    TTL超過を削除し、max_entriesを超えたら古い順に削除する。
    戻り値: 削除した件数
    """
    if ttl_sec <= 0 and (max_entries is None or max_entries <= 0):
        return 0

    now = time.time()
    removed = 0

    # 1) TTL超過を削除
    if ttl_sec > 0:
        expired = []
        for k, ent in _STYLE_PROFILE_CACHE.items():
            try:
                ts = float(ent.get("ts", 0.0))
            except Exception:
                ts = 0.0
            if (now - ts) > ttl_sec:
                expired.append(k)

        for k in expired:
            if _STYLE_PROFILE_CACHE.pop(k, None) is not None:
                removed += 1

    # 2) max_entries超過を削除（古い順）
    if max_entries is not None and max_entries > 0:
        over = len(_STYLE_PROFILE_CACHE) - max_entries
        if over > 0:
            items = sorted(
                _STYLE_PROFILE_CACHE.items(),
                key=lambda kv: float((kv[1] or {}).get("ts", 0.0)),
            )
            for i in range(over):
                k = items[i][0]
                if _STYLE_PROFILE_CACHE.pop(k, None) is not None:
                    removed += 1

    return removed


def _quantize_axes(axes: dict[str, float] | None, step: float = 0.25) -> dict[str, float]:
    """
    小さな揺れでキャッシュが無効化されないよう、軸値を粗く丸める。
    step=0.25 なら -1..1 を 0.25刻み。
    """
    if not axes:
        return {}
    q: dict[str, float] = {}
    for k, v in axes.items():
        try:
            fv = float(v)
        except Exception:
            continue
        fv = max(-1.0, min(1.0, fv))
        q[k] = round(fv / step) * step
    return q


def _quantize_phase_weights(
    phase_weights: dict[str, float] | None,
    *,
    step: float = 0.25,
    scale_by_n: bool = True,
) -> list[tuple[str, float]]:
    """
    phase_weights(総和=1) を「全相・固定順」で量子化して署名にする。

    - 全相を捉える（Top-Kにしない）ので、A/B逆転なども確実に検出できる
    - N相の違いを吸収したい場合は scale_by_n=True にして w*N を量子化する
    - 出力は [(phase_name, bucket), ...] の安定なリスト（name順）
    """
    if not phase_weights:
        return []

    # 安定順（辞書順の揺れを避ける）
    names = sorted([k for k in phase_weights.keys() if isinstance(k, str)])
    n = len(names) if names else 0
    if n <= 0:
        return []

    sig: list[tuple[str, float]] = []
    for name in names:
        v = phase_weights.get(name, 0.0)
        try:
            w = float(v)
        except Exception:
            w = 0.0
        w = max(0.0, min(1.0, w))

        x = (w * n) if scale_by_n else w
        # 量子化
        b = round(x / step) * step
        sig.append((name, round(float(b), 4)))

    return sig


def _style_profile_cache_key(
    persona_name: str,
    phase_weights: dict[str, float] | None,
    relation_axes: dict[str, float] | None,
    emotion_axes: dict[str, float] | None,
    intensity: float,
    *,
    step_axes: float = 0.25,
    step_phase: float = 0.25,
    scale_phase_by_n: bool = True,
) -> str:
    """
    キャッシュキー：persona + 量子化した phase_weights + 量子化した関係/感情 + intensity(粗く)

    phase_fusion(description/refs) は「文字列・順序」が揺れやすいのでキーから外す。
    """
    payload = {
        "persona": persona_name,
        "phase": _quantize_phase_weights(
            phase_weights,
            step=step_phase,
            scale_by_n=scale_phase_by_n,
        ),
        "rel": _quantize_axes(relation_axes, step=step_axes),
        "emo": _quantize_axes(emotion_axes, step=step_axes),
        "int": round(float(intensity), 2),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_persona_profile_cached(persona_name: str) -> Dict[str, Any]:
    """
    既存 load_persona_profile のキャッシュ版（同一プロセス内）
    """
    if persona_name in _PERSONA_CACHE:
        return _PERSONA_CACHE[persona_name]
    data = load_persona_profile(persona_name)
    _PERSONA_CACHE[persona_name] = data
    return data


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

# ============================================================
# 🎭 Expression Injector（表現辞書統合レイヤ）
# ============================================================
import random

def _collect_expression_refs(persona_data: dict, phase_name: str | None):
    """
    persona_data["expression_bank"] と phase 情報から、
    利用対象となる (category, key) の組を集約する。
    - phase.expression_refs に書かれている "cat.key"
    - phase.description 内に書かれた "cat.key"
    """
    bank = persona_data.get("expression_bank") or {}
    refs: set[tuple[str, str]] = set()

    if not bank or not phase_name:
        return bank, refs

    phases = persona_data.get("phases") or {}
    phase = phases.get(phase_name) or {}

    # 1) 明示的な expression_refs
    for ref in phase.get("expression_refs", []):
        if not isinstance(ref, str):
            continue
        if "." not in ref:
            continue
        cat, key = ref.split(".", 1)
        sub = bank.get(cat)
        if isinstance(sub, dict) and key in sub:
            refs.add((cat, key))

    # 2) description 内の "cat.key"
    desc = phase.get("description", "")
    if isinstance(desc, str) and desc:
        found = re.findall(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", desc)
        for cat, key in found:
            sub = bank.get(cat)
            if isinstance(sub, dict) and key in sub:
                refs.add((cat, key))

    return bank, refs


def extract_expression_snippets(persona_data: dict, phase_name: str | None = None) -> str:
    """
    persona_data["expression_bank"] を読み込み、
    対象 phase で利用されるカテゴリからランダムサンプルを生成する。

    - phase_name があれば、その phase に関連するカテゴリから抽出
    - それも無ければ expression_bank 全体からフォールバック
    """
    bank, refs = _collect_expression_refs(persona_data, phase_name)
    if not bank:
        return ""

    samples: list[str] = []

    if refs:
        # 指定カテゴリから抽出
        for (cat, key) in refs:
            sub = bank.get(cat, {})
            if isinstance(sub, dict):
                lst = sub.get(key)
                if isinstance(lst, list) and lst:
                    samples.append(random.choice(lst))
    else:
        # フォールバック：全カテゴリからランダム抽出
        flat: list[str] = []
        for cat, sub in bank.items():
            if isinstance(sub, dict):
                for key, lst in sub.items():
                    if isinstance(lst, list):
                        flat.extend(lst)
        if flat:
            samples.append(random.choice(flat))

    if not samples:
        return ""

    joined = " / ".join(samples[:3])
    return f"【表現ヒントサンプル】{joined}"


def sample_expression_snippets_weighted(
    persona_data: dict,
    expression_refs: list[str] | None,
    max_samples: int = 3,
) -> list[str]:
    """
    フェーズ重畳によって得られた expression_refs に基づき、
    expression_<persona>.json のカテゴリからサンプルを抽選する。

    対応パターン:
      - "talk.intro" のような cat.key 形式
      - "battle_cries" のようなドット無しキー（expression_bank[ref]）

    ・expression_refs が None or 空なら従来の extract_expression_snippets() にフォールバック。
    ・カテゴリ内のフレーズは expression のままコピーせず、"素材" としてそのまま渡す。
      （揺らぎづけはプロンプト生成用LLMが担当）
    """
    expressions = persona_data.get("expression_bank") or {}
    if not expressions:
        return []

    # フォールバック: refs が無い場合は旧ヘルパーを使う
    if not expression_refs:
        try:
            snippet = extract_expression_snippets(persona_data, phase_name=None)
        except Exception:
            snippet = ""
        return [snippet] if snippet else []

    flat_list: list[str] = []

    for ref in expression_refs:
        if not isinstance(ref, str):
            continue

        # 1) "cat.key" 形式
        if "." in ref:
            cat, key = ref.split(".", 1)
            sub = expressions.get(cat)
            if isinstance(sub, dict):
                arr = sub.get(key)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, str):
                            flat_list.append(item)
            continue

        # 2) ドット無しキー → expression_bank[ref] を見る
        val = expressions.get(ref)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    flat_list.append(item)
        elif isinstance(val, dict):
            # サブカテゴリをすべてフラットに集約
            for lst in val.values():
                if isinstance(lst, list):
                    for item in lst:
                        if isinstance(item, str):
                            flat_list.append(item)

    if not flat_list:
        return []

    random.shuffle(flat_list)
    return flat_list[:max_samples]



def build_expression_instruction(
    persona_data: dict,
    phase_name: str | None = None,
    expression_refs: list[str] | None = None,
) -> str:
    """
    相に紐づく expression の使い方を、LLM 向けの「操作ルール」として文章化する。

    対応パターン:
      - "cat.key" 形式（例: "talk.intro"）
      - "flat_key" 形式（例: "battle_cries"）
    """
    bank = persona_data.get("expression_bank") or {}
    if not bank:
        return ""

    pair_refs: set[tuple[str, str]] = set()  # ("cat","key")
    flat_keys: set[str] = set()              # "battle_cries" など

    # 1) phase_name ベースの参照（従来の挙動 + flat key 拡張）
    if phase_name is not None:
        try:
            _, phase_pairs = _collect_expression_refs(persona_data, phase_name)
            pair_refs.update(phase_pairs)
        except Exception:
            pass

        phases = persona_data.get("phases") or {}
        phase = phases.get(phase_name) or {}
        for ref in phase.get("expression_refs", []):
            if not isinstance(ref, str):
                continue
            if "." in ref:
                cat, key = ref.split(".", 1)
                pair_refs.add((cat, key))
            else:
                flat_keys.add(ref)

    # 2) phase_fusion などから渡された expression_refs
    if expression_refs:
        for ref in expression_refs:
            if not isinstance(ref, str):
                continue
            if "." in ref:
                cat, key = ref.split(".", 1)
                pair_refs.add((cat, key))
            else:
                flat_keys.add(ref)

    # 実在するカテゴリだけに絞り込む
    valid_pairs: set[tuple[str, str]] = set()
    for cat, key in pair_refs:
        sub = bank.get(cat)
        if isinstance(sub, dict) and key in sub:
            valid_pairs.add((cat, key))
    pair_refs = valid_pairs

    valid_flat: set[str] = set()
    for k in flat_keys:
        val = bank.get(k)
        if isinstance(val, (list, dict)):
            valid_flat.add(k)
    flat_keys = valid_flat

    if not pair_refs and not flat_keys:
        return ""

    persona_label = persona_data.get("persona_name") or "ペルソナ"
    lines: list[str] = []
    lines.append("【相に基づく表現操作ルール】")
    lines.append("・以下の expression カテゴリは、元の文をそのままコピペするのではなく、意味とノリを保ちながら、類義語・言い換え・語尾変形・カタカナ化・即興造語などで再構成してよい。")
    lines.append("・カテゴリ内のフレーズは「素材」として扱い、複数を組み合わせたり部分的に変形して、新しいセリフや歌詞を作ること。")
    lines.append("・サンプルとしていくつかのフレーズを示すが、そのまま固定文としてではなく、必ず少し揺らぎを加えて使うこと。")

    # まず cat.key 形式
    for cat, key in sorted(pair_refs):
        lines.append(f"・{cat}.{key} : expression_{persona_label}.json 内のフレーズ群を素材として利用せよ。")
        sub = bank.get(cat, {})
        if not isinstance(sub, dict):
            continue
        lst = sub.get(key)
        if not isinstance(lst, list) or not lst:
            continue

        examples = [s for s in lst if isinstance(s, str) and s.strip()]
        random.shuffle(examples)
        for ex in examples[:2]:
            ex_clean = ex.strip()
            lines.append(
                f"    - 例(cat.{key}): 「{ex_clean}」のニュアンスを保ちつつ、語尾や言い回しを少し変形して使ってよい。"
            )

    # 次に flat key 形式
    for k in sorted(flat_keys):
        lines.append(f"・{k} : expression_{persona_label}.json 内のフレーズ群を素材として利用せよ。")
        val = bank.get(k)
        flat: list[str] = []
        if isinstance(val, list):
            flat.extend(s for s in val if isinstance(s, str) and s.strip())
        elif isinstance(val, dict):
            for lst in val.values():
                if isinstance(lst, list):
                    flat.extend(s for s in lst if isinstance(s, str) and s.strip())

        if not flat:
            continue

        random.shuffle(flat)
        for ex in flat[:2]:
            ex_clean = ex.strip()
            lines.append(
                f"    - 例({k}): 「{ex_clean}」のニュアンスを保ちつつ、語尾や言い回しを少し変形して使ってよい。"
            )

    return "\n".join(lines)





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


# ============================================================
# 🧭 Phase Selector（相の選択）
# ============================================================
def select_active_phase(persona_name: str, persona_data: Dict[str, Any]) -> tuple[str | None, str, Dict[str, Any]]:
    """
    現在有効な「相（phase）」を決定する。

    優先順位:
      1. state_<persona_name>.json の "dominant_phase"
      2. state_<persona_name>.json の "phase_weights" 最大値
      3. persona の "基本相"
      4. persona["phases"] の先頭
    """
    phases = persona_data.get("phases") or {}
    if not phases:
        return None, "", {}

    phase_name: str | None = None
    phase_cfg: Dict[str, Any] = {}

    # 1 / 2. state_<persona>.json を見る
    try:
        state_path = Path(get_data_path("personas")) / f"state_{persona_name}.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            dom = state.get("dominant_phase")
            if isinstance(dom, str) and dom in phases:
                phase_name = dom
            else:
                weights = state.get("phase_weights") or {}
                if isinstance(weights, dict):
                    candidates: list[tuple[str, float]] = []
                    for name in phases.keys():
                        w = weights.get(name)
                        if isinstance(w, (int, float)):
                            candidates.append((name, float(w)))
                    if candidates:
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        phase_name = candidates[0][0]
    except Exception:
        # state が壊れていても落ちないようにする
        phase_name = None

    # 3. "基本相" があれば優先
    if phase_name is None and "基本相" in phases:
        phase_name = "基本相"

    # 4. それでもなければ最初のキー
    if phase_name is None:
        phase_name = next(iter(phases.keys()))

    phase_cfg = phases.get(phase_name, {}) or {}
    desc = phase_cfg.get("description", "")
    if not isinstance(desc, str):
        desc = ""

    return phase_name, desc, phase_cfg


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
# 🧮 Phase Fusion（相の重ね合わせ）
# ============================================================
def load_phase_weights(persona_name: str, persona_data: Dict[str, Any]) -> dict[str, float]:
    """
    state_<persona>.json の phase_weights を読み込み、
    なければ persona["phases"] を一様分布で初期化する。
    """
    phases = persona_data.get("phases") or {}
    weights: dict[str, float] = {}

    if not phases:
        return {}

    # state から読む
    try:
        state_path = Path(get_data_path("personas")) / f"state_{persona_name}.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            raw = state.get("phase_weights") or {}
            if isinstance(raw, dict):
                for name, v in raw.items():
                    if name in phases and isinstance(v, (int, float)):
                        weights[name] = float(v)
    except Exception:
        weights = {}

    # 何も取れなかったら一様
    if not weights:
        n = len(phases)
        if n > 0:
            w = 1.0 / n
            weights = {name: w for name in phases.keys()}

    # 正規化
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    #logger.debug(f"phase_weights:{json.dumps(weights, ensure_ascii=False)}")

    return weights


def fuse_phase_config(persona_data: Dict[str, Any], phase_weights: dict[str, float]) -> Dict[str, Any]:
    """
    phase_weights（合計 1.0）に基づき、全相の情報を重ね合わせる。

    戻り値:
      {
        "description": 相ごとの説明を重み付きでまとめたテキスト,
        "expression_refs": 重み付き優先度順の expression 参照リスト,
        "style_bias": 相ごとの style_bias の重み付き合成,
        "emotion_bias": 相ごとの emotion_bias の重み付き合成,
      }
    """
    phases = persona_data.get("phases") or {}
    if not phases or not phase_weights:
        return {"description": "", "expression_refs": [], "style_bias": {}, "emotion_bias": {}}

    desc_chunks: list[str] = []
    expr_weight_map: dict[str, float] = {}
    fused_style: dict[str, float] = {}
    fused_emotion: dict[str, float] = {}

    for name, cfg in phases.items():
        w = phase_weights.get(name)
        if not isinstance(w, (int, float)) or w <= 0:
            continue

        # 説明
        desc = cfg.get("description")
        if isinstance(desc, str) and desc.strip():
            desc_chunks.append(f"【{name}（重み {w:.2f}）】{desc.strip()}")

        # style_bias
        sb = cfg.get("style_bias") or {}
        if isinstance(sb, dict):
            for k, v in sb.items():
                if isinstance(v, (int, float)):
                    fused_style[k] = fused_style.get(k, 0.0) + w * float(v)

        # emotion_bias
        eb = cfg.get("emotion_bias") or {}
        if isinstance(eb, dict):
            for k, v in eb.items():
                if isinstance(v, (int, float)):
                    fused_emotion[k] = fused_emotion.get(k, 0.0) + w * float(v)

        # expression_refs
        for ref in cfg.get("expression_refs", []):
            if isinstance(ref, str):
                expr_weight_map[ref] = expr_weight_map.get(ref, 0.0) + w

    # 優先度順に並べた expression_refs
    sorted_refs = sorted(expr_weight_map.items(), key=lambda x: x[1], reverse=True)
    fused_refs = [r for r, _ in sorted_refs]

    fused_desc = "\n".join(desc_chunks)

    phase_fusion = {
        "description": fused_desc,
        "expression_refs": fused_refs,
        "style_bias": fused_style,
        "emotion_bias": fused_emotion,
    }
    #logger.debug(f"phase_fusion:{json.dumps(phase_fusion, ensure_ascii=False)}")    

    return phase_fusion




# ============================================================
# 🧬 Core Profile Summary + Style Profile LLM
# ============================================================
def summarize_core_profile(persona_data: Dict[str, Any]) -> str:
    """
    core_profile から、応答LLMに渡すための簡潔な日本語サマリを作る。
    """
    core = persona_data.get("core_profile") or {}
    lines: list[str] = []

    summary = core.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(f"・概要: {summary.strip()}")

    values = core.get("values")
    if isinstance(values, list) and values:
        vs = " / ".join(str(v) for v in values)
        lines.append(f"・価値観: {vs}")

    reasoning = core.get("reasoning_pattern")
    if isinstance(reasoning, str) and reasoning.strip():
        lines.append(f"・思考パターン: {reasoning.strip()}")

    speech = core.get("speech_pattern")
    if isinstance(speech, str) and speech.strip():
        lines.append(f"・話し方の傾向: {speech.strip()}")

    # 性別・年代などの属性
    demographic = core.get("demographic")
    if isinstance(demographic, dict):
        gender = demographic.get("gender")
        age_range = demographic.get("age_range")
        parts = []
        if isinstance(gender, str) and gender.strip() and gender.strip() != "不明":
            parts.append(f"性別: {gender.strip()}")
        if isinstance(age_range, str) and age_range.strip() and age_range.strip() != "不明":
            parts.append(f"年代: {age_range.strip()}")
        if parts:
            lines.append("・属性: " + " / ".join(parts))

    # 言語・なまり・口癖など
    lang_prof = core.get("language_profile")
    if isinstance(lang_prof, dict):
        lparts = []
        dialect = lang_prof.get("dialect")
        if isinstance(dialect, str) and dialect.strip() and dialect.strip() != "不明":
            lparts.append(f"方言・なまり: {dialect.strip()}")
        speech_style = lang_prof.get("speech_style")
        if isinstance(speech_style, str) and speech_style.strip():
            lparts.append(f"話し方のスタイル: {speech_style.strip()}")
        samples = lang_prof.get("sample_phrases")
        if isinstance(samples, list) and samples:
            sample_str = " / ".join(str(s) for s in samples[:3])
            lparts.append(f"口癖・表現例: {sample_str}")
        if lparts:
            lines.append("・言語・口調: " + " / ".join(lparts))



    return "\n".join(lines) if lines else "（概要情報なし）"


def build_style_profile_with_llm(
    persona_name: str,
    persona_data: Dict[str, Any],
    phase_fusion: Dict[str, Any],
    relation_axes: Dict[str, float] | None = None,
    emotion_axes: Dict[str, float] | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 384,
) -> str:
    """
    相の重畳結果 + persona 基本情報 + 関係軸 + 感情軸 + expression をまとめて、
    応答LLMに渡す「話法・スタイル指針テキスト」を LLM に生成させる。

    ※ meta.styleNotes / song.chorus / talk.intro などの expression タグは
       あくまで「内部タグ」としてだけ渡し、style_profile 本文には出させない。

    速度最適化:
      - max_tokens は 2048 固定ではなく、呼び出し側から下げられるようにする
    """

    fused_style_bias = phase_fusion.get("style_bias") or {}
    fused_emotion_bias = phase_fusion.get("emotion_bias") or {}
    fused_desc = (phase_fusion.get("description") or "").strip() or "（相の説明なし）"
    expr_refs = phase_fusion.get("expression_refs") or []

    unique_cats: list[str] = []
    if expr_refs:
        cats = {ref.split(".", 1)[0] for ref in expr_refs if isinstance(ref, str) and "." in ref}
        unique_cats = sorted(cats)

    if unique_cats:
        expr_block = "・" + "\n・".join(unique_cats)
    else:
        expr_block = "（指定なし）"

    # 代表フレーズは少数だけ（長文化抑制）
    expr_samples = sample_expression_snippets_weighted(
        persona_data,
        phase_fusion.get("expression_refs"),
        max_samples=2,  # ← 元は3。少し削る
    )

    core_summary = summarize_core_profile(persona_data)
    style = persona_data.get("style", {})
    first_person = style.get("first_person", []) or ["私"]
    second_person = style.get("second_person", []) or ["あなた"]
    keywords = style.get("keywords", []) or []

    rel_hint = synthesize_relation_hint(relation_axes) if relation_axes else "（指定なし）"
    emo_hint = generate_emotion_prompt(emotion_axes) if emotion_axes else "感情指針: （指定なし）"

    prompt = f"""
あなたは「ペルソナ話法設計アシスタント」です。
目的は、以下の情報をもとに、LLM が『{persona_name}』として発話するための
一貫した「話法・スタイル指針」を日本語でまとめることです。

この指針は、別の応答生成用 LLM に system プロンプトとして渡されます。
出力はそのまま貼り付けて使えるように、純粋な日本語テキストのみで記述してください（JSONは禁止）。

【ペルソナ名】
{persona_name}

【コアプロファイル要約】
{core_summary}

【基本スタイル情報】
・一人称候補: {", ".join(first_person)}
・二人称候補: {", ".join(second_person)}
・キーワード例: {", ".join(keywords) if keywords else "（未指定）"}

【相（フェーズ）の重畳情報】
{fused_desc}

【相ベースのスタイルバイアス（合成済み）】
{json.dumps(fused_style_bias, ensure_ascii=False)}

【相ベースの感情バイアス（合成済み）】
{json.dumps(fused_emotion_bias, ensure_ascii=False)}

【関係性ヒント（ユーザ⇄ペルソナ）】
{rel_hint}

【感情ヒント】
{emo_hint}

【内部用の表現カテゴリタグ（expression の参照。出力には書かない）】
{expr_block}

【参考用 expression サンプル（このままコピペせず、ニュアンスだけを使うこと）】
{expr_samples if expr_samples else "（特に指定なし）"}

【出力要件】
- 口調 / 語彙傾向 / 文長・リズム / 揺らぎの付け方 を短めに要点化して書く
- 代表例は 1〜3 個まで（長文化しない）
- タグ名や cat.key は本文に出さない
"""

    style_profile = ask_llm(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return (style_profile or "").strip()


    

# ============================================================
# 🧠 Prompt Construction（応答生成用プロンプト構築）
# ============================================================
def build_prompt(
    input_text: str,
    persona_name: str,
    persona_data: Dict[str, Any],
    intensity: float = 0.7,
    verbose: bool = False,
    relation_axes: Dict[str, float] | None = None,
    relations: Dict[str, Dict[str, float]] | None = None,
    emotion_axes: Dict[str, float] | None = None,
    style_profile: str | None = None,
    expression_instruction: str | None = None,
):
    """
    ユーザー発話に対する『ペルソナとしての応答』を生成するプロンプトを構築。
    相・expression の詳細は style_profile（別LLMの出力）と expression_instruction に織り込まれている前提。
    """
    style = persona_data.get("style", {})
    knowledge = (
        persona_data.get("knowledge_anchors")
        or persona_data.get("core_profile", {}).get("knowledge_anchors", [])
    )

    # 人称ガイダンス（候補提示＋候補外禁止）
    pronoun_guidance = build_pronoun_guidance(persona_data, relations)

    # 冗長さガイド
    expressiveness = (
        "簡潔に1〜2文で答える。" if not verbose
        else "丁寧かつ饒舌に、2〜4文程度で情景や心情も補って答える。"
    )

    # 関係性ヒント（ユーザ⇄persona）
    relation_hint = synthesize_relation_hint(relation_axes) if relation_axes else ""

    # 他ペルソナとの関係
    relation_context = ""
    if relations:
        others = []
        for target, axes in relations.items():
            if target in ["ユーザ", "ユーザー", "User", "user"]:
                continue
            desc = synthesize_relation_hint(axes)
            if desc:
                others.append(f"{target}: {desc}")
        relation_context = " / ".join(others) if others else "（指定なし）"
    else:
        relation_context = "（指定なし）"

    # 感情ヒント
    emotion_hint_text = generate_emotion_prompt(emotion_axes) if emotion_axes else "（指定なし）"

    # core_profile 要約
    core_summary = summarize_core_profile(persona_data)

    # knowledge anchors
    knowledge_lines = []
    if isinstance(knowledge, list):
        for k in knowledge:
            if isinstance(k, dict):
                label = k.get("label") or k.get("type") or ""
                ref = k.get("reference") or k.get("significance") or ""
                if label or ref:
                    knowledge_lines.append(f"- {label}: {ref}")
    knowledge_block = "\n".join(knowledge_lines) if knowledge_lines else "（特記なし）"

    style_profile_text = style_profile or "（話法・スタイル指針は別途定義されているものとする）"
    expr_instruction_text = expression_instruction or "（expression 由来の特別な指針はない）"

    # プロンプト本体
    prompt = f"""
あなたは今から完全に『{persona_name}』として応答します。
口調・語彙・価値観・判断基準は {persona_name} のものを厳守してください。
{pronoun_guidance}

【ペルソナの基本情報】
{core_summary}

【話法・スタイル指針（相・expression・関係性・感情を統合したもの）】
{style_profile_text}

【expression 由来の表現操作ルール（内部ガイド）】
{expr_instruction_text}

スタイル強度: {intensity * 100:.0f}%
他者との関係: {relation_hint if relation_hint else "（指定なし）"}
他ペルソナとの関係: {relation_context}
{emotion_hint_text if emotion_hint_text else "感情指針: （指定なし）"}
関係性や感情指針の内容は、応答の語彙・口調・態度・話法に必ず反映させること。
{expressiveness}

【ペルソナ固有の知識アンカー（過去の出来事など）】
{knowledge_block}

【ユーザー発話】 
{input_text}

【厳守事項】
- 出力は**あなた（{persona_name}）としての応答文のみ**。説明・前置き・メタ記述は禁止。
- 人称は上記候補からのみ選択し、一貫して用いる。候補外の人称は使用禁止。
- 質問返しは避け、まずは**答え**を返す（必要なら最後に1件だけ簡潔な問い返し可）。
- 日本語で書く。

【出力】
""".strip()
    return prompt





# ============================================================
# 💬 LLM Interface with Output Cleaner
# ============================================================
def ask_llm(prompt: str, temperature=0.6, max_tokens=800) -> str:
    return ask_llm_chat([{"role": "user", "content": prompt}])


# ============================================================
# 💬 Chat形式 LLM Interface
# ============================================================
def ask_llm_chat(
    messages: list[dict[str, str]],
    temperature=0.6,
    max_tokens=800,
    top_p: float = 1.0,
    gen_params: dict | None = None,
) -> str:
    """
    Chat形式 (messages[]) 入力対応版。
    gen_params があれば request_llm に extra_params として渡す。
    """
    try:
        response = request_llm(
            messages=messages,
            backend="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_params=gen_params or {},
        )
        cleaned = response.strip()
        return cleaned
    except Exception as e:
        logger.error(f"[response_modulator] Chat LLM error: {e}")
        return ""


# ============================================================
# 🎭 Response Modulation Core
# ============================================================
def modulate_response(
    text: str | list[dict[str, str]],
    persona_name: str,
    intensity: float = 0.7,
    verbose: bool = False,
    relation_axes: dict[str, float] | None = None,
    relations: dict[str, dict[str, float]] | None = None,  # ← relay_server から渡される複数関係
    emotion_axes: dict[str, float] | None = None,
    debug: bool = False,
    log_console: bool = False,
    gen_params: dict | None = None,
):
    """
    text が str なら従来どおり build_prompt() を使う。
    text が list (messages形式) なら Chat形式で LLM を呼び出す。

    構造:
      1. persona/state/relations/emotion から「相の重畳」を計算
      2. スタイル設計用 LLM で「話法・スタイル指針」を生成
      3. 応答生成LLMに、上記スタイル指針＋会話履歴/ユーザ入力を渡す
    """

    # logger instance は既に存在している想定
    '''
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # console handler の追加（必要なら）
    if log_console:
        if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logger.handlers[0].formatter)
            logger.addHandler(console)
    '''
    global logger
    
    # 既存の logger がある場合でも level を更新する
    log_level = "DEBUG" if debug else "INFO"
    logger = get_logger("response_modulator", level=log_level, to_console=log_console)

    persona_data = load_persona_profile_cached(persona_name)

    # relations からユーザ対象の軸だけを抽出（あれば）
    if relations and isinstance(relations, dict):
        # "ユーザ/ユーザー/User/user" を優先
        target_name = None
        for cand in ["ユーザ", "ユーザー", "User", "user"]:
            if cand in relations:
                target_name = cand
                break
        if target_name:
            relation_axes = extract_relation_axes_for_target(relations, target_name)

    # --- 相の重畳 ---
    phase_weights = load_phase_weights(persona_name, persona_data)
    phase_fusion = fuse_phase_config(persona_data, phase_weights)

    # --- スタイル・話法プロファイル（重いのでキャッシュ優先） ---
    # gen_params で挙動を上書き可能:
    #   style_profile_mode: "cached" | "always" | "off"
    #   style_profile_max_tokens: int
    #   style_profile_temperature: float
    sp_mode = (gen_params or {}).get("style_profile_mode", "cached")
    sp_max_tokens = int((gen_params or {}).get("style_profile_max_tokens", 384))
    sp_temp = float((gen_params or {}).get("style_profile_temperature", 0.2))
    sp_ttl_sec = float((gen_params or {}).get("style_profile_ttl_sec", 3600))  # 1h
    sp_cache_max_entries = int((gen_params or {}).get("style_profile_cache_max_entries", 256))

    style_profile = ""
    cache_key = None

    if sp_mode == "off":
        logger.debug("[style_profile] mode=off (skip)")
    else:
        phase_sig = _quantize_phase_weights(
            phase_weights,
            step=0.25,
            scale_by_n=True,
        )
        logger.debug(f"[style_profile] phase_sig={phase_sig}")

        cache_key = _style_profile_cache_key(
            persona_name=persona_name,
            phase_weights=phase_weights,
            relation_axes=relation_axes,
            emotion_axes=emotion_axes,
            intensity=intensity,
            step_axes=0.25,
            step_phase=0.25,
            scale_phase_by_n=True,
        )

        # GC: TTL超過や件数超過を掃除（毎回でOK。重ければ間引き運用に変更可）
        _gc_style_profile_cache(sp_ttl_sec, sp_cache_max_entries)

        if sp_mode == "cached":
            ent = _STYLE_PROFILE_CACHE.get(cache_key)
            if ent:
                ts = float(ent.get("ts", 0.0))
                age = time.time() - ts
                if age <= sp_ttl_sec:
                    style_profile = str(ent.get("profile", "") or "")
                    _STYLE_PROFILE_STATS["hit"] += 1
                    logger.debug(
                        f"[style_profile] HIT key={cache_key[:8]} age={age:.1f}s "
                        f"hits={_STYLE_PROFILE_STATS['hit']} miss={_STYLE_PROFILE_STATS['miss']}"
                    )
                else:
                    logger.debug(f"[style_profile] EXPIRED key={cache_key[:8]} age={age:.1f}s ttl={sp_ttl_sec:.0f}s")
                    # TTL超過はキャッシュからも削除して肥大を防ぐ
                    _STYLE_PROFILE_CACHE.pop(cache_key, None)
                    

        if (sp_mode in ("always", "cached")) and not style_profile:
            _STYLE_PROFILE_STATS["miss"] += 1
            logger.debug(
                f"[style_profile] MISS key={cache_key[:8]} -> build_style_profile_with_llm() "
                f"hits={_STYLE_PROFILE_STATS['hit']} miss={_STYLE_PROFILE_STATS['miss']}"
            )


            t0 = time.time()
            style_profile = build_style_profile_with_llm(
                persona_name=persona_name,
                persona_data=persona_data,
                phase_fusion=phase_fusion,
                relation_axes=relation_axes,
                emotion_axes=emotion_axes,
                temperature=sp_temp,
                max_tokens=sp_max_tokens,
            )
            dt = time.time() - t0
            logger.debug(f"[style_profile] build_style_profile_with_llm() done in {dt:.2f}s key={cache_key[:8]}")

            _STYLE_PROFILE_CACHE[cache_key] = {"profile": style_profile, "ts": time.time()}
            _gc_style_profile_cache(sp_ttl_sec, sp_cache_max_entries)



    # --- expression 由来の表現操作ルール（expression_bank 利用） ---
    expression_instruction = build_expression_instruction(
        persona_data=persona_data,
        phase_name=None,
        expression_refs=phase_fusion.get("expression_refs"),
    )


    # Chat形式の場合（relay_server 経由など）
    if isinstance(text, list):
        logger.debug("Chat-mode messages input detected")
        logger.debug(json.dumps(text, ensure_ascii=False, indent=2))

        style = persona_data.get("style", {})
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
            rel_others_hint = " / ".join(others_hint) if others_hint else "（指定なし）"
        else:
            rel_others_hint = "（指定なし）"

        # 感情ヒント
        emo_hint = generate_emotion_prompt(emotion_axes) if emotion_axes else "（指定なし）"

        core_summary = summarize_core_profile(persona_data)

        persona_system_message = {
            "role": "system",
            "content": (
                f"あなたは今から完全に『{persona_name}』として応答します。\n"
                f"{pronoun_guidance}\n\n"
                f"【ペルソナの基本情報】\n{core_summary}\n\n"
                f"【話法・スタイル指針（相・expression・関係性・感情を統合したもの）】\n"
                f"{style_profile}\n\n"
                f"【expression 由来の表現操作ルール（内部ガイド）】\n"
                f"{expression_instruction or '（expression 由来の特別な指針はない）'}\n\n"
                f"スタイル強度: {intensity*100:.0f}%\n"
                f"関係性（ユーザ⇄{persona_name}）: {rel_user_hint}\n"
                f"他ペルソナとの関係: {rel_others_hint}\n"
                f"{emo_hint}\n"
                f"出力は応答文のみ。メタ発言禁止。"
            )
        }


        messages_with_persona = [persona_system_message] + text

        logger.debug(f"persona_system_message:\n{json.dumps(persona_system_message, ensure_ascii=False, indent=2)}")

        response = ask_llm_chat(
            messages_with_persona,
            # OpenWebUIから来た値があればそれを優先させる（無ければ ask_llm_chat 側デフォルト）
            temperature=(gen_params or {}).get("temperature", 0.6),
            max_tokens=(gen_params or {}).get("max_tokens", 800),
            top_p=(gen_params or {}).get("top_p", 1.0),
            gen_params=gen_params,
        )

        return response.strip() if response else ""

    # テキストモード（旧 CLI 互換）
    prompt = build_prompt(
        input_text=text,
        persona_name=persona_name,
        persona_data=persona_data,
        intensity=intensity,
        verbose=verbose,
        relation_axes=relation_axes,
        relations=relations,
        emotion_axes=emotion_axes,
        style_profile=style_profile,
        expression_instruction=expression_instruction,
    )

  
    logger.debug("generated prompt\n%s\n%s", prompt, "=" * 80)

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
    parser.add_argument("--relations", type=str, default=None, help="関係性構造（JSON: {'ユーザ': {...}, '徳川家康': {...}}）")
    parser.add_argument("--debug", action="store_true", help="デバッグ表示（プロンプト出力）")
    parser.add_argument("--log-console", action="store_true", help="ログをコンソールにも出力") 

    args = parser.parse_args()

    # ------------------------------
    # ロガー設定（--debug で制御）
    # ------------------------------
    global logger
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("response_modulator", level=log_level, to_console=args.log_console)
    logger.info(f"Response modulation log_level={log_level})")

    relation_axes = json.loads(args.relation_axes) if args.relation_axes else None
    relations = json.loads(args.relations) if args.relations else None
    emotion_axes = json.loads(args.emotion_axes) if args.emotion_axes else None

    rewritten = modulate_response(
        text=args.text,
        persona_name=args.persona,
        intensity=args.intensity,
        verbose=args.verbose,
        relation_axes=relation_axes,
        relations=relations,
        emotion_axes=emotion_axes,
        debug=args.debug,
        log_console=args.log_console
    )


    logger.debug("\n==== Rewritten Text ====")
    logger.debug(rewritten)
    logger.debug("=" * 80)

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
