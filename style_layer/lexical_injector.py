#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lexical_injector.py — Style Registry resolver with aliases & hierarchical fallback.

使い方（例）:
    from lexical_injector import load_registry, build_constraints_for_persona
    reg = load_registry("/home/aiuser/modules/style_layer/style_registry.json")
    constraints_text, chosen = build_constraints_for_persona(persona_profile, reg)
    # constraints_text をプロンプトへ注入する
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Tuple, Optional, List


def load_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"style_registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(value: Optional[str], alias_map: Dict[str, str]) -> Optional[str]:
    if not value:
        return value
    return alias_map.get(value, value)


def _deep_get(d: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    """階層辞書から順に辿って最も深いノードを返す（失敗ならNone）"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, dict) else None


def _merge_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """下位(具体)→上位(汎用)の順で足りないキーを補完しつつ統合。"""
    out: Dict[str, Any] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        for k, v in n.items():
            if k not in ("一人称", "二人称", "語尾", "語彙"):
                continue
            if k not in out or not out[k]:
                out[k] = list(v) if isinstance(v, list) else v
            else:
                # リストは重複排除で結合
                if isinstance(v, list) and isinstance(out[k], list):
                    seen = set(out[k])
                    out[k].extend([x for x in v if x not in seen])
    return out


def _collect_candidates(registry: Dict[str, Any],
                        cultural: Optional[str],
                        era: Optional[str],
                        gender: Optional[str],
                        role: Optional[str],
                        aliases: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """正規化＋多段フォールバックで最適スタイルを選ぶ。"""
    # エイリアス正規化
    role = _normalize(role, aliases.get("role", {})) if role else role
    gender = _normalize(gender, aliases.get("gender", {})) if gender else gender
    era = _normalize(era, aliases.get("era", {})) if era else era

    reg = registry.get("registry", {})
    generic = reg.get("汎用", {})

    nodes: List[Dict[str, Any]] = []

    # 1) 完全一致: cultural -> era -> gender -> role
    if cultural and era and gender and role:
        n = _deep_get(reg, [cultural, era, gender, role])
        if n: nodes.append(n)
    # 2) genderまで
    if cultural and era and gender:
        n = _deep_get(reg, [cultural, era, gender])
        if n: nodes.append(n)
    # 3) era直下
    if cultural and era:
        n = _deep_get(reg, [cultural, era, "汎用"]) or _deep_get(reg, [cultural, era])
        if n: nodes.append(n)
    # 4) cultural直下
    if cultural:
        n = _deep_get(reg, [cultural, "汎用"]) or _deep_get(reg, [cultural])
        if n: nodes.append(n)
    # 5) global汎用
    if generic:
        nodes.append(generic)

    return _merge_nodes(nodes)


def derive_context_from_persona(persona_profile: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    persona JSON から推測する文脈。存在しない場合は None。
    将来: personaに cultural_context/era/gender/role の明示を推奨。
    """
    ctx = {
        "cultural_context": None,
        "era": None,
        "gender": None,
        "role": None,
    }
    # ここでは暫定的に expression_prompt/linguistic_style などから拾えるなら拾う（任意）
    meta = persona_profile.get("meta", {})
    for k in ctx.keys():
        if k in meta:
            ctx[k] = meta[k]

    # 明示されていなければ None のまま（フォールバックで救う）
    return ctx

def build_constraints_for_persona(persona_profile: Dict[str, Any],
                                  registry: Dict[str, Any],
                                  debug: bool = False) -> Tuple[str, Dict[str, Any]]:
    aliases = registry.get("aliases", {})
    ctx = derive_context_from_persona(persona_profile)

    if debug:
        print(f"[lexical_injector] context: "
              f"{ctx.get('cultural_context')}/{ctx.get('era')}/"
              f"{ctx.get('gender')}/{ctx.get('role')}")

    selected = _collect_candidates(
        registry=registry,
        cultural=ctx.get("cultural_context"),
        era=ctx.get("era"),
        gender=ctx.get("gender"),
        role=ctx.get("role"),
        aliases=aliases
    )

    # ★ injection スキップ判定
    if not selected or not any(selected.values()):
        if debug:
            print("[lexical_injector] no matching lexical data — skip injection")
        return "", {}

    if debug:
        print(f"[lexical_injector] selected: {json.dumps(selected, ensure_ascii=False)}")

    ich = selected.get("一人称", ["私"])
    nih = selected.get("二人称", ["あなた"])
    tail = selected.get("語尾", ["〜です", "〜ます"])
    lex = selected.get("語彙", [])

    if debug:
        print(f"[lexical_injector] 使用一人称={ich}, 二人称={nih}, 語尾={tail}")

    chosen = {"一人称": ich, "二人称": nih, "語尾": tail, "語彙": lex}
    constraints_lines = [
        f"使用する一人称（優先順）: {', '.join(ich)}",
        f"相手の呼称（優先順）: {', '.join(nih)}",
        f"語尾候補（優先順）: {', '.join(tail)}",
    ]
    if lex:
        constraints_lines.append(f"語彙ヒント: {', '.join(lex[:8])}")

    return "\n".join(constraints_lines), chosen

