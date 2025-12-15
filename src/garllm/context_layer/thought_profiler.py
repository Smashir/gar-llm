#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
thought_profiler.py — GAR Context Layer: Thought Profiler（semantic専用・再設計版）

目的:
  - semantic_*.json（または同等の構造）から人物の思想・価値観・エピソードを抽出し、
    thought_*.json を生成する。
  - episodes / anchors を thought に統合し、persona_generator がそのまま使える形にする。

入力想定:
[
  {
    "title": "記事タイトル",
    "url": "元URL",
    "summary": "人物中心の要約",        # semantic_condenser の出力
    "description": "元本文の要約など"   # あれば fallback 用
  },
  ...
]

出力:
{
  "persona_name": "...",
  "summary": "...",
  "background": "...",
  "values": [...],
  "reasoning_pattern": "...",
  "speech_pattern": "...",
  "episodes": [...],
  "anchors": [...]
}
"""

import os
import sys
import json
import argparse
from pathlib import Path

from garllm.utils.env_utils import get_data_path
from garllm.utils.llm_client import request_llm


from garllm.utils.logger import get_logger

logger = get_logger("thought_profiler", level="DEBUG", to_console=True)


# ================================================================
# 資料読込
# ================================================================
def load_entries(input_path: str):
    path = Path(input_path)
    if not path.exists():
        logger.error(f"入力ファイルが存在しない: {path}")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug(f"load_entries: {len(data) if isinstance(data, list) else 1} 件読込")
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            logger.error(f"予期しないJSON形式: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"JSON読込失敗: {path} ({e})")
        return []



# ================================================================
# 資料束構築
# ================================================================
def build_materials_text(persona, entries):
    if not entries:
        logger.error("資料エントリなし（entries が空）。")
        return ""

    chunks = []
    for i, e in enumerate(entries, 1):
        title = e.get("title", f"資料{i}")
        body = e.get("summary") or e.get("description") or ""
        logger.debug(f"entry[{i}] summary={bool(e.get('summary'))}, description={bool(e.get('description'))}")
        if not body:
            continue
        chunks.append(f"【資料{i}: {title}】\n{body}\n")

    if not chunks:
        logger.error("資料束用テキストなし（summary/description が全て空）。")
        return ""

    materials = "\n".join(chunks)
    logger.info(f"資料束構築: 長さ={len(materials)}文字, 件数={len(chunks)}")
    return materials



# ================================================================
# LLM 呼び出し
# ================================================================
def ask_profile_llm(persona: str, materials: str) -> str:
    """
    LLM に対して、人物の思想・価値観・エピソード・アンカーを
    一括で JSON として出力させる。
    """
    logger.debug(f"LLM呼出開始: persona={persona}, materials_len={len(materials)}")

    prompt = f"""
あなたの役割は、与えられた資料から特定の人物の思想・価値観・話し方を分析し、
指定されたスキーマに従って JSON を生成することです。

対象人物: {persona}

【資料束】
以下はこの人物に関する資料です。重複していても構いません。全体として人物像をつかんでください。

----
{materials}
----

【出力で求めるもの】
以下のキーを持つ JSON オブジェクト 1 つだけを出力してください。

{{
  "summary": "人物の思想や活動全体を一言でまとめた要約（200〜300文字）",
  "background": "出身や生い立ち、所属、活動の舞台などをまとめた説明（150〜250文字）",
  "values": ["価値観や信念を表す短いフレーズ", "..."],
  "reasoning_pattern": "ものごとの考え方・判断の癖・思考パターンの説明（150〜250文字）",
  "speech_pattern": "話し方の特徴（語気、テンポ、距離感、比喩の多さなど）を説明（100〜200文字）",
  "episodes": [
    {{
      "title": "象徴的なエピソードの名前",
      "description": "その出来事の内容（100〜200文字）",
      "impact": "その出来事が人物の価値観や行動に与えた影響"
    }},
    ...
  ],
  "anchors": [
    {{
      "belief": "人物が強く信じていること・前提",
      "origin": "その信念が生まれた背景（エピソードや環境）"
    }},
    ...
  ]
}}

【制約】
- 上記のキー名・構造を変更しないでください。
- episodes と anchors は 2〜5 個程度ずつ挙げてください（資料から推測してよい）。
- JSON 以外のテキスト（説明文やコメント）は一切出力しないでください。
"""

    try:
        res = request_llm(
            messages=[
                {"role": "system", "content": "あなたは人物分析に特化…"},
                {"role": "user", "content": prompt},
            ],
            endpoint_type="chat",
            max_tokens=1200,
            temperature=0.4,
        )
    except Exception as e:
        logger.error(f"LLM呼出エラー: {e}")
        return ""

    logger.debug(f"LLM生出力:\n{res[:800]} ...")
    return (res or "").strip()

def extract_background_profile(persona: str, materials: str) -> dict:
    """
    性別・年代・出身地・言語・方言など、
    会話スタイルに必要な構造化情報を LLM に抽出させる。
    """
    #  f-string の文法エスケープのため、三重引用符で囲む, jsonの中括弧もエスケープのため二重に
    prompt = f"""
あなたは人物の文化的・言語的背景を分析する専門家です。
以下は人物に関する資料束です。この情報から、人物の
会話スタイルに直接影響する属性を抽出してください。

必ず以下の JSON のみを出力してください。

```json
{{
  "demographic": {{
    "gender": "男性/女性/不明/その他から選択。資料に確証が無い場合は必ず不明。",
    "age_range": "10代/20代前半/30代/不明 など概算でよい"
  }},
  "language_profile": {{
    "dialect": "方言名（例: 関西弁/東北訛り/標準語）。決められない場合は候補や曖昧表現でよい",
    "speech_style": "口調・文体の傾向（砕けている/配信者口調/荒い/丁寧 など）",
    "sample_phrases": ["よく使いそうな語尾・口癖を3〜6個、なければ空配列"]
  }}
}}
```
資料束:
{materials}
""".strip()

    try:
        raw = request_llm(
            messages=[
                {"role": "system", "content": "出力は JSON のみ。説明不要。"},
                {"role": "user", "content": prompt}
            ],
            endpoint_type="chat",
            max_tokens=800,
            temperature=0.2,
        )
        raw = raw.strip()
        logger.debug(f"[thought_profiler background raw]\n{raw}")
    except Exception as e:
        logger.error(f"背景抽出 LLM失敗: {e}")
        return {
            "demographic": {"gender": "不明", "age_range": "不明"},
            "language_profile": {"dialect": "不明", "speech_style": "", "sample_phrases": []}
        }

    parsed = extract_json_block(raw)
    if not parsed:
        logger.error("背景 JSON 抽出失敗。デフォルト値を返す。")
        return {
            "demographic": {"gender": "不明", "age_range": "不明"},
            "language_profile": {"dialect": "不明", "speech_style": "", "sample_phrases": []}
        }

    return parsed

# ================================================================
# JSON 抽出
# ================================================================
def extract_json_block(text: str):
    if not text:
        logger.error("extract_json_block: 入力 text が空。")
        return None

    import re
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        json_str = m.group(1)
        logger.debug("```json``` ブロック抽出成功")
    else:
        m2 = re.search(r"\{[\s\S]*\}", text)
        if not m2:
            logger.error("JSON ブロックが見つからない。")
            return None
        json_str = m2.group(0)
        logger.debug("裸の { ... } ブロック抽出")

    try:
        parsed = json.loads(json_str)
        logger.info("JSON パース成功")
        return parsed
    except Exception as e:
        logger.error(f"JSON パース失敗: {e}\n-----\n{json_str}\n-----")
        return None



# ================================================================
# プロファイル構築
# ================================================================
def build_profile(persona: str, entries):
    """
    semantic由来の entries から thought プロファイルを構築する。
    LLM がコケても、最低限 summary は必ず埋める。
    """

    logger.info(f"プロファイル生成開始: persona={persona}, entries={len(entries)}")

    # fallback用ユーティリティ
    def make_fallback_profile(summary_source: str) -> dict:
        logger.warning(f"fallback_profile を構築: summary_len={len(summary_source)}")
        return {
            "persona_name": persona,
            "summary": summary_source,
            "background": "",
            "values": [],
            "reasoning_pattern": "",
            "speech_pattern": "",
            "episodes": [],
            "anchors": [],
        }

    # まず資料束テキストを構築
    materials = build_materials_text(persona, entries)
    logger.debug(f"materials_len={len(materials)}")

    if not materials:
        logger.error("materials が空。entries から summary/description をかき集めて fallback を作る。")
        descs = []
        for e in entries:
            if e.get("summary"):
                descs.append(e["summary"])
            elif e.get("description"):
                descs.append(e["description"])
        summary = "\n".join(descs)[:400] if descs else ""
        return make_fallback_profile(summary)

    # LLM へ投げる
    raw = ask_profile_llm(persona, materials)

    if not raw:
        logger.error("LLM からの応答が空 or 取得失敗 → materials から fallback プロファイルを作る。")
        # materials 自体を summaryのソースとして縮める
        return make_fallback_profile(materials[:400])

    # JSON 抽出
    parsed = extract_json_block(raw)
    if not parsed or not isinstance(parsed, dict):
        logger.error("LLM 出力の JSON パースに失敗 → materials から fallback プロファイルを作る。")
        return make_fallback_profile(materials[:400])

    # 正常系: parsed からプロファイル構築
    logger.info("LLM 出力の JSON パース成功。プロファイルを構築する。")

    profile = {
        "persona_name": persona,
        "summary": parsed.get("summary", ""),
        "background": parsed.get("background", ""),
        "values": parsed.get("values", []),
        "reasoning_pattern": parsed.get("reasoning_pattern", ""),
        "speech_pattern": parsed.get("speech_pattern", ""),
        "episodes": parsed.get("episodes", []),
        "anchors": parsed.get("anchors", []),
    }

    background_info = extract_background_profile(persona, materials)
    profile["demographic"] = background_info.get("demographic", {})
    profile["language_profile"] = background_info.get("language_profile", {})

    # 空っぽになってないか最低限チェック
    if not profile["summary"]:
        logger.warning("summary が空だったため、materials から一部を補完する。")
        profile["summary"] = materials[:400]

    return profile



# ================================================================
# 保存
# ================================================================
def save_profile(profile, persona: str, output_path: str | None = None) -> Path:
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
    parser = argparse.ArgumentParser(description="GAR Thought Profiler（semantic専用・再設計版）")
    parser.add_argument("--input", required=True, help="semantic_*.json などを指定")
    parser.add_argument("--persona", required=True, help="人物名")
    parser.add_argument("--output", type=str, help="出力パス（省略時は data/thoughts/thought_<persona>.json）")
    args = parser.parse_args()

    persona = args.persona
    entries = load_entries(args.input)
    profile = build_profile(persona, entries)
    save_profile(profile, persona, args.output)
    print("[Thought Profiler] 完了。")


if __name__ == "__main__":
    main()
