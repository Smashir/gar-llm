"""Render plan builder with stage-aware segment extraction.

Design goals:
- keep existing OpenAI-compatible chat response untouched
- keep runtime_profile cache untouched
- enrich only the side-channel render_plan
- preserve backward compatibility for direct gar-llm -> WebUI usage

Current behavior:
- display_text: original rewritten text 그대로保持
- speech_text: stage blocks removed text for TTS
- segments:
    * ambience (optional, inferred from stage text)
    * foley    (optional, parsed from 【物理音】)
    * speech   (main spoken text)
"""

from __future__ import annotations

import re
from typing import Any

_STAGE_HEAD_RE = re.compile(r"^\s*【(情景|所作|物理音)】\s*(.*)$")


def _strip_exact_speaker_prefix(text: str, speaker: str | None) -> str:
    if not text:
        return ""
    value = text.strip()
    if speaker:
        exact = re.compile(rf"^\s*{re.escape(speaker)}\s*[:：]\s*")
        value = exact.sub("", value, count=1).strip()
    return value


def _collapse_lines(lines: list[str]) -> str:
    out: list[str] = []
    prev_blank = True

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if not prev_blank:
                out.append("")
            prev_blank = True
            continue
        out.append(line)
        prev_blank = False

    return "\n".join(out).strip()


def _extract_stage_blocks(text: str) -> tuple[str, dict[str, str]]:
    """
    Parse simple stage blocks:

    【情景】
    ...
    【所作】
    ...
    【物理音】
    ...

    Heuristic:
    - a blank line ends the current block
    - lines outside blocks are treated as spoken body
    """
    if not text:
        return "", {}

    body_lines: list[str] = []
    sections: dict[str, list[str]] = {
        "情景": [],
        "所作": [],
        "物理音": [],
    }
    current: str | None = None

    for raw in text.splitlines():
        stripped = raw.strip()

        if current is not None and not stripped:
            current = None
            continue

        m = _STAGE_HEAD_RE.match(stripped)
        if m:
            current = m.group(1)
            tail = (m.group(2) or "").strip()
            if tail:
                sections[current].append(tail)
            continue

        if current is None:
            body_lines.append(raw)
        else:
            sections[current].append(raw)

    body_text = _collapse_lines(body_lines)
    cleaned_sections = {
        name: _collapse_lines(lines)
        for name, lines in sections.items()
        if _collapse_lines(lines)
    }
    return body_text, cleaned_sections


def _normalize_cue_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^[\-\*\u2022・●]+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _split_physical_cues(physical_text: str) -> list[str]:
    """
    Split 【物理音】 block into simple cue strings.
    Keeps ellipsis / repeated onomatopoeia as one cue when possible.
    """
    if not physical_text:
        return []

    cues: list[str] = []
    seen: set[str] = set()

    for raw_line in physical_text.splitlines():
        line = _normalize_cue_text(raw_line)
        if not line:
            continue

        parts = re.split(r"[、,，/／|]", line)
        for part in parts:
            cue = _normalize_cue_text(part)
            if not cue:
                continue
            if cue not in seen:
                seen.add(cue)
                cues.append(cue)

    return cues


_AMBIENCE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("rain", ["雨", "雨音", "土砂降り", "霧雨", "しとしと", "ざぁ", "ざー", "ザー"]),
    ("wind", ["風", "風音", "突風", "木枯らし", "びゅう", "びゅー", "ヒュウ"]),
    ("thunder", ["雷", "稲妻", "ごろごろ", "ゴロゴロ"]),
    ("waves", ["波", "潮騒", "海鳴り", "波音", "海辺"]),
    ("fire", ["焚き火", "暖炉", "炎", "火の粉", "薪", "炉"]),
    ("forest", ["森", "林", "木々", "虫の音", "林間"]),
    ("crowd", ["雑踏", "群衆", "人混み", "市場", "ざわめき", "喧騒"]),
]


def _infer_ambience_cues(scene_text: str, action_text: str, physical_text: str) -> list[str]:
    merged = "\n".join([scene_text or "", action_text or "", physical_text or ""])
    if not merged.strip():
        return []

    found: list[str] = []
    for cue, keywords in _AMBIENCE_KEYWORDS:
        if any(k in merged for k in keywords):
            found.append(cue)
    return found


def build_render_plan(
    *,
    completion_id: str,
    persona_name: str,
    display_text: str,
) -> dict[str, Any]:
    raw_text = _strip_exact_speaker_prefix(display_text, persona_name)

    body_text, stage = _extract_stage_blocks(raw_text)
    speech_text = body_text.strip() if body_text.strip() else raw_text.strip()

    scene_text = stage.get("情景", "")
    action_text = stage.get("所作", "")
    physical_text = stage.get("物理音", "")

    ambience_cues = _infer_ambience_cues(scene_text, action_text, physical_text)
    physical_cues = _split_physical_cues(physical_text)

    segments: list[dict[str, Any]] = []

    for cue in ambience_cues:
        segments.append(
            {
                "kind": "ambience",
                "cue": cue,
                "text": cue,
                "audible": True,
                "visible": False,
                "renderer": "sfx_catalog",
                "placement": "underlay",
                "level_db": -26.0,
            }
        )

    for cue in physical_cues:
        segments.append(
            {
                "kind": "foley",
                "cue": cue,
                "text": cue,
                "audible": True,
                "visible": False,
                "renderer": "sfx_catalog",
                "placement": "lead_in",
                "level_db": -10.0,
            }
        )

    if speech_text:
        segments.append(
            {
                "kind": "speech",
                "text": speech_text,
                "speaker": persona_name,
                "audible": True,
                "visible": True,
                "renderer": "persona_tts",
            }
        )

    return {
        "version": "0.3",
        "completion_id": completion_id,
        "default_speaker": persona_name,
        "speech_text": speech_text,
        "display_text": display_text,
        "stage": {
            "scene": scene_text,
            "action": action_text,
            "physical": physical_text,
        },
        "segments": segments,
    }
