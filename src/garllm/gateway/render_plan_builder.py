"""Render plan builder with typed visible/audible segments.

Design goals:
- keep existing OpenAI-compatible chat response untouched
- keep runtime_profile cache untouched
- enrich only the side-channel render_plan
- registered non-speech tags become structured segments
- preserve backward compatibility for current voice-bridge integration

Notes:
- `display_text` at the top level remains the original response text
- `visible_text` is the tag-stripped text suitable for UI rendering
- `spoken_text` is the speech-only text suitable for TTS rendering
- segment discriminator uses `type` (and keeps legacy `kind` for compatibility)
"""

from __future__ import annotations

import re
from typing import Any


_TAG_HEAD_RE = re.compile(r"^\s*【([^】]+)】\s*(.*)$")

_NON_SPEECH_TAG_TYPES: dict[str, str] = {
    "情景": "scene",
    "所作": "action",
    "物理音": "foley",
    "環境音": "ambience",
    "BGM": "music",
}

_SPEECH_HINT_TAGS: set[str] = {
    "セリフ",
    "台詞",
    "発話",
}


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


def _normalize_cue_text(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^[\-\*\u2022・●]+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _split_physical_cues(physical_text: str) -> list[str]:
    if not physical_text:
        return []

    cues: list[str] = []
    seen: set[str] = set()

    # Prefer explicit onomatopoeia inside parentheses.
    paren_hits = re.findall(r"[（(]([^()（）]+)[)）]", physical_text)
    for hit in paren_hits:
        cue = _normalize_cue_text(hit)
        if cue and cue not in seen:
            seen.add(cue)
            cues.append(cue)

    chunks = re.split(r"[、,，/／|]|(?:\s+と\s+)", physical_text)
    for raw in chunks:
        cue = _normalize_cue_text(raw)
        if not cue:
            continue
        if len(cue) > 48 and not re.search(r"[ァ-ヴーぁ-んA-Za-z0-9]", cue):
            continue
        if cue not in seen:
            seen.add(cue)
            cues.append(cue)

    return cues


def _split_loose_cues(text: str) -> list[str]:
    if not text:
        return []
    cues: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[\n、,，/／|]", text):
        cue = _normalize_cue_text(raw)
        if cue and cue not in seen:
            seen.add(cue)
            cues.append(cue)
    return cues


def _parse_blocks(text: str) -> list[dict[str, Any]]:
    """
    Parse response text into ordered blocks.

    Rules:
    - only registered non-speech tags become special blocks
    - speech-hint tags are stripped and their contents are treated as speech
    - unknown tags remain part of ordinary speech text
    """
    if not text:
        return []

    blocks: list[dict[str, Any]] = []
    speech_lines: list[str] = []
    current_special: dict[str, Any] | None = None

    def flush_speech() -> None:
        nonlocal speech_lines
        collapsed = _collapse_lines(speech_lines)
        if collapsed:
            blocks.append({"type": "speech", "lines": [collapsed]})
        speech_lines = []

    def flush_special() -> None:
        nonlocal current_special
        if current_special is None:
            return
        collapsed = _collapse_lines(current_special.get("lines", []))
        if collapsed:
            blocks.append(
                {
                    "type": current_special["type"],
                    "tag": current_special["tag"],
                    "lines": [collapsed],
                }
            )
        current_special = None

    for raw in text.splitlines():
        stripped = raw.strip()
        m = _TAG_HEAD_RE.match(stripped)

        if m:
            tag = (m.group(1) or "").strip()
            tail = (m.group(2) or "").strip()

            # any new tag ends the previous special block
            if current_special is not None:
                flush_special()

            if tag in _NON_SPEECH_TAG_TYPES:
                flush_speech()
                current_special = {
                    "type": _NON_SPEECH_TAG_TYPES[tag],
                    "tag": tag,
                    "lines": [],
                }
                if tail:
                    current_special["lines"].append(tail)
                continue

            if tag in _SPEECH_HINT_TAGS:
                if tail:
                    speech_lines.append(tail)
                continue

            # Unknown tag -> leave untouched as speech text.
            speech_lines.append(raw)
            continue

        if current_special is not None:
            if stripped:
                current_special["lines"].append(raw)
            else:
                flush_special()
            continue

        speech_lines.append(raw)

    flush_special()
    flush_speech()
    return blocks


def _make_speech_segment(text: str, speaker: str) -> dict[str, Any]:
    return {
        "type": "speech",
        "kind": "speech",
        "display_text": text,
        "spoken_text": text,
        "visible": True,
        "audible": True,
        "speaker": speaker,
        "renderer": "persona_tts",
    }


def _make_visible_only_segment(seg_type: str, text: str) -> dict[str, Any]:
    return {
        "type": seg_type,
        "kind": seg_type,
        "display_text": text,
        "spoken_text": None,
        "visible": True,
        "audible": False,
        "renderer": None,
    }


def _make_audio_segment(seg_type: str, cue: str, prompt: str, renderer: str, placement: str, level_db: float) -> dict[str, Any]:
    return {
        "type": seg_type,
        "kind": seg_type,
        "cue": cue,
        "prompt": prompt,
        "display_text": None,
        "spoken_text": None,
        "visible": False,
        "audible": True,
        "renderer": renderer,
        "placement": placement,
        "level_db": level_db,
    }


def build_render_plan(
    *,
    completion_id: str,
    persona_name: str,
    display_text: str,
) -> dict[str, Any]:
    raw_text = _strip_exact_speaker_prefix(display_text, persona_name)
    parsed_blocks = _parse_blocks(raw_text)

    segments: list[dict[str, Any]] = []

    stage: dict[str, str] = {
        "scene": "",
        "action": "",
        "physical": "",
        "ambience": "",
        "music": "",
    }

    for block in parsed_blocks:
        block_type = str(block.get("type") or "")
        block_text = _collapse_lines(list(block.get("lines") or []))
        if not block_text:
            continue

        if block_type == "speech":
            segments.append(_make_speech_segment(block_text, persona_name))
            continue

        if block_type == "scene":
            stage["scene"] = block_text if not stage["scene"] else stage["scene"] + "\n" + block_text
            segments.append(_make_visible_only_segment("scene", block_text))
            continue

        if block_type == "action":
            stage["action"] = block_text if not stage["action"] else stage["action"] + "\n" + block_text
            segments.append(_make_visible_only_segment("action", block_text))
            continue

        if block_type == "foley":
            stage["physical"] = block_text if not stage["physical"] else stage["physical"] + "\n" + block_text
            cues = _split_physical_cues(block_text) or [block_text]
            for cue in cues:
                segments.append(
                    _make_audio_segment(
                        "foley",
                        cue=cue,
                        prompt=cue,
                        renderer="sfx_catalog",
                        placement="lead_in",
                        level_db=-10.0,
                    )
                )
            continue

        if block_type == "ambience":
            stage["ambience"] = block_text if not stage["ambience"] else stage["ambience"] + "\n" + block_text
            segments.append(
                _make_audio_segment(
                    "ambience",
                    cue=block_text,
                    prompt=block_text,
                    renderer="sfx_or_model",
                    placement="underlay",
                    level_db=-24.0,
                )
            )
            continue

        if block_type == "music":
            stage["music"] = block_text if not stage["music"] else stage["music"] + "\n" + block_text
            segments.append(
                _make_audio_segment(
                    "music",
                    cue=block_text,
                    prompt=block_text,
                    renderer="music_or_model",
                    placement="underlay",
                    level_db=-28.0,
                )
            )
            continue

    visible_parts = [
        str(seg.get("display_text") or "").strip()
        for seg in segments
        if bool(seg.get("visible", False)) and str(seg.get("display_text") or "").strip()
    ]
    visible_text = "\n".join(visible_parts).strip()

    spoken_parts = [
        str(seg.get("spoken_text") or "").strip()
        for seg in segments
        if str(seg.get("type") or seg.get("kind") or "").lower() == "speech"
        and bool(seg.get("audible", False))
        and str(seg.get("spoken_text") or "").strip()
    ]
    spoken_text = "\n".join(spoken_parts).strip()

    return {
        "version": "1.0",
        "completion_id": completion_id,
        "default_speaker": persona_name,
        "display_text": display_text,
        "visible_text": visible_text,
        "spoken_text": spoken_text,
        # backward compatibility
        "speech_text": spoken_text,
        "stage": stage,
        "segments": segments,
    }
