"""Minimal render plan builder for GAR relay_server.

This file does NOT store anything.
It only builds a small per-completion render plan dict.

Current scope:
- keep existing OpenAI-compatible response untouched
- keep existing runtime_profile cache untouched
- build side-channel data only for TTS/UI split
"""

from __future__ import annotations

import re
from typing import Any


def _strip_exact_speaker_prefix(text: str, speaker: str | None) -> str:
    if not text:
        return ""
    value = text.strip()
    if speaker:
        exact = re.compile(rf'^\s*{re.escape(speaker)}\s*[:：]\s*')
        value = exact.sub("", value, count=1).strip()
    return value


def build_render_plan(
    *,
    completion_id: str,
    persona_name: str,
    display_text: str,
) -> dict[str, Any]:
    speech_text = _strip_exact_speaker_prefix(display_text, persona_name)

    return {
        "version": "0.2",
        "completion_id": completion_id,
        "default_speaker": persona_name,
        "speech_text": speech_text,
        "display_text": display_text,
        "segments": [
            {
                "kind": "speech",
                "text": speech_text,
                "speaker": persona_name,
                "audible": True,
                "visible": True,
                "renderer": "persona_tts",
            }
        ],
    }
