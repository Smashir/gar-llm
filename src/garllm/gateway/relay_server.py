#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
relay_server.py — GAR-LLM Gateway Relay Server (env_utils統合版)
-----------------------------------------------------------
OpenAI API 互換のエンドポイントを提供し、
style_layer / context_layer / persona_layer と連携して
会話スタイルとペルソナを自動制御します。

特徴:
- OpenAI 互換 /v1/chat/completions エンドポイント対応
- ペルソナが存在しない場合、自動生成をトリガー
- context_controller と style_modulator を統合呼び出し
- env_utils.py によりデータパスを一元管理

起動例:
    python3 relay_server.py --host 0.0.0.0 --port 8081

API利用例:
    curl -X POST http://localhost:8081/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gar-llm",
        "messages": [
          {"role": "system", "content": "あなたは織田信長です。"},
          {"role": "user", "content": "よくもやってくれたな"}
        ],
        "persona": "織田信長",
        "intensity": 0.8,
        "verbose": true
      }' | jq .
"""

import re
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import garllm
from garllm.utils.env_utils import get_data_path, ensure_data_dirs  # ✅ env_utils統合
from garllm.style_layer.response_modulator import modulate_response
from garllm.utils.logger import get_logger
from garllm.utils.llm_client import request_llm


# ============================================================
# GAR 環境パス設定
# ============================================================
#GAR_ROOT = Path(os.path.expanduser("~/modules/gar-llm/src"))
#sys.path.append(str(GAR_ROOT))

# garllm モジュールが存在するディレクトリをルートとして使う
GAR_ROOT = Path(garllm.__file__).resolve().parent
# もし src 配下がある場合は自動で1階層上がる
if (GAR_ROOT / "garllm").exists():
    GAR_ROOT = GAR_ROOT / "garllm"

# データディレクトリ初期化
ensure_data_dirs()
PERSONA_DIR = Path(get_data_path("personas"))
THOUGHT_DIR = Path(get_data_path("thoughts"))
SEMANTIC_DIR = Path(get_data_path("semantic"))
RETRIEVED_DIR = Path(get_data_path("retrieved"))
CLEANED_DIR = Path(get_data_path("cleaned"))
CONDENSED_DIR = Path(get_data_path("condensed"))

# ============================================================
# ロガー設定（初期値はINFO、mainで上書き）
# ============================================================

logger = get_logger("relay_server", level="INFO", to_console=False)

# ============================================================
# FastAPI 設定
# ============================================================
app = FastAPI(title="GAR-LLM Relay Server", version="1.2.0")

# ============================================================
# 補助関数群
# ============================================================

def _is_internal_prompt(message_text: str) -> bool:
    patterns = ["### Task:", "### Chat History:", "### Output:", "### Guidelines:"]
    return any(p in message_text for p in patterns)

def _state_path_for(persona_name: str) -> str:
    """~/data/personas/state_<persona>.json を返す"""
    base = Path(get_data_path("personas"))
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"state_{persona_name}.json")

def _persona_path_for(persona_name: str) -> str:
    """~/data/personas/persona_<persona>.json を返す（必要なら使用）"""
    base = Path(get_data_path("personas"))
    return str(base / f"persona_{persona_name}.json")

def _load_state(persona_name: str) -> dict:
    """存在しなければ最小初期値を返す（context_controllerの初期形に合わせる）"""
    p = _state_path_for(persona_name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    # からの初期（relationsはユーザのみで0埋め、emotion_axesは8軸0）
    rel_axes = {k: 0.0 for k in ["Trust","Familiarity","Hostility","Dominance","Empathy","Instrumentality"]}
    emo_axes = {k: 0.0 for k in ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]}
    return {"relations":{"ユーザ":rel_axes},"emotion_axes":emo_axes,"phase_weights":{}}

def _save_state(persona_name: str, state: dict) -> None:
    p = _state_path_for(persona_name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _extract_user_axes(relations: dict | None) -> dict | None:
    """
    relations から『ユーザ⇄persona』の軸だけを取り出して返す。
    見つからなければ None。
    """
    if not isinstance(relations, dict):
        return None
    # よく使うキーの表記ゆれを吸収
    for k in ("ユーザ", "ユーザー", "user", "User"):
        axes = relations.get(k)
        if isinstance(axes, dict):
            return axes
    return None

# ユーティリティ: 先頭の {name}: / {name}： を全部はがして、必要なら1回だけ付ける
def _normalize_persona_prefix(text: str, persona_name: str, keep_one: bool) -> str:
    if not text:
        return text
    # ^(織田信長\s*[:：]\s*)+ を全削除
    pattern = re.compile(rf'^(?:{re.escape(persona_name)}\s*[:：]\s*)+', re.UNICODE)
    cleaned = pattern.sub('', text.strip())
    return f"{persona_name}: {cleaned}" if keep_one else cleaned


def _run_step(script_name: str, args: list[str]):
    """各スクリプトを順に起動"""

    if script_name == "persona_generator.py":
        script_path = GAR_ROOT / "persona_layer" / script_name
    else:
        script_path = GAR_ROOT / "context_layer" / script_name
        
    if not script_path.exists():
        print(f"[WARN] Missing script: {script_path}")
        return False


    logger.info(f"Running {script_name} {' '.join(args)}", file=sys.stderr)

    result = subprocess.run(["python3", str(script_path)] + args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed: {script_name}\n{result.stderr}")
        return False
    return True


def _auto_generate_persona(persona_name: str) -> bool:
    """retriever → cleaner → condenser → semantic_condenser → thought_profiler → persona_generator を順次起動"""
    logger.info(f"Persona '{persona_name}' not found, auto-generation triggered.", file=sys.stderr)

    steps = [
        ("retriever.py", ["--query", persona_name, "--output", str(RETRIEVED_DIR / f"retrieved_{persona_name}.json")]),
        ("cleaner.py", ["--input", str(RETRIEVED_DIR / f"retrieved_{persona_name}.json"),
                        "--output", str(CLEANED_DIR / f"cleaned_{persona_name}.json")]),
        ("condenser.py", ["--input", str(CLEANED_DIR / f"cleaned_{persona_name}.json"),
                          "--output", str(CONDENSED_DIR / f"condensed_{persona_name}.json")]),
        ("semantic_condenser.py", ["--input", str(CONDENSED_DIR / f"condensed_{persona_name}.json"),
                                   "--output", str(SEMANTIC_DIR / f"semantic_{persona_name}.json")]),
        ("thought_profiler.py", ["--input", str(SEMANTIC_DIR / f"semantic_{persona_name}.json"),
                                 "--output", str(THOUGHT_DIR / f"thought_{persona_name}.json"),
                                 "--persona", persona_name]),
        ("persona_generator.py", ["--input", str(THOUGHT_DIR / f"thought_{persona_name}.json"),
                                    "--persona", persona_name])
    ]

    for script, args in steps:
        if not _run_step(script, args):
            logger.error(f"Persona generation failed at step: {script}", file=sys.stderr)
            return False

    persona_path = PERSONA_DIR / f"persona_{persona_name}.json"
    if persona_path.exists():
        logger.info(f"Persona successfully generated: {persona_name}", file=sys.stderr)
        return True
    else:
        logger.error(f"Persona file not found after generation: {persona_path}", file=sys.stderr)
        return False


def _ensure_persona_exists(persona_name: str):
    """ペルソナが存在しない場合、自動生成を行う"""
    persona_path = PERSONA_DIR / f"persona_{persona_name}.json"
    if persona_path.exists():
        return True
    return _auto_generate_persona(persona_name)


def _run_context_update(persona_name: str, user_text: str, mode: str = "llm", debug: bool = False):
    """context_controllerに状態更新だけをやらせる（stdoutは無視）。
       結果はstateファイルを読み直して使う。
    """
    state_file = _state_path_for(persona_name)
    script_path = GAR_ROOT / "style_layer" / "context_controller.py"
    cmd = [
        "python3", os.path.expanduser(script_path),
        "--persona", persona_name,
        "--input_text", user_text,
        "--mode", mode,
        "--state_file", state_file,
    ]
    if debug:
        cmd.append("--debug")

    # emit_text はサーバー運用では絶対に付けない（stdoutが混ざる）
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if debug and proc.stdout:
        # デバッグログとしては残してOK（JSONではないのでパースしない）
        print("[context_controller stdout]", proc.stdout.strip())
    if proc.returncode != 0:
        print("[WARN] context_controller non-zero exit:", proc.stderr.strip())
    return _load_state(persona_name)
    

def _run_style_modulator(persona_name: str, text: str, intensity: float, verbose: bool,
                         relation_axes=None, emotion_axes=None):
    """style_modulatorを呼び出して最終出力を生成"""
    style_script = GAR_ROOT / "style_layer" / "style_modulator.py"
    args = [
        "python3", str(style_script),
        "--persona", persona_name,
        "--text", text,
        "--intensity", str(intensity)
    ]
    if verbose:
        args.append("--verbose")
    if relation_axes:
        args += ["--relation_axes", json.dumps(relation_axes)]
    if emotion_axes:
        args += ["--emotion_axes", json.dumps(emotion_axes)]

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"style_modulator failed:\n{result.stderr}", file=sys.stderr)
        return text

    if "==== Rewritten Text ====" in result.stdout:
        return result.stdout.split("==== Rewritten Text ====")[-1].strip().split("===")[0].strip()
    return result.stdout.strip()

# ============================================================
# FastAPI エンドポイント
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

# ============================================================
# GAR Command Parser (新仕様対応)
# ============================================================

GAR_CMD_RE = re.compile(
    r"[\(\{\[]\s*gar\.(?P<cmd>[a-zA-Z0-9_]+)\s*:(?P<body>[^)\}\]]+)[\)\}\]]"
)

def extract_gar_commands(text: str):
    """文中から gar コマンドをすべて抽出"""
    matches = GAR_CMD_RE.finditer(text or "")
    commands = []
    for m in matches:
        cmd = m.group("cmd").strip()
        body = m.group("body").strip()
        commands.append({"cmd": cmd, "body": body})
    return commands

def strip_gar_commands(text: str) -> str:
    """garコマンドを本文から除去しつつ、persona名は残す"""
    def replacer(match):
        cmd = match.group("cmd").strip()
        body = match.group("body").strip().split(";")[0]
        # personaコマンドの場合は名前を残す
        if cmd == "persona":
            return body
        # それ以外は完全除去
        return ""
    return GAR_CMD_RE.sub(replacer, text or "").strip()

def clean_messages(messages):
    # コマンド構文を削除し、全履歴をまとめたテキストを cleaned_text に格納
    joined_messages = []
    for m in messages:
        role = m.get("role", "").upper()
        content = strip_gar_commands(m.get("content", ""))
        joined_messages.append(f"{role}: {content}")
    cleaned_text = "\n".join(joined_messages)
    return cleaned_text


def extract_persona_from_messages(messages):
    """(gar.persona: …) 構文から最後に指定されたペルソナ名を抽出"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        commands = extract_gar_commands(text)
        persona_cmds = [c for c in commands if c["cmd"] == "persona"]
        if persona_cmds:
            persona = persona_cmds[-1]["body"].split(";")[0].strip()
            return persona
    return None


def inject_system_message(messages: list[dict], content: str):
    """
    chat履歴に system メッセージを正しい形式で挿入する。
    通常は最後のユーザーメッセージの直後に追加される。
    """
    # 挿入位置：最後の user の直後
    insert_index = len(messages)
    for i in reversed(range(len(messages))):
        if messages[i].get("role") == "user":
            insert_index = i + 1
            break

    system_entry = {"role": "system", "content": content}
    messages.insert(insert_index, system_entry)
    return messages
    

def get_last_message(messages):
    """メッセージ履歴から最後のユーザーメッセージ本文を取得"""
    last_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    text = last_message.get("content", "") if last_message else ""
    return text


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req = await request.json()
    messages = req.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages is required"})

    # logger.debug(f"Received /v1/chat/completions request\n{req}")

    cleaned_text = clean_messages(messages)
    last_message = get_last_message(messages)
    intensity = float(req.get("intensity", 0.8))
    verbose = bool(req.get("verbose", False))

    # --- OpenWebUIのフォローアップクエスチョンやタイトルなど内部メタタスクを検知した場合は、LLMへ直接パススルー ---
    if _is_internal_prompt(last_message):
        #logger.debug("Internal meta task detected — skipping response_modulator and passing through.")
        # LLMへ直接パススルー
        raw_response = request_llm(
            messages=messages,  # オリジナルのまま
            backend="auto",
            temperature=0.7,
            max_tokens=800,
        )
        return JSONResponse(
            content={
                "id": f"chatcmpl-{os.urandom(8).hex()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gar-llm",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": raw_response},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
            }
        )


    # persona 指定を検出する
    persona_name = (
        req.get("persona")
        or extract_persona_from_messages(messages)
        or args.persona
        or "default"
    )

    # gar.persona が新たに指定されていた場合のみ切り替え通知
    commands = extract_gar_commands(last_message)
    persona_cmds = [c for c in commands if c["cmd"] == "persona"]

    if persona_cmds and args.inject_system == "on":
        already_injected = any(
            m.get("role") == "system" and persona_name in m.get("content", "")
            for m in messages
        )

        if not already_injected:
            switch_text = f"assistantはここから {persona_name} の人格として応答しています。"
            logger.info(f"Persona switch -> '{persona_name}' (history preserved)")
            # LLMが履歴を読み直した際に、GARからの指示が通るようSystem役のメッセージを差し込む
            switch_text = f"assistantはここから {persona_name} の人格として応答しています。"
            messages = inject_system_message(messages, switch_text)
       
        # ================================================================
        # 🧠 Persona Stabilization Handshake Patch
        # ================================================================
        import asyncio

        ENABLE_PERSONA_HANDSHAKE = os.getenv("GAR_PERSONA_HANDSHAKE", "false").lower() == "true"
        HANDSHAKE_TIMEOUT = int(os.getenv("GAR_PERSONA_HANDSHAKE_TIMEOUT", "10"))

        if ENABLE_PERSONA_HANDSHAKE:
            print(f"[HANDSHAKE] Starting persona stabilization handshake for '{persona_name}'", file=sys.stderr)
            handshake_messages = messages.copy()
            if handshake_messages and handshake_messages[-1].get("role") == "system":
                handshake_messages.append({
                    "role": "user",
                    "content": "了解しましたか？ あなたの名前を答えてください。"
                })

            stabilization_sequence = [{
                "role": "system",
                "content": (
                    f"あなたは今から {persona_name} の人格に切り替わりました。"
                    f"{persona_name} の記憶・話法・語彙・価値観・歴史的背景をすべて再構成し、"
                    f"人格が安定したら自分の名前を名乗ってください。"
                )
            }]
            handshake_messages.extend(stabilization_sequence)

            loop = asyncio.get_running_loop()
            try:
                task = loop.run_in_executor(
                    None,
                    lambda: modulate_response(
                        text=handshake_messages,
                        persona_name=persona_name,
                        intensity=float(intensity),
                        verbose=False,
                        debug=False,
                        relations=None,
                        emotion_axes=None,
                    )
                )
                handshake_response = await asyncio.wait_for(task, timeout=HANDSHAKE_TIMEOUT)
                print(f"[HANDSHAKE] Response: {handshake_response[:80]!r}", file=sys.stderr)
                print(f"[HANDSHAKE] Persona '{persona_name}' stabilized.", file=sys.stderr)
            except asyncio.TimeoutError:
                print(f"[HANDSHAKE] Timeout during persona stabilization for '{persona_name}'", file=sys.stderr)
            except Exception as e:
                print(f"[HANDSHAKE] Error during stabilization: {e}", file=sys.stderr)

    # personaが存在しなければ自動生成
    if not _ensure_persona_exists(persona_name):
        return JSONResponse(
            status_code=500,
            content={"error": f"Persona generation failed for '{persona_name}'"}
        )

    # 状態更新を実行（context_controllerがstateファイルに結果を書き込む）
    # _run_context_update(persona_name, cleaned_text, mode="llm", debug=args.debug)
    context_input = json.dumps(messages, ensure_ascii=False)
    _run_context_update(persona_name, context_input, mode="llm", debug=args.debug)


    # 更新後のstateを読み出す
    context_data = _load_state(persona_name)
    relations = context_data.get("relations", {})
    emotion_axes = context_data.get("emotion_axes", {})

    # 💬 LLMにリレーするmessages全体を確認
    logger.debug("Messages before response modulation:\n" + json.dumps(messages, ensure_ascii=False, indent=2))

    rewritten = modulate_response(
        text=messages,
        persona_name=persona_name,
        intensity=intensity,
        verbose=verbose,
        debug=args.debug,
        relations=relations,
        emotion_axes=emotion_axes
    )

    keep_one = (args.prefix_persona == "on") and (persona_name and persona_name != "default")
    rewritten = _normalize_persona_prefix(rewritten, persona_name, keep_one)

    response = {
        "id": f"chatcmpl-{os.urandom(8).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gar-llm",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": rewritten},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    }
    return JSONResponse(content=response)

# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="GAR-LLM Relay Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--persona", type=str, default="default", help="(任意) デフォルトペルソナ名。リクエストに persona がない場合に使用。")
    parser.add_argument("--handshake", choices=["on", "off", "auto"],
                        default=os.getenv("GAR_HANDSHAKE", "off"),
                        help="ペルソナ切替時の名乗りハンドシェイク（on/off/auto）")
    parser.add_argument("--inject-system", choices=["on", "off"], default=os.getenv("GAR_INJECT_SYSTEM", "on"))
    parser.add_argument("--prefix-persona", choices=["on", "off"], default=os.getenv("GAR_PREFIX_PERSONA", "on"))
    parser.add_argument("--debug", action="store_true", help="デバッグ出力を有効化（--log-console 併用可）")
    parser.add_argument("--log-console", action="store_true", help="ログをコンソールにも出力")

    args = parser.parse_args()

    # ============================================================
    # ログレベル制御（--debug オプションを唯一のトリガに）
    # ============================================================
    log_level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("relay_server", level=log_level, to_console=args.log_console)

    logger.info(f"Starting Ghost Assimilation Relay Server on {args.host}:{args.port} (log_level={log_level})")

    uvicorn.run(app, host=args.host, port=args.port)
