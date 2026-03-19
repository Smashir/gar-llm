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
    - modules/gar-llm/src/garllm/gateway/relay_server.py を直接起動
        python3 relay_server.py --host 0.0.0.0 --port 8081
    - editable install 後は以下のようにモジュール経由で起動可能
        python -m garllm.gateway.relay_server --host 0.0.0.0 --port 8081 --debug

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

from collections import OrderedDict
from fastapi import FastAPI, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse

import garllm
from garllm.utils.env_utils import get_data_path, ensure_data_dirs  # ✅ env_utils統合
from garllm.style_layer.response_modulator import modulate_response
from garllm.utils.logger import get_logger
from garllm.utils.llm_client import request_llm

from garllm.gateway.render_plan_builder import build_render_plan

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

# ============================================================
# Completion Runtime Profile Cache (observer API)
# ============================================================
_PROFILE_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_PROFILE_TTL_SEC = int(os.getenv("GAR_PROFILE_TTL_SEC", "600"))   # 10 min
_PROFILE_MAX_ITEMS = int(os.getenv("GAR_PROFILE_MAX_ITEMS", "2048"))

def _cache_profile(completion_id: str, profile: dict) -> None:
    now = time.time()

    # purge expired (oldest first)
    expire_before = now - _PROFILE_TTL_SEC
    keys_to_delete = []
    for k, (ts, _) in _PROFILE_CACHE.items():
        if ts < expire_before:
            keys_to_delete.append(k)
        else:
            break
    for k in keys_to_delete:
        _PROFILE_CACHE.pop(k, None)

    # upsert
    _PROFILE_CACHE[completion_id] = (now, profile)
    _PROFILE_CACHE.move_to_end(completion_id, last=True)

    # enforce max size
    while len(_PROFILE_CACHE) > _PROFILE_MAX_ITEMS:
        _PROFILE_CACHE.popitem(last=False)

def _get_cached_profile(completion_id: str) -> dict | None:
    item = _PROFILE_CACHE.get(completion_id)
    if not item:
        return None
    ts, profile = item
    if (time.time() - ts) > _PROFILE_TTL_SEC:
        _PROFILE_CACHE.pop(completion_id, None)
        return None
    return profile


# ============================================================
# Completion Render Plan Cache (observer API)
# ============================================================
_RENDER_PLAN_CACHE: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_RENDER_PLAN_TTL_SEC = int(os.getenv("GAR_RENDER_PLAN_TTL_SEC", "600"))   # 10 min
_RENDER_PLAN_MAX_ITEMS = int(os.getenv("GAR_RENDER_PLAN_MAX_ITEMS", "2048"))

def _cache_render_plan(completion_id: str, plan: dict) -> None:
    now = time.time()

    expire_before = now - _RENDER_PLAN_TTL_SEC
    keys_to_delete = []
    for k, (ts, _) in _RENDER_PLAN_CACHE.items():
        if ts < expire_before:
            keys_to_delete.append(k)
        else:
            break
    for k in keys_to_delete:
        _RENDER_PLAN_CACHE.pop(k, None)

    _RENDER_PLAN_CACHE[completion_id] = (now, plan)
    _RENDER_PLAN_CACHE.move_to_end(completion_id, last=True)

    while len(_RENDER_PLAN_CACHE) > _RENDER_PLAN_MAX_ITEMS:
        _RENDER_PLAN_CACHE.popitem(last=False)

def _get_cached_render_plan(completion_id: str) -> dict | None:
    item = _RENDER_PLAN_CACHE.get(completion_id)
    if not item:
        return None
    ts, plan = item
    if (time.time() - ts) > _RENDER_PLAN_TTL_SEC:
        _RENDER_PLAN_CACHE.pop(completion_id, None)
        return None
    return plan

def _load_persona_voice_block(persona_name: str) -> dict:
    """persona_<name>.json から voice ブロックを安全に読み出す。無ければ {}。"""
    p = _persona_path_for(persona_name)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("voice")
        return v if isinstance(v, dict) else {}
    except Exception as e:
        logger.warning(f"[profile] failed to load persona voice: {p}: {e}")
        return {}


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
    return {"relations":{"user":rel_axes},"emotion_axes":emo_axes,"phase_weights":{}}

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


def parse_persona_with_constraint(raw: str):
    """
    'スネーク@メタルギアソリッド'
      -> ('スネーク', 'メタルギアソリッド')

    '@' がなければ拘束条件なし
    """
    if "@" in raw:
        name, constraint = raw.split("@", 1)
        return name.strip(), constraint.strip()
    return raw.strip(), None


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
        logger.error(f"[WARN] Missing script: {script_path}")
        return False


    logger.info(f"Running {script_name} {' '.join(args)}")

    result = subprocess.run(["python3", str(script_path)] + args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Step failed: {script_name}\n{result.stderr}")
        return False
    return True


#def _auto_generate_persona(persona_name: str) -> bool:
def _auto_generate_persona(persona_name: str, constraint: str | None = None) -> bool:

    """retriever → semantic_condenser → thought_profiler → persona_generator を順次起動"""
    logger.info(f"Persona '{persona_name}' not found, auto-generation triggered.")
    base = persona_name
    if constraint:
        base = f"{persona_name} {constraint}"

    steps = [
        # retriever: 生情報取得＋clean_text生成（cleaned_*.json）
        (
            "retriever.py",
            [
                "--queries",
                json.dumps(
                    [
                        base,
                        f"{base} 性別",
                        f"{base} 話し方",
                        f"{base} 口調",
                        f"{base} 方言",
                        f"{base} なまり",
                        f"{base} キャラクター",
                        f"{base} 自己紹介",
                    ],
                    ensure_ascii=False
                ),
                "--output", str(RETRIEVED_DIR / f"retrieved_{persona_name}.json"),
                "--limit", "5",
            ],
        ),
        # semantic_condenser: 人物要約（summary）生成（semantic_*.json）
        (
            "semantic_condenser.py",
            [
                "--input", str(RETRIEVED_DIR / f"retrieved_{persona_name}.json"),
                "--persona", persona_name,
                "--output", str(SEMANTIC_DIR / f"semantic_{persona_name}.json"),
            ],
        ),
        # thought_profiler: episodes / anchors / core_profile 生成
        (
            "thought_profiler.py",
            [
                "--input", str(SEMANTIC_DIR / f"semantic_{persona_name}.json"),
                "--output", str(THOUGHT_DIR / f"thought_{persona_name}.json"),
                "--persona", persona_name,
            ],
        ),
        # persona_generator: 最終 persona_*.json 生成
        (
            "persona_generator.py",
            [
                "--input", str(THOUGHT_DIR / f"thought_{persona_name}.json"),
                "--persona", persona_name,
            ],
        ),
    ]

    for script, args in steps:
        if not _run_step(script, args):
            logger.error(f"Persona generation failed at step: {script}")
            return False

    persona_path = PERSONA_DIR / f"persona_{persona_name}.json"
    if persona_path.exists():
        logger.info(f"Persona successfully generated: {persona_name}")
        return True
    else:
        logger.error(f"Persona file not found after generation: {persona_path}")
        return False



def _ensure_persona_exists(persona_name: str, constraint: str | None = None):
    """ペルソナが存在しない場合、自動生成を行う"""
    persona_path = PERSONA_DIR / f"persona_{persona_name}.json"
    if persona_path.exists():
        return True
    return _auto_generate_persona(persona_name, constraint)



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

    '''
    if proc.stdout:
        # デバッグログとしては残してOK（JSONではないのでパースしない）
        logger.debug(f"[context_controller stdout] {proc.stdout.strip()}")
    '''
    if proc.returncode != 0:
        logger.error(f"[WARN] context_controller non-zero exit: {proc.stderr.strip()}")

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
        logger.error(f"style_modulator failed:\n{result.stderr}")
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
# GAR Observer API: runtime_profile by completion_id
# ============================================================
@app.get("/v1/gar/runtime_profile")
async def get_runtime_profile(
    completion_id: str = Query(..., description="OpenAI互換レスポンスのid (chatcmpl-...)")
):
    prof = _get_cached_profile(completion_id)
    if prof is None:
        return JSONResponse(status_code=404, content={"error": "profile_not_found", "completion_id": completion_id})
    return JSONResponse(content=prof)


@app.get("/v1/gar/render_plan")
async def get_render_plan(
    completion_id: str = Query(..., description="OpenAI互換レスポンスのid (chatcmpl-...)")
):
    plan = _get_cached_render_plan(completion_id)
    if plan is None:
        return JSONResponse(
            status_code=404,
            content={"error": "render_plan_not_found", "completion_id": completion_id}
        )
    return JSONResponse(content=plan)


# ============================================================
# OpenAI API 互換: モデル一覧
# ============================================================
@app.get("/v1/models")
async def list_models():
    """OpenAI API 互換用の /v1/models エンドポイント"""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "gar-llm",      # OpenWebUI 上で表示されるモデルID
                "object": "model",
                "created": now,
                "owned_by": "garllm",
                "permission": [],
            }
        ],
    }

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
            #persona = persona_cmds[-1]["body"].split(";")[0].strip()
            #return persona
            raw = persona_cmds[-1]["body"].split(";")[0].strip()
            persona_name, constraint = parse_persona_with_constraint(raw)
            return persona_name, constraint
    return None

def _normalize_stage_value(raw: str) -> str | None:
    """
    gar.stage の body を正規化して "on" / "off" / "auto" のいずれかにする。
    受け付ける値:
      - on/off/auto
      - true/false
      - 1/0
      - yes/no
      - jp: 有効/無効/自動
    """
    if raw is None:
        return None
    v = raw.strip().lower()

    # 末尾に ;key=val が付いている場合は先頭だけ使う
    v = v.split(";")[0].strip()

    mapping = {
        "on": "on", "enable": "on", "enabled": "on", "true": "on", "1": "on", "yes": "on",
        "off": "off", "disable": "off", "disabled": "off", "false": "off", "0": "off", "no": "off",
        "auto": "auto", "automatic": "auto", "default": "auto",
        "有効": "on", "無効": "off", "自動": "auto",
    }
    return mapping.get(v, None)


def extract_stage_from_messages(messages):
    """
    (gar.stage: on/off/auto) 構文から最後に指定された stage モードを抽出。
    返り値: "on" / "off" / "auto" / None
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        commands = extract_gar_commands(text)
        stage_cmds = [c for c in commands if c["cmd"] == "stage"]
        if stage_cmds:
            raw = stage_cmds[-1]["body"].strip()
            return _normalize_stage_value(raw)
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
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    req = await request.json()

    logger.info("OpenWebUI req keys: %s", sorted(req.keys()))
    # 値は長いので、まずは “パラメータだけ”
    logger.info("OpenWebUI gen-ish params: %s", json.dumps(
        {k: req.get(k) for k in sorted(req.keys()) if k not in ["messages"]},
        ensure_ascii=False
    ))

    # ---- OpenWebUIから来た「生成系パラメータ」を抽出（指定されているキーだけ）----
    # messages/model/stream は生成パラメータではないので除外
    GAR_RESERVED = {"messages", "model", "stream", "intensity", "verbose", "persona", "stage"}    
    gen_params = {k: req.get(k) for k in req.keys() if k not in GAR_RESERVED and req.get(k) is not None}

    messages = req.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages is required"})

    # logger.debug(f"Received /v1/chat/completions request\n{req}")

    cleaned_text = clean_messages(messages)
    last_message = get_last_message(messages)
    intensity = float(req.get("intensity", 0.8))
    verbose = bool(req.get("verbose", False))

    # --- OpenWebUIのフォローアップクエスチョンやタイトルなど内部メタタスクを検知した場合は、LLMへ直接パススルー ---
    internal = _is_internal_prompt(last_message)
    logger.info(f"[internal_check] last_message_head={last_message[:120]!r} internal={internal}")

    if internal:
        # internal task の可観測性を上げる（messages全文は長いので要点だけ）
        try:
            logger.info(f"[internal_task] last_user_len={len(last_message)}")
            # 直近数件だけ（長すぎるログを避ける）
            tail = messages[-6:] if isinstance(messages, list) else []
            logger.debug("[internal_task] messages_tail=\n" + json.dumps(tail, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"[internal_task] log_failed: {e}")

        raw_response = request_llm(
            messages=messages,
            backend="auto",
            temperature=float(req.get("temperature", 0.7)),
            max_tokens=int(req.get("max_tokens", 800)),
            top_p=float(req.get("top_p", 1.0)),
            extra_params=gen_params,
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
    '''
    persona_name = (
        req.get("persona")
        or extract_persona_from_messages(messages)
        or args.persona
        or "default"
    )
    '''
    persona_info = extract_persona_from_messages(messages)

    if isinstance(persona_info, tuple):
        persona_name, persona_constraint = persona_info
    else:
        persona_name = persona_info or req.get("persona") or args.persona or "default"
        persona_constraint = None


    # stage（演出）モードを検出する（gar.stage: on/off/auto）
    # 明示指定がない場合は None（=AUTO相当の既定動作はstyle側で決める）
    stage_mode = extract_stage_from_messages(messages)
    if stage_mode:
        # response_modulator 側で解釈する内部パラメータとして渡す
        gen_params["gar_stage"] = stage_mode


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
            logger.info(f"[HANDSHAKE] Starting persona stabilization handshake for '{persona_name}'")
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
                        relations=None,
                        emotion_axes=None,
                        debug=args.debug,
                        log_console=args.log_console
                    )
                )
                handshake_response = await asyncio.wait_for(task, timeout=HANDSHAKE_TIMEOUT)
                logger.info(f"[HANDSHAKE] Response: {handshake_response[:80]!r}")
                logger.info(f"[HANDSHAKE] Persona '{persona_name}' stabilized.")
            except asyncio.TimeoutError:
                logger.error(f"[HANDSHAKE] Timeout during persona stabilization for '{persona_name}'")
            except Exception as e:
                logger.error(f"[HANDSHAKE] Error during stabilization: {e}")

    # personaが存在しなければ自動生成
    if not _ensure_persona_exists(persona_name, persona_constraint):
        return JSONResponse(
            status_code=500,
            content={"error": f"Persona generation failed for '{persona_name}'"}
        )
    
    # ============================================================
    # 🧠 Context update (ASYNC) — レイテンシ改善のため非同期モードも実装
    # ============================================================
    context_input = json.dumps(messages, ensure_ascii=False)

    # このターンは「前回までの state」を使う（即応優先）
    context_data = _load_state(persona_name)
    relations = context_data.get("relations", {})
    emotion_axes = context_data.get("emotion_axes", {})

    # 次ターン以降の state 更新は、設定に応じて同期/非同期を切り替える
    async_mode = getattr(args, "async_context", "on")

    # ペルソナ切替コマンドが含まれるターンは、直後の1ターン遅れが目立つので同期に強制する
    force_sync = False
    try:
        # last_message はこの関数内で既に使っている（persona_cmds判定に使っている）前提
        cmds = extract_gar_commands(last_message)
        persona_cmds = [c for c in cmds if c.get("cmd") == "persona"]
        if persona_cmds:
            force_sync = True
    except Exception:
        force_sync = False

    if async_mode == "on" and not force_sync:
        try:
            background_tasks.add_task(_run_context_update, persona_name, context_input, "llm", args.debug)
        except Exception as e:
            logger.error(f"[WARN] failed to schedule async context update: {e}")
    else:
        # async-context off または persona 切替ターンは同期更新
        _run_context_update(persona_name, context_input, mode="llm", debug=args.debug)


    # 💬 LLMにリレーするmessages全体を確認
    logger.debug("Messages before response modulation:\n" + json.dumps(messages, ensure_ascii=False, indent=2))

    rewritten = modulate_response(
        text=messages,
        persona_name=persona_name,
        intensity=intensity,
        verbose=verbose,
        emotion_axes=emotion_axes,
        relations=relations,
        debug=args.debug,
        log_console=args.log_console,
        gen_params=gen_params,
    )

    keep_one = (args.prefix_persona == "on") and (persona_name and persona_name != "default")
    rewritten = _normalize_persona_prefix(rewritten, persona_name, keep_one)

    # --- completion_id を先に確定（このIDが参照キーになる） ---
    completion_id = f"chatcmpl-{os.urandom(8).hex()}"

    # --- この応答生成に使った state（= このターンの basis）と persona voice をスナップショット ---
    voice_block = _load_persona_voice_block(persona_name)
    runtime_profile = {
        "completion_id": completion_id,
        "persona": {"id": persona_name},
        "emotion": {"axes": emotion_axes},
        "voice": voice_block,
        "created": int(time.time()),
    }

    response = {
        "id": completion_id,
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

    _cache_profile(completion_id, runtime_profile)

    render_plan = build_render_plan(
        completion_id=completion_id,
        persona_name=persona_name,
        display_text=rewritten,
    )
    _cache_render_plan(completion_id, render_plan)

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
    parser.add_argument("--async-context", choices=["on", "off"], default="on",
                    help="context_controller をバックグラウンドで更新（on=非同期/off=同期）")
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
