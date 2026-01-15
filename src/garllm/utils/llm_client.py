# modules/utils/llm_client.py
# ------------------------------------------------------------
# 汎用 LLM クライアント
# - vLLM / Ollama / OpenAI互換（LM Studio含む）対応
# - backend="auto" にすると自動判別（優先: vLLM → Ollama）
# - persona_assimilator, response_modulator 等から共通呼び出し可
#
# 追加:
# - extra_params: OpenWebUI等から来た任意パラメータを受け取り、
#   backendごとに「通せるものだけ」通す（未知は落とす/ログ）
# - repeat_penalty -> repetition_penalty の安全な正規化
#   （repetition_penaltyが来ていればそれを優先）
# ------------------------------------------------------------
import os
import sys
import json
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, List, Dict, Any, Literal, Tuple

sys.path.append(os.path.expanduser("~/modules/"))
from garllm.utils.env_utils import get_base_url  # vLLM用
from garllm.utils.logger import get_logger

BackendType = Literal["vllm", "ollama", "openai", "auto"]
EndpointType = Literal["chat", "completions", "auto"]

__all__ = ["request_llm"]

logger = get_logger("llm_client", level="INFO", to_console=False)

def _http_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _detect_backend() -> BackendType:
    """稼働中のバックエンドを自動検出。優先順: vLLM → Ollama → OpenAI"""
    # 1. vLLM 確認
    try:
        base = get_base_url()
        with urllib.request.urlopen(base + "/models", timeout=1) as r:
            if r.status == 200:
                return "vllm"
    except Exception:
        pass

    # 2. Ollama デフォルトエンドポイント確認
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            if r.status == 200:
                return "ollama"
    except Exception:
        pass

    # 3. OpenAI 環境変数（例: LM Studio, API proxy）
    if os.getenv("OPENAI_API_BASE"):
        return "openai"

    return "vllm"  # fallback


# ---- 正規化/フィルタ ----

# vLLM(OpenAI互換)にそのまま渡しやすいキー（拡張含む）
_VLLM_ALLOWED = {
    "temperature", "top_p", "max_tokens",
    "frequency_penalty", "presence_penalty",
    "repetition_penalty",
    "stop", "seed", "n", "logit_bias",
    # OpenAI互換の新しめのキーが来ても壊さないために入れておく（未対応ならvLLM側が拒否しうるので必要なら絞って）
    "response_format", "tool_choice", "tools",
}

# OpenAI互換に投げるときに通すキー（安全寄り。repetition_penalty は非標準だがLM Studio/vLLM互換で通ることがある）
_OPENAI_ALLOWED = {
    "temperature", "top_p", "max_tokens",
    "frequency_penalty", "presence_penalty",
    "stop", "seed", "n", "logit_bias",
    "response_format", "tool_choice", "tools",
    "repetition_penalty",
}

# Ollama options の代表キー（ここは「変換」ではなく「通せるものだけ通す」）
_OLLAMA_ALLOWED_OPTIONS = {
    "temperature", "top_p", "top_k",
    "repeat_penalty", "repeat_last_n",
    "presence_penalty", "frequency_penalty",
    "num_ctx", "num_predict",
    "seed",
    "mirostat", "mirostat_eta", "mirostat_tau",
    "tfs_z", "typical_p",
}


def _normalize_repeat_keys(params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    repeat_penalty / repetition_penalty の事故回避。
    - 両方来たら repetition_penalty を優先し repeat_penalty を捨てる
    - repeat_penalty だけなら repetition_penalty に写す（vLLM向け）
    """
    p = dict(params or {})
    notes: List[str] = []

    if "repetition_penalty" in p and "repeat_penalty" in p:
        notes.append("repeat_penalty ignored (repetition_penalty is present)")
        p.pop("repeat_penalty", None)

    # vLLM/OpenAI互換向けに、repeat_penalty を repetition_penalty に寄せたいケース
    if "repeat_penalty" in p and "repetition_penalty" not in p:
        p["repetition_penalty"] = p.pop("repeat_penalty")
        notes.append("repeat_penalty -> repetition_penalty")

    return p, notes


def _drop_none(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if v is not None}


def _filter_allowed(params: Dict[str, Any], allowed: set[str]) -> Tuple[Dict[str, Any], List[str]]:
    p = dict(params or {})
    dropped = []
    for k in list(p.keys()):
        if k not in allowed:
            dropped.append(k)
            p.pop(k, None)
    return p, dropped


def request_llm(
    *,
    backend: BackendType = "auto",
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    endpoint_type: EndpointType = "auto",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    任意の LLM バックエンドにリクエストしてテキストを返す。

    - extra_params:
        OpenWebUI等から来た追加パラメータ（指定されているキーだけ入れる前提）
        -> backendごとに allowlist で通す
        -> repeat_penalty/repetition_penalty を安全に正規化
    """

    backend = _detect_backend() if backend == "auto" else backend
    extra_params = _drop_none(extra_params or {})

    # ここで「上位引数（temperature等）」を extra_params にも反映しておく（上書きは extra_params 優先）
    base_overrides = _drop_none({
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    })
    merged = dict(base_overrides)
    merged.update(extra_params)  # extra_params が優先

    # === vLLM ===
    if backend == "vllm":
        base = get_base_url()
        if endpoint_type == "auto":
            endpoint_type = "chat" if messages else "completions"
        url = base + ("/chat/completions" if endpoint_type == "chat" else "/completions")

        # 正規化 + allowlist
        norm, notes = _normalize_repeat_keys(merged)
        norm, dropped = _filter_allowed(_drop_none(norm), _VLLM_ALLOWED)

        if notes:
            logger.info("[vLLM] normalize: %s", notes)
        if dropped:
            logger.info("[vLLM] dropped params: %s", dropped)

        payload: Dict[str, Any] = {
            "model": model or "",
            **norm,
        }

        if endpoint_type == "chat":
            payload["messages"] = messages or [{"role": "user", "content": prompt or ""}]
        else:
            payload["prompt"] = prompt or "\n".join(m.get("content", "") for m in (messages or []))

        # logger.info("[vLLM payload] %s", json.dumps(payload, ensure_ascii=False))

        data = _http_post(url, payload)
        return (
            data["choices"][0]["message"]["content"]
            if endpoint_type == "chat"
            else data["choices"][0]["text"]
        )

    # === Ollama ===
    elif backend == "ollama":
        url = "http://localhost:11434/api/generate"

        # Ollamaは repetition_penalty ではなく repeat_penalty が一般的なので、
        # vLLM向けの repetition_penalty を repeat_penalty に戻す（ただしrepeat_penaltyが既にあればそちら優先）
        p = dict(merged)
        if "repeat_penalty" not in p and "repetition_penalty" in p:
            p["repeat_penalty"] = p.pop("repetition_penalty")

        # max_tokens -> num_predict
        if "num_predict" not in p and "max_tokens" in p:
            p["num_predict"] = p.pop("max_tokens")

        options, dropped = _filter_allowed(_drop_none(p), _OLLAMA_ALLOWED_OPTIONS)
        if dropped:
            logger.info("[Ollama] dropped params: %s", dropped)

        payload = {
            "model": model or "llama3",
            "prompt": prompt or "\n".join(m.get("content", "") for m in (messages or [])),
            "stream": False,
            "options": options,
        }
        data = _http_post(url, payload)
        return data.get("response", "")

    # === OpenAI互換 (LM Studio含む) ===
    elif backend == "openai":
        base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        key = os.getenv("OPENAI_API_KEY")
        url = base + "/chat/completions"

        # 正規化 + allowlist
        norm, notes = _normalize_repeat_keys(merged)
        norm, dropped = _filter_allowed(_drop_none(norm), _OPENAI_ALLOWED)

        if notes:
            logger.info("[OpenAI] normalize: %s", notes)
        if dropped:
            logger.info("[OpenAI] dropped params: %s", dropped)

        payload = {
            "model": model or "gpt-3.5-turbo",
            "messages": messages or [{"role": "user", "content": prompt or ""}],
            **norm,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unsupported backend: {backend}")
