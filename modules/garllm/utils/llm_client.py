# modules/utils/llm_client.py
# ------------------------------------------------------------
# 汎用 LLM クライアント
# - vLLM / Ollama / OpenAI互換（LM Studio含む）対応
# - backend="auto" にすると自動判別（優先: vLLM → Ollama）
# - persona_assimilator, style_modulator 等から共通呼び出し可
# ------------------------------------------------------------
import os
import sys
import json
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, List, Dict, Any, Literal

sys.path.append(os.path.expanduser("~/modules/"))
from garllm.utils.env_utils import get_base_url  # vLLM用

BackendType = Literal["vllm", "ollama", "openai", "auto"]
EndpointType = Literal["chat", "completions", "auto"]

__all__ = ["request_llm"]

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
) -> str:
    """
    任意の LLM バックエンドにリクエストしてテキストを返す。
    - backend="auto" で自動判別
    - endpoint_type="auto" で messages/prompt に応じて切替
    """

    backend = _detect_backend() if backend == "auto" else backend

    # === vLLM ===
    if backend == "vllm":
        base = get_base_url()
        if endpoint_type == "auto":
            endpoint_type = "chat" if messages else "completions"
        url = base + ("/chat/completions" if endpoint_type == "chat" else "/completions")
        payload = {
            "model": model or "",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if endpoint_type == "chat":
            payload["messages"] = messages or [{"role": "user", "content": prompt or ""}]
        else:
            payload["prompt"] = prompt or "\n".join(m.get("content", "") for m in (messages or []))
        data = _http_post(url, payload)
        return (
            data["choices"][0]["message"]["content"]
            if endpoint_type == "chat"
            else data["choices"][0]["text"]
        )

    # === Ollama ===
    elif backend == "ollama":
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model or "llama3",
            "prompt": prompt or "\n".join(m.get("content", "") for m in (messages or [])),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = _http_post(url, payload)
        return data.get("response", "")

    # === OpenAI互換 (LM Studio含む) ===
    elif backend == "openai":
        base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        key = os.getenv("OPENAI_API_KEY", "sk-local")
        url = base + "/chat/completions"
        payload = {
            "model": model or "gpt-3.5-turbo",
            "messages": messages or [{"role": "user", "content": prompt or ""}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
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
