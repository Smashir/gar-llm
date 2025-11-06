# modules/utils/vllm_client.py
# ------------------------------------------------------------
# OpenAI 互換 vLLM クライアント（chat/completions を自動切替）
# - systemctl からポート取得
# - /v1/models が通ることを前提にベースURLを確定
# - endpoint_type="auto" の場合：messages があれば chat、prompt があれば completions
# ------------------------------------------------------------
import json as _json
from typing import List, Dict, Any, Literal, Optional as _Optional
import urllib.request as _req
from urllib.error import URLError as _URLError, HTTPError as _HTTPError

sys.path.append(os.path.expanduser("~/modules/"))
from garllm.utils.env_utils import get_base_url

EndpointType = Literal["chat", "completions", "auto"]


def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = _json.dumps(payload).encode("utf-8")
    req = _req.Request(url, data=body, headers={"Content-Type": "application/json"})
    with _req.urlopen(req, timeout=60) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def request_openai(
    *,
    messages: _Optional[List[Dict[str, str]]] = None,
    prompt: _Optional[str] = None,
    endpoint_type: EndpointType = "auto",
    model_name: _Optional[str] = None,
    # 代表的生成パラメータ（必要に応じて増やす）
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    stop: _Optional[List[str]] = None,
    extra: _Optional[Dict[str, Any]] = None,
) -> str:
    """vLLM OpenAI 互換 API を叩いてテキストを返す。"""
    base = get_base_url(model_name)
    # endpoint 自動決定
    if endpoint_type == "auto":
        endpoint_type = "chat" if messages is not None else "completions"

    if endpoint_type == "chat":
        url = base + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": "",  # vLLM は単一モデルサーブなので空で可（指定してもよい）
            "messages": messages or [{"role": "user", "content": prompt or ""}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        data = _http_post(url, payload)
        return data["choices"][0]["message"]["content"]

    elif endpoint_type == "completions":
        url = base + "/completions"
        payload = {
            "model": "",
            "prompt": prompt if prompt is not None else "\n".join(
                m.get("content", "") for m in (messages or [])
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if stop:
            payload["stop"] = stop
        if extra:
            payload.update(extra)
        data = _http_post(url, payload)
        return data["choices"][0]["text"]

    else:
        raise ValueError(f"Unsupported endpoint_type: {endpoint_type}")
