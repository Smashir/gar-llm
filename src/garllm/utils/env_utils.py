# modules/utils/env_utils.py
# ------------------------------------------------------------
# vLLM 実行環境の情報を systemd から動的に取得するユーティリティ
# - .env は見ない（ユーザ指定：最新版は systemctl の情報）
# - 稼働中の vLLM ユニット名、ExecStart、--port を解析
# - /v1/models を叩いて実サーブ中の model id も取得可能
# ------------------------------------------------------------
import os
import re
import json
import subprocess
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional, Tuple

__all__ = [
    "get_active_service", "get_active_model_name", "get_vllm_port",
    "get_base_url", "get_model_path", "get_served_model_id",
]

SERVICE_PREFIX = "vllm@"


def _sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True).strip()


def get_active_service() -> Optional[str]:
    """起動中の vLLM systemd ユニット名（vllm@<name>.service）を 1 件返す。"""
    try:
        out = _sh("systemctl list-units --type=service --state=running | grep -oP 'vllm@[^ ]+\.service' | head -n1")
        return out or None
    except subprocess.CalledProcessError:
        return None


def get_active_model_name() -> Optional[str]:
    svc = get_active_service()
    if not svc:
        return None
    m = re.match(r"vllm@(.+)\.service", svc)
    return m.group(1) if m else None


def _get_execstart_for(service: str) -> Optional[str]:
    try:
        # ExecStart= 行丸ごとを取得
        out = _sh(f"systemctl show -p ExecStart {service}")
        # 例: ExecStart={ path=/bin/bash ; argv[]=/bin/bash -lc '... --port 8000 ...' ; ...}
        # もしくは: ExecStart=/bin/bash -lc '... --port 8000 ...'
        if "ExecStart=" in out:
            return out.split("ExecStart=", 1)[1]
        return out
    except subprocess.CalledProcessError:
        return None


def get_vllm_port(model_name: Optional[str] = None, default: int = 8000) -> int:
    """systemctl の ExecStart から --port <N> を抜き出す。見つからなければ default。"""
    if model_name is None:
        model_name = get_active_model_name()
    if not model_name:
        return default
    service = f"{SERVICE_PREFIX}{model_name}.service"
    line = _get_execstart_for(service) or ""
    # --port 8000 または --port=8000 の両方に対応
    m = re.search(r"--port(?:\s+|=)(\d+)", line)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default


def get_base_url(model_name: Optional[str] = None) -> str:
    port = get_vllm_port(model_name)
    return f"http://localhost:{port}/v1"


def get_served_model_id(model_name: Optional[str] = None) -> Optional[str]:
    """/v1/models から現在サーブされているモデル ID を 1 件返す。"""
    url = get_base_url(model_name) + "/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            arr = data.get("data", [])
            return arr[0].get("id") if arr else None
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def get_model_path(model_name: Optional[str] = None) -> Optional[str]:
    """互換のために残す：~/models/<name> 形式で返す。未起動なら None。"""
    name = model_name or get_active_model_name()
    return f"~/models/{name}" if name else None

def get_data_path(subdir: str = "") -> str:
    """
    GAR のデータルートを返す。
    subdir が与えられた場合、その下のパスを生成する。
    """
    root = os.environ.get("GAR_DATA_ROOT", os.path.expanduser("~/data"))
    path = os.path.join(root, subdir) if subdir else root
    os.makedirs(path, exist_ok=True)
    return path

# ============================================================
# GAR: Data path management (追加)
# ============================================================

from pathlib import Path

# データルート（環境変数 or ~/data）
GAR_DATA_ROOT = Path(os.environ.get("GAR_DATA_ROOT", os.path.expanduser("~/data")))

# サブディレクトリ定義
DATA_SUBDIRS = {
    "retrieved": GAR_DATA_ROOT / "retrieved",
    "cleaned": GAR_DATA_ROOT / "cleaned",
    "condensed": GAR_DATA_ROOT / "condensed",
    "semantic": GAR_DATA_ROOT / "semantic",
    "thoughts": GAR_DATA_ROOT / "thoughts",
    "personas": GAR_DATA_ROOT / "personas",
}

def ensure_data_dirs():
    """GARで使用する全データディレクトリを作成"""
    for p in DATA_SUBDIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def get_data_path(subdir: str = "") -> str:
    """
    既存仕様を拡張:
    GAR のデータルート、または指定サブディレクトリを返す。
    subdir が存在しない場合は自動作成。
    """
    if subdir and subdir in DATA_SUBDIRS:
        path = DATA_SUBDIRS[subdir]
    else:
        path = GAR_DATA_ROOT / subdir if subdir else GAR_DATA_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
