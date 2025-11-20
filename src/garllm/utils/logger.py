import os
import sys
import logging
from datetime import datetime

# === 共通ログ設定 ===
LOG_ROOT = os.path.expanduser("~/logs")
os.makedirs(LOG_ROOT, exist_ok=True)

def get_logger(module_name: str, level: str = "INFO", to_console: bool = False):
    """
    GAR 全体で共通のロガーを取得。

    - 同じ module_name は同じ logger を共有する
    - level は呼び出し毎に上書きされる
    - FileHandler / stderr 用 StreamHandler は重複作成しない
    - to_console=True のとき stdout 用ハンドラを追加、
      False のとき stdout 用ハンドラを削除する
    """
    # ログ出力ディレクトリ作成
    log_dir = os.path.join(LOG_ROOT, module_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{module_name}.log")

    logger = logging.getLogger(module_name)

    # レベルは毎回更新
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- FileHandler（モジュールごとに 1 つ） ---
    has_file = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(log_file)
        for h in logger.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # --- stderr 用エラーハンドラ（1 つ）---
    has_stderr = any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        for h in logger.handlers
    )
    if not has_stderr:
        error_handler = logging.StreamHandler(sys.stderr)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    # --- stdout コンソールハンドラ（オンオフ可能）---
    stdout_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
    ]

    if to_console:
        # なければ追加
        if not stdout_handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    else:
        # あれば削除
        for h in stdout_handlers:
            logger.removeHandler(h)

    return logger

