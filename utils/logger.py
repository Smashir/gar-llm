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
    
    module_name: ログ出力モジュール名 (例: "relay_server", "context_controller")
    level: ログレベル ("DEBUG" / "INFO" / "WARNING" / "ERROR")
    to_console: Trueならコンソールにも出力（デバッグ用）
    """

    # ログ出力ディレクトリ作成
    log_dir = os.path.join(LOG_ROOT, module_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{module_name}.log")

    # すでにロガーが存在する場合は再利用
    logger = logging.getLogger(module_name)
    if logger.handlers:
        return logger

    # ロガー基本設定
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ファイル出力ハンドラ

    # ログローテーションをする場合
    #from logging.handlers import RotatingFileHandler
    #file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- コンソール出力（オプション） ---
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # --- エラーだけ常時stderr出力 ---
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
