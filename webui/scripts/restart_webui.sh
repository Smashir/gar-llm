#!/bin/bash
set -e

# ==============================
# Open WebUI 再起動スクリプト（最終版）
# ==============================

# 絶対パスを使ってカレントに依存しない
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"
CONTAINER_NAME="webui"
IMAGE_NAME="ghcr.io/open-webui/open-webui:main"

# ==============================
# IP 検出
# ==============================
WSL_IP=$(ip -o -4 addr show eth0 | awk '{print $4}' | cut -d/ -f1)

# ==============================
# 停止・削除（存在チェック付き）
# ==============================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🛑 Stopping existing container: $CONTAINER_NAME ..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# ==============================
# コンテナ起動（絶対パス指定）
# ==============================
echo "🚀 Starting Open WebUI container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart=always \
  -p 3000:8080 \
  -v "$DATA_DIR:/app/backend/data" \
  -e OPENAI_API_KEY=not-needed \
  -e OPENAI_API_BASE_URL="http://$WSL_IP:8000/v1" \
  "$IMAGE_NAME"

# ==============================
# 起動確認
# ==============================
sleep 2
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "✅ Open WebUI is running at: http://$WSL_IP:3000"
else
  echo "❌ Failed to start Open WebUI. Check with 'docker logs $CONTAINER_NAME'."
fi
