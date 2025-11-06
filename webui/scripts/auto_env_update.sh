#!/usr/bin/env bash
# auto_env_update.sh — regenerate .env file for Open WebUI
# Author: aiuser
# Last updated: 2025-10-11

set -e
cd "$(dirname "$0")/.."

ENV_FILE=".env"
BACKUP_FILE=".env.bak"

echo -e "\033[1;34m🔧 Updating Open WebUI environment file (.env)\033[0m"

# Detect WSL IP
WSL_IP=$(ip -o -4 addr show eth0 | awk '{print $4}' | cut -d/ -f1)
if [[ -z "$WSL_IP" ]]; then
  echo -e "\033[1;33m⚠️  Could not detect WSL IP. Using localhost.\033[0m"
  WSL_IP="127.0.0.1"
fi

# Detect active vLLM port (default: 8000)
VLLM_PORT=8000
if sudo lsof -i :8000 >/dev/null 2>&1; then
  echo "✅ vLLM detected on port 8000"
else
  echo "⚠️  vLLM not detected on port 8000. You may need to start it first."
fi

# Detect Ollama service (default: 11434)
if sudo lsof -i :11434 >/dev/null 2>&1; then
  echo "✅ Ollama detected on port 11434"
  OLLAMA_ACTIVE=true
else
  echo "⚠️  Ollama not detected on port 11434."
  OLLAMA_ACTIVE=false
fi

# Backup existing .env if exists
if [[ -f "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$BACKUP_FILE"
  echo "🗂️  Backup created: $BACKUP_FILE"
fi

# Write new .env
{
  echo "### Auto-generated Open WebUI environment file"
  echo "### Generated on $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  echo "OPENAI_API_KEY=not-needed"
  echo "OPENAI_API_BASE_URL=http://${WSL_IP}:${VLLM_PORT}/v1"
  echo "OLLAMA_API_BASE_URL=http://${WSL_IP}:11434"
  echo
  echo "### Notes:"
  echo "# - This file is managed by auto_env_update.sh"
  echo "# - Do not edit manually unless necessary"
} > "$ENV_FILE"

echo -e "\033[1;32m✅ .env successfully updated:\033[0m"
cat "$ENV_FILE"

# Restart WebUI container if running
if docker ps --format '{{.Names}}' | grep -q '^webui$'; then
  echo -e "\033[1;36m🔄 Restarting webui container...\033[0m"
  docker restart webui >/dev/null
  echo "✅ webui restarted."
else
  echo "ℹ️  webui container not running — skipped restart."
fi
