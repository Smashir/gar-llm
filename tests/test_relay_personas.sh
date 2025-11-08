#!/bin/bash
# ===========================================
# GAR-LLM Relay Server Persona Test Script
# ===========================================
# このスクリプトは relay_server.py が稼働している状態で、
# 既存および未生成のペルソナ呼び出しを確認するためのもの。
#
# 既存：織田信長
# 未生成：徳川家康（自動生成ルートを確認）
# ===========================================

# テストAPIのベースURL
URL="http://localhost:8081/v1/chat/completions"

# JSON出力を整形して jq で確認（jqがない場合は省略可）
echo "===== Persona Test: 織田信長（既存） ====="
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gar-llm",
    "messages": [
      {"role": "system", "content": "あなたは織田信長です。"},
      {"role": "user", "content": "また会おう。"}
    ],
    "persona": "織田信長",
    "intensity": 0.8,
    "verbose": true
  }' | jq .

echo ""
echo "================================================================================"
echo ""

echo "===== Persona Test: 徳川家康（未生成 → 自動生成） ====="
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gar-llm",
    "messages": [
      {"role": "system", "content": "あなたは徳川家康です。"},
      {"role": "user", "content": "どんな食べ物が好き？"}
    ],
    "persona": "徳川家康",
    "intensity": 0.8,
    "verbose": true
  }' | jq .

echo ""
echo "================================================================================"
echo ""
echo "✅ テスト完了。
