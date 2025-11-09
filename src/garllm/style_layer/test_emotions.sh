#!/bin/bash
# 感情軸テストスクリプト
# 各感情状態に応じて style_modulator の出力を比較する
# Persona: 織田信長
# Text: "この戦いが終われば酒を飲もう。"

PERSONA="織田信長"
TEXT="この戦いが終われば酒を飲もう。"

echo "===== Emotion Test: Joy (喜び) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Joy": 0.8}' \
  --intensity 0.7 --verbose

echo "===== Emotion Test: Trust (信頼・安堵) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Trust": 0.8}' \
  --intensity 0.7 --verbose

echo "===== Emotion Test: Fear (恐れ・慎重) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Fear": 0.8}' \
  --intensity 0.7 --verbose

echo "===== Emotion Test: Surprise (驚き・混乱) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Surprise": 0.8}' \
  --intensity 0.7 --verbose

echo "===== Mixed Emotion: Joy + Trust (喜び＋安堵) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Joy": 0.5, "Trust": 0.5}' \
  --intensity 0.7 --verbose

echo "===== Mixed Emotion: Fear + Surprise (恐れ＋驚き) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Fear": 0.5, "Surprise": 0.5}' \
  --intensity 0.7 --verbose

echo "===== Opposite Emotion: Joy + Fear (喜び＋恐れ) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Joy": 0.5, "Fear": 0.5}' \
  --intensity 0.7 --verbose

echo "===== Opposite Emotion: Trust + Surprise (信頼＋驚き) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Trust": 0.5, "Surprise": 0.5}' \
  --intensity 0.7 --verbose

echo "===== Negative Emotion: Joy -0.7 (悲しみ寄り) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Joy": -0.7}' \
  --intensity 0.7 --verbose

echo "===== Negative Emotion: Fear -0.8 (怒り寄り) ====="
python3 style_modulator.py --persona "$PERSONA" --text "$TEXT" \
  --emotion_axes '{"Fear": -0.8}' \
  --intensity 0.7 --verbose
