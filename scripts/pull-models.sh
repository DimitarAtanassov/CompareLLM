#!/bin/sh
set -eu

: "${MODELS_CONFIG:?MODELS_CONFIG is required}"
: "${OLLAMA_HOST:?OLLAMA_HOST is required}"

echo "ENV: MODELS_CONFIG='${MODELS_CONFIG}'  OLLAMA_HOST='${OLLAMA_HOST}'"

# tools
apk add --no-cache yq curl jq >/dev/null

# Optional: allow disabling via env
if [ "${ENABLE_AUTO_PULL:-true}" != "true" ]; then
  echo "Auto-pull disabled (ENABLE_AUTO_PULL=${ENABLE_AUTO_PULL:-})"
  exit 0
fi

# sanity
if [ ! -f "$MODELS_CONFIG" ]; then
  echo "ERROR: Config not found at $MODELS_CONFIG"
  exit 1
fi

echo "Reading models from $MODELS_CONFIG ..."

# Get all models
CHAT_MODELS="$(yq -r '.providers.ollama.models // [] | .[]' < "$MODELS_CONFIG" || true)"
EMBEDDING_MODELS="$(yq -r '.providers.ollama.embedding_models // [] | .[]' < "$MODELS_CONFIG" || true)"

# Combine all models
ALL_MODELS=""
if [ -n "$CHAT_MODELS" ]; then
  ALL_MODELS="$CHAT_MODELS"
fi
if [ -n "$EMBEDDING_MODELS" ]; then
  if [ -n "$ALL_MODELS" ]; then
    ALL_MODELS="$ALL_MODELS
$EMBEDDING_MODELS"
  else
    ALL_MODELS="$EMBEDDING_MODELS"
  fi
fi

if [ -z "$ALL_MODELS" ]; then
  echo "No ollama models listed in config. Done."
  exit 0
fi

# Wait for Ollama API to be available
echo "🔄 Waiting for Ollama API..."
WAIT_RETRIES=60  # Wait up to 2 minutes
WAIT_SLEEP=2
i=1
OLLAMA_READY=false

until [ $i -gt "$WAIT_RETRIES" ]; do
  if curl -fsS "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    OLLAMA_READY=true
    echo "✅ Ollama API is ready!"
    break
  fi
  echo "⏳ Waiting for Ollama API... ($i/$WAIT_RETRIES)"
  i=$((i+1))
  sleep "$WAIT_SLEEP"
done

if [ "$OLLAMA_READY" = "false" ]; then
  echo "❌ Ollama API not ready. Falling back to pull all models."
fi

# Simple function to check if model exists (with error handling)
model_exists() {
  local model_name="$1"
  if [ "$OLLAMA_READY" = "false" ]; then
    return 1  # Assume it doesn't exist if we can't check
  fi
  
  # Try to get the model list and check if our model is in it
  EXISTING_LIST=$(curl -fsS "$OLLAMA_HOST/api/tags" 2>/dev/null | jq -r '.models[]?.name // empty' 2>/dev/null || echo "")
  echo "$EXISTING_LIST" | grep -q "^${model_name}$" 2>/dev/null
}

RETRIES="${PULL_RETRIES:-5}"
SLEEP="${PULL_SLEEP_SECS:-3}"

echo ""
echo "📋 Processing models:"
echo "$ALL_MODELS" | sed 's/^/  - /'

echo ""
echo "🔍 Checking and pulling models..."
echo "$ALL_MODELS" | while IFS= read -r m; do
  if [ -n "$m" ]; then
    if model_exists "$m"; then
      echo "✅ $m (already exists, skipping)"
    else
      echo "📥 $m (pulling...)"
      i=1
      until [ $i -gt "$RETRIES" ]; do
        if curl -fsS -X POST "$OLLAMA_HOST/api/pull" \
          -H 'Content-Type: application/json' \
          -d "{\"name\":\"$m\"}" >/dev/null 2>&1; then
          echo "✅ $m (pulled successfully)"
          break
        fi
        echo "⏳ retry $i/$RETRIES for $m in ${SLEEP}s..."
        i=$((i+1)); sleep "$SLEEP"
      done
      if [ $i -gt "$RETRIES" ]; then
        echo "❌ FAILED to pull $m after $RETRIES attempts" >&2
      fi
    fi
  fi
done

echo ""
echo "🎉 Pull phase complete."