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
ls -la "$(dirname "$MODELS_CONFIG")" || true
if [ ! -f "$MODELS_CONFIG" ]; then
  echo "ERROR: Config not found at $MODELS_CONFIG"
  exit 1
fi

echo "Reading models from $MODELS_CONFIG ..."
echo "----- BEGIN models.yaml -----"
sed -n '1,200p' "$MODELS_CONFIG" || true
echo "-----  END models.yaml  -----"

# Get chat models under providers.ollama.models[]
CHAT_MODELS="$(yq -r '.providers.ollama.models // [] | .[]' < "$MODELS_CONFIG" || true)"

# Get embedding models under providers.ollama.embedding_models[]
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

echo "Will pull chat models:"
if [ -n "$CHAT_MODELS" ]; then
  echo "$CHAT_MODELS" | sed 's/^/  - /'
else
  echo "  (none)"
fi

echo "Will pull embedding models:"
if [ -n "$EMBEDDING_MODELS" ]; then
  echo "$EMBEDDING_MODELS" | sed 's/^/  - /'
else
  echo "  (none)"
fi

RETRIES="${PULL_RETRIES:-5}"
SLEEP="${PULL_SLEEP_SECS:-3}"

echo "Starting model pulls..."
echo "$ALL_MODELS" | while IFS= read -r m; do
  if [ -n "$m" ]; then
    echo "==> Pull $m"
    i=1
    until [ $i -gt "$RETRIES" ]; do
      if curl -fsS -X POST "$OLLAMA_HOST/api/pull" \
        -H 'Content-Type: application/json' \
        -d "{\"name\":\"$m\"}" >/dev/null; then
        echo "Pulled $m"
        break
      fi
      echo "retry $i/$RETRIES for $m in ${SLEEP}s..."
      i=$((i+1)); sleep "$SLEEP"
    done
    if [ $i -gt "$RETRIES" ]; then
      echo "FAILED to pull $m after $RETRIES attempts" >&2
    fi
  fi
done

echo "Pull phase complete."