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

# models under providers.ollama.models[]
MODELS="$(yq -r '.providers.ollama.models // [] | .[]' < "$MODELS_CONFIG" || true)"

if [ -z "$MODELS" ]; then
  echo "No ollama models listed in config. Done."
  exit 0
fi

echo "Will pull:"
echo "$MODELS" | sed 's/^/  - /'

RETRIES="${PULL_RETRIES:-5}"
SLEEP="${PULL_SLEEP_SECS:-3}"

for m in $MODELS; do
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
done

echo "Pull phase complete."
