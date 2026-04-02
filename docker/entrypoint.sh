#!/bin/sh
# =============================================================
# VOIDER Entrypoint — applies config from environment vars
# =============================================================
set -e

CONFIG_FILE="/app/config/config.yaml"
mkdir -p /app/data/vector_store /app/logs

# ── If using Ollama, wait for it to be ready ─────────────────
PROVIDER_TYPE=$(echo "$VOIDER_PROVIDER" | tr '[:upper:]' '[:lower:]')
if [ "$PROVIDER_TYPE" = "ollama" ] || [ -z "$VOIDER_PROVIDER" ]; then
    OLLAMA_URL="${VOIDER_OLLAMA_HOST:-http://ollama:11434}"
    echo "⏳ Waiting for Ollama at $OLLAMA_URL ..."
    MAX_WAIT=60
    WAITED=0
    until curl -sf "$OLLAMA_URL" > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ "$WAITED" -ge "$MAX_WAIT" ]; then
            echo "⚠️  Ollama not reachable after ${MAX_WAIT}s — starting anyway."
            break
        fi
    done
    echo "✅ Ollama is up."
fi

# ── Patch config.yaml with env-override values ───────────────
if [ -f "$CONFIG_FILE" ]; then
    # Inject provider/model/api_key into the YAML using Python
    python3 - <<'PYEOF'
import os, yaml, sys

cfg_path = "/app/config/config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

provider  = os.environ.get("VOIDER_PROVIDER", "ollama")
model     = os.environ.get("VOIDER_MODEL", "")
api_key   = os.environ.get("VOIDER_API_KEY", "")
base_url  = os.environ.get("VOIDER_OLLAMA_HOST", "http://ollama:11434")

llm = cfg.setdefault("llm", {})
llm["provider"] = provider
if model:
    llm["model"] = model
if api_key:
    llm["api_key"] = api_key
if provider == "ollama":
    llm["base_url"] = base_url

with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"✅ Config updated: provider={provider}, model={llm.get('model','<default>')}")
PYEOF
fi

# ── Start server ──────────────────────────────────────────────
echo "🚀 Starting VOIDER on 0.0.0.0:${AIOS_PORT:-8000} ..."
exec uvicorn backend.main:app \
    --host "${AIOS_HOST:-0.0.0.0}" \
    --port "${AIOS_PORT:-8000}" \
    --workers 1
