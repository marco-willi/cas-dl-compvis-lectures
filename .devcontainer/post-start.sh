#!/bin/bash
# Post-start command: Runs every time the container starts

set -e

echo "Running post-start tasks..."

# Ensure uv is on PATH (installed during post-create)
export PATH="$HOME/.local/bin:$PATH"

# Ensure .venv symlink points to the native-fs venv (survives container rebuilds)
# Also re-link a stale symlink: /workspace is host-mounted, so an old link persists
VENV_REAL="$UV_PROJECT_ENVIRONMENT"
if [ -n "$VENV_REAL" ] && [ "$(readlink /workspace/.venv 2>/dev/null)" != "$VENV_REAL" ]; then
    rm -rf /workspace/.venv
    ln -s "$VENV_REAL" /workspace/.venv
    echo "Re-linked .venv -> $VENV_REAL"
fi

# Redirect Quarto cache to native filesystem (avoids slow SQLite WAL on 9p)
# Always recreate the /tmp target — it is wiped on every container restart
mkdir -p /tmp/quarto-cache
if [ ! -L /workspace/.quarto ]; then
    rm -rf /workspace/.quarto
    ln -s /tmp/quarto-cache /workspace/.quarto
fi

# Install CLI tools
bash "$(dirname "$0")/cli-tools.sh"

echo ""
echo "Python : $(which python) ($(python --version 2>&1))"
echo "uv     : $(uv --version)"
echo "Venv   : /workspace/.venv -> $(readlink -f /workspace/.venv)"

echo ""
echo "Container ready!"
