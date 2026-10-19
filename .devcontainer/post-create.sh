#!/bin/bash
# Post-create command: Runs once after container is created

set -e

echo "Running post-create setup..."

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# # Install NPX
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
# nvm install --lts

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Ensure we're in the workspace
cd /workspace

# UV_PROJECT_ENVIRONMENT (set via devcontainer.json containerEnv) places the venv
# on the native Linux filesystem for performance (avoids slow 9p scan by Quarto)
VENV_REAL="$UV_PROJECT_ENVIRONMENT"

uv sync

# Auto-activate the venv in every new terminal session
VENV_ACTIVATE=$VENV_REAL/bin/activate

for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$RC" ]; then
        if ! grep -q "source $VENV_ACTIVATE" "$RC"; then
            {
                echo ""
                echo "# Auto-activate uv venv"
                echo "[ -f $VENV_ACTIVATE ] && source $VENV_ACTIVATE"
            } >> "$RC"
        fi
    fi
done

# Point Quarto at the uv-managed interpreter (stable path, no hash suffix)
echo "QUARTO_PYTHON=$VENV_REAL/bin/python3" > /workspace/_environment

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo ""
    echo "Installing pre-commit hooks..."
    uv run pre-commit install || echo "Warning: Failed to install pre-commit hooks (continuing anyway)"
fi



echo ""
echo "Post-create setup complete!"
echo "Venv : /workspace/.venv -> $VENV_REAL (native fs)"
echo "Python: $($VENV_REAL/bin/python3 --version)"
