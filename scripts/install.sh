#!/bin/bash
set -e

echo "Installing poetry"
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

echo "setting up workspace"
cd /workspace
poetry config virtualenvs.in-project true
poetry install --with dev

echo "Poetry environment installed successfully."

echo "Installing zsh and oh-my-zsh"
apt-get update && apt-get install -y zsh git curl sudo build-essential libgl1-mesa-glx
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
