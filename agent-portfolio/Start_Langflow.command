#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo "请先按 README.md 安装 Python 3.12 环境。"
  exit 1
fi
exec .venv/bin/python launch_local.py
