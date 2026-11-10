#!/bin/zsh
set -eu
export NANOBOT_STREAM_IDLE_TIMEOUT_S=1800
export NANOBOT_OPENAI_COMPAT_TIMEOUT_S=1800
cd /Users/huynhnguyentrunghieu/Documents/NanoBot
export SSL_CERT_FILE="$PWD/.private/ca-bundle.pem"
session_id=robot-console
if [[ "${1:-}" == "--new" ]]; then
  session_id="robot-console-$(uuidgen)"
  shift
fi
exec .venv/bin/python launch_nanobot.py agent --classic --session "$session_id" "$@"
