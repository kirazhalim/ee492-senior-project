#!/usr/bin/env bash
# Convenience launcher for the EE492 cough-detection live demo.
#
# Usage:
#   ./run_demo.sh                  # opens the Streamlit app in your browser
#   ./run_demo.sh --warmup         # only run the warmup pass (pre-load models)
#
# The script lives at the project root so paths resolve unambiguously.
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Could not find ${VENV_PY}. Create the project venv first." >&2
  exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--warmup" ]]; then
  exec "${VENV_PY}" -m app.warmup
fi

exec "${VENV_PY}" -m streamlit run app/streamlit_app.py \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
