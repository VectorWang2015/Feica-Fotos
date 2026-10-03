#!/usr/bin/env bash
# Offline launcher: uses only this checkout's existing virtual environment.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="$ROOT/.venv/bin/python"
cd -- "$ROOT"

show_setup() {
    printf '%s\n' 'Feica Fotos requires a project .venv with Python 3.12 and its runtime dependencies.' >&2
    printf 'From %q run these commands yourself:\n' "$ROOT" >&2
    printf '%s\n' '  python3.12 -m venv .venv' \
        '  .venv/bin/python -m pip install -r requirements-app.txt' >&2
    printf '%s\n' 'The install command may access package indexes. This launcher never installs or downloads anything.' \
        'For offline setup, use a trusted pre-downloaded wheel directory with pip --no-index --find-links.' >&2
}

if [[ ! -x "$PYTHON" ]]; then
    show_setup
    exit 1
fi
if ! "$PYTHON" -E -s -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"; import numpy; from PIL import Image, ImageCms; from PySide6.QtWidgets import QApplication'; then
    printf '%s\n' 'The project environment is missing dependencies or compatible system Qt libraries.' >&2
    show_setup
    exit 1
fi
exec "$PYTHON" -E -s -m apps.feica_fotos "$@"
