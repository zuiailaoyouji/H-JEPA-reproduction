#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use an activated Python 3.11 environment"'
"$python_bin" -m pip install -r "$package_dir/environment/requirements-portable.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128
"$python_bin" -m pip install --no-deps -e "$package_dir/code"
printf '%s\n' 'Set HJEPA_HOME to your restored data directory before running evaluation.'
