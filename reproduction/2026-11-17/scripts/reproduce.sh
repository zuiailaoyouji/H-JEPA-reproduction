#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${package_dir}/code${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
command_name="${1:-verify}"
if (($#)); then shift; fi
case "$command_name" in
  verify)
    exec "$python_bin" "$package_dir/scripts/verify_archive.py" "$@"
    ;;
  tests)
    cd "$package_dir/code"
    exec "$python_bin" -m pytest -p no:cacheprovider h_jepa/tests "$@"
    ;;
  dry-run)
    cd "$package_dir/code/h_jepa"
    exec "$python_bin" scripts/run_official_reproduction.py --dry-run "$@"
    ;;
  aggregate|evaluate|probes|protocol)
    : "${HJEPA_HOME:?Set HJEPA_HOME to a restored or downloaded data directory.}"
    cd "$package_dir/code/h_jepa"
    case "$command_name" in
      aggregate)
        exec "$python_bin" scripts/analyze_official_release.py \
          --output-dir "$HJEPA_HOME/analyse/reaggregated_20261117" "$@"
        ;;
      evaluate)
        if [[ " $* " != *" --gpus "* && " $* " != *" --gpus="* ]]; then
          printf '%s\n' 'Supply --gpus with GPUs assigned to this run.' >&2
          exit 2
        fi
        exec "$python_bin" scripts/run_official_reproduction.py "$@"
        ;;
      probes)
        exec "$python_bin" scripts/analyze_probes.py "$HJEPA_HOME/ckpts" \
          --model hjepa_l2 --model hjepa_l4 --require-seeds 3 \
          --output-dir "$HJEPA_HOME/analyse/probes_reaggregated_20261117" "$@"
        ;;
      protocol)
        exec "$python_bin" scripts/analyze_protocol_tables.py \
          --output-dir "$HJEPA_HOME/analyse/protocol_tables_recomputed" "$@"
        ;;
    esac
    ;;
  *)
    printf '%s\n' 'Usage: reproduce.sh {verify|tests|dry-run|aggregate|evaluate|probes|protocol} [options]' >&2
    exit 2
    ;;
esac
