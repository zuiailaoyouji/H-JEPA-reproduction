# Source this file from Bash: source /home/songxw/H-JEPA/local_env/activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo 'Use: source /home/songxw/H-JEPA/local_env/activate.sh' >&2
    exit 1
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if declare -F deactivate >/dev/null; then
        deactivate
    else
        # Child shells may inherit VIRTUAL_ENV/PATH without the deactivate function.
        IFS=: read -r -a _hjepa_path_parts <<< "$PATH"
        _hjepa_clean_path=""
        for _hjepa_part in "${_hjepa_path_parts[@]}"; do
            if [[ "$_hjepa_part" != "$VIRTUAL_ENV/bin" ]]; then
                _hjepa_clean_path="${_hjepa_clean_path}:${_hjepa_part}"
            fi
        done
        export PATH="${_hjepa_clean_path#:}"
        unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
        unset _hjepa_path_parts _hjepa_clean_path _hjepa_part
        hash -r
    fi
fi
source /home/songxw/miniconda3/etc/profile.d/conda.sh
conda activate hjepa || return
export HJEPA_HOME="${HJEPA_HOME:-/home/songxw/hjepa-data}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
