#!/usr/bin/env bash
# Rerun the official end-of-training evaluations using the saved final model.
set -euo pipefail

export HJEPA_HOME=/home/songxw/hjepa-data
export CUDA_VISIBLE_DEVICES=4
export MUJOCO_EGL_DEVICE_ID=4
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

cd /home/songxw/H-JEPA/h_jepa
python_bin=/home/songxw/miniconda3/envs/hjepa/bin/python
run_dir="$HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l2/seed42"
recovery_dir="$run_dir/eval_recovery"
mkdir -p "$recovery_dir"
exec 9>"$recovery_dir/run.lock"
flock -n 9 || exit 1
exec >>"$recovery_dir/run.log" 2>&1

stage=starting
on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        printf 'failed: %s (exit %s)\n' "$stage" "$exit_code" >"$recovery_dir/status.txt"
        printf '[%s] Failed during %s (exit %s)\n' "$(date -Is)" "$stage" "$exit_code"
    fi
}
trap on_exit EXIT

printf '[%s] Recovering official final evaluations on GPU %s\n' "$(date -Is)" "$CUDA_VISIBLE_DEVICES"
stage=planning_eval
printf '%s\n' "$stage" >"$recovery_dir/status.txt"
"$python_bin" -u eval.py --config-name fourroom_l2 \
    seed=42 \
    policy="$run_dir/fourroom_hjepa_l2_object.ckpt" \
    load_eval_trajs_path=/home/songxw/H-JEPA/h_jepa/assets/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt \
    output.dir=planning_eval/epoch_0008 \
    hydra.run.dir="$recovery_dir/hydra/planning"

stage=final_probing_decoding_eval
printf '%s\n' "$stage" >"$recovery_dir/status.txt"
printf '[%s] Planning complete; starting official 60-epoch probes and decoders\n' "$(date -Is)"
"$python_bin" -u main_probing_decoding_eval.py --config-name fourroom \
    seed=42 \
    policy="$run_dir/fourroom_hjepa_l2_object.ckpt" \
    +level1.train_decoder=true +level1.probes.enabled=true \
    +level2.train_decoder=true +level2.probes.enabled=true \
    hydra.run.dir="$recovery_dir/hydra/probing"

stage=complete
printf '%s\n' "$stage" >"$recovery_dir/status.txt"
printf '[%s] Both official final evaluations completed\n' "$(date -Is)"
