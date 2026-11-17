from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from aggregate_results import aggregate, planning_summaries
from metrics_lineage import merge_metric_sessions


def write_metrics(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["step,fit/loss,model_opt/level1"]
    for step, loss in rows:
        lines.extend([f"{step},,{0.001 - step * 1e-6}", f"{step},{loss},"])
    path.write_text("\n".join(lines) + "\n")


def test_metric_sessions_replace_abandoned_suffix(tmp_path):
    first = tmp_path / "spt/runs/20261027/090000/first/metrics.csv"
    resumed = tmp_path / "spt/runs/20261027/120000/resumed/metrics.csv"
    write_metrics(first, [(0, 4.0), (50, 3.0), (100, 2.0), (150, 1.0)])
    write_metrics(resumed, [(100, 20.0), (150, 10.0), (200, 5.0)])

    rows, lineage = merge_metric_sessions([first, resumed])

    assert [row["step"] for row in rows] == [0, 50, 100, 150, 200]
    assert [row["fit/loss"] for row in rows] == [4.0, 3.0, 20.0, 10.0, 5.0]
    assert rows[2]["_metrics_session_id"] == "resumed"
    assert lineage["session_count"] == 2
    assert lineage["has_resume_or_restart"] is True
    assert lineage["superseded_logged_step_count"] == 2
    assert lineage["sessions"][1]["superseded_prior_first_step"] == 100
    assert lineage["sessions"][1]["superseded_prior_last_step"] == 150


def test_incomplete_trailing_csv_row_is_ignored(tmp_path):
    path = tmp_path / "spt/runs/20261027/090000/first/metrics.csv"
    write_metrics(path, [(0, 4.0)])
    with path.open("ab") as stream:
        stream.write(b"50,3")

    rows, lineage = merge_metric_sessions([path])

    assert [row["step"] for row in rows] == [0]
    assert lineage["sessions"][0]["incomplete_trailing_bytes_ignored"] == 4


def test_planning_summary_averages_planners_before_training_seeds():
    rows = []
    for training_seed, values in ((42, (10.0, 30.0)), (43, (30.0, 50.0))):
        for planner_seed, value in enumerate(values, start=1):
            rows.append(
                {
                    "environment": "ant",
                    "model": "ant_hjepa_l2",
                    "evaluation_kind": "automatic_ant_l2",
                    "metric_name": "success_rate_pct",
                    "training_protocol": "v2_b128_accum1",
                    "model_weight_source": "local_training",
                    "training_seed": training_seed,
                    "planner_seed": planner_seed,
                    "metric_value": value,
                    "status": "complete",
                    "is_latest_for_evaluation_key": True,
                }
            )

    summary = planning_summaries(rows)[0]

    assert summary["mean"] == 30.0
    assert summary["standard_error"] == 10.0
    assert summary["training_seed_count"] == 2
    assert summary["complete_evaluation_count"] == 4


def test_aggregator_separates_protocol_and_preserves_missing_values(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run = ckpt_root / "fourroom/fourroom_hjepa_l2/seed42"
    run.mkdir(parents=True)
    config = {
        "output_model_name": "fourroom_hjepa_l2",
        "env": "fourroom",
        "seed": 42,
        "num_levels": 2,
        "training_protocol_version": 2,
        "loader": {"batch_size": 128},
        "trainer": {"accumulate_grad_batches": 1},
        "planning_eval": {"enabled": True, "config_name": "fourroom_l2"},
        "final_probing_decoding_eval": {"enabled": True},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "fourroom_hjepa_l2_object.ckpt").write_bytes(b"checkpoint")
    write_metrics(
        run / "spt/runs/20261027/090000/train/metrics.csv",
        [(0, 2.0), (50, 1.0)],
    )

    task = tmp_path / "task.pt"
    task.write_bytes(b"task")
    planning = run / "planning_eval/epoch_0008"
    planning.mkdir(parents=True)
    (planning / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "policy": str(run / "fourroom_hjepa_l2_object.ckpt"),
                "load_eval_trajs_path": str(task),
                "eval": {"num_eval": 2},
            }
        )
    )
    (planning / "metrics.yaml").write_text(
        yaml.safe_dump({"success_rate": 50.0, "episode_successes": [True, False]})
    )

    posthoc = run / "final_probing_decoding_eval_v2"
    posthoc.mkdir()
    (posthoc / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "optimizer": {"probe_lr": 0.001, "decoder_lr": 0.0001},
            }
        )
    )
    (posthoc / "metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "final_probing_decoding_eval/eval/level1_probe_xy_nmse": 0.2,
                "final_probing_decoding_eval/eval/decoder_loss_level1": 0.1,
            }
        )
    )

    result = aggregate(ckpt_root, tmp_path / "output")

    assert len(result["runs"]) == 1
    assert result["runs"][0]["training_protocol"] == "v2_b128_accum1"
    assert result["runs"][0]["model_weight_source"] == "local_training"
    assert result["planning_results"][0]["status"] == "complete"
    assert result["planning_results"][0]["success_rate_pct"] == 50.0
    assert result["planning_results"][0]["frechet_fidelity_pct"] is None
    assert result["planning_summaries"][0]["standard_error"] is None
    assert result["coverage"][0]["planning_status"] == "complete"
    assert result["coverage"][0]["posthoc_status"] == "complete"
    assert (tmp_path / "output/aggregated_results.json").is_file()
