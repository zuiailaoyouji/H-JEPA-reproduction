from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analysis_utils import accepted_run, mean_se, pareto_indices, run_metadata
from analyze_cost_monotonicity import negative_spearman
from analyze_frequency import _figure5_points, _spectral_terms
from analyze_official_release import build_cost_tables
from analyze_protocol_tables import level1_frames, table11
from analyze_training_dynamics import collect, read_run
from progress_dashboard import _latest_attempt
from run_official_reproduction import cost_jobs, depth_jobs, droid_jobs, probe_jobs


def test_mean_se_and_pareto_front():
    mean, se, count = mean_se([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert math.isclose(se, 1.0 / math.sqrt(3.0))
    assert count == 3
    assert pareto_indices([1, 2, 3, 4], [1, 3, 2, 4]).tolist() == [0, 1, 3]


def test_spectral_centroid_orders_slow_and_fast_sinusoids():
    t = np.arange(128)
    slow = np.sin(2 * np.pi * 2 * t / len(t))[:, None]
    fast = np.sin(2 * np.pi * 16 * t / len(t))[:, None]
    slow_num, slow_den, _ = _spectral_terms(slow)
    fast_num, fast_den, _ = _spectral_terms(fast)
    assert math.isclose(slow_num / slow_den, 2 / 128, rel_tol=1e-6)
    assert math.isclose(fast_num / fast_den, 16 / 128, rel_tol=1e-6)
    assert math.isclose((fast_num / fast_den) / (slow_num / slow_den), 8.0, rel_tol=1e-12)


def test_spectral_centroid_uses_periodogram_power():
    t = np.arange(128)
    mixed = np.sin(2 * np.pi * 2 * t / len(t))
    mixed += 0.5 * np.sin(2 * np.pi * 10 * t / len(t))
    numerator, denominator, _ = _spectral_terms(mixed[:, None])

    power_weighted_centroid = (2 + 0.5**2 * 10) / ((1 + 0.5**2) * len(t))
    magnitude_weighted_centroid = (2 + 0.5 * 10) / (1.5 * len(t))
    assert math.isclose(numerator / denominator, power_weighted_centroid, rel_tol=1e-6)
    assert not math.isclose(numerator / denominator, magnitude_weighted_centroid, rel_tol=1e-3)


def test_figure5_aggregates_no_gap_environment_as_all_retained():
    centroids = [
        {"env": "cube", "entity": "block", "centroid_cycles_per_frame": 0.10, "frequency_gap": 1.5},
        {"env": "cube", "entity": "arm", "centroid_cycles_per_frame": 0.15, "frequency_gap": 1.5},
    ]
    probes = {
        ("cube", "block", 1): {"nmse_mean": 0.10},
        ("cube", "block", 2): {"nmse_mean": 0.14},
        ("cube", "arm", 1): {"nmse_mean": 0.20},
        ("cube", "arm", 2): {"nmse_mean": 0.22},
    }
    points = _figure5_points(centroids, probes)
    assert len(points) == 1
    assert points[0]["entity"] == "all"
    assert points[0]["role"] == "retained"
    assert points[0]["n_entities"] == 2
    assert points[0]["delta_nmse"] == pytest.approx(0.03)


def test_negative_spearman_sign():
    assert math.isclose(negative_spearman(np.array([4.0, 3.0, 2.0, 1.0])), 1.0)
    assert math.isclose(negative_spearman(np.array([1.0, 2.0, 3.0, 4.0])), -1.0)
    assert math.isnan(negative_spearman(np.ones(4)))


def _write_metrics(path: Path, rows: list[dict]) -> None:
    fields = ["step", "epoch", "fit/pred_loss_level1", "fit/sigreg_loss_level1", "fit/loss"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_training_fragments_overwrite_repeated_global_steps(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_metrics(
        first,
        [
            {"step": 0, "epoch": 0, "fit/pred_loss_level1": 2, "fit/sigreg_loss_level1": 4, "fit/loss": 6},
            {"step": 1, "epoch": 0, "fit/pred_loss_level1": 4, "fit/sigreg_loss_level1": 6, "fit/loss": 8},
        ],
    )
    _write_metrics(
        second,
        [{"step": 1, "epoch": 0, "fit/pred_loss_level1": 8, "fit/sigreg_loss_level1": 10, "fit/loss": 12}],
    )
    first.touch()
    second.touch()
    rows = read_run([first, second], {"env": "x", "model": "m", "seed": 1, "run_dir": str(tmp_path)})
    values = {(row["metric"], row["level"]): row["value"] for row in rows}
    assert values[("pred", 1)] == 5.0
    assert values[("sigreg", 1)] == 7.0
    assert values[("total", 0)] == 9.0


def test_training_collection_keeps_latest_retry_for_same_seed(tmp_path):
    files = []
    for name, prediction, modified in (("seed42", 99, 1), ("seed42_retry", 0, 2)):
        run = tmp_path / "ant" / "ant_hjepa_l2" / name
        metrics = run / "spt" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        protocol = "" if name == "seed42" else "training_protocol_version: 2\n"
        (run / "config.yaml").write_text(
            f"env: ant\noutput_model_name: ant_hjepa_l2\nseed: 42\n{protocol}"
        )
        _write_metrics(
            metrics,
            [{"step": 0, "epoch": 0, "fit/pred_loss_level1": prediction, "fit/sigreg_loss_level1": 1, "fit/loss": 1}],
        )
        os.utime(metrics, (modified, modified))
        files.append(metrics)

    rows = collect([str(tmp_path)])
    prediction_rows = [row for row in rows if row["metric"] == "pred"]
    assert len(prediction_rows) == 1
    assert prediction_rows[0]["value"] == 0
    assert prediction_rows[0]["run_dir"].endswith("seed42_retry")


def test_verified_release_manifest_accepts_only_listed_run(tmp_path):
    root = tmp_path / "ckpts"
    official = root / "ant" / "ant_hjepa_l2" / "seed42"
    local = root / "ant" / "ant_hjepa_l2" / "seed42_paper_v2_20261027"
    official.mkdir(parents=True)
    local.mkdir(parents=True)
    (root / "OFFICIAL_WEIGHTS_MANIFEST.json").write_text(
        '{"revision":"release-sha","files":{"ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt":{}}}'
    )
    official_result = official / "final_probing_decoding_eval_v2" / "metrics.yaml"
    official_result.parent.mkdir()
    official_result.write_text("eval/x: 1\n")
    local_result = local / "final_probing_decoding_eval_v2" / "metrics.yaml"
    local_result.parent.mkdir()
    local_result.write_text("eval/x: 1\n")

    official_metadata = run_metadata(official_result)
    local_metadata = run_metadata(local_result)
    assert accepted_run(official_metadata)
    assert official_metadata["source"] == "official_release"
    assert official_metadata["release_revision"] == "release-sha"
    assert not accepted_run(local_metadata)


def test_official_protocol_enumerates_every_release_cell(tmp_path):
    depth = depth_jobs(tmp_path)
    cost = cost_jobs(tmp_path)
    droid = droid_jobs(tmp_path)
    probes = probe_jobs(tmp_path)
    assert len(depth) == 4 * 7 * 3 == 84
    assert len(cost) == 4 * (2 + 3 + 4) * 3 == 108
    assert len(droid) == 3 * 3 * 3 == 27
    assert len(probes) == (4 + 2) * 3 == 18
    assert sum(job.kind == "probe_decoder" for job in probes) == 2
    all_jobs = depth + cost + droid + probes
    assert len({job.name for job in all_jobs}) == len(all_jobs)
    assert len({job.result for job in all_jobs}) == len(all_jobs)


def test_dashboard_uses_only_latest_attempt_progress():
    text = """=== launch 2026-10-28 01:00:00 gpu=1 ===
epoch 58/60
=== launch 2026-10-28 11:00:00 gpu=4 ===
epoch 1/60
"""
    launched, gpu, attempt = _latest_attempt(text)
    assert launched is not None
    assert gpu == "4"
    assert "epoch 1/60" in attempt
    assert "epoch 58/60" not in attempt


def test_cost_tables_include_projection_and_hierarchical_references():
    cost = []
    depth = []
    for env in ("ant", "fourroom", "cube", "pusht"):
        depth.append({"env": env, "model": "lewm", "depth": 1, "mean": 10.0, "se": 1.0, "n": 3})
        for model_depth in (2, 3, 4):
            depth.append(
                {
                    "env": env,
                    "model": f"hjepa_l{model_depth}",
                    "depth": model_depth,
                    "mean": 20.0 + model_depth,
                    "se": 2.0,
                    "n": 3,
                }
            )
            for cost_level in range(1, model_depth + 1):
                cost.append(
                    {
                        "env": env,
                        "model": f"hjepa_l{model_depth}",
                        "depth": model_depth,
                        "cost_level": cost_level,
                        "mean": 30.0 + cost_level,
                        "se": 3.0,
                        "n": 3,
                    }
                )

    rows = build_cost_tables(cost, depth)
    assert len(rows) == 4 * 4
    ant_l4 = next(row for row in rows if row["env"] == "ant" and row["model"] == "hjepa_l4")
    assert ant_l4["native_l1_mean"] == 31.0
    assert ant_l4["l4_projection_mean"] == 34.0
    assert ant_l4["hierarchical_mean"] == 24.0
    assert ant_l4["n_seeds"] == 3


def test_table3_hierarchical_temporal_layout():
    frames = [level1_frames(depth) for depth in range(1, 5)]
    assert frames == [4, 7, 13, 25]
    assert [5 * (value - 1) + 1 for value in frames] == [16, 31, 61, 121]


def test_table11_global_sigreg_is_invariant_to_equal_sharding():
    rows = table11(seed=0)
    assert [row["shards"] for row in rows] == [1, 2, 4]
    assert all(abs(row["global_tep_delta_from_1_shard"]) < 1e-12 for row in rows)
    assert all(abs(row["global_gradient_delta_from_1_shard"]) < 1e-12 for row in rows)
    assert any(abs(row["local_reference_drift_percent"]) > 1.0 for row in rows[1:])
