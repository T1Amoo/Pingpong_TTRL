import csv
import json
from pathlib import Path

import numpy as np

from fit_0729_response import (
    discover_trials,
    estimate_velocity_limit,
    format_python_dict,
    load_trial_series,
    write_merged_csv,
)


FIELDS = [
    "source",
    "run_id",
    "stamp",
    "t",
    "phase",
    "joint_index",
    "joint_name",
    "signal_type",
    "freq_hz",
    "chirp_start_hz",
    "chirp_end_hz",
    "amplitude_rad",
    "sample_index",
    *[f"target_q{i}" for i in range(1, 8)],
    *[f"target_dq{i}" for i in range(1, 8)],
    *[f"actual_q{i}" for i in range(1, 8)],
    *[f"actual_dq{i}" for i in range(1, 8)],
    *[f"actual_tau{i}" for i in range(1, 8)],
]


def _write_trial(path: Path, *, joint: int, signal: str, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "joint_index": joint,
        "joint_name": f"joint{joint}-a1_r",
        "signal_type": signal,
        "rate_hz": 100.0,
        "chirp_start_hz": 0.1,
        "chirp_end_hz": 6.0 if signal == "chirp" else 0.1,
        "freq_hz": 0.4 if signal == "step" else 0.1,
        "real_kp": [300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 60.0],
        "real_kd": [3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5],
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for sample in range(rows):
            row = {
                "source": "real",
                "run_id": "test",
                "stamp": str(1000.0 + 0.01 * sample),
                "t": str(2.0 + 0.01 * sample),
                "phase": "active",
                "joint_index": str(joint),
                "joint_name": f"joint{joint}-a1_r",
                "signal_type": signal,
                "freq_hz": "0.4" if signal == "step" else "0.1",
                "chirp_start_hz": "0.1",
                "chirp_end_hz": "6.0" if signal == "chirp" else "0.1",
                "amplitude_rad": "0.06",
                "sample_index": str(sample),
            }
            for i in range(1, 8):
                row[f"target_q{i}"] = str(i + 0.1 * sample)
                row[f"target_dq{i}"] = "0.0"
                row[f"actual_q{i}"] = str(i + 0.1 * sample + 0.02)
                row[f"actual_dq{i}"] = "0.0"
                row[f"actual_tau{i}"] = "0.0"
            writer.writerow(row)


def test_discover_trials_excludes_empty_timeouts_and_empty_csv(tmp_path):
    valid = tmp_path / "backhand" / "j7" / "real" / "j7_backhand_chirp.csv"
    empty_timeout = tmp_path / "backhand" / "_empty_timeouts" / "j7_backhand_chirp_empty.csv"
    empty_real = tmp_path / "backhand" / "j7" / "real" / "j7_backhand_step_empty.csv"
    _write_trial(valid, joint=7, signal="chirp", rows=3)
    _write_trial(empty_timeout, joint=7, signal="chirp", rows=3)
    _write_trial(empty_real, joint=7, signal="step", rows=0)

    trials = discover_trials(tmp_path, "backhand")

    assert [trial.csv_path for trial in trials] == [valid]
    assert trials[0].joint == 7
    assert trials[0].signal_type == "chirp"
    assert trials[0].real_kp[6] == 60.0
    assert trials[0].real_kd[6] == 0.5


def test_load_trial_series_maps_selected_joint_and_relative_time(tmp_path):
    csv_path = tmp_path / "backhand" / "j3" / "real" / "j3_backhand_chirp.csv"
    _write_trial(csv_path, joint=3, signal="chirp", rows=4)
    trial = discover_trials(tmp_path, "backhand")[0]

    series = load_trial_series(trial)

    np.testing.assert_allclose(series.relative_s, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(series.command_q, [3.0, 3.1, 3.2, 3.3])
    np.testing.assert_allclose(series.actual_q, [3.02, 3.12, 3.22, 3.32])


def test_write_merged_csv_uses_existing_fit_column_names(tmp_path):
    csv_path = tmp_path / "backhand" / "j2" / "real" / "j2_backhand_chirp.csv"
    merged_path = tmp_path / "merged" / "j2_chirp.csv"
    _write_trial(csv_path, joint=2, signal="chirp", rows=2)
    trial = discover_trials(tmp_path, "backhand")[0]
    series = load_trial_series(trial)

    write_merged_csv(merged_path, series)

    with merged_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0].keys() == {"relative_s", "target_raw_q", "cmd_q", "real_actual_q"}
    assert rows[1]["relative_s"] == "0.01"
    assert rows[1]["cmd_q"] == "2.1"
    assert rows[1]["real_actual_q"] == "2.12"


def test_format_python_dict_is_stable_for_training_constants():
    text = format_python_dict({"r2": -0.762, "r1": 1.769}, indent=4)

    assert text == '{\n    "r1": 1.769,\n    "r2": -0.762,\n}'


def test_estimate_velocity_limit_uses_real_dq_with_margin():
    max_real_dq = {
        "r1": 2.777,
        "r2": 2.371,
        "r3": 3.377,
        "r4": 3.944,
        "r5": 5.238,
        "r6": 3.895,
        "r7": 6.484,
    }

    assert estimate_velocity_limit(max_real_dq) == {
        "r1": 4.0,
        "r2": 4.0,
        "r3": 5.0,
        "r4": 6.0,
        "r5": 8.0,
        "r6": 6.0,
        "r7": 9.0,
    }
