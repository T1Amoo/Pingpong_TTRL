from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = (ROOT / "legged_lab/envs/a1_tt/a1_tt_config.py").read_text()
REGISTRY_SOURCE = (ROOT / "legged_lab/envs/__init__.py").read_text()
ASSET_SOURCE = (ROOT / "legged_lab/assets/a1/a1.py").read_text()
WATCHDOG_SOURCE = (ROOT / "watchdog_a1_tt_backhand_v7.sh").read_text()
V1_3_URDF = ROOT / "legged_lab/assets/a1/X1_URDF_V1_3/urdf/X1_URDF_V1_3.urdf"
V2_DIR = ROOT / "legged_lab/assets/a1/X1_URDF_V2"
V2_URDF = V2_DIR / "urdf/X1_URDF_V2.urdf"
V2_USD = V2_DIR / "X1_URDF_V2_paddle.usd"
READY_Q = (1.369, -0.651, 1.656, -1.767, 0.145, 0.684, -2.153)


def _joint(root: ET.Element, name: str) -> ET.Element:
    return next(joint for joint in root.findall("joint") if joint.attrib["name"] == name)


def _origin_z(joint: ET.Element) -> float:
    return float(joint.find("origin").attrib["xyz"].split()[2])


def test_v2_is_an_independent_fixed_sj_108cm_asset_without_changing_v1_3():
    v1_root = ET.parse(V1_3_URDF).getroot()
    v2_root = ET.parse(V2_URDF).getroot()
    v1_sj = _joint(v1_root, "sj")
    v2_sj = _joint(v2_root, "sj")
    v2_ro = _joint(v2_root, "ro")

    assert v1_sj.attrib["type"] == "fixed"
    assert v2_sj.attrib["type"] == "fixed"
    assert _origin_z(v1_sj) == 1.09680245585163
    assert _origin_z(v2_sj) == 1.02680245585163
    assert abs((_origin_z(v1_sj) - _origin_z(v2_sj)) - 0.07) < 1.0e-12
    assert abs((0.0282 + _origin_z(v2_sj) + _origin_z(v2_ro)) - 1.08) < 5.0e-6

    assert V2_USD.is_file() and V2_USD.stat().st_size > 1000
    assert (V2_DIR / "configuration/X1_URDF_V2_paddle_base.usd").stat().st_size > 50_000_000
    assert (V2_DIR / "meshes").resolve() == (
        ROOT / "legged_lab/assets/a1/X1_URDF_V1_3/meshes"
    ).resolve()
    assert "A1_USD_PATH_V2" in ASSET_SOURCE
    assert '"X1_URDF_V2", "X1_URDF_V2_paddle.usd"' in ASSET_SOURCE


def test_v7_ready_pose_is_inside_every_urdf_joint_limit():
    root = ET.parse(V2_URDF).getroot()
    for index, q in enumerate(READY_Q, start=1):
        limit = _joint(root, f"r{index}").find("limit").attrib
        assert float(limit["lower"]) <= q <= float(limit["upper"])


def test_v7_inherits_v6_and_changes_only_asset_ready_pose_and_response_anchor():
    assert (
        "class A1TableTennisBackhandV7EnvCfg(A1TableTennisBackhandV6EnvCfg)"
        in CONFIG_SOURCE
    )
    body = CONFIG_SOURCE.split(
        "class A1TableTennisBackhandV7EnvCfg", 1
    )[1].split("class A1TableTennisBackhandV7EvalEnvCfg", 1)[0]
    assert "super().__post_init__()" in body
    assert "self.scene.robot.spawn.usd_path = A1_USD_PATH_V2" in body
    assert "self.scene.robot.init_state.joint_pos.update" in body
    assert "self.robot.action_response_u_mean = A1_BACKHAND_V7_READY_Q" in body
    for forbidden in (
        "self.reward.",
        "self.ball.",
        "action_response_fn_hz =",
        "action_response_zeta =",
        "action_response_delay_s =",
        "action_response_gain =",
        "action_response_bias_rad =",
        "action_target_lowpass",
        "self.domain_rand.",
    ):
        assert forbidden not in body
    assert (
        "A1_BACKHAND_V7_READY_Q = (1.369, -0.651, 1.656, -1.767, 0.145, 0.684, -2.153)"
        in CONFIG_SOURCE
    )


def test_v7_has_dedicated_train_eval_agent_registry_and_watchdog_namespaces():
    for class_name in (
        "A1TableTennisBackhandV7EnvCfg",
        "A1TableTennisBackhandV7EvalEnvCfg",
        "A1TableTennisBackhandV7AgentCfg",
    ):
        assert f"class {class_name}" in CONFIG_SOURCE
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v7",') == 1
    assert REGISTRY_SOURCE.count('"a1_tt_backhand_v7_eval",') == 1
    assert "A1TableTennisBackhandV7EnvCfg()" in REGISTRY_SOURCE
    assert "A1TableTennisBackhandV7EvalEnvCfg()" in REGISTRY_SOURCE
    assert 'experiment_name: str = "a1_tt_backhand_real_v7_r108_readypose"' in CONFIG_SOURCE
    assert "max_iterations = backhand_v6.MAX_ITERATIONS" in CONFIG_SOURCE

    assert "TASK=a1_tt_backhand_v7" in WATCHDOG_SOURCE
    assert "EXP=logs/a1_tt_backhand_real_v7_r108_readypose" in WATCHDOG_SOURCE
    assert "TARGET=${A1_TT_TARGET:-19999}" in WATCHDOG_SOURCE
    assert "--num_envs 4096 --headless --predictor" in WATCHDOG_SOURCE
    assert "pkill" not in WATCHDOG_SOURCE
    assert "rm -" not in WATCHDOG_SOURCE
