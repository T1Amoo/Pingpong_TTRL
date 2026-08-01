import numpy as np
import mujoco
from pathlib import Path
import json
import os
import pytest
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a1_scene import (
    DEFAULT_SCENE_XML,
    PHYSICS_DT,
    PROFILES,
    RIGHT_ARM_EFFORT_LIMITS,
    RIGHT_ARM_JOINTS,
    ball_addresses,
    build_scene_xml,
    load_scene,
)
from ball_gate import BallGateOutput, BallValidityGate
from policy_io import PRED_SENTINEL
from policy_io import (
    A1PolicyIO,
    BRIDGE_MAX_DELTA_PER_TICK,
    DAMIAO_MIT_KD,
    DAMIAO_MIT_KP,
    DAMIAO_MIT_VEL,
    DEFAULT_POLICY,
    DEFAULT_RIGHT_Q,
    EFFORT,
    FittedSecondOrderActionResponse,
    OBS_SIZE,
    OnnxPolicy,
    REAL_RESPONSE_MAX_DELTA_PER_TICK,
    REAL_RESPONSE_DELAY_S,
    REAL_RESPONSE_FN_HZ,
)
from run_a1_tt_sim2sim import (
    action_for_gate,
    next_serve_step_after_inactive,
    parse_args,
    run,
    velocity_clip_limit_for_mode,
)
from serve import BALL_DRAG_K, BOUNCE_X, BOUNCE_VZ, SERVE_LAUNCH, SERVE_Y_CENTER, SERVE_Y_HALF, Serve


def test_scene_profiles_include_v1_3_backhand_geometry():
    from a1_scene import PROFILES, ACTIVE_PROFILE

    assert ACTIVE_PROFILE.name == "old_v1_1"
    assert "old_v1_1" in PROFILES
    assert "model_10999" in PROFILES
    assert "old_v1_1_backhand" in PROFILES
    assert "old_v1_1_backhand_2100" in PROFILES
    assert "v1_3_backhand" in PROFILES
    assert "v1_3_backhand_2100" in PROFILES
    assert "v1_3_backhand_low_arm_2100" in PROFILES
    assert "v1_3_backhand_low_arm_hitplane020" in PROFILES

    model_10999 = PROFILES["model_10999"]
    assert model_10999.urdf.name == "X1_URDF_V1_1_sj_fixed.urdf"
    np.testing.assert_allclose(model_10999.robot_table_pos, [-1.8, 0.76, 0.0282])
    assert model_10999.hit_plane_x == -1.43
    assert model_10999.home_y == 0.76
    assert model_10999.paddle_y_offset == -0.03
    np.testing.assert_allclose(model_10999.paddle_offset, [0.0, 0.0, 0.085])
    np.testing.assert_allclose(model_10999.pred_sentinel, [-1.43, 0.64, 1.11])
    np.testing.assert_allclose(
        [model_10999.default_qpos[j] for j in RIGHT_ARM_JOINTS],
        [1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043],
    )

    old_backhand = PROFILES["old_v1_1_backhand"]
    assert old_backhand.urdf.name == "X1_URDF_V1_1.urdf"
    np.testing.assert_allclose(old_backhand.robot_table_pos, [-1.8, 0.0, 0.0282])
    assert old_backhand.hit_plane_x == -1.159
    np.testing.assert_allclose(old_backhand.paddle_offset, [0.0, 0.0, 0.085])
    np.testing.assert_allclose(
        [old_backhand.default_qpos[j] for j in RIGHT_ARM_JOINTS],
        [1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043],
    )

    new_profile = PROFILES["v1_3_backhand"]
    assert new_profile.urdf.name == "X1_URDF_V1_3.urdf"
    assert "X1_URDF_V1_3" in str(new_profile.meshdir)
    np.testing.assert_allclose(new_profile.robot_table_pos, [-1.8, 0.0, 0.0282])
    assert new_profile.hit_plane_x == -1.159
    assert new_profile.home_y == 0.0
    assert new_profile.paddle_y_offset == -0.03
    np.testing.assert_allclose(new_profile.paddle_offset, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        [new_profile.default_qpos[j] for j in RIGHT_ARM_JOINTS],
        [1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043],
    )

    ckpt_2100 = PROFILES["v1_3_backhand_2100"]
    assert ckpt_2100.urdf.name == "X1_URDF_V1_3.urdf"
    assert ckpt_2100.hit_plane_x == -1.279
    np.testing.assert_allclose(ckpt_2100.paddle_offset, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(ckpt_2100.pred_sentinel, [-1.279, 0.066, 1.22])
    assert ckpt_2100.serve_bounce_x_range == (-1.089, -0.809)
    assert ckpt_2100.serve_bounce_vz_range == (1.95, 2.35)
    assert ckpt_2100.serve_y_center == 0.045


def test_v1_3_low_arm_2100_profile_matches_current_training_snapshot():
    from a1_scene import PROFILES

    low_arm = PROFILES["v1_3_backhand_low_arm_2100"]
    assert low_arm.urdf.name == "X1_URDF_V1_3.urdf"
    np.testing.assert_allclose(
        [low_arm.default_qpos[j] for j in RIGHT_ARM_JOINTS],
        [1.45, -0.762, -2.05, 1.445, 0.206, -0.827, 1.043],
    )
    assert low_arm.hit_plane_x == -1.203
    np.testing.assert_allclose(low_arm.paddle_offset, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(low_arm.pred_sentinel, [-1.203, 0.041, 0.92], atol=1e-7)
    assert low_arm.hit_target_y_range == (-0.025, 0.107)
    assert low_arm.hit_target_z_range == (0.86, 0.98)
    assert low_arm.serve_bounce_x_range == (-1.013, -0.733)
    assert low_arm.serve_bounce_vz_range == (0.0, 0.45)
    assert low_arm.serve_y_center == 0.041
    assert low_arm.serve_y_half == 0.02


def test_v1_3_profile_reaches_policy_io_at_import_time():
    env = os.environ.copy()
    env["A1_SIM2SIM_PROFILE"] = "v1_3_backhand"
    script = """
import json
import a1_scene
import policy_io
print(json.dumps({
    "profile": a1_scene.ACTIVE_PROFILE.name,
    "urdf": a1_scene.DEFAULT_URDF.name,
    "meshdir": str(a1_scene.DEFAULT_MESHDIR),
    "robot_table_pos": a1_scene.ROBOT_TABLE_POS.tolist(),
    "default_right_q": policy_io.DEFAULT_RIGHT_Q.tolist(),
    "hit_plane_x": policy_io.HIT_PLANE_X,
    "hit_target_y_range": list(policy_io.HIT_TARGET_Y_RANGE),
    "hit_target_z_range": list(policy_io.HIT_TARGET_Z_RANGE),
    "home_y": policy_io.HOME_Y,
    "paddle_y_offset": policy_io.PADDLE_Y_OFFSET,
    "paddle_offset": policy_io.PADDLE_OFFSET.tolist(),
    "pred_sentinel": policy_io.PRED_SENTINEL.tolist(),
}))
"""
    out = subprocess.check_output([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent, env=env)
    data = json.loads(out.decode())
    assert data["profile"] == "v1_3_backhand"
    assert data["urdf"] == "X1_URDF_V1_3.urdf"
    assert "X1_URDF_V1_3" in data["meshdir"]
    np.testing.assert_allclose(data["robot_table_pos"], [-1.8, 0.0, 0.0282])
    np.testing.assert_allclose(data["default_right_q"], [1.769, -0.762, -1.863, 1.445, 0.206, -0.827, 1.043])
    assert data["hit_plane_x"] == -1.159
    np.testing.assert_allclose(data["hit_target_y_range"], [0.00, 0.135])
    np.testing.assert_allclose(data["hit_target_z_range"], [1.08, 1.16])
    assert data["home_y"] == 0.0
    assert data["paddle_y_offset"] == -0.03
    np.testing.assert_allclose(data["paddle_offset"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(data["pred_sentinel"], [-1.159, 0.0675, 1.12], atol=1e-7)


def test_v1_3_2100_profile_reaches_policy_and_serve_at_import_time():
    env = os.environ.copy()
    env["A1_SIM2SIM_PROFILE"] = "v1_3_backhand_2100"
    script = """
import json
import a1_scene
import policy_io
import serve
print(json.dumps({
    "profile": a1_scene.ACTIVE_PROFILE.name,
    "hit_plane_x": policy_io.HIT_PLANE_X,
    "pred_sentinel": policy_io.PRED_SENTINEL.tolist(),
    "bounce_x": list(serve.BOUNCE_X),
    "bounce_vz": list(serve.BOUNCE_VZ),
    "serve_y_center": serve.SERVE_Y_CENTER,
}))
"""
    out = subprocess.check_output([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent, env=env)
    data = json.loads(out.decode())
    assert data["profile"] == "v1_3_backhand_2100"
    assert data["hit_plane_x"] == -1.279
    np.testing.assert_allclose(data["pred_sentinel"], [-1.279, 0.066, 1.22], atol=1e-7)
    np.testing.assert_allclose(data["bounce_x"], [-1.089, -0.809])
    np.testing.assert_allclose(data["bounce_vz"], [1.95, 2.35])
    assert data["serve_y_center"] == 0.045


def test_v1_3_low_arm_2100_profile_reaches_policy_and_serve_at_import_time():
    env = os.environ.copy()
    env["A1_SIM2SIM_PROFILE"] = "v1_3_backhand_low_arm_2100"
    script = """
import json
import a1_scene
import policy_io
import serve
print(json.dumps({
    "profile": a1_scene.ACTIVE_PROFILE.name,
    "default_right_q": policy_io.DEFAULT_RIGHT_Q.tolist(),
    "hit_plane_x": policy_io.HIT_PLANE_X,
    "hit_target_y_range": list(policy_io.HIT_TARGET_Y_RANGE),
    "hit_target_z_range": list(policy_io.HIT_TARGET_Z_RANGE),
    "pred_sentinel": policy_io.PRED_SENTINEL.tolist(),
    "bounce_x": list(serve.BOUNCE_X),
    "bounce_vz": list(serve.BOUNCE_VZ),
    "serve_y_center": serve.SERVE_Y_CENTER,
    "serve_y_half": serve.SERVE_Y_HALF,
}))
"""
    out = subprocess.check_output([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent, env=env)
    data = json.loads(out.decode())
    assert data["profile"] == "v1_3_backhand_low_arm_2100"
    np.testing.assert_allclose(data["default_right_q"], [1.45, -0.762, -2.05, 1.445, 0.206, -0.827, 1.043])
    assert data["hit_plane_x"] == -1.203
    np.testing.assert_allclose(data["hit_target_y_range"], [-0.025, 0.107])
    np.testing.assert_allclose(data["hit_target_z_range"], [0.86, 0.98])
    np.testing.assert_allclose(data["pred_sentinel"], [-1.203, 0.041, 0.92], atol=1e-7)
    np.testing.assert_allclose(data["bounce_x"], [-1.013, -0.733])
    np.testing.assert_allclose(data["bounce_vz"], [0.0, 0.45])
    assert data["serve_y_center"] == 0.041
    assert data["serve_y_half"] == 0.02


def test_v1_3_low_arm_hitplane020_profile_reaches_policy_and_serve_at_import_time():
    env = os.environ.copy()
    env["A1_SIM2SIM_PROFILE"] = "v1_3_backhand_low_arm_hitplane020"
    script = """
import json
import a1_scene
import policy_io
import serve
print(json.dumps({
    "profile": a1_scene.ACTIVE_PROFILE.name,
    "default_right_q": policy_io.DEFAULT_RIGHT_Q.tolist(),
    "hit_plane_x": policy_io.HIT_PLANE_X,
    "hit_target_y_range": list(policy_io.HIT_TARGET_Y_RANGE),
    "hit_target_z_range": list(policy_io.HIT_TARGET_Z_RANGE),
    "pred_sentinel": policy_io.PRED_SENTINEL.tolist(),
    "bounce_x": list(serve.BOUNCE_X),
    "bounce_vz": list(serve.BOUNCE_VZ),
    "serve_y_center": serve.SERVE_Y_CENTER,
    "serve_y_half": serve.SERVE_Y_HALF,
}))
"""
    out = subprocess.check_output([sys.executable, "-c", script], cwd=Path(__file__).resolve().parent, env=env)
    data = json.loads(out.decode())
    assert data["profile"] == "v1_3_backhand_low_arm_hitplane020"
    np.testing.assert_allclose(data["default_right_q"], [1.45, -0.762, -2.05, 1.445, 0.206, -0.827, 1.043])
    assert data["hit_plane_x"] == -1.243
    np.testing.assert_allclose(data["hit_target_y_range"], [-0.025, 0.107])
    np.testing.assert_allclose(data["hit_target_z_range"], [0.86, 0.98])
    np.testing.assert_allclose(data["pred_sentinel"], [-1.243, 0.041, 0.92], atol=1e-7)
    np.testing.assert_allclose(data["bounce_x"], [-1.053, -0.773])
    np.testing.assert_allclose(data["bounce_vz"], [0.0, 0.45])
    assert data["serve_y_center"] == 0.041
    assert data["serve_y_half"] == 0.02


def test_scene_loads_with_robot_table_and_ball(tmp_path):
    xml_path = build_scene_xml(out_path=tmp_path / "a1_tt_scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert {"a1_root", "Link_r7", "table", "ball"}.issubset(names)
    geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    assert "floor" in geom_names
    assert model.nlight >= 2
    for joint in RIGHT_ARM_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
    ball_addresses(model)


def test_generated_mjcf_matches_training_robot_collision_setup(tmp_path):
    xml_path = build_scene_xml(out_path=tmp_path / "a1_tt_scene.xml")
    root = ET.parse(xml_path).getroot()
    robot_root = root.find(".//body[@name='a1_root']")
    assert robot_root is not None

    paddle_count = 0
    for geom in robot_root.iter("geom"):
        if geom.get("name") == "paddle_blade":
            paddle_count += 1
            assert geom.get("contype") == "1"
            assert geom.get("conaffinity") == "1"
        else:
            assert geom.get("contype") == "0"
            assert geom.get("conaffinity") == "0"
    assert paddle_count == 1

    model, data, _ = load_scene(xml_path)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_low_arm_profile_uses_small_ball_contact_margin():
    assert PROFILES["v1_3_backhand_low_arm_2100"].ball_contact_margin == pytest.approx(0.015)


def test_generated_mjcf_has_motor_metadata(tmp_path):
    xml_path = build_scene_xml(out_path=tmp_path / "a1_tt_scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nu == len(RIGHT_ARM_JOINTS)
    for joint in RIGHT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = int(model.jnt_dofadr[jid])
        effort = RIGHT_ARM_EFFORT_LIMITS[joint]
        assert np.allclose(model.jnt_actfrcrange[jid], [-effort, effort])
        assert model.dof_armature[dof] > 0.0
        assert model.dof_damping[dof] > 0.0


def test_checked_in_default_mjcf_has_current_motor_metadata():
    root = ET.parse(DEFAULT_SCENE_XML).getroot()
    for joint in RIGHT_ARM_JOINTS:
        effort = RIGHT_ARM_EFFORT_LIMITS[joint]
        joint_node = root.find(f".//joint[@name='{joint}']")
        assert joint_node is not None
        assert joint_node.get("actuatorfrcrange") == f"{-effort:.12g} {effort:.12g}"
        motor_node = root.find(f".//motor[@name='{joint}_motor']")
        assert motor_node is not None
        assert motor_node.get("ctrlrange") == f"{-effort:.12g} {effort:.12g}"


def test_policy_obs_and_action_shape():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    obs = io.observe(np.array([1.0, 0.0, 1.0]), np.array([-3.0, 0.0, 1.0]))
    assert obs.shape == (OBS_SIZE,)
    if not DEFAULT_POLICY.exists():
        pytest.skip(f"default exported policy not found: {DEFAULT_POLICY}")
    policy = OnnxPolicy(DEFAULT_POLICY)
    action = policy(obs)
    assert action.shape == (7,)


def test_damiao_constants_use_0729_backhand_real_fit():
    np.testing.assert_allclose(DAMIAO_MIT_KP, [300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 60.0])
    np.testing.assert_allclose(DAMIAO_MIT_KD, [3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5])
    np.testing.assert_allclose(DAMIAO_MIT_VEL, [4.0, 4.0, 5.0, 6.0, 8.0, 6.0, 9.0])
    assert not np.allclose(REAL_RESPONSE_DELAY_S, np.full(7, 0.08))
    assert np.all(REAL_RESPONSE_DELAY_S >= 0.0)
    assert np.all(REAL_RESPONSE_FN_HZ > 0.0)


def test_invalid_ball_zero_action_gate_masks_policy_output():
    class FakePolicy:
        def __call__(self, obs):
            return np.arange(1, 8, dtype=np.float32)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    invalid = BallGateOutput(
        live=False,
        engaged=False,
        reason="paddle_hit",
        first_bounce_x=float("nan"),
        own_bounces=0,
        hit_seen=True,
    )
    action = action_for_gate(FakePolicy(), obs, invalid, invalid_ball_action="zero")
    np.testing.assert_allclose(action, np.zeros(7, dtype=np.float32))


def test_valid_ball_zero_action_gate_keeps_policy_output():
    class FakePolicy:
        def __call__(self, obs):
            return np.arange(1, 8, dtype=np.float32)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    valid = BallGateOutput(
        live=True,
        engaged=True,
        reason="valid",
        first_bounce_x=-0.9,
        own_bounces=1,
        hit_seen=False,
    )
    action = action_for_gate(FakePolicy(), obs, valid, invalid_ball_action="zero")
    np.testing.assert_allclose(action, np.arange(1, 8, dtype=np.float32))


def test_invalid_ball_policy_action_gate_preserves_legacy_policy_output():
    class FakePolicy:
        def __call__(self, obs):
            return np.arange(1, 8, dtype=np.float32)

    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    invalid = BallGateOutput(
        live=False,
        engaged=False,
        reason="out_volume",
        first_bounce_x=float("nan"),
        own_bounces=0,
        hit_seen=False,
    )
    action = action_for_gate(FakePolicy(), obs, invalid, invalid_ball_action="policy")
    np.testing.assert_allclose(action, np.arange(1, 8, dtype=np.float32))


def test_mjcf_motor_metadata_matches_training_damiao_effort_limits():
    np.testing.assert_allclose(
        [RIGHT_ARM_EFFORT_LIMITS[joint] for joint in RIGHT_ARM_JOINTS],
        [28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0],
    )


def test_damiao_mode_uses_real_velocity_clip_unless_explicitly_disabled():
    np.testing.assert_allclose(
        velocity_clip_limit_for_mode("damiao_mit", no_qvel_clip=False),
        DAMIAO_MIT_VEL,
    )
    assert velocity_clip_limit_for_mode("damiao_mit", no_qvel_clip=True) is None


def test_mit_pd_torque_is_clipped_to_motor_limits():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    io.update_action(np.array([100, -100, 100, -100, 100, -100, 100], dtype=np.float32))
    data.qfrc_applied[:] = 0.0
    tau = io.apply_mit_pd()
    assert np.all(np.abs(tau) <= EFFORT + 1e-6)
    assert np.isclose(np.max(np.abs(tau[:4])), 28.0)
    assert np.isclose(np.max(np.abs(tau[4:])), 8.0)


def test_bridge_qdes_delta_limit_matches_deploy_envelope():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    q_des = io.update_action(np.full(7, 100.0, dtype=np.float32), max_delta_per_tick=BRIDGE_MAX_DELTA_PER_TICK)
    assert np.allclose(q_des, DEFAULT_RIGHT_Q + BRIDGE_MAX_DELTA_PER_TICK)
    q_des = io.update_action(np.full(7, -100.0, dtype=np.float32), max_delta_per_tick=BRIDGE_MAX_DELTA_PER_TICK)
    assert np.allclose(q_des, DEFAULT_RIGHT_Q)


def test_real_response_model_smooths_limited_qdes_step():
    response = FittedSecondOrderActionResponse(PHYSICS_DT, DEFAULT_RIGHT_Q)
    command = DEFAULT_RIGHT_Q + REAL_RESPONSE_MAX_DELTA_PER_TICK
    prev = response.response.copy()
    max_step = 0.0
    for _ in range(10):
        current = response.step(command)
        max_step = max(max_step, float(np.max(np.abs(current - prev))))
        prev = current.copy()
    assert np.isfinite(response.response).all()
    assert max_step < float(np.max(REAL_RESPONSE_MAX_DELTA_PER_TICK))
    assert np.max(np.abs(response.response - DEFAULT_RIGHT_Q)) < float(np.max(REAL_RESPONSE_MAX_DELTA_PER_TICK))


def test_serve_bounce_sampler_ranges():
    s = Serve(rng=np.random.default_rng(0))
    for _ in range(200):
        pos, vel = s.sample()
        assert np.allclose(pos, [1.35, 0.0, 1.03])
        assert vel[0] < 0.0
        assert BOUNCE_VZ[0] <= vel[2] <= BOUNCE_VZ[1]
        # Reconstruct sampled bounce y from launch vy and time-to-bounce.
        t_b = (vel[2] + np.sqrt(vel[2] * vel[2] + 2.0 * 9.81 * (1.03 - 0.78))) / 9.81
        x_b = 1.35 + np.sign(vel[0]) * np.log1p(BALL_DRAG_K * abs(vel[0]) * t_b) / BALL_DRAG_K
        y_b = np.sign(vel[1]) * np.log1p(BALL_DRAG_K * abs(vel[1]) * t_b) / BALL_DRAG_K
        assert BOUNCE_X[0] <= x_b <= BOUNCE_X[1]
        assert SERVE_Y_CENTER - SERVE_Y_HALF <= y_b <= SERVE_Y_CENTER + SERVE_Y_HALF


def test_serve_sampler_inverts_horizontal_drag_like_isaac_training():
    s = Serve(rng=np.random.default_rng(0))
    for _ in range(200):
        pos, vel = s.sample()
        t_b = (vel[2] + np.sqrt(vel[2] * vel[2] + 2.0 * 9.81 * (SERVE_LAUNCH[2] - 0.78))) / 9.81

        def drag_displacement(v0: float, duration: float) -> float:
            return np.sign(v0) * np.log1p(BALL_DRAG_K * abs(v0) * duration) / BALL_DRAG_K

        x_b = float(pos[0] + drag_displacement(float(vel[0]), float(t_b)))
        y_b = float(pos[1] + drag_displacement(float(vel[1]), float(t_b)))
        assert BOUNCE_X[0] <= x_b <= BOUNCE_X[1]
        assert SERVE_Y_CENTER - SERVE_Y_HALF <= y_b <= SERVE_Y_CENTER + SERVE_Y_HALF

        no_drag_x = float(pos[0] + vel[0] * t_b)
        assert not np.isclose(no_drag_x, x_b, atol=1e-4)


def test_ball_gate_accepts_valid_serve_and_rejects_invalid_ball():
    gate = BallValidityGate()
    pos, vel = Serve(rng=np.random.default_rng(1)).sample()
    out = gate.update(pos, vel)
    assert out.engaged
    assert out.reason == "valid"
    assert -1.37 <= out.first_bounce_x <= 0.0

    out = gate.update(np.array([-1.66, 0.0, 1.0]), np.array([-2.0, 0.0, 0.0]))
    assert not out.engaged
    assert out.reason == "behind_hit_plane"


def test_invalid_ball_observation_uses_home_sentinel():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    obs = io.observe(np.array([-1.65, 0.3, 0.6]), np.array([1.0, 0.0, 0.0]), valid_ball=False)
    newest = obs[-39:]
    assert np.allclose(newest[27:30], PRED_SENTINEL)
    assert np.allclose(newest[33:36], PRED_SENTINEL)


def test_prediction_is_projected_to_profile_hit_target_geometry():
    from policy_io import (
        HIT_PLANE_X,
        HIT_TARGET_Y_RANGE,
        HIT_TARGET_Z_RANGE,
        project_prediction_to_target_geometry,
    )

    pred = project_prediction_to_target_geometry(np.array([-2.0, 10.0, -1.0], dtype=np.float32))

    assert pred[0] == HIT_PLANE_X
    assert pred[1] == HIT_TARGET_Y_RANGE[1]
    assert pred[2] == HIT_TARGET_Z_RANGE[0]


def test_short_headless_rollout_without_policy():
    args = parse_args([])
    args.headless_steps = 5
    args.no_policy = True
    stats = run(args)
    assert stats["control_steps"] == 5
    assert stats["steps"] == 50
    assert stats["max_abs_tau"] <= 28.0 + 1e-6


def test_dead_ball_schedules_next_serve_without_long_no_ball_gap():
    assert next_serve_step_after_inactive(control_step=14, pause_steps=0) == 15
    assert next_serve_step_after_inactive(control_step=14, pause_steps=1) == 15
    assert next_serve_step_after_inactive(control_step=14, pause_steps=12) == 26


def test_damiao_clip_effort_torque_speed_envelope():
    from policy_io import damiao_clip_effort
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    effort_limit = np.array([28, 28, 28, 8, 8, 8, 8], dtype=np.float64)
    brake = effort_limit.copy()

    # 静止(dq=0): 两侧都用 brake_limit=effort_limit
    tau = np.array([100, -100, 0, 100, -100, 0, 5], dtype=np.float64)
    dq0 = np.zeros(7, dtype=np.float64)
    out = damiao_clip_effort(tau, dq0, vel_limit, effort_limit, brake)
    np.testing.assert_allclose(out, np.array([28, -28, 0, 8, -8, 0, 5], dtype=np.float64))

    # 正速度且加速方向(tau>0): 受 speed_scale 限制; 反向(刹车)用 brake
    # r1: dq=4, vel_limit=8 -> speed_scale=0.5 -> accel_limit=14; 加速 tau=100 -> 14
    # r2: dq=4 -> 反向 tau=-100 -> -brake=-28
    dq = np.array([4, 4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau2 = np.array([100, -100, 0, 0, 0, 0, 0], dtype=np.float64)
    out2 = damiao_clip_effort(tau2, dq, vel_limit, effort_limit, brake)
    assert out2[0] == 14.0
    assert out2[1] == -28.0

    # 负速度: 加速方向是负, 正向是刹车
    # r1: dq=-4 -> speed_scale=0.5 -> accel_limit=14; tau=-100(加速) -> -14;
    dq3 = np.array([-4, -4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau3 = np.array([-100, 100, 0, 0, 0, 0, 0], dtype=np.float64)
    out3 = damiao_clip_effort(tau3, dq3, vel_limit, effort_limit, brake)
    assert out3[0] == -14.0   # 加速方向受限
    assert out3[1] == 28.0    # 正向刹车用 brake


def test_damiao_slew_limits_per_step_delta():
    from policy_io import damiao_slew
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    dt = 0.002
    cmd = np.zeros(7, dtype=np.float64)
    # 目标远大于 max_delta=vel_limit*dt=[0.016,...,0.04,...]
    q_des = np.ones(7, dtype=np.float64)
    out = damiao_slew(cmd, q_des, vel_limit, dt)
    np.testing.assert_allclose(out[:3], 0.016)   # 8*0.002
    np.testing.assert_allclose(out[3:], 0.040)   # 20*0.002
    # 目标在步长内则直达
    cmd2 = np.zeros(7, dtype=np.float64)
    q_des2 = np.full(7, 0.001, dtype=np.float64)
    out2 = damiao_slew(cmd2, q_des2, vel_limit, dt)
    np.testing.assert_allclose(out2, 0.001)


def test_apply_damiao_mit_torque_signs_and_limit(tmp_path):
    # 用现有 scene 构建一个 io，验证力矩方向与受限
    import mujoco
    from a1_scene import build_scene_xml, load_scene
    from policy_io import A1PolicyIO, DAMIAO_MIT_EFFORT
    # Adaptation: load_scene returns (model, data, path); use use_predictor=False
    # instead of non-existent policy= kwarg.
    model, data, _ = load_scene(build_scene_xml(out_path=tmp_path / "a1_tt_scene.xml"))
    io = A1PolicyIO(model, data, use_predictor=False)
    q = io.right_q()
    # 目标设在当前位置 +0.5rad(远超步长)，期望力矩为正且不超 effort_limit
    io.q_des = (q + 0.5).astype(np.float64)
    io.set_motor_q_des(io.q_des)
    io.damiao_cmd = q.copy()
    tau = io.apply_damiao_mit(0.002)
    assert tau.shape == (7,)
    assert np.all(tau >= -DAMIAO_MIT_EFFORT - 1e-9)
    assert np.all(tau <= DAMIAO_MIT_EFFORT + 1e-9)
    assert tau[0] > 0.0   # 目标在正方向 -> 正力矩
    # qfrc_applied 已写入
    assert np.allclose(data.qfrc_applied[io.dof_addr], tau)


def test_apply_damiao_mit_uses_response_velocity_in_damping(tmp_path):
    from a1_scene import build_scene_xml, load_scene
    from policy_io import A1PolicyIO, DAMIAO_MIT_EFFORT, DAMIAO_MIT_KD

    model, data, _ = load_scene(build_scene_xml(out_path=tmp_path / "a1_tt_scene.xml"))
    io = A1PolicyIO(model, data, use_predictor=False)
    q = io.right_q()
    io.q_des = q.copy()
    io.set_motor_q_des(q)
    io.damiao_cmd = q.copy()

    desired_vel = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    tau = io.apply_damiao_mit(0.002, desired_vel=desired_vel)

    expected = np.clip(DAMIAO_MIT_KD * desired_vel, -DAMIAO_MIT_EFFORT, DAMIAO_MIT_EFFORT)
    np.testing.assert_allclose(tau, expected, atol=1e-9)
