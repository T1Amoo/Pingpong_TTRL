# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.legged_env import LeggedEnv
from legged_lab.envs.base.tt_env import TTEnv

from legged_lab.envs.t1_tt.t1_tt_config import (
    T1TableTennisEnvCfg,
    T1TableTennisAgentCfg,
    T1TT_EvalEnvCfg,
)


from legged_lab.envs.g1_tt.g1_tt_config import (
    G1TableTennisEnvCfg,
    G1TableTennisAgentCfg,
    G1TT_EvalEnvCfg,
    G1TT_EvalHardEnvCfg,
    G1TableTennisDREnvCfg,
    G1TableTennisDRAgentCfg,
)

from legged_lab.envs.g1_locomotion.g1_locomotion_config import (
    G1LocomotionEnvCfg,
    G1LocomotionAgentCfg,
)


from legged_lab.envs.a1_tt.a1_tt_config import (
    A1TableTennisEnvCfg,
    A1TableTennisAgentCfg,
    A1TableTennisDeployEnvCfg,
    A1TableTennisDeployAgentCfg,
    A1TableTennisTorqueOnlyEnvCfg,
    A1TableTennisTorqueOnlyAgentCfg,
    A1TableTennisTorqueLowpassEnvCfg,
    A1TableTennisTorqueLowpassAgentCfg,
    A1TableTennisV13TestEnvCfg,
    A1TableTennisV13TestAgentCfg,
    A1TableTennisV9EnvCfg,
    A1TableTennisV9AgentCfg,
    A1TableTennisV10EnvCfg,
    A1TableTennisV10AgentCfg,
    A1TableTennisV11EnvCfg,
    A1TableTennisV11AgentCfg,
    A1TableTennisV12EnvCfg,
    A1TableTennisV12AgentCfg,
    A1TableTennisV13EnvCfg,
    A1TableTennisV13AgentCfg,
    A1TableTennisV14EnvCfg,
    A1TableTennisV14AgentCfg,
    A1TableTennisBackhandEnvCfg,
    A1TableTennisBackhandEvalEnvCfg,
    A1TableTennisBackhandAgentCfg,
    A1TableTennisBackhandV2EnvCfg,
    A1TableTennisBackhandV2EvalEnvCfg,
    A1TableTennisBackhandV2AgentCfg,
    A1TableTennisBackhandV3EnvCfg,
    A1TableTennisBackhandV3EvalEnvCfg,
    A1TableTennisBackhandV3AgentCfg,
    A1TableTennisBackhandV4EnvCfg,
    A1TableTennisBackhandV4EvalEnvCfg,
    A1TableTennisBackhandV4AgentCfg,
    A1TableTennisBackhandV5EnvCfg,
    A1TableTennisBackhandV5EvalEnvCfg,
    A1TableTennisBackhandV5AgentCfg,
    A1TableTennisBackhandV6EnvCfg,
    A1TableTennisBackhandV6EvalEnvCfg,
    A1TableTennisBackhandV6AgentCfg,
    A1TableTennisBackhandV7EnvCfg,
    A1TableTennisBackhandV7EvalEnvCfg,
    A1TableTennisBackhandV7AgentCfg,
    A1TableTennisBackhandV8EnvCfg,
    A1TableTennisBackhandV8EvalEnvCfg,
    A1TableTennisBackhandV8AgentCfg,
    A1TableTennisBackhandV9EnvCfg,
    A1TableTennisBackhandV9EvalEnvCfg,
    A1TableTennisBackhandV9AgentCfg,
    A1TableTennisDamiaoEnvCfg,
    A1TableTennisDamiaoAgentCfg,
    A1TableTennisOpenArmEnvCfg,
    A1TableTennisOpenArmAgentCfg,
    A1TT_EvalEnvCfg,
)
from legged_lab.envs.a1_tt.a1_tt_env import A1TTEnv

from legged_lab.utils.task_registry import task_registry
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg()) #TTEnv
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg())
task_registry.register("g1_tt", TTEnv, G1TableTennisEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_eval", TTEnv, G1TT_EvalEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_eval_hard", TTEnv, G1TT_EvalHardEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_dr", TTEnv, G1TableTennisDREnvCfg(), G1TableTennisDRAgentCfg())
task_registry.register("g1_locomotion", LeggedEnv, G1LocomotionEnvCfg(), G1LocomotionAgentCfg())
# Unversioned A1 task names always mean the current backhand route.  Historical
# forehand behavior remains available only through explicit a1_tt_v9..v14 names.
task_registry.register("a1_tt", A1TTEnv, A1TableTennisBackhandEnvCfg(), A1TableTennisBackhandAgentCfg())
task_registry.register("a1_tt_real", A1TTEnv, A1TableTennisBackhandEnvCfg(), A1TableTennisBackhandAgentCfg())
task_registry.register("a1_tt_torque_only", A1TTEnv, A1TableTennisTorqueOnlyEnvCfg(), A1TableTennisTorqueOnlyAgentCfg())
task_registry.register("a1_tt_real_lowpass", A1TTEnv, A1TableTennisTorqueLowpassEnvCfg(), A1TableTennisTorqueLowpassAgentCfg())
task_registry.register("a1_tt_v13_test", A1TTEnv, A1TableTennisV13TestEnvCfg(), A1TableTennisV13TestAgentCfg())
task_registry.register("a1_tt_v9", A1TTEnv, A1TableTennisV9EnvCfg(), A1TableTennisV9AgentCfg())
task_registry.register("a1_tt_v10", A1TTEnv, A1TableTennisV10EnvCfg(), A1TableTennisV10AgentCfg())
task_registry.register("a1_tt_v11", A1TTEnv, A1TableTennisV11EnvCfg(), A1TableTennisV11AgentCfg())
task_registry.register("a1_tt_v12", A1TTEnv, A1TableTennisV12EnvCfg(), A1TableTennisV12AgentCfg())
task_registry.register("a1_tt_v13", A1TTEnv, A1TableTennisV13EnvCfg(), A1TableTennisV13AgentCfg())
task_registry.register("a1_tt_v14", A1TTEnv, A1TableTennisV14EnvCfg(), A1TableTennisV14AgentCfg())
task_registry.register(
    "a1_tt_backhand",
    A1TTEnv,
    A1TableTennisBackhandEnvCfg(),
    A1TableTennisBackhandAgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v2",
    A1TTEnv,
    A1TableTennisBackhandV2EnvCfg(),
    A1TableTennisBackhandV2AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v2_eval",
    A1TTEnv,
    A1TableTennisBackhandV2EvalEnvCfg(),
    A1TableTennisBackhandV2AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v3",
    A1TTEnv,
    A1TableTennisBackhandV3EnvCfg(),
    A1TableTennisBackhandV3AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v3_eval",
    A1TTEnv,
    A1TableTennisBackhandV3EvalEnvCfg(),
    A1TableTennisBackhandV3AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v4",
    A1TTEnv,
    A1TableTennisBackhandV4EnvCfg(),
    A1TableTennisBackhandV4AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v4_eval",
    A1TTEnv,
    A1TableTennisBackhandV4EvalEnvCfg(),
    A1TableTennisBackhandV4AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v5",
    A1TTEnv,
    A1TableTennisBackhandV5EnvCfg(),
    A1TableTennisBackhandV5AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v5_eval",
    A1TTEnv,
    A1TableTennisBackhandV5EvalEnvCfg(),
    A1TableTennisBackhandV5AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v6",
    A1TTEnv,
    A1TableTennisBackhandV6EnvCfg(),
    A1TableTennisBackhandV6AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v6_eval",
    A1TTEnv,
    A1TableTennisBackhandV6EvalEnvCfg(),
    A1TableTennisBackhandV6AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v7",
    A1TTEnv,
    A1TableTennisBackhandV7EnvCfg(),
    A1TableTennisBackhandV7AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v7_eval",
    A1TTEnv,
    A1TableTennisBackhandV7EvalEnvCfg(),
    A1TableTennisBackhandV7AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v8",
    A1TTEnv,
    A1TableTennisBackhandV8EnvCfg(),
    A1TableTennisBackhandV8AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v8_eval",
    A1TTEnv,
    A1TableTennisBackhandV8EvalEnvCfg(),
    A1TableTennisBackhandV8AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v9",
    A1TTEnv,
    A1TableTennisBackhandV9EnvCfg(),
    A1TableTennisBackhandV9AgentCfg(),
)
task_registry.register(
    "a1_tt_backhand_v9_eval",
    A1TTEnv,
    A1TableTennisBackhandV9EvalEnvCfg(),
    A1TableTennisBackhandV9AgentCfg(),
)
task_registry.register("a1_tt_damiao", A1TTEnv, A1TableTennisDamiaoEnvCfg(), A1TableTennisDamiaoAgentCfg())
task_registry.register("a1_tt_openarm", A1TTEnv, A1TableTennisOpenArmEnvCfg(), A1TableTennisOpenArmAgentCfg())
task_registry.register("a1_tt_eval", A1TTEnv, A1TableTennisBackhandEvalEnvCfg(), A1TableTennisBackhandAgentCfg())
