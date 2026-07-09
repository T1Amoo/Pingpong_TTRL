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
task_registry.register("a1_tt", A1TTEnv, A1TableTennisEnvCfg(), A1TableTennisAgentCfg())
task_registry.register("a1_tt_real", A1TTEnv, A1TableTennisDeployEnvCfg(), A1TableTennisDeployAgentCfg())
task_registry.register("a1_tt_eval", A1TTEnv, A1TT_EvalEnvCfg(), A1TableTennisAgentCfg())
