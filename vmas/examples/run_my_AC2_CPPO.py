#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time
from typing import Type

import torch

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from agent_A2C_CPPO_v1 import *


def run_heuristic(
    scenario_name: str = "transport",
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = {},
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
    batch_size: int=10
):

    assert not (save_render and not render), "To save the video you have to render it"

    # Scenario specific variables
    policy = heuristic(continuous_action=True)

    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )
    cfg = get_args()
    n_states = env.observation_space[0].shape[0]
    n_actions =  env.action_space[0].shape[0]
    agent = Agent(n_states, n_actions, cfg,len(env.observation_space))
    train(cfg,env,agent,n_steps)
    

if __name__ == "__main__":
    from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic


    run_heuristic(
        scenario_name="balance",
        heuristic=TransportHeuristic,
        n_envs=3000,
        n_steps=200,
        render=False,
        save_render=False,
        batch_size = 500,
        device="cuda"
    )