#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
from typing import Dict, Optional

import numpy as np
import ray
import wandb
from ray import tune
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

from vmas import make_env, Wrapper

scenario_name = "transport"

# Scenario specific variables.
# When modifying this also modify env_config and env_creator
n_agents = 2

# Common variables
continuous_actions = True
max_steps = 200
num_vectorized_envs = 96
num_workers = 1
vmas_device = "cuda"  # or cuda


def env_creator(config: Dict):
    env = make_env(
        scenario_name=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env


if not ray.is_initialized():
    ray.init()
    print("Ray init!")
register_env(scenario_name, lambda config: env_creator(config))


class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                try:
                    episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                except KeyError:
                    episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()


class RenderingCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        vid = np.transpose(self.frames, (0, 3, 1, 2))
        episode.media["rendering"] = wandb.Video(
            vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
        )
        self.frames = []

import time
from typing import Type

import torch

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video
from ray import tune
import ray
"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

import argparse
import numpy as np
from gym.spaces import Discrete
import os

import ray
from ray import air, tune

from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    PPOTF1Policy,
    PPOTF2Policy,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import (
    CentralizedCriticModel,
    TorchModelV2,
    ModelV2,
    TorchFC,
    SlimFC
    #TorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=7.99, help="Reward at which we stop training."
)


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function
            )
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (
        not pytorch and policy.loss_initialized()
    ):
        assert other_agent_batches is not None
        if policy.config["enable_connectors"]:
            [(_, _, opponent_batch)] = list(other_agent_batches.values())
        else:
            [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # f = open('log.txt','w')
        # f.write("obs", obs.shape)
        # f.write("opponent_obs", sample_batch[0].shape)
        # f.write("opponent_actions", opponent_batch[0].shape)
        # f.close()
        # overwrite default VF prediction with the central VF
        if args.framework == "torch":
            sample_batch[SampleBatch.VF_PREDS] = (
                policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch[SampleBatch.CUR_OBS], policy.device
                    ),
                    convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
                    convert_to_torch_tensor(
                        sample_batch[OPPONENT_ACTION], policy.device
                    ),
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            sample_batch[SampleBatch.VF_PREDS] = convert_to_numpy(
                policy.compute_central_vf(
                    sample_batch[SampleBatch.CUR_OBS],
                    sample_batch[OPPONENT_OBS],
                    sample_batch[OPPONENT_ACTION],
                )
            )
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    class CCPPOTFPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            # Use super() to get to the base PPO policy.
            # This special loss function utilizes a shared
            # value function defined on self, and the loss function
            # defined on PPO policies.
            return loss_with_central_critic(
                self, super(), model, dist_class, train_batch
            )

        @override(base)
        def postprocess_trajectory(
            self, sample_batch, other_agent_batches=None, episode=None
        ):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def stats_fn(self, train_batch: SampleBatch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTFPolicy


CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )


class CentralizedCritic(PPO):
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        if config["framework"] == "torch":
            return CCPPOTorchPolicy
        elif config["framework"] == "tf":
            return CCPPOStaticGraphTFPolicy
        else:
            return CCPPOEagerTFPolicy

class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Base of the model
        self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        input_size = 22  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 64, activation_fn=nn.Tanh),
            SlimFC(64, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        # f = open('log.txt','w')
        # f.write("obs", obs.shape)
        # f.write("opponent_obs", opponent_obs.shape)
        # f.write("opponent_actions", opponent_actions.shape)
        # f.close()
        input_ = torch.cat(
            [
                obs, # obs.shape=32 * 11 
                opponent_obs

                # torch.nn.functional.one_hot(opponent_actions.long(), 2).float(),
            ],
            1, # opponent_obs.shape = 32 * 11
        )
        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

def train():

    RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    num_gpus = 0.001 if RLLIB_NUM_GPUS > 0 else 0  # Driver GPU
    num_gpus_per_worker = (
        (RLLIB_NUM_GPUS - num_gpus) / (num_workers + 1) if vmas_device == "cuda" else 0
    )


    # ray.init()
    args = parser.parse_args()

    ModelCatalog.register_custom_model(
        "cc_model",
        TorchCentralizedCriticModel
        if args.framework == "torch"
        else CentralizedCriticModel,
    )


    env = make_env(
        
                device= vmas_device,
                num_envs= num_vectorized_envs,
                scenario_name= scenario_name,
                continuous_actions= continuous_actions,
                max_steps= max_steps,
                
    )
    
    config = (
        PPOConfig()
        .environment(scenario_name)
        .framework(args.framework)
        .rollouts(batch_mode="complete_episodes", num_rollout_workers=0)
        .training(model={"custom_model": "cc_model"})
        .multi_agent(
            policies={
                "pol1": (
                    None,
                    env.observation_space[0],
                    env.action_space[0],
                    # `framework` would also be ok here.
                    {}#PPOConfig.to_dict()#.overrides(framework_str=args.framework),
                ),
                "pol2": (
                    None,
                    env.observation_space[1],
                    env.action_space[1],
                    # `framework` would also be ok here.
                    {}#PPOConfig.to_dict()#.overrides(framework_str=args.framework),
                ),
                # "pol3": (
                #     None,
                #     env.observation_space[2],
                #     env.action_space[2],
                #     # `framework` would also be ok here.
                #     {}#PPOConfig.to_dict()#.overrides(framework_str=args.framework),
                # ),
                # "pol4": (
                #     None,
                #     env.observation_space[3],
                #     env.action_space[3],
                #     # `framework` would also be ok here.
                #     {}#PPOConfig#.overrides(framework_str=args.framework),
                # ),
            },
            policy_mapping_fn=lambda agent_id, **kwargs: "pol1"
            if agent_id == 0
            else "pol2" ,
            # policy_mapping_fn=lambda agent_id, **kwargs: "pol1"
            # if agent_id == 0
            # else "pol2" if agent_id == 1 else "pol3" if agent_id == 2 else "pol4",
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    config2={
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 60000,
            "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096,
            "num_sgd_iter": 40,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": vmas_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                # Scenario specific variables
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },
            "evaluation_interval": 5,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks([RenderingCallbacks, EvaluationCallbacks]),
            },
            "callbacks": EvaluationCallbacks,
        }
    # print("original:",config.to_dict())
    config = config.to_dict()
    
    config.update(config2)
    
    # print("\n\n\nupdated:",config)
    # os.system('pause')
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    tuner = tune.Tuner(
        CentralizedCritic,
        param_space= config,
        run_config=air.RunConfig(stop=stop, verbose=1),
    )
    results = tuner.fit()

    
if __name__ == "__main__":
    train()
