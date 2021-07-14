import yaml
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from torch.optim.adam import Adam

from scripts.dqn import nn
from scripts.dqn.env import WaterNetworkEnvironment
from scripts.epynetUtils import time_parameters_summary

config_file = 'anytown.yaml'

with open(config_file, 'r') as fin:
    hparams = yaml.safe_load(fin)


# Build the environment
env = WaterNetworkEnvironment(
    town=hparams['env']['town'],
    state_vars=hparams['env']['state_vars'],
    action_vars=hparams['env']['action_vars'],
    duration=hparams['env']['duration'],
    hyd_step=hparams['env']['hyd_step'],
    pattern_step=hparams['env']['pattern_step'],
    pattern_file=hparams['env']['pattern_file']
)

# Creating the epsilon greedy policy
epsilon = LinearParameter(value=1., threshold_value=.1, n=1000000)
# epsilon_test = Parameter(value=.05)
# epsilon_random = Parameter(value=1)
pi = EpsGreedy(epsilon=epsilon)


# Set parameters of approximator
nn_params = dict(
    input_shape = env.info.observation_space.n,
    output_shape = env.info.action_space.n,
    hidden_size = hparams['nn']['hidden_size']
)


# Create the approximator from the neural network we have implemented
approximator = LinearApproximator



# Build the agent
agent = DQN(
    mdp_info=env.info,
    policy=pi,
    approximator=,
    approximator_params=approximator,
    batch_size=hparams['agent']['batch_size'],
    target_update_frequency=hparams['agent']['target_update_frequency'],
    replay_memory=None,
    initial_replay_size=hparams['agent']['initial_replay_memory'],
    max_replay_size=hparams['agent']['max_replay_size'],
    fit_params=hparams['nn']['fit_params'],
    predict_params=hparams['nn']['predict_params'],
    clip_reward=False
)

core = Core(agent, env)




