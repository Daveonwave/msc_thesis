import yaml
from mushroom_rl.core import Core
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.callbacks import CollectDataset, CollectMaxQ
from torch.optim.adam import Adam
from torch.nn import functional as F
import matplotlib.pyplot as plt

from scripts.dqn import nn
from scripts.dqn.env import WaterNetworkEnvironment
from scripts.dqn.logger import InfoLogger

results_path = '../../results/DQN'
config_file = 'anytown.yaml'

logger = InfoLogger(config_file[:-5], results_path)

config_file = 'anytown.yaml'

with open(config_file, 'r') as fin:
    hparams = yaml.safe_load(fin)


class DQNAgent(DQN):
    """

    """
    def __init__(self):
        self.env = WaterNetworkEnvironment(
            town=hparams['env']['town'],
            state_vars=hparams['env']['state_vars'],
            action_vars=hparams['env']['action_vars'],
            duration=hparams['env']['duration'],
            hyd_step=hparams['env']['hyd_step'],
            pattern_step=hparams['env']['pattern_step'],
            pattern_file=hparams['env']['pattern_file'],
            update_every=hparams['env']['update_every'],
            bounds=hparams['env']['bounds'],
            logger=logger,
            show_plot=None
        )

        # Creating the epsilon greedy policy
        self.epsilon_train = LinearParameter(value=1., threshold_value=.1, n=1000000)
        self.epsilon_test = Parameter(value=.05)
        self.epsilon_random = Parameter(value=1)
        self.pi = EpsGreedy(epsilon=self.epsilon_random)

        # Create the optimizer dictionary
        self.optimizer = dict()
        self.optimizer['class'] = Adam
        self.optimizer['params'] = hparams['optimizer']

        # Set parameters of neural network taken by the torch approximator
        nn_params = dict(hidden_size=hparams['nn']['hidden_size'])

        # Create the approximator from the neural network we have implemented
        approximator = TorchApproximator

        # Set parameters of approximator
        approximator_params = dict(
            network=nn.NN10Layers,
            input_shape=self.env.info.observation_space.shape,
            output_shape=(self.env.info.action_space.n,),
            n_actions=self.env.info.action_space.n,
            optimizer=self.optimizer,
            loss=F.smooth_l1_loss,
            batch_size=0,
            use_cuda=True,
            **nn_params
        )

        # Build replay buffer
        self.replay_buffer = ReplayMemory(initial_size=hparams['agent']['initial_replay_memory'],
                                          max_size=hparams['agent']['max_replay_size'])

        super().__init__(mdp_info=self.env.info,
                         policy=self.pi,
                         approximator=approximator,
                         approximator_params=approximator_params,
                         batch_size=hparams['agent']['batch_size'],
                         target_update_frequency=hparams['agent']['target_update_frequency'],
                         replay_memory=self.replay_buffer,
                         initial_replay_size=hparams['agent']['initial_replay_memory'],
                         max_replay_size=hparams['agent']['max_replay_size']
                         )

        # Callbacks
        dataset = CollectDataset()
        self.core = Core(self, self.env, callbacks_fit=[dataset])

        self.scores = []

    def fill_replay_buffer(self):
        self.pi.set_epsilon(self.epsilon_random)

        if self.replay_buffer.size < hparams['agent']['initial_replay_memory']:
            # Fill replay memory with random data
            self.core.learn(n_steps=hparams['agent']['initial_replay_memory'] - self.replay_buffer.size,
                            n_steps_per_fit=hparams['agent']['initial_replay_memory'], render=False)

    def learn(self):
        self.env.on_eval = False
        self.pi.set_epsilon(dqn.epsilon_train)
        logger.training_phase()
        self.core.learn(n_episodes=hparams['learning']['train_episodes'],
                        n_steps_per_fit=hparams['learning']['train_frequency'],
                        render=False)
        logger.end_phase()

    def evaluate(self):
        self.env.on_eval = True
        self.pi.set_epsilon(self.epsilon_test)
        logger.evaluation_phase()
        dataset = self.core.evaluate(n_episodes=1, render=True)
        self.scores.append(logger.get_stats(dataset))
        logger.end_phase()


if __name__ == '__main__':
    dqn = DQNAgent()

    n_epochs = hparams['learning']['epochs']
    dqn.fill_replay_buffer()

    for epoch in range(1, n_epochs + 1):
        dqn.learn()
        dqn.evaluate()
