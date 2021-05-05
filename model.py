# import ...
# This is just a prototype of how the RL algorithm class will look like.
# The environment will fit with the gym environment and the structure is similar to stable baselines 3 library.


class Model:
    def __init__(self, wds, env):
        self.wds = wds
        self.n_episodes = 10
        self.env = env

    def run(self):
        for i in range(self.n_episodes):
            cumulated_reward = 0
            # Here we only initialize the epanet simulation
            self.wds.initSimulation()
            obs = self.env.reset()

            curr_time = 0
            # First epanet simulation step (maybe can be predicted by RL)
            timestep = self.wds.simulate_step(curr_time)

            while timestep > 0:
                action, _states = self.predict(obs)
                # In the step function there will be also a step in the epanet simulation
                obs, reward, _, _ = self.step(action)
                self.wds.update_pumps_status()
                cumulated_reward += reward

    # ...something else...

    def step(self, action):
        obs, reward, done, info = 0
        return obs, reward, done, info

    def predict(self, obs):
        action, states = 0
        return action, states


