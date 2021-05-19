import scipy as sp
import numpy as np
from scripts import network
from scripts import objFunction


class DifferentialEvolution:

    def __init__(self, wds: network.WaterDistributionNetwork, obj_func, vars_list, n_pop, n_generations, CR, F):
        """
        Initializes differential evolution and simulation handling parameters
        :param wds: water distribution network
        :param obj_func: function to maximize
        """
        self.wds = wds
        self.obj_func = obj_func
        self.vars_list = vars_list
        self.n_pop = n_pop
        self.n_generations = n_generations
        self.CR = CR
        self.F = F

        self.duration = 0
        self.hyd_step = 0
        self.update_interval = 0
        self.n_updates = 0

        self.best_candidate = {'config': {}, 'value': 0}
        self.population = []
        self.best_population = []


    def populate(self):
        """
        Gets the population among which we want to apply differential evolution
        :return: list of lists with length equal to n_updates * n_actuators
        """
        # Floor division between duration and update_interval gives the number of update we will perform
        self.n_updates = self.duration // self.update_interval
        population = [np.random.randint(2, size=self.n_updates * len(self.vars_list)) for i in range(self.n_pop)]
        return population

    def run(self, duration=86400, hyd_step = 600, update_interval=14400, n_simulations=1, keep_same_pop=False):
        """
        :param duration:
        :param update_interval:
        :param n_simulations:
        :param keep_same_pop:
        :return:
        """
        self.duration = duration
        self.hyd_step = hyd_step
        self.update_interval = update_interval

        # If we want to run multiple serial simulations (for example, a week with one day duration each)
        if n_simulations > 1:
            for i in range(n_simulations):
                # Keep the last computed pop_dict from the day before
                if not keep_same_pop:
                    pop = self.populate()
                    self.__run(pop)
                else:
                    self.__run(self.best_population)
                    self.best_population = self.population
                # save_results()

        # Runs only one simulation (for example, just one day)
        else:
            pop = self.populate()
            self.__run(pop)

    def __run(self, pop):
        """
        Runs differential evolution algorithm
        :return:
        """
        self.wds.set_time_params(duration=self.duration, hydraulic_step=self.hyd_step)

        self.population = pop
        # while iteration < max_iterations:
        for i in range(self.n_generations):
            # for individual in population:
            print("> Genaration " + str(i) + ": ...")
            for k, target in enumerate(self.population):
                mutant = self.mutatation(k)
                candidate = self.recombination(target, mutant)
                if self.selection(target, candidate):
                    self.population[k] = candidate
                #TODO: DE stopping conditions
            print(">>> DONE")
            print("> Best generation member: " + str(self.best_candidate['config']))
            print("> Best generation value: " + str(self.best_candidate['value']))

        print("Best candidate: " + str(self.best_candidate['config']))
        print("Best candidate value: " + str(self.best_candidate['value']))

    def mutatation(self, k):
        """
        Compute the new mutant by using binary mutation as explained in Gong-Tuson paper on BDE.
        # TODO: check if iBDE mutation explained by Doerr-Zheng works better
        :param index:
        :return:
        """
        pop_range = [idx for idx in range(self.n_pop) if idx != k]
        idx_selection = np.random.choice(pop_range, 3, replace=False)
        [a, b, c] = [self.population[i] for i in idx_selection ]

        # Rand/1 strategy
        mutant = []
        for j in range(len(a)):
            rand_var = np.random.uniform(0, 1)
            if b[j] != c[j] and rand_var < 0.7:
                mutant.append(1 - a[j])
            else:
                mutant.append(a[j])
        return mutant

    def recombination(self, target, mutant):
        """
        Binomial recombination step. TODO: check with exponential
        :return:
        """
        # Binomial strategy
        cross_vars = np.random.rand(len(target)) <= self.CR
        candidate = np.where(cross_vars, mutant, target)
        return candidate

    def selection(self, target, candidate):
        """
        Selects which candidate is the better one after the evaluation phase
        :param target:
        :param candidate:
        :return:
        """
        parent = {}
        child = {}
        idx = 0
        # Builds candidates as dictionaries of pumps
        for pump_id in self.vars_list:
            parent[pump_id] = target[idx * self.n_updates : idx * self.n_updates + self.n_updates]
            child[pump_id] = candidate[idx * self.n_updates : idx * self.n_updates + self.n_updates]
            idx += 1

        parent_value = self.evaluation(parent)
        print("Parent: candidate = {}, value = {}".format(parent, parent_value))
        child_value = self.evaluation(child)
        print("Child: candidate = {}, value = {}".format(child, child_value))

        update_pop = False
        # We maximize the objective function
        if parent_value >= child_value:
            update_pop = True
            # Update the best candidate with the new candidate
            if child_value > self.best_candidate['value']:
                self.best_candidate['config'] = candidate
                self.best_candidate['value'] = child_value
        else:
            # Update the best candidate with the old target
            if parent_value > self.best_candidate['value']:
                self.best_candidate['config'] = target
                self.best_candidate['value'] = parent_value
        return update_pop

    def evaluation(self, candidate):
        """
        Wrapper method which runs the simulation and evaluation with the current candidate
        :param vars_list:
        :return:
        """
        self.simulate_episode(candidate)
        return self.obj_func(self.wds, self.vars_list)

    def simulate_episode(self, candidate):
        """
        Run a complete simulation of the entire duration to evaluate the current candidate
        :param candidate: dictionary of pumps with, as values, the list of their updates
        :return:
        """
        self.wds.init_simulation()
        curr_time = 0
        update_time = 0
        timestep = 1

        # Initialize status of pumps with the first element in the update list of the candidate
        init_status = {pump_id: candidate[pump_id][0] for pump_id in candidate.keys()}
        self.wds.update_actuators_status(init_status)
        print("Initial update")

        # timestep becomes 0 the last hydraulic step
        while timestep > 0:
            # Check if it's needed an update
            if update_time >= self.update_interval:
                update_index = curr_time // self.update_interval
                new_status = {pump_id: candidate[pump_id][update_index] for pump_id in candidate.keys()}
                # new status= {'PU1':1, 'PU2':0}
                self.wds.update_actuators_status(new_status)
                update_time -= self.update_interval
                print("-> update")

            # Changed simulate_step method since it doesn't update status by itself
            timestep, state = self.wds.simulate_step(curr_time=curr_time)
            curr_time += timestep
            update_time += timestep

        self.wds.ep.ENcloseH()
        # TODO: check if it is faster without reports and only with result series
        self.wds.create_df_reports()


if __name__ == '__main__':
    DE_params = {
        'vars_list': ['78', '79'],
        # 'vars_list': ['PU1', 'PU2'],
        'n_pop': 20,
        'n_generations': 5,
        'CR': 0.7,
        'F': 0.5
    }

    net = network.WaterDistributionNetwork("anytown.inp")
    de = DifferentialEvolution(net, objFunction.supply_demand_ratio, **DE_params)
    duration = 86400
    hyd_step = 600
    update_interval = 14400
    de.run(duration, hyd_step, update_interval)