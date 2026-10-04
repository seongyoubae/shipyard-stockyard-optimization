import numpy as np

import copy
import math
import random
import time
import matplotlib.pyplot as plt

from stockyard.baselines.legacy.sa_move import *

class Annealer(object):
    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """
    # defaults
    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    def __init__(self, initial_state=None, piles_from=None, piles_to=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')

        self.piles_from = piles_from
        self.piles_to = piles_to

        self.energy_list = []
        self.time_list = []

    def move(self):
        prev_energy, prev_result = self.energy()
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

        self.state[a][1] = random.choice(list(self.piles_to.keys()))
        self.state[b][1] = random.choice(list(self.piles_to.keys()))
        E, result = self.energy()

        return E - prev_energy

    def energy(self):
        reversal, piles_to_copy = SA_move(self.state, self.piles_from, self.piles_to)

        return reversal, piles_to_copy

    def set_schedule(self, schedule):
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps = int(schedule['steps'])

    def copy_state(self, state):
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def anneal(self):
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E, result = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0

        while step < self.steps:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E, result = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E

            self.time_list.append(time.time() - start)
            self.energy_list.append(E)


        self.state = self.copy_state(self.best_state)

        return self.best_state, self.best_energy
