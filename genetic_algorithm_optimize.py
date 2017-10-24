#!/usr/bin/python

"""
Copyright (C) 2017, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: genetic_algorithm_optimize.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a genetic algorithm to search the parameter space. Data
from all subjects is pooled such that a single set of optimal parameters is
estimated.
"""

from __future__ import division, absolute_import

import argparse
import numpy as np
import os
import random
import sys

from builtins import range, str, zip
from deap import base, creator, tools
from multiprocessing import Pool

from addm_toolbox.addm import aDDM
from addm_toolbox.util import load_data_from_csv, convert_item_values


# Global variables.
dataTrials = []


def evaluate(individual):
    """
    Computes the negative log likelihood of the global data set given the
    parameters of the aDDM.
    Args:
      individual: list containing the 3 model parameters, in the following
          order: d, theta, sigma.
    Returns:
      A list containing the negative log likelihood for the global data set and
          the given model.
    """
    d = individual[0]
    theta = individual[1]
    sigma = individual[2]
    model = aDDM(d, sigma, theta) 

    logLikelihood = 0
    for trial in dataTrials:
        try:
            likelihood = model.get_trial_likelihood(trial)
        except:
            print(u"An exception occurred during the likelihood " +
                  "computations for model " + str(model.params) + u".")
            raise
        if likelihood != 0:
            logLikelihood += np.log(likelihood)

    print(u"NLL for " + str(individual) + u": " + str(-logLikelihood))
    if logLikelihood != 0:
        return -logLikelihood,
    else:
        return sys.maxint,


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(u"--subject-ids", nargs=u"+", type=str, default=[],
                        help=u"List of subject ids. If not provided, all "
                        "existing subjects will be used.")
    parser.add_argument(u"--num-threads", type=int, default=9,
                        help=u"Size of the thread pool.")
    parser.add_argument(u"--trials-per-subject", type=int, default=100,
                        help=u"Number of trials from each subject to be used "
                        "in the analysis; if smaller than 1, all trials are "
                        "used.")
    parser.add_argument(u"--pop-size", type=int, default=18,
                        help=u"Number of individuals in each population.")
    parser.add_argument(u"--num-generations", type=int, default=20,
                        help=u"Number of generations.")
    parser.add_argument(u"--crossover-rate", type=float, default=0.5,
                        help=u"Crossover rate.")
    parser.add_argument(u"--mutation-rate", type=float, default=0.3,
                        help=u"Mutation rate.")
    parser.add_argument(u"--lower-bound-d", type=float, default=0.0001,
                        help=u"Lower search bound for parameter d.")
    parser.add_argument(u"--upper-bound-d", type=float, default=0.01,
                        help=u"Upper search bound for parameter d.")
    parser.add_argument(u"--lower-bound-theta", type=float, default=0,
                        help=u"Lower search bound for parameter theta.")
    parser.add_argument(u"--upper-bound-theta", type=float, default=1,
                        help=u"Upper search bound for parameter theta.")
    parser.add_argument(u"--lower-bound-sigma", type=float, default=0.001,
                        help=u"Lower search bound for parameter sigma.")
    parser.add_argument(u"--upper-bound-sigma", type=float, default=0.1,
                        help=u"Upper search bound for parameter sigma.")
    parser.add_argument(u"--expdata-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            u"addm_toolbox/data/expdata.csv"),
                        help=u"Name of experimental data file.")
    parser.add_argument(u"--fixations-file-name", type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.realpath(__file__)),
                            u"addm_toolbox/data/fixations.csv"),
                        help=u"Name of fixations file.")
    parser.add_argument(u"--verbose", default=False, action=u"store_true",
                        help=u"Increase output verbosity.")
    args = parser.parse_args()

    global dataTrials

    # Load experimental data from CSV file.
    if args.verbose:
        print(u"Loading experimental data...")
    data = load_data_from_csv(
        args.expdata_file_name, args.fixations_file_name,
        convertItemValues=convert_item_values)

    # Get correct subset of trials.
    subjectIds = args.subject_ids if args.subject_ids else list(data)
    for subjectId in subjectIds:
        numTrials = (args.trials_per_subject if args.trials_per_subject >= 1
                     else len(data[subjectId]))
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))],
            numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    creator.create(u"FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create(u"Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Create thread pool.
    pool = Pool(args.num_threads)
    toolbox.register(u"map", pool.map)

    # Create individual.
    toolbox.register(u"attr_d", random.uniform, args.lower_bound_d,
                     args.upper_bound_d)
    toolbox.register(u"attr_theta", random.uniform, args.lower_bound_theta,
                     args.upper_bound_theta)
    toolbox.register(u"attr_sigma", random.uniform, args.lower_bound_sigma,
                     args.upper_bound_sigma)
    toolbox.register(u"individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_d, toolbox.attr_theta, toolbox.attr_sigma),
                     n=1)

    # Create population.
    toolbox.register(u"population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=args.pop_size)

    # Create operators.
    toolbox.register(u"mate", tools.cxUniform, indpb=0.4)
    toolbox.register(u"mutate", tools.mutGaussian, mu=0,
                     sigma=[0.0005, 0.05, 0.005], indpb=0.4)
    toolbox.register(u"select", tools.selTournament, tournsize=3)
    toolbox.register(u"evaluate", evaluate)

    # Evaluate the entire population.
    try:
        fitnesses = list(map(toolbox.evaluate, pop))
    except:
        print(u"An exception occurred during the first population evaluation.")
        raise
    bestFit = sys.float_info.max
    bestInd = None
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

        # Get best individual.
        currFit = fit[0] if isinstance(fit, tuple) else fit
        if currFit < bestFit:
            bestInd = ind

    for g in range(args.num_generations):
        if args.verbose:
            print(u"Generation " + str(g) + u"...")

        # Select the next generation individuals.
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals.
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < args.crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < args.mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals which are valid but have an invalid fitness.
        invalidInd = list()
        for ind in offspring:
            if (ind[0] < args.lower_bound_d or
                ind[0] > args.upper_bound_d or
                ind[1] < args.lower_bound_theta or
                ind[1] > args.upper_bound_theta or
                ind[2] < args.lower_bound_sigma or
                ind[2] > args.upper_bound_sigma):
                ind.fitness.values = sys.float_info.max,
            elif not ind.fitness.valid:
                invalidInd.append(ind)
        try:
            fitnesses = list(map(toolbox.evaluate, invalidInd))
        except:
            print(u"An exception occurred during the population evaluation "
                  "for generation " + str(g) + u".")
            raise
        for ind, fit in zip(invalidInd, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring.
        pop[:] = offspring

        # Update best individual.
        for ind in pop:
            if ind.fitness.values[0] < bestFit:
                bestFit = ind.fitness.values[0]
                bestInd = ind

    print(u"Best individual: " + str(bestInd))
    print(u"Fitness of best individual: " + str(bestFit))


if __name__ == u"__main__":
    main()
