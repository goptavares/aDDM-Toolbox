#!/usr/bin/python

"""
genetic_algorithm_optimize.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood estimation procedure for the attentional drift-diffusion
model (aDDM), using a genetic algorithm to search the parameter space. Data from
all subjects is pooled such that a single set of optimal parameters is
estimated.
"""

from deap import base, creator, tools
from multiprocessing import Pool

import argparse
import numpy as np
import random
import sys

from addm import get_trial_likelihood
from util import load_data_from_csv


# Global variables.
choice = dict()
valueLeft = dict()
valueRight = dict()
fixItem = dict()
fixTime = dict()
trialsPerSubject = 0


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

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        trials = choice[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = get_trial_likelihood(
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, sigma=sigma,
                plotResults=False)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    print("NLL for " + str(individual) + ": " + str(-logLikelihood))
    return -logLikelihood,


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--trials-per-subject", type=int, default=100,
                        help="Number of trials from each subject to be used in "
                        "the analysis; if smaller than 1, all trials are used.")
    parser.add_argument("--pop-size", type=int, default=18,
                        help="Number of individuals in each population.")
    parser.add_argument("--num-generations", type=int, default=20,
                        help="Number of generations.")
    parser.add_argument("--crossover-rate", type=float, default=0.5,
                        help="Crossover rate.")
    parser.add_argument("--mutation-rate", type=float, default=0.3,
                        help="Mutation rate.")
    parser.add_argument("--lower-bound-d", type=float, default=0.0001,
                        help="Lower search bound for parameter d.")
    parser.add_argument("--upper-bound-d", type=float, default=0.01,
                        help="Upper search bound for parameter d.")
    parser.add_argument("--lower-bound-theta", type=float, default=0,
                        help="Lower search bound for parameter theta.")
    parser.add_argument("--upper-bound-theta", type=float, default=1,
                        help="Upper search bound for parameter theta.")
    parser.add_argument("--lower-bound-sigma", type=float, default=0.001,
                        help="Lower search bound for parameter sigma.")
    parser.add_argument("--upper-bound-sigma", type=float, default=0.1,
                        help="Upper search bound for parameter sigma.")
    parser.add_argument("--expdata-file-name", type=str, default="expdata.csv",
                        help="Name of experimental data file.")
    parser.add_argument("--fixations-file-name", type=str,
                        default="fixations.csv", help="Name of fixations file.")
    args = parser.parse_args()

    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime
    global trialsPerSubject

    # Load experimental data from CSV file and update global variables.
    data = load_data_from_csv(args.expdata_file_name, args.fixations_file_name,
                              useAngularDists=True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    trialsPerSubject = args.trials_per_subject

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Create thread pool.
    pool = Pool(args.num_threads)
    toolbox.register("map", pool.map)

    # Create individual.
    toolbox.register("attr_d", random.uniform, args.lower_bound_d,
                     args.upper_bound_d)
    toolbox.register("attr_theta", random.uniform, args.lower_bound_theta,
                     args.upper_bound_theta)
    toolbox.register("attr_sigma", random.uniform, args.lower_bound_sigma,
                     args.upper_bound_sigma)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_d, toolbox.attr_theta, toolbox.attr_sigma),
                     n=1)

    # Create population.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=args.pop_size)

    # Create operators.
    toolbox.register("mate", tools.cxUniform, indpb=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0,
                     sigma=[0.0005, 0.05, 0.005], indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Evaluate the entire population.
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    bestFit = sys.float_info.max
    bestInd = None
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        # Get best individual.
        if fit < bestFit:
            bestInd = ind

    for g in xrange(args.num_generations):
        print("Generation " + str(g) + "...")

        # Select the next generation individuals.
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals.
        offspring = map(toolbox.clone, offspring)

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
        fitnesses = map(toolbox.evaluate, invalidInd)
        for ind, fit in zip(invalidInd, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring.
        pop[:] = offspring

        # Update best individual.
        for ind in pop:
            if ind.fitness.values[0] < bestFit:
                bestFit = ind.fitness.values[0]
                bestInd = ind

    print("Best individual: " + str(bestInd))
    print("Fitness of best individual: " + str(bestFit))


if __name__ == '__main__':
    main()
