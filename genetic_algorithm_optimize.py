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

    trialsPerSubject = 200  # Number of trials to be used from each subject.
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
    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime

    # Load experimental data from CSV file and update global variables.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Constants.
    dMin, dMax = 0.0002, 0.08
    thetaMin, thetaMax = 0, 1
    sigmaMin, sigmaMax = 0.05, 0.15
    crossoverRate = 0.5
    mutationRate = 0.3
    numGenerations = 30

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Create thread pool.
    numThreads = 8
    pool = Pool(numThreads)
    toolbox.register("map", pool.map)

    # Create individual.
    toolbox.register("attr_d", random.uniform, dMin, dMax)
    toolbox.register("attr_theta", random.uniform, thetaMin, thetaMax)
    toolbox.register("attr_sigma", random.uniform, sigmaMin, sigmaMax)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_d, toolbox.attr_theta, toolbox.attr_sigma),
                     n=1)

    # Create population.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=16)

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

    for g in xrange(numGenerations):
        print("Generation " + str(g) + "...")

        # Select the next generation individuals.
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals.
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossoverRate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutationRate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals which are valid but have an invalid fitness.
        invalidInd = list()
        for ind in offspring:
            if (ind[0] <= dMin or ind[0] >= dMax or
                ind[1] <= thetaMin or ind[1] >= thetaMax or
                ind[2] <= sigmaMin or ind[2] >= sigmaMax):
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
