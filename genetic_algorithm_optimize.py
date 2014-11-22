#!/usr/bin/python

# genetic_algorithm_optimize.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from deap import base, creator, tools
from multiprocessing import Pool

import numpy as np
import operator
import random
import sys

from dyn_prog_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)


# Global variables.
rt = dict()
choice = dict()
valueLeft = dict()
valueRight = dict()
fixItem = dict()
fixTime = dict()


def evaluate(individual):
    trialsPerSubject = 100
    d = individual[0]
    theta = individual[1]
    mu = individual[2]

    logLikelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, mu, plotResults=False)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    print("NLL for " + str(individual) + ": " + str(-logLikelihood))
    return -logLikelihood,


def main():
    global rt
    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime

    # Load experimental data from CSV file and update global variables.
    data = load_data_from_csv()
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Constants.
    dMin, dMax = 0.00005, 0.001
    thetaMin, thetaMax = 0, 1
    muMin, muMax = 50, 1000
    crossoverRate = 0.5
    mutationRate = 0.2
    numGenerations = 10

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
    toolbox.register("attr_mu", random.randint, muMin, muMax)
    toolbox.register("individual", tools.initCycle, creator.Individual,
        (toolbox.attr_d, toolbox.attr_theta, toolbox.attr_mu), n=1)

    # Create population.
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=16)

    # Create operators.
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Evaluate the entire population.
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(numGenerations):
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
            if (ind[0] <= dMin or ind[0] >= dMax or ind[1] <= thetaMin or
                ind[1] >= thetaMax or ind[2] <= muMin or ind[2] >= muMax):
                ind.fitness.values = sys.float_info.max,
            elif not ind.fitness.valid:
                invalidInd.append(ind)
        fitnesses = map(toolbox.evaluate, invalidInd)
        for ind, fit in zip(invalidInd, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring.
        pop[:] = offspring

    # Get best individual.
    minValue = sys.float_info.max
    bestInd = []
    for ind in pop:
        if ind.fitness.values[0] < minValue:
            minValue = ind.fitness.values[0]
            bestInd = ind

    print minValue
    print bestInd


if __name__ == '__main__':
    main()