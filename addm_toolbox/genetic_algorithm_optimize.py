#!/usr/bin/env python

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

from __future__ import absolute_import, division

import numpy as np
import pkg_resources
import random

from builtins import range, str, zip
from deap import base, creator, tools
from multiprocessing import Pool

from .addm import aDDM
from .util import load_data_from_csv, convert_item_values


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
        return float("inf"),


def main(lowerBoundD=0.0001, upperBoundD=0.09, lowerBoundSigma=0.001,
         upperBoundSigma=0.9, lowerBoundTheta=0, upperBoundTheta=1,
         expdataFileName=None, fixationsFileName=None, trialsPerSubject=100,
         popSize=18, numGenerations=20, crossoverRate=0.5, mutationRate=0.3,
         subjectIds=[], numThreads=9, verbose=False):
    """
    Args:
      lowerBoundD: float, lower search bound for parameter d.
      upperBoundD: float, upper search bound for parameter d.
      lowerBoundSigma: float, lower search bound for parameter sigma.
      upperBoundSigma: float, upper search bound for parameter sigma.
      lowerBoundTheta: float, lower search bound for parameter theta.
      upperBoundTheta: float, upper search bound for parameter theta.
      expdataFileName: string, path of experimental data file.
      fixationsFileName: string, path of fixations file.
      trialsPerSubject: int, number of trials from each subject to be used in
          the analysis. If smaller than 1, all trials are used.
      popSize: int, number of individuals in each population.
      numGenerations: int, number of generations.
      crossoverRate: float, crossover rate.
      mutationRate: float, mutation rate.
      subjectIds: list of strings corresponding to the subject ids. If not
          provided, all existing subjects will be used.
      numThreads: int, size of the thread pool.
      verbose: boolean, whether or not to increase output verbosity.
    """
    global dataTrials

    # Load experimental data from CSV file.
    if verbose:
        print(u"Loading experimental data...")
    if not expdataFileName:
        expdataFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"data/expdata.csv")
    if not fixationsFileName:
        fixationsFileName = pkg_resources.resource_filename(
            u"addm_toolbox", u"data/fixations.csv")
    data = load_data_from_csv(expdataFileName, fixationsFileName,
                              convertItemValues=convert_item_values)

    # Get correct subset of trials.
    subjectIds = ([str(subj) for subj in subjectIds] if subjectIds
                  else list(data))
    for subjectId in subjectIds:
        numTrials = (trialsPerSubject if trialsPerSubject >= 1
                     else len(data[subjectId]))
        trialSet = np.random.choice(
            [trialId for trialId in range(len(data[subjectId]))],
            numTrials, replace=False)
        dataTrials.extend([data[subjectId][t] for t in trialSet])

    creator.create(u"FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create(u"Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Create thread pool.
    pool = Pool(numThreads)
    toolbox.register(u"map", pool.map)

    # Create individual.
    toolbox.register(u"attr_d", random.uniform, lowerBoundD, upperBoundD)
    toolbox.register(u"attr_sigma", random.uniform, lowerBoundSigma,
                     upperBoundSigma)
    toolbox.register(u"attr_theta", random.uniform, lowerBoundTheta,
                     upperBoundTheta)
    toolbox.register(u"individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_d, toolbox.attr_theta, toolbox.attr_sigma),
                     n=1)

    # Create population.
    toolbox.register(u"population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=popSize)

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
    bestFit = float("inf")
    bestInd = None
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

        # Get best individual.
        currFit = fit[0] if isinstance(fit, tuple) else fit
        if currFit < bestFit:
            bestInd = ind

    for g in range(numGenerations):
        if verbose:
            print(u"Generation " + str(g) + u"...")

        # Select the next generation individuals.
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals.
        offspring = list(map(toolbox.clone, offspring))

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
            if (ind[0] < lowerBoundD or
                ind[0] > upperBoundD or
                ind[1] < lowerBoundTheta or
                ind[1] > upperBoundTheta or
                ind[2] < lowerBoundSigma or
                ind[2] > upperBoundSigma):
                ind.fitness.values = float("inf"),
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
