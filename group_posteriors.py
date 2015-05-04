#!/usr/bin/python

# group_posteriors.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import numpy as np

from addm import (analysis_per_trial, get_empirical_distributions,
    generate_probabilistic_simulations)
from util import load_data_from_csv, save_simulations_to_csv


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    trialsPerSubject = 100
    numThreads = 10
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("kirby_1000_trials.csv", "kirby_1000_trials_fixations.csv", False)
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Posteriors estimation for the parameters of the model.
    print("Starting grid search...")
    rangeD = [0.02, 0.03]
    rangeTheta = [0.02, 0.04]
    rangeStd = [0.3, 0.5]
    rangeDecay = [0.0011, 0.0014]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeStd) * len(rangeDecay)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                for decay in rangeDecay:
                    model = (d, theta, std, decay)
                    models.append(model)
                    posteriors[model] = 1./ numModels

    subjects = rt.keys()
    for subject in subjects:
        print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            listParams = list()
            for model in models:
                listParams.append((rt[subject][trial], choice[subject][trial],
                    valueLeft[subject][trial], valueRight[subject][trial],
                    fixItem[subject][trial], fixTime[subject][trial], model[0],
                    model[1], model[2], model[3]))
            likelihoods = pool.map(run_analysis_wrapper, listParams)

            # Get the denominator for normalizing the posteriors.
            i = 0
            denominator = 0
            for model in models:
                denominator += posteriors[model] * likelihoods[i]
                i += 1
            if denominator == 0:
                continue

            # Calculate the posteriors after this trial.
            i = 0
            for model in models:
                prior = posteriors[model]
                posteriors[model] = likelihoods[i] * prior / denominator
                i += 1

            # Debugging. Simulation bypass code. Just gives random posteriors
            # so that we can test the simulation code.  Note that these
            # posteriors are total bullshit, don't even add to 1
        print("DEBUG:Bypassing model fitting, posteriors are not reliable")
        for model in models:
            posteriors[model] = 1.0/numModels
                
            
        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))

    # Get empirical distributions for the data.
    dists = get_empirical_distributions(rt, choice, valueLeft, valueRight,
        fixItem, fixTime, useOddTrials=True, useEvenTrials=True, 
        valueDiffs = range(-4,5,1), numFixDists = 2)
    probLeftFixFirst = dists.probLeftFixFirst
    distTransitions = dists.distTransitions
    distFixations = dists.distFixations

    # Trial conditions for generating simulations.
    # For perceptual (oriented bar) stimuli
#    orientations = range(-15,20,5)
#    trialConditions = list()
#    for oLeft in orientations:
#        for oRight in orientations:
#            if oLeft != oRight:
#                vLeft = np.absolute((np.absolute(oLeft) - 15) / 5)
#                vRight = np.absolute((np.absolute(oRight) - 15) / 5)
#                trialConditions.append((vLeft, vRight))

    # For McGinty and Newsome value stimui
    # Beware: this list of values reflects an experiment-specific stimulus scheme
    values = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5]
    numOfValues = np.size(values)
    trialConditions = list()
    #valDiffList = list()  # For debugging purposes
    for i in xrange(numOfValues):
        for j in xrange(numOfValues):    
            if i == j:  # In our task, we never show two of the same stimuli
                continue
            vLeft = values[i]
            vRight = values[j]
            absValDiff = np.absolute(vLeft-vRight)
            if absValDiff > 2: # Debugging, want to simulate only some conditions
                continue
            trialConditions.append((values[i], values[j]))
            #valDiffList.append(lVal-rVal)


    # Generate probabilistic simulations using the posteriors distribution.
    simul = generate_probabilistic_simulations(probLeftFixFirst,
        distTransitions, distFixations, trialConditions, posteriors,
        numSamples = 10) # Debug; 10 samples instetad of 100
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valueLeft
    simulValueRight = simul.valueRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    simulFixRDV = simul.fixRDV

    totalTrials = len(simulRt.keys())
    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
        simulValueRight, simulFixItem, simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()
