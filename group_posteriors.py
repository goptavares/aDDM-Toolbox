#!/usr/bin/python

# group_fitting.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from handle_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)

def generate_probabilistic_simulations(probLeftFixFirst, distTransition,
    distFirstFix, distSecondFix, distThirdFix, distOtherFix, posteriors,
    numSamples=100, numSimulationsPerSample=10):
    posteriorsList = list()
    models = dict()
    i = 0
    for model, posterior in posteriors.iteritems():
        posteriorsList.append(posterior)
        models[i] = model
        i += 1

    # Parameters for generating simulations.
    values = range(1,6,1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    rt = dict()
    choice = dict()
    valLeft = dict()
    valRight = dict()
    fixItem = dict()
    fixTime = dict()
    fixRDV = dict()

    numModels = len(models.keys())
    trialCount = 0
    for i in xrange(numSamples):
        # Sample model from posteriors distribution.
        modelIndex = np.random.choice(np.array(range(numModels)),
            p=np.array(posteriorsList))
        model = models[modelIndex]
        d = model[0]
        theta = model[1]
        std = model[2]

        # Generate simulations with the sampled model.
        simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
            distSecondFix, distThirdFix, distOtherFix, numSimulationsPerSample,
            trialConditions, d, theta, std=std, nonFixDiffusion = True)
        for trial in simul.rt.keys():
            rt[trialCount] = simul.rt[trial]
            choice[trialCount] = simul.choice[trial]
            fixTime[trialCount] = simul.fixTime[trial]
            fixItem[trialCount] = simul.fixItem[trial]
            fixRDV[trialCount] = simul.fixRDV[trial]
            valLeft[trialCount] = simul.valLeft[trial]
            valRight[trialCount] = simul.valRight[trial]
            trialCount += 1
    
    allSimul = collections.namedtuple('Simul', ['rt', 'choice', 'valLeft',
        'valRight', 'fixItem', 'fixTime', 'fixRDV'])
    return allSimul(rt, choice, valLeft, valRight, fixItem, fixTime, fixRDV)

# This is how Gaby's version ends.  For my version, I want an output
#    numTrials = len(rt.keys())
#    save_simulations_to_csv(choice, rt, valueLeft, valueRight, fixItem, fixTime,
#        fixRDV, numTrials)

def generate_choice_curves(choicesData, valueLeftData, valueRightData,
    choicesSimul, valueLeftSimul, valueRightSimul, numTrials):
    countTotal = np.zeros(9)
    countLeftChosen = np.zeros(9)

    subjects = choicesData.keys()
    for subject in subjects:
        trials = choicesData[subject].keys()
        for trial in trials:
            valueDiff = (valueLeftData[subject][trial] -
                valueRightData[subject][trial])
            idx = valueDiff + 4
            if choicesData[subject][trial] == -1:  # Choice was left.
                countLeftChosen[idx] +=1
                countTotal[idx] += 1
            elif choicesData[subject][trial] == 1:  # Choice was right.
                countTotal[idx] += 1

    stdProbLeftChosen = np.zeros(9)
    probLeftChosen = np.zeros(9)
    for i in xrange(0,9):
        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
        stdProbLeftChosen[i] = np.sqrt((probLeftChosen[i] *
            (1 - probLeftChosen[i])) / countTotal[i])

    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
    plt.errorbar(range(-4,5,1), probLeftChosen, yerr=stdProbLeftChosen,
        color=colors[0], label='Data')

    countTotal = np.zeros(9)
    countLeftChosen = np.zeros(9)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeftSimul[trial] - valueRightSimul[trial]
        idx = valueDiff + 4
        if choicesSimul[trial] == -1:  # Choice was left.
            countLeftChosen[idx] +=1
            countTotal[idx] += 1
        elif choicesSimul[trial] == 1:  # Choice was right.
            countTotal[idx] += 1

    stdProbLeftChosen = np.zeros(9)
    probLeftChosen = np.zeros(9)
    for i in xrange(0,9):
        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
        stdProbLeftChosen[i] = np.sqrt((probLeftChosen[i] *
            (1 - probLeftChosen[i])) / countTotal[i])

    plt.errorbar(range(-4,5,1), probLeftChosen, yerr=stdProbLeftChosen,
        color=colors[5], label='Simulations')
    plt.xlabel('Value difference')
    plt.ylabel('P(choose left)')
    plt.legend()
    return fig


def generate_rt_curves(rtsData, valueLeftData, valueRightData, rtsSimul,
    valueLeftSimul, valueRightSimul, numTrials):
    rtsPerValueDiff = dict()
    for valueDiff in xrange(-4,5,1):
        rtsPerValueDiff[valueDiff] = list()

    subjects = rtsData.keys()
    for subject in subjects:
        trials = rtsData[subject].keys()
        for trial in trials:
            valueDiff = (valueLeftData[subject][trial] -
                valueRightData[subject][trial])
            rtsPerValueDiff[valueDiff].append(rtsData[subject][trial])

    meanRts = np.zeros(9)
    stdRts = np.zeros(9)
    for valueDiff in xrange(-4,5,1):
        idx = valueDiff + 4
        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
        stdRts[idx] = (np.std(np.array(rtsPerValueDiff[valueDiff])) /
            np.sqrt(len(rtsPerValueDiff[valueDiff])))

    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
    plt.errorbar(range(-4,5,1), meanRts, yerr=stdRts, label='Data',
        color=colors[0])

    rtsPerValueDiff = dict()
    for valueDiff in xrange(-4,5,1):
        rtsPerValueDiff[valueDiff] = list()

    for trial in xrange(0, numTrials):
        valueDiff = valueLeftSimul[trial] - valueRightSimul[trial]
        rtsPerValueDiff[valueDiff].append(rtsSimul[trial])

    meanRts = np.zeros(9)
    stdRts = np.zeros(9)
    for valueDiff in xrange(-4,5,1):
        idx = valueDiff + 4
        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
        stdRts[idx] = (np.std(np.array(rtsPerValueDiff[valueDiff])) /
            np.sqrt(len(rtsPerValueDiff[valueDiff])))

    plt.errorbar(range(-4,5,1), meanRts, yerr=stdRts, label='Simulations',
        color=colors[5])
    plt.xlabel('Value difference')
    plt.ylabel('Mean RT')
    plt.legend()
    return fig


def save_simulations_to_csv(choice, rt, valueLeft, valueRight, fixItem,
    fixTime, fixRDV, numTrials):
    df = pd.DataFrame(choice, index=range(1))
    df.to_csv('choice.csv', header=0, sep=',', index_col=None)

    df = pd.DataFrame(rt, index=range(1))
    df.to_csv('rt.csv', header=0, sep=',', index_col=None)

    dictValueLeft = dict()
    dictValueRight = dict()
    dictItem = dict()
    dictTime = dict()
    dictRDV = dict()
    for trial in xrange(0, numTrials):
        dictValueLeft[trial] = valueLeft[trial]
        dictValueRight[trial] = valueRight[trial]
        dictItem[trial] = pd.Series(fixItem[trial])
        dictTime[trial] = pd.Series(fixTime[trial])
        dictRDV[trial] = pd.Series(fixRDV[trial])
    df = pd.DataFrame(dictValueLeft, index=range(1))
    df.to_csv('value_left.csv', header=0, sep=',', index_col=None)
    df = pd.DataFrame(dictValueRight, index=range(1))
    df.to_csv('value_right.csv', header=0, sep=',', index_col=None)
    df = pd.DataFrame(dictItem)
    df.to_csv('fix_item.csv', header=0, sep=',', index_col=None)
    df = pd.DataFrame(dictTime)
    df.to_csv('fix_time.csv', header=0, sep=',', index_col=None)
    df = pd.DataFrame(dictRDV)
    df.to_csv('fix_rdv.csv', header=0, sep=',', index_col=None)


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    trialsPerSubject = 200
    numThreads = 4
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    exFile =  "kirby_7647_trials.csv"
    fixFile = "kirby_7647_trials_fixations.csv"
    data = load_data_from_csv(exFile, fixFile)
    rt = data.rt
    choice = data.choice
    valueLeft = data.distLeft  # In my code, distance means value
    valueRight = data.distRight
    fixItem = data.fixItem
    fixTime = data.fixTime


    # Maximum likelihood estimation using odd trials only.
    # Grid search on the parameters of the model.
    print("Starting grid search...")

    rangeD = [0.005, 0.01, 0.02]
    rangeTheta = [0.05, 0.1, 0.15]
    rangeStd = [0.05, 0.1, 0.2]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeStd)    

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                model = (d, theta, std)
                models.append(model)
                posteriors[model] = 1./ numModels

    subjects = rt.keys()
    for subject in subjects:
        print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            print("Fitting trial " + str(trial))
            listParams = list()
            for model in models:
                listParams.append((rt[subject][trial], choice[subject][trial],
                    valueLeft[subject][trial], valueRight[subject][trial],
                    fixItem[subject][trial], fixTime[subject][trial], model[0],
                    model[1], model[2], 0, 10, 1, 0, 0, False, True))
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

        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))


    # Get empirical distributions from even trials.
    evenDists = get_empirical_distributions(rt, choice, valueLeft, valueRight,
        fixItem, fixTime, useOddTrials=False, useEvenTrials=True)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distTransition = evenDists.distTransition
    distFirstFix = evenDists.distFirstFix
    distSecondFix = evenDists.distSecondFix
    distThirdFix = evenDists.distThirdFix
    distOtherFix = evenDists.distOtherFix


    # Parameters for generating simulations.
    numTrials = 50
    values = range(1,6,1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    simul = generate_probabilistic_simulations(probLeftFixFirst, distTransition,
        distFirstFix, distSecondFix, distThirdFix, distOtherFix, posteriors)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valLeft
    simulValueRight = simul.valRight

    # For the sake of naming the output, we'll find the best parameters
    highestPosteriorParams = max(posteriors, key = posteriors.get)
    optimD = highestPosteriorParams[0]
    optimTheta = highestPosteriorParams[1]
    optimStd = highestPosteriorParams[2]

    # Create pdf file to save figures.
    pp = PdfPages("figures_bayes_consensus_params_" + 
        str(optimD) + "_" + str(optimTheta) + "_" + str(optimStd) + "_" + str(numTrials) + ".pdf")

    # Generate choice and rt curves for real data (odd trials) and
    # simulations (generated from even trials).
    totalTrials = numTrials * len(trialConditions)
    fig1 = generate_choice_curves(choice, valueLeft, valueRight, simulChoice,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig1)
    fig2 = generate_rt_curves(rt, valueLeft, valueRight, simulRt,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig2)
    pp.close()

#    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
#        simulValueRight, simulFixItem, simulFixTime, simulFixRDV, totalTrials)


if __name__ == '__main__':
    main()