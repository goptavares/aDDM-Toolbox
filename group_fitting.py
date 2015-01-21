#!/usr/bin/python

# group_fitting.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd

from handle_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)


def generate_choice_curves(choicesData, valueLeftData, valueRightData,
    choicesSimul, valueLeftSimul, valueRightSimul, numTrials):
    countTotal = np.zeros(7)
    countLeftChosen = np.zeros(7)

    subjects = choicesData.keys()
    for subject in subjects:
        trials = choicesData[subject].keys()
        for trial in trials:
            valueDiff = (valueLeftData[subject][trial] -
                valueRightData[subject][trial])
            idx = valueDiff + 3
            if choicesData[subject][trial] == -1:  # Choice was left.
                countLeftChosen[idx] +=1
                countTotal[idx] += 1
            elif choicesData[subject][trial] == 1:  # Choice was right.
                countTotal[idx] += 1

    stdProbLeftChosen = np.zeros(7)
    probLeftChosen = np.zeros(7)
    for i in xrange(0,7):
        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
        stdProbLeftChosen[i] = np.sqrt((probLeftChosen[i] *
            (1 - probLeftChosen[i])) / countTotal[i])

    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
    plt.errorbar(range(-3,4,1), probLeftChosen, yerr=stdProbLeftChosen,
        color=colors[0], label='Data')

    countTotal = np.zeros(7)
    countLeftChosen = np.zeros(7)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeftSimul[trial] - valueRightSimul[trial]
        idx = valueDiff + 3
        if choicesSimul[trial] == -1:  # Choice was left.
            countLeftChosen[idx] +=1
            countTotal[idx] += 1
        elif choicesSimul[trial] == 1:  # Choice was right.
            countTotal[idx] += 1

    stdProbLeftChosen = np.zeros(7)
    probLeftChosen = np.zeros(7)
    for i in xrange(0,7):
        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
        stdProbLeftChosen[i] = np.sqrt((probLeftChosen[i] *
            (1 - probLeftChosen[i])) / countTotal[i])

    plt.errorbar(range(-3,4,1), probLeftChosen, yerr=stdProbLeftChosen,
        color=colors[5], label='Simulations')
    plt.xlabel('Value difference')
    plt.ylabel('P(choose left)')
    plt.legend()
    return fig


def generate_rt_curves(rtsData, valueLeftData, valueRightData, rtsSimul,
    valueLeftSimul, valueRightSimul, numTrials):
    rtsPerValueDiff = dict()
    for valueDiff in xrange(-3,4,1):
        rtsPerValueDiff[valueDiff] = list()

    subjects = rtsData.keys()
    for subject in subjects:
        trials = rtsData[subject].keys()
        for trial in trials:
            valueDiff = (valueLeftData[subject][trial] -
                valueRightData[subject][trial])
            rtsPerValueDiff[valueDiff].append(rtsData[subject][trial])

    meanRts = np.zeros(7)
    stdRts = np.zeros(7)
    for valueDiff in xrange(-3,4,1):
        idx = valueDiff + 3
        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
        stdRts[idx] = (np.std(np.array(rtsPerValueDiff[valueDiff])) /
            np.sqrt(len(rtsPerValueDiff[valueDiff])))

    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
    plt.errorbar(range(-3,4,1), meanRts, yerr=stdRts, label='Data',
        color=colors[0])

    rtsPerValueDiff = dict()
    for valueDiff in xrange(-3,4,1):
        rtsPerValueDiff[valueDiff] = list()

    for trial in xrange(0, numTrials):
        valueDiff = valueLeftSimul[trial] - valueRightSimul[trial]
        rtsPerValueDiff[valueDiff].append(rtsSimul[trial])

    meanRts = np.zeros(7)
    stdRts = np.zeros(7)
    for valueDiff in xrange(-3,4,1):
        idx = valueDiff + 3
        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
        stdRts[idx] = (np.std(np.array(rtsPerValueDiff[valueDiff])) /
            np.sqrt(len(rtsPerValueDiff[valueDiff])))

    plt.errorbar(range(-3,4,1), meanRts, yerr=stdRts, label='Simulations',
        color=colors[5])
    plt.xlabel('Value difference')
    plt.ylabel('Mean RT')
    plt.legend()
    return fig


def save_simulations_to_csv(choice, rt, valueLeft, valueRight, fixItem,
    fixTime, numTrials):
    # Psychometric choice curve.
    countTotal = np.zeros(7)
    countLeftChosen = np.zeros(7)
    trialCount = np.zeros(7)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeft[trial] - valueRight[trial]
        idx = valueDiff + 3
        trialCount[idx] += 1
        if choice[trial] == -1:  # Choice was left.
            countLeftChosen[idx] +=1
            countTotal[idx] += 1
        elif choice[trial] == 1:  # Choice was right.
            countTotal[idx] += 1

    probLeftChosen = np.zeros(7)
    stdProbLeftChosen = np.zeros(7)

    for i in xrange(0,7):
        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
        stdProbLeftChosen[i] = np.sqrt(probLeftChosen[i] *
            (1 - probLeftChosen[i]))

    d = {'a': probLeftChosen, 'b': stdProbLeftChosen, 'c': trialCount}
    df = pd.DataFrame(d)
    df.to_csv('choices.csv', header=0, sep=',', index_col=None)

    # Reaction times.
    rtsPerValueDiff = dict()
    for valueDiff in xrange(-3,4,1):
        rtsPerValueDiff[valueDiff] = list()

    trialCount = np.zeros(7)
    for trial in xrange(0, numTrials):
        valueDiff = valueLeft[trial] - valueRight[trial]
        idx = valueDiff + 3
        trialCount[idx] += 1
        rtsPerValueDiff[valueDiff].append(rt[trial])

    meanRts = np.zeros(7)
    stdRts = np.zeros(7)
    for valueDiff in xrange(-3,4,1):
        idx = valueDiff + 3
        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
        stdRts[idx] = np.std(np.array(rtsPerValueDiff[valueDiff]))

    d = {'a': meanRts, 'b': stdRts, 'c': trialCount}
    df = pd.DataFrame(d)
    df.to_csv('rts.csv', header=0, sep=',', index_col=None)

    maxLen = 0
    for valueDiff in xrange(-3,4,1):
        if len(rtsPerValueDiff[valueDiff]) > maxLen:
            maxLen = len(rtsPerValueDiff[valueDiff])

    d = dict()
    for valueDiff in xrange(-3,4,1):
        rts = np.zeros(maxLen)
        for i in xrange(len(rtsPerValueDiff[valueDiff])):
            rts[i] = rtsPerValueDiff[valueDiff][i]
        d[valueDiff] = rts
    df = pd.DataFrame(d)
    df.to_csv('all_rts.csv', header=0, sep=',', index_col=None)

    # Pyschometric choice curve grouped by first fixation.
    countTotalLeft = np.zeros(7)
    countLeftChosenLeft = np.zeros(7)
    trialCountLeft = np.zeros(7)
    countTotalRight = np.zeros(7)
    countLeftChosenRight = np.zeros(7)
    trialCountRight = np.zeros(7)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeft[trial] - valueRight[trial]
        idx = valueDiff + 3
        if fixItem[trial][1] == 1:  # First item was left.
            trialCountLeft[idx] += 1
            countTotalLeft[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenLeft[idx] +=1
        elif fixItem[trial][1] == 2:  # First item was right.
            trialCountRight[idx] += 1
            countTotalRight[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenRight[idx] +=1

    probLeftChosenLeft = np.zeros(7)
    stdProbLeftChosenLeft = np.zeros(7)
    probLeftChosenRight = np.zeros(7)
    stdProbLeftChosenRight = np.zeros(7)
    for i in xrange(0,7):
        probLeftChosenLeft[i] = countLeftChosenLeft[i] / countTotalLeft[i]
        stdProbLeftChosenLeft[i] = np.sqrt(probLeftChosenLeft[i] *
            (1 - probLeftChosenLeft[i]))
        probLeftChosenRight[i] = countLeftChosenRight[i] / countTotalRight[i]
        stdProbLeftChosenRight[i] = np.sqrt(probLeftChosenRight[i] *
            (1 - probLeftChosenRight[i]))

    d = {'a': probLeftChosenLeft, 'b': stdProbLeftChosenLeft,
        'c': trialCountLeft, 'd': probLeftChosenRight,
        'e': stdProbLeftChosenRight, 'f': trialCountRight}
    df = pd.DataFrame(d)
    df.to_csv('choices_first_fix.csv', header=0, sep=',', index_col=None)

    # Pyschometric choice curve grouped by last fixation.
    countTotalLeft = np.zeros(7)
    countLeftChosenLeft = np.zeros(7)
    trialCountLeft = np.zeros(7)
    countTotalRight = np.zeros(7)
    countLeftChosenRight = np.zeros(7)
    trialCountRight = np.zeros(7)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeft[trial] - valueRight[trial]
        idx = valueDiff + 3
        if fixItem[trial][-1] == 1:  # Last item was left.
            trialCountLeft[idx] += 1
            countTotalLeft[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenLeft[idx] +=1
        elif fixItem[trial][-1] == 2:  # Last item was right.
            trialCountRight[idx] += 1
            countTotalRight[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenRight[idx] +=1

    probLeftChosenLeft = np.zeros(7)
    stdProbLeftChosenLeft = np.zeros(7)
    probLeftChosenRight = np.zeros(7)
    stdProbLeftChosenRight = np.zeros(7)
    for i in xrange(0,7):
        probLeftChosenLeft[i] = countLeftChosenLeft[i] / countTotalLeft[i]
        stdProbLeftChosenLeft[i] = np.sqrt(probLeftChosenLeft[i] *
            (1 - probLeftChosenLeft[i]))
        probLeftChosenRight[i] = countLeftChosenRight[i] / countTotalRight[i]
        stdProbLeftChosenRight[i] = np.sqrt(probLeftChosenRight[i] *
            (1 - probLeftChosenRight[i]))

    d = {'a': probLeftChosenLeft, 'b': stdProbLeftChosenLeft,
        'c': trialCountLeft, 'd': probLeftChosenRight,
        'e': stdProbLeftChosenRight, 'f': trialCountRight}
    df = pd.DataFrame(d)
    df.to_csv('choices_last_fix.csv', header=0, sep=',', index_col=None)

    # Pyschometric choice curve grouped by longest fixation time.
    countTotalLeft = np.zeros(7)
    countLeftChosenLeft = np.zeros(7)
    trialCountLeft = np.zeros(7)
    countTotalRight = np.zeros(7)
    countLeftChosenRight = np.zeros(7)
    trialCountRight = np.zeros(7)

    for trial in xrange(0, numTrials):
        valueDiff = valueLeft[trial] - valueRight[trial]
        idx = valueDiff + 3

        # Get total fixation time for each item.
        fixTimeLeft = 0
        fixTimeRight = 0
        for i in xrange(0, len(fixItem[trial])):
            if fixItem[trial][i] == 1:
                fixTimeLeft += fixTime[trial][i]
            elif fixItem[trial][i] == 2:
                fixTimeRight += fixTime[trial][i]

        if fixTimeLeft >= fixTimeRight:  # Longest fixated item was left.
            trialCountLeft[idx] += 1
            countTotalLeft[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenLeft[idx] +=1
        else:  # Longest fixated item was right.
            trialCountRight[idx] += 1
            countTotalRight[idx] += 1
            if choice[trial] == -1:  # Choice was left.
                countLeftChosenRight[idx] +=1

    probLeftChosenLeft = np.zeros(7)
    stdProbLeftChosenLeft = np.zeros(7)
    probLeftChosenRight = np.zeros(7)
    stdProbLeftChosenRight = np.zeros(7)
    for i in xrange(0,7):
        probLeftChosenLeft[i] = countLeftChosenLeft[i] / countTotalLeft[i]
        stdProbLeftChosenLeft[i] = np.sqrt(probLeftChosenLeft[i] *
            (1 - probLeftChosenLeft[i]))
        probLeftChosenRight[i] = countLeftChosenRight[i] / countTotalRight[i]
        stdProbLeftChosenRight[i] = np.sqrt(probLeftChosenRight[i] *
            (1 - probLeftChosenRight[i]))

    d = {'a': probLeftChosenLeft, 'b': stdProbLeftChosenLeft,
        'c': trialCountLeft, 'd': probLeftChosenRight,
        'e': stdProbLeftChosenRight, 'f': trialCountRight}
    df = pd.DataFrame(d)
    df.to_csv('choices_most_fix.csv', header=0, sep=',', index_col=None)


def run_analysis(rt, choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
    std, useOddTrials=True, useEvenTrials=True, verbose=True):
    trialsPerSubject = 200
    logLikelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        if verbose:
            print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            likelihood = analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, std=std)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("NLL for " + str(d) + ", " + str(theta) + ", "
            + str(std) + ": " + str(-logLikelihood))
    return -logLikelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 9
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt = data.rt
    choice = data.choice
    distLeft = data.distLeft
    distRight = data.distRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get item values.
    valueLeft = dict()
    valueRight = dict()
    subjects = distLeft.keys()
    for subject in subjects:
        valueLeft[subject] = dict()
        valueRight[subject] = dict()
        trials = distLeft[subject].keys()
        for trial in trials:
            valueLeft[subject][trial] = np.absolute((np.absolute(
                distLeft[subject][trial])-15)/5)
            valueRight[subject][trial] = np.absolute((np.absolute(
                distRight[subject][trial])-15)/5)

    # Maximum likelihood estimation using odd trials only.
    # Grid search on the parameters of the model.
    print("Starting grid search...")
    rangeD = [0.0015, 0.0025, 0.0035]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeStd = [0.03, 0.06, 0.09]

    models = list()
    listParams = list()
    for d in rangeD:
        for theta in rangeTheta:
            for std in rangeStd:
                models.append((d, theta, std))
                params = (rt, choice, valueLeft, valueRight, fixItem, fixTime,
                    d, theta, std, True, False)
                listParams.append(params)

    print("Starting pool of workers...")
    results = pool.map(run_analysis_wrapper, listParams)

    # Get optimal parameters.
    minNegLogLikeIdx = results.index(min(results))
    optimD = models[minNegLogLikeIdx][0]
    optimTheta = models[minNegLogLikeIdx][1]
    optimStd = models[minNegLogLikeIdx][2]
    print("Finished coarse grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal std: " + str(optimStd))
    print("Min NLL: " + str(min(results)))

    # Get empirical distributions from even trials.
    evenDists = get_empirical_distributions(rt, choice, distLeft, distRight,
        fixItem, fixTime, useOddTrials=False, useEvenTrials=True)
    probLeftFixFirst = evenDists.probLeftFixFirst
    distTransition = evenDists.distTransition
    distFirstFix = evenDists.distFirstFix
    distMiddleFix = evenDists.distMiddleFix

    # Parameters for generating simulations.
    numTrials = 800
    orientations = range(-15,20,5)
    trialConditions = list()
    for oLeft in orientations:
        for oRight in orientations:
            if oLeft != oRight:
                trialConditions.append((oLeft, oRight))

    # Generate simulations using the even trials distributions and the
    # estimated parameters.
    simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
        distMiddleFix, numTrials, trialConditions, optimD, optimTheta,
        std=optimStd)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulDistLeft = simul.distLeft
    simulDistRight = simul.distRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime

    # Get item values for simulations.
    totalTrials = numTrials * len(trialConditions)
    simulValueLeft = dict()
    simulValueRight = dict()
    for trial in xrange(totalTrials):
        simulValueLeft[trial] = np.absolute((np.absolute(
            simulDistLeft[trial])-15)/5)
        simulValueRight[trial] = np.absolute((np.absolute(
            simulDistRight[trial])-15)/5)

    # Create pdf file to save figures.
    pp = PdfPages("figures_" + str(optimD) + "_" + str(optimTheta) + "_" +
        str(optimStd) + "_" + str(numTrials) + ".pdf")

    # Generate choice and rt curves for real data (odd trials) and
    # simulations (generated from even trials).
    fig1 = generate_choice_curves(choice, valueLeft, valueRight, simulChoice,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig1)
    fig2 = generate_rt_curves(rt, valueLeft, valueRight, simulRt,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig2)
    pp.close()

    save_simulations_to_csv(simulChoice, simulRt, simulValueLeft,
        simulValueRight, simulFixItem, simulFixTime, totalTrials)


if __name__ == '__main__':
    main()