# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:23:16 2015

@author: vince
"""

# Here, we make "nice" fake data, by importing some "ok" fake data, and using
# it's RT distributions to feed into the run_simulation module, with known
# parameters

# Load packages

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from handle_fixations import (load_data_from_csv, 
    get_empirical_distributions, run_simulations)

def generate_choice_curves(choicesData, valueLeftData, valueRightData,
    choicesSimul, valueLeftSimul, valueRightSimul, numTrials):
#    countTotal = np.zeros(9)
#    countLeftChosen = np.zeros(9)
#
#    subjects = choicesData.keys()
#    for subject in subjects:
#        trials = choicesData[subject].keys()
#        for trial in trials:
#            valueDiff = (valueLeftData[subject][trial] -
#                valueRightData[subject][trial])
#            idx = valueDiff + 4
#            if choicesData[subject][trial] == -1:  # Choice was left.
#                countLeftChosen[idx] +=1
#                countTotal[idx] += 1
#            elif choicesData[subject][trial] == 1:  # Choice was right.
#                countTotal[idx] += 1
#
#    stdProbLeftChosen = np.zeros(9)
#    probLeftChosen = np.zeros(9)
#    for i in xrange(0,9):
#        probLeftChosen[i] = countLeftChosen[i] / countTotal[i]
#        stdProbLeftChosen[i] = np.sqrt((probLeftChosen[i] *
#            (1 - probLeftChosen[i])) / countTotal[i])
#
    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
#    plt.errorbar(range(-4,5,1), probLeftChosen, yerr=stdProbLeftChosen,
#        color=colors[0], label='Data')

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
#    rtsPerValueDiff = dict()
#    for valueDiff in xrange(-4,5,1):
#        rtsPerValueDiff[valueDiff] = list()
#
#    subjects = rtsData.keys()
#    for subject in subjects:
#        trials = rtsData[subject].keys()
#        for trial in trials:
#            valueDiff = (valueLeftData[subject][trial] -
#                valueRightData[subject][trial])
#            rtsPerValueDiff[valueDiff].append(rtsData[subject][trial])
#
#    meanRts = np.zeros(9)
#    stdRts = np.zeros(9)
#    for valueDiff in xrange(-4,5,1):
#        idx = valueDiff + 4
#        meanRts[idx] = np.mean(np.array(rtsPerValueDiff[valueDiff]))
#        stdRts[idx] = (np.std(np.array(rtsPerValueDiff[valueDiff])) /
#            np.sqrt(len(rtsPerValueDiff[valueDiff])))
#
    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
#    plt.errorbar(range(-4,5,1), meanRts, yerr=stdRts, label='Data',
#        color=colors[0])

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


def main():
    
    data = load_data_from_csv("fake_data_9mar15_choices1_nobias.csv", "fake_data_9mar15_fixations.csv")
    rt = data.rt
    choice = data.choice
    distLeft = data.distLeft
    distRight = data.distRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get empirical distributions.
    dists = get_empirical_distributions(rt, choice, distLeft, distRight,
        fixItem, fixTime)
    probLeftFixFirst = dists.probLeftFixFirst
    distTransition = dists.distTransition
    distFirstFix = dists.distFirstFix
    distSecondFix = dists.distSecondFix
    distThirdFix = dists.distThirdFix
    distOtherFix = dists.distOtherFix

    # Parameters for artificial data generation.
    # Change them as needed to produce different effects in the data
    numTrials = 50
    d = 0.01
    theta = 0.3
    std = 0.1

    values = range(1,6,1)
    trialConditions = list()
    for vLeft in values:
        for vRight in values:
            trialConditions.append((vLeft, vRight))

    # Generate simulations using the even trials distributions and the
    # estimated parameters.
    simul = run_simulations(probLeftFixFirst, distTransition, distFirstFix,
        distSecondFix, distThirdFix, distOtherFix, numTrials, trialConditions,
        d, theta, std)
    simulRt = simul.rt
    simulChoice = simul.choice
    simulValueLeft = simul.valLeft
    simulValueRight = simul.valRight
    simulFixItem = simul.fixItem
    simulFixTime = simul.fixTime
    #simulFixRDV = simul.fixRDV

    # Write artificial data to CSV.

    totalTrials = numTrials * len(trialConditions)
    
    with open("./fake_data/expdata_" + str(d) + "_" + str(theta) + "_" + str(std) + "_" +
        str(numTrials) + ".csv", "wb") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["parcode", "trial", "rt", "choice", "val_left",
            "val_right"])
        for trial in xrange(totalTrials):
            csvWriter.writerow(["dummy_subj", str(trial), str(simulRt[trial]),
                str(simulChoice[trial]), str(simulValueLeft[trial]),
                str(simulValueRight[trial])])

    with open("./fake_data/fixations_" + str(d) + "_" + str(theta) + "_" + str(std) + "_" +
        str(numTrials) + ".csv", "wb") as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(["parcode", "trial", "fix_item", "fix_time"])
        for trial in xrange(totalTrials):
            for fix in xrange(len(simulFixItem[trial])):
                csvWriter.writerow(["dummy_subj", str(trial),
                    str(simulFixItem[trial][fix]),
                    str(simulFixTime[trial][fix])])
                    
    # Make plots choice and RT plots for artificial data

        # Create pdf file to save figures.
    pp = PdfPages("./fake_data/figures_" + str(d) + "_" + str(theta) + "_" +
        str(std) + "_" + str(numTrials) + ".pdf")

    # Generate choice and rt curves for fake data.  The dict() arguments indicate
    # where real data would go, if we had it in this case.  (see functions above)
    totalTrials = numTrials * len(trialConditions)
    fig1 = generate_choice_curves(choice, dict(), dict(), simulChoice,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig1)
    fig2 = generate_rt_curves(rt, dict(), dict(), simulRt,
        simulValueLeft, simulValueRight, totalTrials)
    pp.savefig(fig2)
    pp.close()
                    
if __name__ == '__main__':
    main()
