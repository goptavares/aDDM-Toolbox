#!/usr/bin/python

# util.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import matplotlib
matplotlib.use('Agg')

import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data_from_csv(expdataFile, fixationsFile):
    # Load experimental data from CSV file.
    # Format: parcode, trial, rt, choice, dist_left, dist_right.
    # Angular distances to target are transformed to values in [0, 1, 2, 3].
    df = pd.DataFrame.from_csv(expdataFile, header=0, sep=',', index_col=None)
    subjects = df.parcode.unique()

    rt = dict()
    choice = dict()
    distLeft = dict()
    distRight = dict()

    for subject in subjects:
        rt[subject] = dict()
        choice[subject] = dict()
        distLeft[subject] = dict()
        distRight[subject] = dict()
        dataSubject = np.array(df.loc[df['parcode']==subject,
            ['trial','rt','choice','dist_left','dist_right']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(df.loc[(df['trial']==trial) &
                (df['parcode']==subject), ['rt','choice','dist_left',
                'dist_right']])
            rt[subject][trial] = dataTrial[0,0]
            choice[subject][trial] = dataTrial[0,1]
            distLeft[subject][trial] = dataTrial[0,2]
            distRight[subject][trial] = dataTrial[0,3]

    # Load fixation data from CSV file.
    # Format: parcode, trial, fix_item, fix_time.
    df = pd.DataFrame.from_csv(fixationsFile, header=0, sep=',',
        index_col=None)
    subjects = df.parcode.unique()

    fixItem = dict()
    fixTime = dict()

    for subject in subjects:
        fixItem[subject] = dict()
        fixTime[subject] = dict()
        dataSubject = np.array(df.loc[df['parcode']==subject,
            ['trial','fix_item','fix_time']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(df.loc[(df['trial']==trial) &
                (df['parcode']==subject), ['fix_item','fix_time']])
            fixItem[subject][trial] = dataTrial[:,0]
            fixTime[subject][trial] = dataTrial[:,1]

    data = collections.namedtuple('Data', ['rt', 'choice', 'distLeft',
        'distRight', 'fixItem', 'fixTime'])
    return data(rt, choice, distLeft, distRight, fixItem, fixTime)


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
        dictValueLeft[trial] = (valueLeft[trial] - 3) * 5
        dictValueRight[trial] = (valueRight[trial] - 3) * 5
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