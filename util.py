#!/usr/bin/python

"""
util.py
Author: Gabriela Tavares, gtavares@caltech.edu

Utility functions for the aDDM toolbox.
"""

import matplotlib
matplotlib.use('Agg')

import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data_from_csv(expdataFileName, fixationsFileName,
                       useAngularDists=False):
    """
    Loads experimental data from two CSV files: an experimental data file and a
    fixations file. If angular distances are used, they are expected to be from
    the set [-15, -10, -5, 0, 5, 10, 15] and will be converted into values in
    [0, 1, 2, 3]. Format expected for experimental data file: parcode, trial,
    rt, choice, item_left, item_right. Format expected for fixations file:
    parcode, trial, fix_item, fix_time.
    Args:
      expdataFileName: string, name of experimental data file.
      fixationsFileName: string, name of fixations file.
      useAngularDists: boolean, must be True when the data is from a perceptual
          task and contains angular distances instead of item values.
    Returns:
      A named tuple containing the following fields:
        RT: dict of dicts, indexed first by subject then by trial number. Each
            entry is a number corresponding to the reaction time in the trial.
        choice: dict of dicts with same indexing as RT. Each entry is an integer
            corresponding to the decision made in that trial.
        valueLeft: dict of dicts with same indexing as RT. Each entry is an
            integer corresponding to the value of the left item.
        valueRight: dict of dicts with same indexing as RT. Each entry is an
            integer corresponding to the value of the right item.
        fixItem: dict of dicts with same indexing as RT. Each entry is an
            ordered list of fixated items in the trial.
        fixTime: dict of dicts with same indexing as RT. Each entry is an
            ordered list of fixation durations in the trial.
        isCisTrial: dict of dicts with same indexing as RT. Applies to
            perceptual decisions only. Each entry is a boolean indicating if the
            trial is cis (both bars on the same side of the target).
        isTransTrial: dict of dicts with same indexing as RT. Applies to
            perceptual decisions only. Each entry is a boolean indicating if the
            trial is trans (bars on either side of the target).
    """

    # Load experimental data from CSV file.
    try:
        df = pd.DataFrame.from_csv(expdataFileName, header=0, sep=',',
                                   index_col=None)
    except:
        print ("Error while reading file " + expdataFileName)
        raise

    if ("parcode" not in df.columns or "trial" not in df.columns or
        "rt" not in df.columns or "choice" not in df.columns or
        "item_left" not in df.columns or "item_right" not in df.columns):
        raise RuntimeError("Missing field in experimental data file. Fields "
                           "required: parcode, trial, rt, choice, item_left "
                           "item_right")

    subjects = df.parcode.unique()
    RT = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict()
    isCisTrial = dict()
    isTransTrial = dict()

    for subject in subjects:
        RT[subject] = dict()
        choice[subject] = dict()
        valueLeft[subject] = dict()
        valueRight[subject] = dict()
        isCisTrial[subject] = dict()
        isTransTrial[subject] = dict()
        dataSubject = np.array(
            df.loc[df['parcode']==subject,
            ['trial','rt','choice','item_left','item_right']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(
                df.loc[(df['trial']==trial) & (df['parcode']==subject),
                ['rt','choice','item_left', 'item_right']])
            RT[subject][trial] = dataTrial[0,0]
            choice[subject][trial] = dataTrial[0,1]
            itemLeft = dataTrial[0,2]
            itemRight = dataTrial[0,3]
            isCisTrial[subject][trial] = False
            isTransTrial[subject][trial] = False

            if useAngularDists:
                valueLeft[subject][trial] = np.absolute(
                    (np.absolute(itemLeft) - 15) / 5)
                valueRight[subject][trial] = np.absolute(
                    (np.absolute(itemRight) - 15) / 5)
                if itemLeft * itemRight >= 0:
                    isCisTrial[subject][trial] = True
                if itemLeft * itemRight <= 0:
                    isTransTrial[subject][trial] = True
            else:
                valueLeft[subject][trial] = itemLeft
                valueRight[subject][trial] = itemRight

    # Load fixation data from CSV file.
    try:
        df = pd.DataFrame.from_csv(
            fixationsFileName, header=0, sep=',', index_col=None)
    except:
        print ("Error while reading file " + fixationsFileName)
        raise

    if ("parcode" not in df.columns or "trial" not in df.columns or
        "fix_item" not in df.columns or "fix_time" not in df.columns):
        raise RuntimeError("Missing field in fixations file. Fields required: "
                           "parcode, trial, fix_item, fix_time")

    subjects = df.parcode.unique()
    fixItem = dict()
    fixTime = dict()

    for subject in subjects:
        fixItem[subject] = dict()
        fixTime[subject] = dict()
        dataSubject = np.array(
            df.loc[df['parcode']==subject, ['trial','fix_item','fix_time']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(
                df.loc[(df['trial']==trial) & (df['parcode']==subject),
                ['fix_item','fix_time']])
            fixItem[subject][trial] = dataTrial[:,0]
            fixTime[subject][trial] = dataTrial[:,1]

    data = collections.namedtuple(
        'Data', ['RT', 'choice', 'valueLeft', 'valueRight', 'fixItem',
        'fixTime', 'isCisTrial', 'isTransTrial'])
    return data(RT, choice, valueLeft, valueRight, fixItem, fixTime, isCisTrial,
                isTransTrial)


def save_simulations_to_csv(choice, RT, valueLeft, valueRight, fixItem,
                            fixTime, fixRDV, numTrials):
    """
    Saves the simulations generated with the aDDM algorithm into 7 CSV files.
    The following files contain a single row where each entry corresponds to a
    simulated trial: choice.csv contains the chosen items; rt.csv contains the
    reaction times; value_left.csv contains the value of the left items; and
    value_right.csv contains the value of the right items. In the following
    files, each column corresponds to a simulated trial, and each column entry
    corresponds to a fixation within the trial: fix_item.csv contains the
    fixated items; fix_time.csv contains the fixation durations; and fix_rdv.csv
    contains the value of the RDV at the beginning of each fixation.
    Args:
      choice: dict indexed by trial number, where each entry is an integer
          corresponding to the decision made in that trial.
      RT: dict indexed by trial number, where each entry is a number
          corresponding to the reaction time in that trial.
      valueLeft: dict indexed by trial number, where each entry is an integer
          corresponding to the value of the left item.
      valueRight: dict indexed by trial number, where each entry is an integer
          corresponding to the value of the right item.
      fixItem: dict indexed by trial number, where each entry is an ordered list
          of fixated items in the trial.
      fixTime: dict indexed by trial number, where each entry is an ordered list
          of fixation durations in the trial.
      fixRDV: dict indexed by trial number, where each entry is an ordered list
          of floats corresponding to the value of the RDV at the start of each
          fixation in the trial.
      numTrials: integer, number of trials to be saved.
    """

    df = pd.DataFrame(choice, index=range(1))
    try:
        df.to_csv('choice.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save choices to CSV file.")
        raise
    df = pd.DataFrame(RT, index=range(1))
    try:
        df.to_csv('rt.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save RTs to CSV file.")
        raise

    dictItem = dict()
    dictTime = dict()
    dictRDV = dict()
    for trial in xrange(0, numTrials):
        dictItem[trial] = pd.Series(fixItem[trial])
        dictTime[trial] = pd.Series(fixTime[trial])
        dictRDV[trial] = pd.Series(fixRDV[trial])

    df = pd.DataFrame(valueLeft, index=range(1))
    try:
        df.to_csv('value_left.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save left item values to CSV file.")
        raise
    df = pd.DataFrame(valueRight, index=range(1))
    try:
        df.to_csv('value_right.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save right item values to CSV file.")
        raise
    df = pd.DataFrame(dictItem)
    try:
        df.to_csv('fix_item.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save fixated items to CSV file.")
        raise
    df = pd.DataFrame(dictTime)
    try:
        df.to_csv('fix_time.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save fixation durations to CSV file.")
        raise
    df = pd.DataFrame(dictRDV)
    try:
        df.to_csv('fix_rdv.csv', header=0, sep=',', index_col=None)
    except:
        print("Failed to save RDV values to CSV file.")
        raise


def generate_choice_curves(choicesData, valueLeftData, valueRightData,
                           choicesSimul, valueLeftSimul, valueRightSimul,
                           numTrials):
    """
    Plots the psychometric choice curves for data and simulations.
    Args:
      choicesData: dict of dicts, indexed first by subject then by trial number.
          Each entry is either -1 (choice was left) or +1 (choice was right).
      valueLeftData: dict of dicts with same indexing as choicesData. Each entry
          is an integer corresponding to the value of the left item.
      valueRightData: dict of dicts with same indexing as choicesData. Each
          entry is an integer corresponding to the value of the right item.
      choicesSimul: dict indexed by trial number, where each entry is either -1
          (choice was left) or +1 (choice was right).
      valueLeftSimul: dict indexed by trial number, where each entry is an
          integer corresponding to the value of the left item.
      valueRightSimul: dict indexed by trial number, where each entry is an
          integer corresponding to the value of the right item.
      numTrials: integer, number of trials to be used from the simulations.
    Returns:
      A handle to a figure with the plotted choice curves.
    """

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
        stdProbLeftChosen[i] = np.sqrt(
            (probLeftChosen[i] * (1 - probLeftChosen[i])) / countTotal[i])

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
        stdProbLeftChosen[i] = np.sqrt(
            (probLeftChosen[i] * (1 - probLeftChosen[i])) / countTotal[i])

    plt.errorbar(range(-4,5,1), probLeftChosen, yerr=stdProbLeftChosen,
                 color=colors[5], label='Simulations')
    plt.xlabel('Value difference')
    plt.ylabel('P(choose left)')
    plt.legend()
    return fig


def generate_rt_curves(RTsData, valueLeftData, valueRightData, RTsSimul,
                       valueLeftSimul, valueRightSimul, numTrials):
    """
    Plots the reaction times for data and simulations.
    Args:
      RTsData: dict of dicts, indexed first by subject then by trial number.
          Each entry is a number corresponding to the reaction time in the
          trial.
      valueLeftData: dict of dicts with same indexing as RTsData. Each entry is
          an integer corresponding to the value of the left item.
      valueRightData: dict of dicts with same indexing as RTsData. Each entry is
          an integer corresponding to the value of the right item.
      RTsSimul: dict indexed by trial number, where each entry is a number
          correponding to the reaction time in the trial.
      valueLeftSimul: dict indexed by trial number, where each entry is an
          integer corresponding to the value of the left item.
      valueRightSimul: dict indexed by trial number, where each entry is an
          integer corresponding to the value of the right item.
      numTrials: integer, number of trials to be used from the simulations.
    Returns:
      A handle to a figure with the plotted reaction times.
    """

    RTsPerValueDiff = dict()
    for valueDiff in xrange(-4,5,1):
        RTsPerValueDiff[valueDiff] = list()

    subjects = RTsData.keys()
    for subject in subjects:
        trials = RTsData[subject].keys()
        for trial in trials:
            valueDiff = (valueLeftData[subject][trial] -
                         valueRightData[subject][trial])
            RTsPerValueDiff[valueDiff].append(RTsData[subject][trial])

    meanRTs = np.zeros(9)
    stdRTs = np.zeros(9)
    for valueDiff in xrange(-4,5,1):
        idx = valueDiff + 4
        meanRTs[idx] = np.mean(np.array(RTsPerValueDiff[valueDiff]))
        stdRTs[idx] = (np.std(np.array(RTsPerValueDiff[valueDiff])) /
                       np.sqrt(len(RTsPerValueDiff[valueDiff])))

    colors = cm.rainbow(np.linspace(0, 1, 9))
    fig = plt.figure()
    plt.errorbar(range(-4,5,1), meanRTs, yerr=stdRTs, label='Data',
                 color=colors[0])

    RTsPerValueDiff = dict()
    for valueDiff in xrange(-4,5,1):
        RTsPerValueDiff[valueDiff] = list()

    for trial in xrange(0, numTrials):
        valueDiff = valueLeftSimul[trial] - valueRightSimul[trial]
        RTsPerValueDiff[valueDiff].append(RTsSimul[trial])

    meanRTs = np.zeros(9)
    stdRTs = np.zeros(9)
    for valueDiff in xrange(-4,5,1):
        idx = valueDiff + 4
        meanRTs[idx] = np.mean(np.array(RTsPerValueDiff[valueDiff]))
        stdRTs[idx] = (np.std(np.array(RTsPerValueDiff[valueDiff])) /
                       np.sqrt(len(RTsPerValueDiff[valueDiff])))

    plt.errorbar(range(-4,5,1), meanRTs, yerr=stdRTs, label='Simulations',
                 color=colors[5])
    plt.xlabel('Value difference')
    plt.ylabel('Mean RT')
    plt.legend()
    return fig
