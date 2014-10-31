#!/usr/bin/python

# dyn_prog_fixations.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool
from numba import jit
from scipy.stats import norm

import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data_from_csv():
    # Load experimental data from CSV file.
    # Format: parcode, trial, rt, choice, dist_left, dist_right.
    # Angular distances to target are transformed to values in [-3, -1, 1, 3].
    df = pd.DataFrame.from_csv('expdata.csv', header=0, sep=',', index_col=None)
    subjects = df.parcode.unique()

    rt = dict()
    choice = dict()
    valueLeft = dict()
    valueRight = dict() 

    for subject in subjects:
        rt[subject] = dict()
        choice[subject] = dict()
        valueLeft[subject] = dict()
        valueRight[subject] = dict()
        dataSubject = np.array(df.loc[df['parcode']==subject,
            ['trial','rt','choice','dist_left','dist_right']])
        trials = np.unique(dataSubject[:,0]).tolist()
        for trial in trials:
            dataTrial = np.array(df.loc[(df['trial']==trial) &
                (df['parcode']==subject), ['rt','choice','dist_left',
                'dist_right']])
            rt[subject][trial] = dataTrial[0,0]
            choice[subject][trial] = dataTrial[0,1]
            valueLeft[subject][trial] = (-np.absolute(dataTrial[0,2])/2.5) + 3
            valueRight[subject][trial] = (-np.absolute(dataTrial[0,3])/2.5) + 3

    # Load fixation data from CSV file.
    # Format: parcode, trial, fix_item, fix_time.
    df = pd.DataFrame.from_csv('fixations.csv', header=0, sep=',',
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

    data = collections.namedtuple('Data', ['rt', 'choice', 'valueLeft',
        'valueRight', 'fixItem', 'fixTime'])
    return data(rt, choice, valueLeft, valueRight, fixItem, fixTime)


@jit("(f8,f8,f8,f8,f8[:],f8[:],f8,f8,f8)")
def analysis_per_trial(rt, choice, valueLeft, valueRight, fixItem, fixTime, d,
    theta, mu, plotResults=False):
    # Parameters of the grid.
    stateStep = 0.1
    timeStep = 10
    initialBarrierUp = 1
    initialBarrierDown = -1
    std = mu * d

    # Iterate over the fixations and get the transition time for this trial.
    itemFixTime = 0
    transitionTime = 0
    for fItem, fTime in zip(fixItem, fixTime):
        if fItem == 1 or fItem == 2:
            itemFixTime += fTime // timeStep
        else:
            transitionTime += fTime // timeStep

    # The total time of this trial is given by the sum of all fixations in the
    # trial.
    maxTime = int(itemFixTime + transitionTime)
    if maxTime == 0:
        return 0

    # We start couting the trial time at the end of the transition time.
    time = int(transitionTime)

    # The values of the barriers can change over time.
    decay = 0  # decay = 0 means barriers are constant.
    barrierUp = initialBarrierUp * np.ones(maxTime)
    barrierDown = initialBarrierDown * np.ones(maxTime)
    for t in xrange(0, int(maxTime)):
        barrierUp[t] = float(initialBarrierUp) / float(1+decay*(t+1))
        barrierDown[t] = float(initialBarrierDown) / float(1+decay*(t+1))

    # The vertical axis (RDV space) is divided into states.
    states = np.arange(initialBarrierDown, initialBarrierUp + stateStep,
        stateStep)
    idx = np.where(np.logical_and(states<0.01, states>-0.01))[0]
    states[idx] = 0

    # Initial probability for all states is zero, except for the zero state,
    # which has initial probability equal to one.
    prStates = np.zeros(states.size)
    idx = np.where(states==0)[0]
    prStates[idx] = 1

    # The probability of crossing each barrier over the time of the trial.
    probUpCrossing = np.zeros(maxTime)
    probDownCrossing = np.zeros(maxTime)

    # Create matrix of traces to keep track of the RDV position probabilities.
    traces = np.zeros((states.size + 2, maxTime))
    for i in xrange(int(transitionTime)):
        traces[1:traces.shape[0]-1,i] = prStates
        traces[0,i] = 0
        traces[traces.shape[0]-1,i] = 0

    # Iterate over all fixations in this trial.
    for fItem, fTime in zip(fixItem, fixTime):
        # We use a normal distribution to model changes in RDV stochastically.
        # The mean of the distribution (the change most likely to occur) is
        # calculated from the model parameters and from the item values.
        if fItem == 1:  # Subject is looking left.
            mean = d * (valueLeft - (theta * valueRight))
        elif fItem == 2:  # Subject is looking right.
            mean = d * (-valueRight + (theta * valueLeft))
        else:
            continue

        # Iterate over the time interval of this fixation.
        for t in xrange(0, int(fTime // timeStep)):
            prStatesNew = np.zeros(states.size)

            # Update the probability of the states that remain inside the
            # barriers.
            for s in xrange(0, states.size):
                currState = states[s]
                if (currState > barrierDown[time] and
                    currState < barrierUp[time]):
                    change = (currState * np.ones(states.size)) - states
                    # The probability of being in state B is the sum, over all
                    # states A, of the probability of being in A at the previous
                    # timestep times the probability of changing from A to B.
                    # We multiply the probability by the stateStep to ensure
                    # that the area under the curve for the probability
                    # distributions probUpCrossing and probDownCrossing each add
                    # up to 1.
                    prStatesNew[s] = (stateStep * np.sum(np.multiply(prStates,
                        norm.pdf(change,mean,std))))

            # Calculate the probabilities of crossing the up barrier and the
            # down barrier. This is given by the sum, over all states A, of the
            # probability of being in A at the previous timestep times the
            # probability of crossing the barrier if A is the previous state.
            changeUp = (barrierUp[time] * np.ones(states.size)) - states
            tempUpCross = np.sum(np.multiply(prStates,
                (1 - norm.cdf(changeUp,mean,std))))
            changeDown = (barrierDown[time] * np.ones(states.size)) - states
            tempDownCross = np.sum(np.multiply(prStates,
                (norm.cdf(changeDown,mean,std))))

            # Renormalize to cope with numerical approximations.
            sumIn = np.sum(prStates)
            sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
            prStatesNew = (prStatesNew * float(sumIn)) / float(sumCurrent)
            tempUpCross = (tempUpCross * float(sumIn)) / float(sumCurrent)
            tempDownCross = (tempDownCross * float(sumIn)) / float(sumCurrent)

            # Update the probabilities of each state and the probabilities of
            # crossing each barrier at this timestep.
            prStates = prStatesNew
            probUpCrossing[time] = tempUpCross
            probDownCrossing[time] = tempDownCross

            # Update traces matrix.
            traces[1:traces.shape[0]-1,time] = prStates
            traces[0,time] = tempUpCross
            traces[traces.shape[0]-1,time] = tempDownCross

            time += 1

    # Compute the log likelihood contribution of this trial based on the final
    # choice.
    likelihood = 0
    if choice == -1:  # Choice was left.
        if probUpCrossing[-1] > 0:
            likelihood = probUpCrossing[-1]
    elif choice == 1:  # Choice was right.
        if probDownCrossing[-1] > 0:
            likelihood = probDownCrossing[-1]

    if plotResults:
        fig1 = plt.figure()
        xAxis = np.arange(0, maxTime * timeStep, timeStep)
        yAxis = np.arange(initialBarrierDown, initialBarrierUp + stateStep,
            stateStep)
        heatmap = plt.pcolor(xAxis, yAxis, np.flipud(traces))
        plt.xlim(0, maxTime * timeStep - timeStep)
        plt.xlabel('Time')
        plt.ylabel('RDV')
        plt.colorbar(heatmap)

        fig2 = plt.figure()
        plt.plot(range(len(probUpCrossing)), probUpCrossing, label='Up')
        plt.plot(range(len(probDownCrossing)), probDownCrossing, label='Down')
        plt.xlabel('Time')
        plt.ylabel('P(crossing)')
        plt.legend()
        plt.show()

    return likelihood


def run_analysis(rt, choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
    mu, useOddTrials=True, useEvenTrials=True, verbose=True):
    likelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        if verbose:
            print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        for trial in trials:
            if verbose and trial % 200 == 0:
                print("Running trial " + str(trial) + "...")
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            likelihood += analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, mu, plotResults=False)

    if verbose:
        print("Likelihood for " + str(d) + ", " + str(theta) + ", " + str(mu) +
            ": " + str(likelihood))
    return likelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 8
    pool = Pool(numThreads)

    data = load_data_from_csv()
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Coarse grid search on the parameters of the model.
    print("Starting coarse grid search...")
    rangeD = [0.0002, 0.0003, 0.0004]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeMu = [80, 100, 120]

    models = list()
    list_params = list()
    for d in rangeD:
        for theta in rangeTheta:
            for mu in rangeMu:
                models.append((d, theta, mu))
                params = (rt, choice, valueLeft, valueRight, fixItem, fixTime,
                    d, theta, mu)
                list_params.append(params)

    print("Starting pool of workers...")
    results_coarse = pool.map(run_analysis_wrapper, list_params)

    # Get optimal parameters.
    max_likelihood_idx = results_coarse.index(max(results_coarse))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimMu = models[max_likelihood_idx][2]
    print("Finished coarse grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal mu: " + str(optimMu))

    # Fine grid search on the parameters of the model.
    print("Starting fine grid search...")
    rangeD = [optimD-0.000025, optimD, optimD+0.000025]
    rangeTheta = [optimTheta-0.1, optimTheta, optimTheta+0.1]
    rangeMu = [optimMu-10, optimMu, optimMu+10]

    models = list()
    list_params = list()
    for d in rangeD:
        for theta in rangeTheta:
            for mu in rangeMu:
                models.append((d, theta, mu))
                params = (rt, choice, valueLeft, valueRight, fixItem, fixTime,
                    d, theta, mu)
                list_params.append(params)

    print("Starting pool of workers...")
    results_fine = pool.map(run_analysis_wrapper, list_params)

    # Get optimal parameters.
    max_likelihood_idx = results_fine.index(max(results_fine))
    optimD = models[max_likelihood_idx][0]
    optimTheta = models[max_likelihood_idx][1]
    optimMu = models[max_likelihood_idx][2]
    print("Finished fine grid search!")
    print("Optimal d: " + str(optimD))
    print("Optimal theta: " + str(optimTheta))
    print("Optimal mu: " + str(optimMu))


if __name__ == '__main__':
    main()
