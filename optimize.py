#!/usr/bin/python

# optimize.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from scipy.optimize import basinhopping

import numpy as np

from addm import analysis_per_trial
from util import load_data_from_csv


# Global variables.
choice = dict()
valueLeft = dict()
valueRight = dict()
fixItem = dict()
fixTime = dict()


def run_analysis(x):
    trialsPerSubject = 200
    d = x[0]
    theta = x[1]
    std = x[2]

    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        trials = choice[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = analysis_per_trial(choice[subject][trial],
                valueLeft[subject][trial], valueRight[subject][trial],
                fixItem[subject][trial], fixTime[subject][trial], d, theta,
                std=std)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    print("NLL for " + str(x) + ": " + str(-logLikelihood))
    return -logLikelihood


def main():
    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime

    # Load experimental data from CSV file and update global variables.
    data = load_data_from_csv("expdata.csv", "fixations.csv", True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Initial guess: d, theta, std.
    x0 = [0.0002, 0.5, 0.08]

    # Search bounds.
    xmin = [0.00005, 0., 0.05]
    xmax = [0.01, 1., 0.1]
    bounds = [(lower, upper) for lower, upper in zip(xmin, xmax)]

    # Optimize using Basinhopping algorithm.
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
    res = basinhopping(run_analysis, x0, minimizer_kwargs=minimizer_kwargs)
    print res


if __name__ == '__main__':
    main()