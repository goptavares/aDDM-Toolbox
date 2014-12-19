#!/usr/bin/python

# optimize.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from scipy.optimize import basinhopping

import numpy as np

from dyn_prog_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions)


# Global variables.
rt = dict()
choice = dict()
valueLeft = dict()
valueRight = dict()
fixItem = dict()
fixTime = dict()


def run_analysis(x):
    trialsPerSubject = 200
    d = x[0]
    theta = x[1]
    mu = x[2]

    logLikelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, mu=mu, plotResults=False)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    print("NLL for " + str(x) + ": " + str(-logLikelihood))
    return -logLikelihood


def main():
    global rt
    global choice
    global valueLeft
    global valueRight
    global fixItem
    global fixTime

    # Load experimental data from CSV file and update global variables.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Initial guess: d, theta, mu.
    x0 = [0.0002, 0.5, 200]

    # Search bounds.
    xmin = [0.00005, 0., 10]
    xmax = [0.01, 1., 1000]
    bounds = [(lower, upper) for lower, upper in zip(xmin, xmax)]

    # Optimize using Basinhopping algorithm.
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
    res = basinhopping(run_analysis, x0, minimizer_kwargs=minimizer_kwargs)
    print res


if __name__ == '__main__':
    main()