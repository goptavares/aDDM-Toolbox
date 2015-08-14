#!/usr/bin/python

"""
run_model.py
Author: Gabriela Tavares, gtavares@caltech.edu
"""

import numpy as np
import sys

from addm import get_trial_likelihood
from util import load_data_from_csv


def get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
                  sigma, useOddTrials=True, useEvenTrials=True):
    trialsPerSubject = 1200
    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        print("Running subject " + subject + "...")
        trials = choice[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            likelihood = get_trial_likelihood(
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, sigma=sigma)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)
    return -logLikelihood


def main(argv):
    d = float(argv[0])
    sigma = float(argv[1])
    theta = float(argv[2])

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    NLL = get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d,
                        theta, sigma)

    print("d: " + str(d))
    print("theta: " + str(theta))
    print("sigma: " + str(sigma))
    print("NLL: " + str(NLL))


if __name__ == '__main__':
    main(sys.argv[1:])
