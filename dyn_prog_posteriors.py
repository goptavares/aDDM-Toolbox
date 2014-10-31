#!/usr/bin/python

# dyn_prog_posteriors.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import collections
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd

from dyn_prog_fixations import load_data_from_csv, analysis_per_trial


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    numThreads = 1
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv()
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    rangeD = [0.0002]
    rangeTheta = [0.3]
    rangeMu = [490]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeMu)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for mu in rangeMu:
                model = (d, theta, mu)
                models.append(model)
                posteriors[model] = 1./ numModels

    useOddTrials = True
    useEvenTrials = True

    subjects = rt.keys()
    for subject in subjects:
        print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        for trial in trials:
            if trial % 50 == 0:
                print("Running trial " + str(trial) + "...")
            if not useOddTrials and trial % 2 != 0:
                continue
            if not useEvenTrials and trial % 2 == 0:
                continue
            list_params = list()
            for model in models:
                list_params.append((rt[subject][trial], choice[subject][trial],
                    valueLeft[subject][trial], valueRight[subject][trial],
                    fixItem[subject][trial], fixTime[subject][trial], model[0],
                    model[1], model[2], True))
            likelihoods = pool.map(run_analysis_wrapper, list_params)

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

            if trial % 50 == 0:
                for model in posteriors:
                    print("P" + str(model) + " = " + str(posteriors[model]))
                print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
