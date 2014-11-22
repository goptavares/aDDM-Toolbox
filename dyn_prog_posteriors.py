#!/usr/bin/python

# dyn_prog_posteriors.py
# Author: Gabriela Tavares, gtavares@caltech.edu

from multiprocessing import Pool

import numpy as np

from dyn_prog_fixations import (load_data_from_csv, analysis_per_trial,
    get_empirical_distributions, run_simulations)


def run_analysis_wrapper(params):
    return analysis_per_trial(*params)


def main():
    trialsPerSubject = 200
    numThreads = 9
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv()
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    rangeD = [0.0002, 0.0005, 0.0008]
    rangeTheta = [0.3, 0.5, 0.7]
    rangeMu = [100, 300, 500]
    numModels = len(rangeD) * len(rangeTheta) * len(rangeMu)

    models = list()
    posteriors = dict()
    for d in rangeD:
        for theta in rangeTheta:
            for mu in rangeMu:
                model = (d, theta, mu)
                models.append(model)
                posteriors[model] = 1./ numModels

    subjects = rt.keys()
    for subject in subjects:
        print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            list_params = list()
            for model in models:
                list_params.append((rt[subject][trial], choice[subject][trial],
                    valueLeft[subject][trial], valueRight[subject][trial],
                    fixItem[subject][trial], fixTime[subject][trial], model[0],
                    model[1], model[2], False))
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
                if likelihoods[i] != 0:
                    prior = posteriors[model]
                    posteriors[model] = likelihoods[i] * prior / denominator
                i += 1

        for model in posteriors:
            print("P" + str(model) + " = " + str(posteriors[model]))
        print("Sum: " + str(sum(posteriors.values())))


if __name__ == '__main__':
    main()
