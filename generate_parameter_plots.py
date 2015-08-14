#!/usr/bin/python

"""
generate_parameter_plots.py
Author: Gabriela Tavares, gtavares@caltech.edu
"""

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from addm import get_trial_likelihood
from util import load_data_from_csv


def get_model_nll(choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
                  sigma, verbose=True):
    trialsPerSubject = 100
    logLikelihood = 0
    subjects = choice.keys()
    for subject in subjects:
        if verbose:
            print("Running subject " + subject + "...")
        trials = choice[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = get_trial_likelihood(
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, sigma=sigma)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("Negative log likelihood for " + str(d) + ", " + str(theta) +
              ", " + str(sigma) + ": " + str(-logLikelihood))
    return -logLikelihood


def get_model_nll_wrapper(params):
    return get_model_nll(*params)


def main():
    numThreads = 3
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv",
                              useAngularDists=True)
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    coarseRangeD = [0.0008, 0.001, 0.0012]
    coarseRangeTheta = [0.3, 0.5, 0.7]
    coarseRangeSigma = [0.03, 0.06, 0.09]

    fineRangeD = np.arange(0.0001, 0.001, 0.0001)
    fineRangeTheta = np.arange(0.1, 1.0, 0.1)
    fineRangeSigma = np.arange(0.02, 0.11, 0.01)

    likelihoods = dict()

    # Coarse grid search for d.
    models = list()
    listParams = list()
    for d in fineRangeD:
        for theta in coarseRangeTheta:
            for sigma in coarseRangeSigma:
                if not (d, theta, sigma) in likelihoods:
                    models.append((d, theta, sigma))
                    params = (choice, valueLeft, valueRight, fixItem, fixTime,
                              d, theta, sigma)
                    listParams.append(params)

    print("Starting pool of workers for d search...")
    results = pool.map(get_model_nll_wrapper, listParams)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Coarse grid search for theta.
    models = list()
    listParams = list()
    for d in coarseRangeD:
        for theta in fineRangeTheta:
            for sigma in coarseRangeSigma:
                if not (d, theta, sigma) in likelihoods:
                    models.append((d, theta, sigma))
                    params = (choice, valueLeft, valueRight, fixItem, fixTime,
                              d, theta, sigma)
                    listParams.append(params)

    print("Starting pool of workers for theta search...")
    results = pool.map(get_model_nll_wrapper, listParams)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Coarse grid search for sigma.
    models = list()
    listParams = list()
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            for sigma in fineRangeSigma:
                if not (d, theta, sigma) in likelihoods:
                    models.append((d, theta, sigma))
                    params = (choice, valueLeft, valueRight, fixItem, fixTime,
                              d, theta, sigma)
                    listParams.append(params)

    print("Starting pool of workers for sigma search...")
    results = pool.map(get_model_nll_wrapper, listParams)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Create pdf file to save figures.
    pp = PdfPages("figures.pdf")

    # Create color map.
    colors = cm.rainbow(np.linspace(0, 1, 9))

    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 

    # Generate d plots.
    fig1 = plt.figure()
    ax = plt.subplot(111)
    c = 0
    for theta in coarseRangeTheta:
        for sigma in coarseRangeSigma:
            dLikelihoods = list()
            for d in fineRangeD:
                dLikelihoods.append(likelihoods[(d, theta, sigma)])
            ax.plot(fineRangeD, dLikelihoods, color=colors[c],
                    label=(str(theta) + ", " + str(sigma)))
            c += 1
    plt.xlabel("d")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig1)

    # Generate theta plots.
    fig2 = plt.figure()
    ax = plt.subplot(111)
    c = 0
    for d in coarseRangeD:
        for sigma in coarseRangeSigma:
            thetaLikelihoods = list()
            for theta in fineRangeTheta:
                thetaLikelihoods.append(likelihoods[(d, theta, sigma)])
            ax.plot(fineRangeTheta, thetaLikelihoods, color=colors[c],
                    label=(str(d) + ", " + str(sigma)))
            c += 1
    plt.xlabel("theta")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig2)

    # Generate sigma plots.
    fig3 = plt.figure()
    ax = plt.subplot(111)
    c = 0
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            sigmaLikelihoods = list()
            for sigma in fineRangeSigma:
                sigmaLikelihoods.append(likelihoods[(d, theta, sigma)])
            ax.plot(fineRangeSigma, sigmaLikelihoods,
                    color=colors[c], label=(str(d) + ", " + str(theta)))
            c += 1
    plt.xlabel("sigma")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig3)

    # Generate zoomed-in d plots.
    for theta in coarseRangeTheta:
        for sigma in coarseRangeSigma:
            dLikelihoods = list()
            for d in fineRangeD:
                dLikelihoods.append(likelihoods[(d, theta, sigma)])
            fig = plt.figure()
            plt.plot(fineRangeD, dLikelihoods)
            plt.title(str(theta) + ", " + str(sigma))
            plt.xlabel("d")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in theta plots.
    for d in coarseRangeD:
        for sigma in coarseRangeSigma:
            thetaLikelihoods = list()
            for theta in fineRangeTheta:
                thetaLikelihoods.append(likelihoods[(d, theta, sigma)])
            fig = plt.figure()
            plt.plot(fineRangeTheta, thetaLikelihoods)
            plt.title(str(d) + ", " + str(sigma))
            plt.xlabel("theta")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in sigma plots.
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            sigmaLikelihoods = list()
            for sigma in fineRangeSigma:
                sigmaLikelihoods.append(likelihoods[(d, theta, sigma)])
            fig = plt.figure()
            plt.plot(fineRangeSigma, sigmaLikelihoods)
            plt.title(str(d) + ", " + str(theta))
            plt.xlabel("sigma")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    pp.close()
   

if __name__ == '__main__':
    main()
