#!/usr/bin/python

# generate_parameter_plots.py
# Author: Gabriela Tavares, gtavares@caltech.edu

import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd

from dyn_prog_fixations import (load_data_from_csv, analysis_per_trial)


def run_analysis(rt, choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
    mu, verbose=True):
    trialsPerSubject = 100
    logLikelihood = 0
    subjects = rt.keys()
    for subject in subjects:
        if verbose:
            print("Running subject " + subject + "...")
        trials = rt[subject].keys()
        trialSet = np.random.choice(trials, trialsPerSubject, replace=False)
        for trial in trialSet:
            likelihood = analysis_per_trial(rt[subject][trial],
                choice[subject][trial], valueLeft[subject][trial],
                valueRight[subject][trial], fixItem[subject][trial],
                fixTime[subject][trial], d, theta, mu=mu, plotResults=False)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("Negative log likelihood for " + str(d) + ", " + str(theta) + ", "
            + str(mu) + ": " + str(-logLikelihood))
    return -logLikelihood


def run_analysis_wrapper(params):
    return run_analysis(*params)


def main():
    numThreads = 3
    pool = Pool(numThreads)

    # Load experimental data from CSV file.
    data = load_data_from_csv("expdata.csv", "fixations.csv")
    rt = data.rt
    choice = data.choice
    valueLeft = data.valueLeft
    valueRight = data.valueRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    print("Starting coarse grid search...")
    coarseRangeD = [0.0008, 0.001, 0.0012]
    coarseRangeTheta = [0.3, 0.5, 0.7]
    coarseRangeMu = [50, 100, 150]

    fineRangeD = np.arange(0.0001, 0.001, 0.0001)
    fineRangeTheta = np.arange(0.1, 1.0, 0.1)
    fineRangeMu = np.arange(100, 1000, 100)

    likelihoods = dict()

    # Coarse grid search for d.
    models = list()
    list_params = list()
    for d in fineRangeD:
        for theta in coarseRangeTheta:
            for mu in coarseRangeMu:
                if not (d, theta, mu) in likelihoods:
                    models.append((d, theta, mu))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, mu)
                    list_params.append(params)

    print("Starting pool of workers for d search...")
    results = pool.map(run_analysis_wrapper, list_params)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Coarse grid search for theta.
    models = list()
    list_params = list()
    for d in coarseRangeD:
        for theta in fineRangeTheta:
            for mu in coarseRangeMu:
                if not (d, theta, mu) in likelihoods:
                    models.append((d, theta, mu))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, mu)
                    list_params.append(params)

    print("Starting pool of workers for theta search...")
    results = pool.map(run_analysis_wrapper, list_params)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Coarse grid search for mu.
    models = list()
    list_params = list()
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            for mu in fineRangeMu:
                if not (d, theta, mu) in likelihoods:
                    models.append((d, theta, mu))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, mu)
                    list_params.append(params)

    print("Starting pool of workers for mu search...")
    results = pool.map(run_analysis_wrapper, list_params)

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
        for mu in coarseRangeMu:
            d_likelihoods = list()
            for d in fineRangeD:
                d_likelihoods.append(likelihoods[(d, theta, mu)])
            ax.plot(fineRangeD, d_likelihoods,
                color=colors[c], label=(str(theta) + ", " + str(mu)))
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
        for mu in coarseRangeMu:
            theta_likelihoods = list()
            for theta in fineRangeTheta:
                theta_likelihoods.append(likelihoods[(d, theta, mu)])
            ax.plot(fineRangeTheta, theta_likelihoods,
                color=colors[c], label=(str(d) + ", " + str(mu)))
            c += 1
    plt.xlabel("theta")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig2)

    # Generate mu plots.
    fig3 = plt.figure()
    ax = plt.subplot(111)
    c = 0
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            mu_likelihoods = list()
            for mu in fineRangeMu:
                mu_likelihoods.append(likelihoods[(d, theta, mu)])
            ax.plot(fineRangeMu, mu_likelihoods,
                color=colors[c], label=(str(d) + ", " + str(theta)))
            c += 1
    plt.xlabel("mu")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig3)

    # Generate zoomed-in d plots.
    for theta in coarseRangeTheta:
        for mu in coarseRangeMu:
            d_likelihoods = list()
            for d in fineRangeD:
                d_likelihoods.append(likelihoods[(d, theta, mu)])
            fig = plt.figure()
            plt.plot(fineRangeD, d_likelihoods)
            plt.title(str(theta) + ", " + str(mu))
            plt.xlabel("d")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in theta plots.
    for d in coarseRangeD:
        for mu in coarseRangeMu:
            theta_likelihoods = list()
            for theta in fineRangeTheta:
                theta_likelihoods.append(likelihoods[(d, theta, mu)])
            fig = plt.figure()
            plt.plot(fineRangeTheta, theta_likelihoods)
            plt.title(str(d) + ", " + str(mu))
            plt.xlabel("theta")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in mu plots.
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            mu_likelihoods = list()
            for mu in fineRangeMu:
                mu_likelihoods.append(likelihoods[(d, theta, mu)])
            fig = plt.figure()
            plt.plot(fineRangeMu, mu_likelihoods)
            plt.title(str(d) + ", " + str(theta))
            plt.xlabel("mu")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    pp.close()
   

if __name__ == '__main__':
    main()