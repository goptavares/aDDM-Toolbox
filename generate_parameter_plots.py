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

from handle_fixations import load_data_from_csv, analysis_per_trial


def run_analysis(rt, choice, valueLeft, valueRight, fixItem, fixTime, d, theta,
    std, verbose=True):
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
                fixTime[subject][trial], d, theta, std=std)
            if likelihood != 0:
                logLikelihood += np.log(likelihood)

    if verbose:
        print("Negative log likelihood for " + str(d) + ", " + str(theta) + ", "
            + str(std) + ": " + str(-logLikelihood))
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
    distLeft = data.distLeft
    distRight = data.distRight
    fixItem = data.fixItem
    fixTime = data.fixTime

    # Get item values.
    valueLeft = dict()
    valueRight = dict()
    subjects = distLeft.keys()
    for subject in subjects:
        valueLeft[subject] = dict()
        valueRight[subject] = dict()
        trials = distLeft[subject].keys()
        for trial in trials:
            valueLeft[subject][trial] = np.absolute((np.absolute(
                distLeft[subject][trial])-15)/5)
            valueRight[subject][trial] = np.absolute((np.absolute(
                distRight[subject][trial])-15)/5)

    coarseRangeD = [0.0008, 0.001, 0.0012]
    coarseRangeTheta = [0.3, 0.5, 0.7]
    coarseRangeStd = [0.03, 0.06, 0.09]

    fineRangeD = np.arange(0.0001, 0.001, 0.0001)
    fineRangeTheta = np.arange(0.1, 1.0, 0.1)
    fineRangeStd = np.arange(0.02, 0.11, 0.01)

    likelihoods = dict()

    # Coarse grid search for d.
    models = list()
    list_params = list()
    for d in fineRangeD:
        for theta in coarseRangeTheta:
            for std in coarseRangeStd:
                if not (d, theta, std) in likelihoods:
                    models.append((d, theta, std))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, std)
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
            for std in coarseRangeStd:
                if not (d, theta, std) in likelihoods:
                    models.append((d, theta, std))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, std)
                    list_params.append(params)

    print("Starting pool of workers for theta search...")
    results = pool.map(run_analysis_wrapper, list_params)

    for i in xrange(0, len(results)):
        likelihoods[models[i]] = results[i]

    # Coarse grid search for std.
    models = list()
    list_params = list()
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            for std in fineRangeStd:
                if not (d, theta, std) in likelihoods:
                    models.append((d, theta, std))
                    params = (rt, choice, valueLeft, valueRight, fixItem,
                        fixTime, d, theta, std)
                    list_params.append(params)

    print("Starting pool of workers for std search...")
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
        for std in coarseRangeStd:
            d_likelihoods = list()
            for d in fineRangeD:
                d_likelihoods.append(likelihoods[(d, theta, std)])
            ax.plot(fineRangeD, d_likelihoods,
                color=colors[c], label=(str(theta) + ", " + str(std)))
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
        for std in coarseRangeStd:
            theta_likelihoods = list()
            for theta in fineRangeTheta:
                theta_likelihoods.append(likelihoods[(d, theta, std)])
            ax.plot(fineRangeTheta, theta_likelihoods,
                color=colors[c], label=(str(d) + ", " + str(std)))
            c += 1
    plt.xlabel("theta")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig2)

    # Generate std plots.
    fig3 = plt.figure()
    ax = plt.subplot(111)
    c = 0
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            std_likelihoods = list()
            for std in fineRangeStd:
                std_likelihoods.append(likelihoods[(d, theta, std)])
            ax.plot(fineRangeStd, std_likelihoods,
                color=colors[c], label=(str(d) + ", " + str(theta)))
            c += 1
    plt.xlabel("std")
    plt.ylabel("Negative log likelihood")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pp.savefig(fig3)

    # Generate zoomed-in d plots.
    for theta in coarseRangeTheta:
        for std in coarseRangeStd:
            d_likelihoods = list()
            for d in fineRangeD:
                d_likelihoods.append(likelihoods[(d, theta, std)])
            fig = plt.figure()
            plt.plot(fineRangeD, d_likelihoods)
            plt.title(str(theta) + ", " + str(std))
            plt.xlabel("d")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in theta plots.
    for d in coarseRangeD:
        for std in coarseRangeStd:
            theta_likelihoods = list()
            for theta in fineRangeTheta:
                theta_likelihoods.append(likelihoods[(d, theta, std)])
            fig = plt.figure()
            plt.plot(fineRangeTheta, theta_likelihoods)
            plt.title(str(d) + ", " + str(std))
            plt.xlabel("theta")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    # Generate zoomed-in std plots.
    for d in coarseRangeD:
        for theta in coarseRangeTheta:
            std_likelihoods = list()
            for std in fineRangeStd:
                std_likelihoods.append(likelihoods[(d, theta, std)])
            fig = plt.figure()
            plt.plot(fineRangeStd, std_likelihoods)
            plt.title(str(d) + ", " + str(theta))
            plt.xlabel("std")
            plt.ylabel("Negative log likelihood")
            pp.savefig(fig)

    pp.close()
   

if __name__ == '__main__':
    main()