#!/usr/bin/python

"""
Copyright (C) 2017, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: ddm_mla.py
Author: Gabriela Tavares, gtavares@caltech.edu

Maximum likelihood algorithm for the classic drift-diffusion model (DDM). This
algorithm uses reaction time histograms conditioned on choice from both data
and simulations to estimate each model's log-likelihood. Here we perform a
test to check the validity of this algorithm. Artificil data is generated using
specific parameters for the model. These parameters are then recovered through
a maximum likelihood estimation procedure, using a grid search over the 2 free
parameters of the model.
"""

import argparse
import numpy as np
import os

from multiprocessing import Pool

from ddm import DDMTrial
from util import load_trial_conditions_from_csv


def wrap_ddm_get_model_log_likelihood(args):
    """
    Wrapper for DDM.get_model_log_likelihood(), intended for parallel
    computation using a threadpool.
    Args:
      args: a tuple where the first item is a DDM object, and the remaining
          item are the same arguments required by
          DDM.get_model_log_likelihood().
    Returns:
      The output of DDM.get_model_log_likelihood().
    """
    model = args[0]
    return model.get_model_log_likelihood(*args[1:])


class DDM(object):
    """
    Implementation of the traditional drift-diffusion model (DDM), as described
    by Ratcliff et al. (1998).
    """
    def __init__(self, d, sigma, barrier=1, nonDecisionTime=0, bias=0):
        """
        Args:
          d: float, parameter of the model which controls the speed of
              integration of the signal.
          sigma: float, parameter of the model, standard deviation for the
              normal distribution.
          barrier: positive number, magnitude of the signal thresholds.
          nonDecisionTime: non-negative integer, the amount of time in
              milliseconds during which only noise is added to the decision
              variable.
          bias: number, corresponds to the initial value of the decision
              variable. Must be smaller than barrier.
        """
        if barrier <= 0:
            raise ValueError("Error: barrier parameter must larger than zero.")
        if bias >= barrier:
            raise ValueError("Error: bias parameter must be smaller than "
                "barrier parameter.")
        self.d = d
        self.sigma = sigma
        self.barrier = barrier
        self.nonDecisionTime = nonDecisionTime
        self.bias = bias
        self.params = (d, sigma)


    def simulate_trial(self, valueLeft, valueRight, timeStep=10): 
        """
        DDM algorithm. Given the parameters of the model and the trial
        conditions, returns the choice and reaction time as predicted by the
        model.
        Args:
          valueLeft: integer, value of the left item.
          valueRight: integer, value of the right item.
          timeStep: integer, value in milliseconds which determines how often
              the RDV signal is updated.
        Returns:
          A DDMTrial object resulting from the simulation.
        """
        RT = 0
        choice = 0
        RDV = self.bias
        elapsedNDT = 0

        while RDV < self.barrier and RDV > -self.barrier:
            RT = RT + timeStep
            epsilon = np.random.normal(0, self.sigma)
            if elapsedNDT < int(self.nonDecisionTime // timeStep):
                RDV += epsilon
                elapsedNDT += 1
            else:
                RDV += (self.d * (valueLeft - valueRight)) + epsilon

        if RDV >= self.barrier:
            choice = -1
        elif RDV <= -self.barrier:
            choice = 1
        return DDMTrial(RT, choice, valueLeft, valueRight)


    def get_model_log_likelihood(self, trialConditions, numSimulations,
                                 histBins, dataHistLeft, dataHistRight):
        """
        Computes the log-likelihood of a data set given the model. Data set is
        provided in the form of reaction time histograms conditioned on choice.
        Args:
          trialConditions: list of pairs corresponding to the different trial
              conditions. Each pair contains the values of left and right
              items.
          numSimulations: integer, number of simulations per trial condition to
              be generated when creating reaction time histograms.
          histBins: list of numbers corresponding to the time bins used to
              create the reaction time histograms.
          dataHistLeft: dict indexed by trial condition (where each trial
              condition is a pair (valueLeft, valueRight)). Each entry is a
              numpy array corresponding to the reaction time histogram
              conditioned on left choice for the data. It is assumed that this
              histogram was created using the same time bins as argument
              histBins.
          dataHistRight: same as dataHistLeft, except that the reaction time
              histograms are conditioned on right choice.
          Returns:
              The log-likelihood for the data given the model.
        """
        logLikelihood = 0
        for trialCondition in trialConditions:
            RTsLeft = list()
            RTsRight = list()
            sim = 0
            while sim < numSimulations:
                try:
                    ddmTrial = self.simulate_trial(trialCondition[0],
                                                   trialCondition[1])
                except:
                    print("An exception occurred while generating " +
                          "artificial trial " + str(sim) + " for condition " +
                          str(trialCondition[0]) + ", " +
                          str(trialCondition[1]) + ", during the " +
                          "log-likelihood computation for model " +
                          str(self.params) + ".")
                    raise
                if ddmTrial.choice == -1:
                    RTsLeft.append(ddmTrial.RT)
                elif ddmTrial.choice == 1:
                    RTsRight.append(ddmTrial.RT)
                sim += 1

            simulLeft = np.histogram(RTsLeft, bins=histBins)[0]
            if np.sum(simulLeft) != 0:
                simulLeft = simulLeft / float(np.sum(simulLeft))
            with np.errstate(divide="ignore"):
                logSimulLeft = np.where(simulLeft > 0, np.log(simulLeft), 0)
            dataLeft = np.array(dataHistLeft[trialCondition])
            logLikelihood += np.dot(logSimulLeft, dataLeft)

            simulRight = np.histogram(RTsRight, bins=histBins)[0]
            if np.sum(simulRight) != 0:
                simulRight = simulRight / float(np.sum(simulRight))
            with np.errstate(divide="ignore"):
                logSimulRight = np.where(simulRight > 0, np.log(simulRight), 0)
            dataRight = np.array(dataHistRight[trialCondition])
            logLikelihood += np.dot(logSimulRight, dataRight)

        return logLikelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=int, default=9,
                        help="Size of the thread pool.")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of artificial data trials to be "
                        "generated per trial condition.")
    parser.add_argument("--num-simulations", type=int, default=10,
                        help="Number of simulations to be generated per trial "
                        "condition, to be used in the RT histograms.")
    parser.add_argument("--bin-step", type=int, default=100,
                        help="Size of the bin step to be used in the RT "
                        "histograms.")
    parser.add_argument("--max-rt", type=int, default=8000,
                        help="Maximum RT to be used in the RT histograms.")
    parser.add_argument("--d", type=float, default=0.006,
                        help="DDM parameter for generating artificial data.")
    parser.add_argument("--sigma", type=float, default=0.08,
                        help="DDM parameter for generating artificial data.")
    parser.add_argument("--range-d", nargs="+", type=float,
                        default=[0.005, 0.006, 0.007],
                        help="Search range for parameter d.")
    parser.add_argument("--range-sigma", nargs="+", type=float,
                        default=[0.065, 0.08, 0.095],
                        help="Search range for parameter sigma.")
    parser.add_argument("--trials-file-name", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "data/trial_conditions.csv"),
                        help="Name of trial conditions file.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    pool = Pool(args.num_threads)

    histBins = range(0, args.max_rt + args.bin_step, args.bin_step)

    # Load trial conditions.
    trialConditions = load_trial_conditions_from_csv(args.trials_file_name)

    # Generate artificial data.
    dataRTLeft = dict()
    dataRTRight = dict()
    for trialCondition in trialConditions:
        dataRTLeft[trialCondition] = list()
        dataRTRight[trialCondition] = list()
    model = DDM(args.d, args.sigma)
    for trialCondition in trialConditions:
        trial = 0
        while trial < args.num_trials:
            try:
                ddmTrial = model.simulate_trial(trialCondition[0],
                                                trialCondition[1])
            except:
                print("An exception occurred while generating artificial " +
                      "trial " + str(trial) + " for condition " +
                      str(trialCondition[0]) + ", " + str(trialCondition[1]) +
                      ".")
                raise
            if ddmTrial.choice == -1:
                dataRTLeft[trialCondition].append(ddmTrial.RT)
            elif ddmTrial.choice == 1:
                dataRTRight[trialCondition].append(ddmTrial.RT)
            trial += 1

    # Generate histograms for artificial data.
    dataHistLeft = dict()
    dataHistRight = dict()
    for trialCondition in trialConditions:
        dataHistLeft[trialCondition] = np.histogram(
            dataRTLeft[trialCondition], bins=histBins)[0]
        dataHistRight[trialCondition] = np.histogram(
            dataRTRight[trialCondition], bins=histBins)[0]

    # Grid search on the parameters of the model.
    if args.verbose:
        print("Performing grid search over the model parameters...")
    listParams = list()
    models = list()
    for d in args.range_d:
        for sigma in args.range_sigma:
            model = DDM(d, sigma)
            models.append(model)
            listParams.append((model, trialConditions, args.num_simulations,
                              histBins, dataHistLeft, dataHistRight))
    logLikelihoods = pool.map(wrap_ddm_get_model_log_likelihood, listParams)
    pool.close()

    if args.verbose:
        for i, model in enumerate(models):
            print("L" + str(model.params) + " = " + str(logLikelihoods[i]))
        bestIndex = logLikelihoods.index(max(logLikelihoods))
        print("Best fit: " + str(models[bestIndex].params))


if __name__ == "__main__":
    main()
