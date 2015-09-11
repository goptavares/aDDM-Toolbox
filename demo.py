#!/usr/bin/python

"""
demo.py
Author: Gabriela Tavares, gtavares@caltech.edu

Demo of the attentional drift-diffusion model (aDDM), as described by Krajbich
set al. (2010).
"""

from scipy.stats import norm

import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean", type=float, default=0.05,
                        help="Mean of the normal distribution.")
    parser.add_argument("--sigma", type=float, default=0.25,
                        help="Standard deviation of the normal distribution.")
    parser.add_argument("--state-step", type=float, default=0.1,
                        help="Step size for the RDV states.")
    parser.add_argument("--max-time", type=int, default=200,
                        help="Amount of time to run the algorithm, in "
                        "miliseconds.")
    parser.add_argument("--barrier-decay", type=float, default=0,
                        help="Parameter that controls the decay of the "
                        "barriers over time. A decay of zero means the "
                        "barriers are constant.")
    parser.add_argument("--display-figure", default=False, action="store_true",
                        help="Display a plot of the computed probabilities at "
                        "the end of execution.")
    parser.add_argument("--verbose", default=False, action="store_true",
                        help="Increase output verbosity.")
    args = parser.parse_args()

    initialBarrierUp = 1
    initialBarrierDown = -1

    # The values of the barriers can change over time.
    barrierUp = initialBarrierUp * np.ones(args.max_time)
    barrierDown = initialBarrierDown * np.ones(args.max_time)
    for t in xrange(1, args.max_time):
        barrierUp[t] = initialBarrierUp / (1 + args.barrier_decay * (t + 1))
        barrierDown[t] = initialBarrierDown / (1 + args.barrier_decay * (t + 1))

    # The vertical axis is divided into states.
    states = np.arange(initialBarrierDown, initialBarrierUp + args.state_step,
                       args.state_step)
    states[(states < 0.001) & (states > -0.001)] = 0

    # Initial probability for all states is zero, except the zero state, for
    # which the initial probability is one.
    prStates = np.zeros(states.size)
    prStates[states == 0] = 1

    probUpCrossing = np.zeros(args.max_time)
    probDownCrossing = np.zeros(args.max_time)

    for t in xrange(1, args.max_time):
        prStatesNew = np.zeros(states.size)
        
        # Update the probability of the states that remain inside the barriers.
        for s in xrange(0,states.size):
            currState = states[s]
            if currState > barrierDown[t] and currState < barrierUp[t]:
                change = (currState * np.ones(states.size)) - states
                # The probability of being in state B is the sum, over all
                # states A, of the probability of being in A at the previous
                # time step times the probability of changing from A to B. We
                # multiply the probability by the state step to ensure that the
                # area under the curve for the probability distributions
                # probUpCrossing and probDownCrossing each add up to 1.
                prStatesNew[s] = (args.state_step * np.sum(np.multiply(prStates,
                                  norm.pdf(change, args.mean, args.sigma))))

        # Calculate the probabilities of crossing the up barrier and the down
        # barrier. This is given by the sum, over all states A, of the
        # probability of being in A at the previous timestep times the
        # probability of crossing the barrier if A is the previous state.
        changeUp = (barrierUp[t] * np.ones(states.size)) - states
        tempUpCross = np.sum(
            np.multiply(prStates,
            1 - norm.cdf(changeUp, args.mean, args.sigma)))
        changeDown = (barrierDown[t] * np.ones(states.size)) - states
        tempDownCross = np.sum(
            np.multiply(prStates, norm.cdf(changeDown, args.mean, args.sigma)))

        # Renormalize to cope with numerical approximation.
        sumIn = np.sum(prStates)
        sumCurrent = np.sum(prStatesNew) + tempUpCross + tempDownCross
        prStatesNew = prStatesNew * sumIn / sumCurrent
        tempUpCross = tempUpCross * sumIn / sumCurrent
        tempDownCross = tempDownCross * sumIn / sumCurrent

        # Update the probabilities of each state and the probabilities of
        # crossing each barrier at this timestep.
        prStates = prStatesNew
        probUpCrossing[t] = tempUpCross
        probDownCrossing[t] = tempDownCross

        # Probabilities at each time step DO NOT add up to 1. These
        # probabilities account only for the probability of the signal staying
        # inside the barriers or crossing a barrier at this time step, but not
        # the probability of already having crossed a barrier at an earlier
        # time.
        if args.verbose:
            print("Sum of probabilities at time " + str(t) + ": " +
                  str(np.sum(prStates) + probUpCrossing[t] +
                  probDownCrossing[t]))

    # Probabilities of crossing the two barriers over time add up to 1.
    if args.verbose:
        print("Total probability over time of crossing up barrier: " +
              str(np.sum(probUpCrossing)))
        print("Total probability over time of crossing down barrier: " +
              str(np.sum(probDownCrossing)))
        print("Total probability over time of crossing either barrier: " +
              str(np.sum(probUpCrossing) + np.sum(probDownCrossing)))

    plt.figure
    plt.plot(range(1, args.max_time + 1), probUpCrossing, label='Up')
    plt.plot(range(1, args.max_time + 1), probDownCrossing, label='Down')
    plt.xlabel('Time')
    plt.ylabel('P(crossing)')
    plt.legend()
    plt.show(block=args.display_figure)


if __name__ == '__main__':
    main()
