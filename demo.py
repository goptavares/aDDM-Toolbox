#!/usr/bin/python

# demo.py
# Author: Gabriela Tavares, gtavares@caltech.edu

# Demo script for the attentional drift-diffusion model (aDDM), as described by
# Krajbich et al. (2010).

from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np


# Normal distribution parameters.
mean = 0.05
sigma = 0.25

# Parameters of the grid.
stateStep = 0.1
maxTime = 200
initialBarrierUp = 1
initialBarrierDown = -1

# The values of the barriers can change over time.
decay = 0  # decay = 0 means barriers are constant.
barrierUp = initialBarrierUp * np.ones(maxTime)
barrierDown = initialBarrierDown * np.ones(maxTime)
for t in xrange(1, maxTime):
    barrierUp[t] = initialBarrierUp / (1 + decay * (t + 1))
    barrierDown[t] = initialBarrierDown / (1 + decay * (t + 1))

# The vertical axis is divided into states.
states = np.arange(initialBarrierDown, initialBarrierUp + stateStep, stateStep)
states[(states < 0.001) & (states > -0.001)] = 0

# Initial probability for all states is zero, except the zero state, for
# which the initial probability is one.
prStates = np.zeros(states.size)
prStates[states == 0] = 1

probUpCrossing = np.zeros(maxTime)
probDownCrossing = np.zeros(maxTime)

for t in xrange(1, maxTime):
    prStatesNew = np.zeros(states.size)
    
    # Update the probability of the states that remain inside the barriers.
    for s in xrange(0,states.size):
        currState = states[s]
        if currState > barrierDown[t] and currState < barrierUp[t]:
            change = (currState * np.ones(states.size)) - states
            # The probability of being in state B is the sum, over all states A,
            # of the probability of being in A at the previous time step times
            # the probability of changing from A to B. We multiply the
            # probability by the stateStep to ensure that the area under the
            # curve for the probability distributions probUpCrossing and
            # probDownCrossing each add up to 1.
            prStatesNew[s] = (stateStep *
                np.sum(np.multiply(prStates, norm.pdf(change, mean, sigma))))

    # Calculate the probabilities of crossing the up barrier and the down
    # barrier. This is given by the sum, over all states A, of the
    # probability of being in A at the previous timestep times the
    # probability of crossing the barrier if A is the previous state.
    changeUp = (barrierUp[t] * np.ones(states.size)) - states
    tempUpCross = np.sum(np.multiply(prStates,
        (1 - norm.cdf(changeUp, mean, sigma))))
    changeDown = (barrierDown[t] * np.ones(states.size)) - states
    tempDownCross = np.sum(np.multiply(prStates,
        (norm.cdf(changeDown, mean, sigma))))

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

    # Probabilities at each time step DO NOT add up to 1. These probabilities
    # account only for the probability of the signal staying inside the barriers
    # or crossing a barrier at this time step, but not the probability of
    # already having crossed a barrier at an earlier time.
    print("Sum of probabilities at time " + str(t) + ": " +
        str(np.sum(prStates) + probUpCrossing[t] + probDownCrossing[t]))

# Probabilities of crossing the two barriers over time add up to 1.
print("Total probability over time of crossing up barrier: " +
    str(np.sum(probUpCrossing)))
print("Total probability over time of crossing down barrier: " +
    str(np.sum(probDownCrossing)))
print("Total probability over time of crossing either barrier: " +
    str(np.sum(probUpCrossing) + np.sum(probDownCrossing)))

plt.figure
plt.plot(range(1, maxTime + 1), probUpCrossing, label='Up')
plt.plot(range(1, maxTime + 1), probDownCrossing, label='Down')
plt.xlabel('Time')
plt.ylabel('P(crossing)')
plt.legend()
plt.show()
