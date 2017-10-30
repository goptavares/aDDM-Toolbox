#!/usr/bin/env python

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

Module: addm_toolbox_tests.py
Author: Gabriela Tavares, gtavares@caltech.edu

Tests all scripts in the addm_toolbox.
"""

from .addm_mla_test import main as addm_mla_test_main
from .addm_pta_map import main as addm_pta_map_main
from .addm_pta_mle import main as addm_pta_mle_main
from .addm_pta_test import main as addm_pta_test_main
from .basinhopping_optimize import main as basinhopping_main
from .cis_trans_fitting import main as cis_trans_fit_main
from .ddm_mla_test import main as ddm_mla_test_main
from .ddm_pta_test import main as ddm_pta_test_main
from .demo import main as demo_main
from .genetic_algorithm_optimize import main as genetic_algorithm_main
from .simulate_addm_true_distributions import main as simul_true_dists_main


def main():
    print("\n----------Testing demo.py----------")
    demo_main()

    print("\n----------Testing ddm_pta_test.py----------")
    ddm_pta_test_main(d=0.006, sigma=0.07, rangeD=[0.006, 0.007],
                      rangeSigma=[0.07, 0.08], trialsPerCondition=1,
                      verbose=True)

    print("\n----------Testing addm_pta_test.py----------")
    addm_pta_test_main(d=0.006, sigma=0.07, theta=0.4, trialsPerCondition=1,
                       rangeD=[0.006, 0.007], rangeSigma=[0.07, 0.08],
                       rangeTheta=[0.4, 0.5], verbose=True)
    addm_pta_test_main(d=0.006, sigma=0.07, theta=0.4, subjectIds=[15],
                       trialsPerCondition=1, rangeD=[0.006, 0.007],
                       rangeSigma=[0.07, 0.08], rangeTheta=[0.4, 0.5],
                       verbose=True)

    print("\n----------Testing addm_pta_mle.py----------")
    addm_pta_mle_main(trialsPerSubject=1, simulationsPerCondition=1,
                      rangeD=[0.006, 0.007], rangeSigma=[0.07, 0.08],
                      rangeTheta=[0.4, 0.5], verbose=True)
    addm_pta_mle_main(subjectIds=[15], trialsPerSubject=1,
                      simulationsPerCondition=1, rangeD=[0.006, 0.007],
                      rangeSigma=[0.07, 0.08], rangeTheta=[0.4, 0.5],
                      verbose=True)

    print("\n----------Testing addm_pta_map.py----------")
    addm_pta_map_main(trialsPerSubject=1, numSamples=10, numSimulations=1,
                      rangeD=[0.006, 0.007], rangeSigma=[0.07, 0.08],
                      rangeTheta=[0.4, 0.5], verbose=True)
    addm_pta_map_main(subjectIds=[15], trialsPerSubject=1, numSamples=10,
                      numSimulations=1, rangeD=[0.006, 0.007],
                      rangeSigma=[0.07, 0.08], rangeTheta=[0.4, 0.5],
                      verbose=True)

    print("\n----------Testing ddm_mla_test.py----------")
    ddm_mla_test_main(d=0.006, sigma=0.07, rangeD=[0.006, 0.007],
                      rangeSigma=[0.07, 0.08], numTrials=100,
                      numSimulations=100, verbose=True)

    print("\n----------Testing addm_mla_test.py----------")
    addm_mla_test_main(d=0.006, sigma=0.07, theta=0.4, rangeD=[0.006, 0.007],
                       rangeSigma=[0.07, 0.08], rangeTheta=[0.4, 0.5],
                       numTrials=100, numSimulations=100, verbose=True)

    print("\n----------Testing basinhopping_optimize.py----------")
    basinhopping_main(initialD=0.01, initialSigma=0.1, initialTheta=0.5,
                      trialsPerSubject=1, numIterations=1, stepSize=0.005,
                      verbose=True)

    print("\n----------Testing genetic_algorithm_optimize.py----------")
    genetic_algorithm_main(trialsPerSubject=1, popSize=5, numGenerations=2,
                           verbose=True)

    print("\n----------Testing simulate_addm_true_distributions.py----------")
    simul_true_dists_main(d=0.006, sigma=0.07, theta=0.4, numIterations=2,
                          simulationsPerCondition=1, verbose=True)

    print("\n----------Testing cis_trans_fitting.py for cis trials----------")
    cis_trans_fit_main(trialsPerSubject=1, simulationsPerCondition=1,
                       rangeD=[0.006, 0.007], rangeSigma=[0.07, 0.08],
                       rangeTheta=[0.4, 0.5], useCisTrials=True,
                       useTransTrials=False, verbose=True)

    print("\n---------Testing cis_trans_fitting.py for trans trials---------")
    cis_trans_fit_main(trialsPerSubject=1, simulationsPerCondition=1,
                       rangeD=[0.006, 0.007], rangeSigma=[0.07, 0.08],
                       rangeTheta=[0.4, 0.5], useCisTrials=False,
                       useTransTrials=True, verbose=True)
