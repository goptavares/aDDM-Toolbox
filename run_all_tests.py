#!/usr/bin/python

"""
run_all_tests.py
Author: Gabriela Tavares, gtavares@caltech.edu

Test all modules in the aDDM toolbox.
"""

import os


def main():
    print("\n----------Testing demo.py----------")
    os.system("python demo.py")

    print("\n----------Testing util_test.py----------")
    os.system("python util_test.py")

    print("\n----------Testing ddm_test.py----------")
    os.system("python ddm_test.py --num-values 3 --num-trials 1 --verbose")

    print("\n----------Testing addm_test.py----------")
    os.system("python addm_test.py --num-trials 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")
    os.system("python addm_test.py --subject-ids cai --num-trials 1 --range-d "
              "0.006 0.007 --range-sigma 0.07 0.08 --range-theta 0.4 0.5 "
              "--verbose")

    print("\n----------Testing addm_mle.py----------")
    os.system("python addm_mle.py --trials-per-subject 1 --num-simulations 1 "
              "--range-d 0.006 0.007 --range-sigma 0.07 0.08 --range-theta "
              "0.4 0.5 --verbose")
    os.system("python addm_mle.py --subject-ids cai --trials-per-subject 1 "
              "--num-simulations 1 --range-d 0.006 0.007 --range-sigma "
              "0.07 0.08 --range-theta 0.4 0.5 --verbose")

    print("\n----------Testing addm_posteriors.py----------")
    os.system("python addm_posteriors.py --trials-per-subject 1 "
              "--num-samples 10 --num-simulations-per-sample 1 "
              "--range-d 0.006 0.007 --range-sigma 0.07 0.08 "
              "--range-theta 0.4 0.5 --verbose")
    os.system("python addm_posteriors.py --subject-ids cai "
              "--trials-per-subject 1 --num-samples 10 "
              "--num-simulations-per-sample 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")

    print("\n----------Testing cis_trans_fitting.py for cis trials----------")
    os.system("python cis_trans_fitting.py --trials-per-subject 1 "
              "--num-simulations 1 --range-d 0.006 0.007 --range-sigma 0.07 "
              "0.08 --range-theta 0.4 0.5 --use-cis-trials --verbose")

    print("\n---------Testing cis_trans_fitting.py for trans trials---------")
    os.system("python cis_trans_fitting.py --trials-per-subject 1 "
              "--num-simulations 1 --range-d 0.006 0.007 --range-sigma 0.07 "
              "0.08 --range-theta 0.4 0.5 --use-trans-trials --verbose")

    print("\n----------Testing simulate_addm_true_distributions.py----------")
    os.system("python simulate_addm_true_distributions.py --num-iterations 2 "
              "--num-simulations 1 --verbose")

    print("\n----------Testing optimize.py----------")
    os.system("python optimize.py --trials-per-subject 1 --num-iterations 1 "
              "--step-size 0.005")

    print("\n----------Testing genetic_algorithm_optimize.py----------")
    os.system("python genetic_algorithm_optimize.py --trials-per-subject 1 "
              "--pop-size 5 --num-generations 2")

    print("\n----------Testing old_ddm.py----------")
    os.system("python old_ddm.py --num-values 3 --num-trials 1 "
              "--num-simulations 1 --verbose")

    print("\n----------Testing old_addm.py----------")
    os.system("python old_addm.py --num-trials 1 --num-simulations 1 "
              "--verbose")


if __name__ == '__main__':
    main()
