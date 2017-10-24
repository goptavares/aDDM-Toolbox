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

Module: run_all_tests.py
Author: Gabriela Tavares, gtavares@caltech.edu

Tests all modules in the addm_toolbox.
"""

import os


def main():
    print("\n----------Testing demo.py----------")
    os.system("addm_demo")

    print("\n----------Testing ddm_pta_test.py----------")
    os.system("ddm_pta_test --trials-per-condition 1 --verbose")

    print("\n----------Testing addm_pta_test.py----------")
    os.system("addm_pta_test --trials-per-condition 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")
    os.system("addm_pta_test --subject-ids 15 --trials-per-condition 1 "
              "--range-d 0.006 0.007 --range-sigma 0.07 0.08 --range-theta "
              "0.4 0.5 --verbose")

    print("\n----------Testing addm_pta_mle.py----------")
    os.system("addm_pta_mle --trials-per-subject 1 "
              "--simulations-per-condition 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")
    os.system("addm_pta_mle --subject-ids 15 --trials-per-subject 1 "
              "--simulations-per-condition 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")

    print("\n----------Testing addm_pta_map.py----------")
    os.system("addm_pta_map --trials-per-subject 1 --num-samples 10 "
              "--num-simulations 1 --range-d 0.006 0.007 --range-sigma "
              "0.07 0.08 --range-theta 0.4 0.5 --verbose")
    os.system("addm_pta_map --subject-ids 15 --trials-per-subject 1 "
              "--num-samples 10 --num-simulations 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --verbose")

    print("\n----------Testing cis_trans_fitting.py for cis trials----------")
    os.system("addm_cis_trans_fit --trials-per-subject 1 "
              "--simulations-per-condition 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 --use-cis-trials "
              "--verbose")

    print("\n---------Testing cis_trans_fitting.py for trans trials---------")
    os.system("addm_cis_trans_fit --trials-per-subject 1 "
              "--simulations-per-condition 1 --range-d 0.006 0.007 "
              "--range-sigma 0.07 0.08 --range-theta 0.4 0.5 "
              "--use-trans-trials --verbose")

    print("\n----------Testing simulate_addm_true_distributions.py----------")
    os.system("addm_simulate_true_distributions --num-iterations 2 "
              "--simulations-per-condition 1 --verbose")

    print("\n----------Testing basinhopping_optimize.py----------")
    os.system("addm_basinhopping --trials-per-subject 1 --num-iterations 1 "
              "--step-size 0.005 --verbose")

    print("\n----------Testing genetic_algorithm_optimize.py----------")
    os.system("addm_genetic_algorithm --trials-per-subject 1 --pop-size 5 "
              "--num-generations 2 --verbose")

    print("\n----------Testing ddm_mla.py----------")
    os.system("ddm_mla --num-trials 100 --num-simulations 100 --verbose")

    print("\n----------Testing addm_mla.py----------")
    os.system("addm_mla --num-trials 100 --num-simulations 100 --verbose")


if __name__ == "__main__":
    main()
