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

Module: util_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Unit tests for the util.py module.
"""

import numpy as np
import os
import unittest

from datetime import datetime

from addm import aDDMTrial
from util import (load_trial_conditions_from_csv, load_data_from_csv,
                  save_simulations_to_csv, convert_item_values)


class TestLoadTrialConditions(unittest.TestCase):
    def test_load_trials_from_nonexistent_file(self):
        self.assertRaisesRegexp(
            Exception, "No such file or directory: 'addm_toolbox/test_data/"
            "dummy_file.csv'", load_trial_conditions_from_csv,
            "addm_toolbox/test_data/dummy_file.csv")

    def test_load_trials_from_empty_file(self):
        self.assertRaisesRegexp(
            Exception, "argument of type 'NoneType' is not iterable",
            load_trial_conditions_from_csv,
            "addm_toolbox/test_data/empty_file.csv")

    def test_load_trials_from_file_with_missing_field(self):
        self.assertRaisesRegexp(
            RuntimeError, "Missing field in trial conditions file.",
            load_trial_conditions_from_csv,
            "addm_toolbox/test_data/sample_trial_conditions_incomplete.csv")


class TestLoadData(unittest.TestCase):
    def compare_trials(self, trial1, trial2):
        self.assertEqual(trial1.RT, trial2.RT)
        self.assertEqual(trial1.choice, trial2.choice)
        self.assertEqual(trial1.valueLeft, trial2.valueLeft)
        self.assertEqual(trial1.valueRight, trial2.valueRight)
        np.testing.assert_equal(trial1.fixItem, trial2.fixItem)
        np.testing.assert_equal(trial1.fixTime, trial2.fixTime)
        np.testing.assert_equal(trial1.fixRDV, trial2.fixRDV)
        self.assertEqual(trial1.uninterruptedLastFixTime,
                         trial2.uninterruptedLastFixTime)

    def test_load_data_from_nonexistent_data_file(self):
        self.assertRaisesRegexp(
            Exception, "File addm_toolbox/test_data/dummy_file.csv does not "
            "exist", load_data_from_csv, "addm_toolbox/test_data/"
            "dummy_file.csv", "addm_toolbox/test_data/sample_fixations.csv")

    def test_load_data_from_nonexistent_fixations_file(self):
        self.assertRaisesRegexp(
            Exception, "File addm_toolbox/test_data/dummy_file.csv does not "
            "exist", load_data_from_csv, "addm_toolbox/test_data/"
            "sample_trial_data.csv", "addm_toolbox/test_data/dummy_file.csv")

    def test_load_data_from_empty_data_file(self):
        self.assertRaisesRegexp(
            Exception, "No columns to parse from file",
            load_data_from_csv, "addm_toolbox/test_data/empty_file.csv",
            "addm_toolbox/test_data/sample_fixations.csv")

    def test_load_data_from_empty_fixations_file(self):
        self.assertRaisesRegexp(
            Exception, "No columns to parse from file",
            load_data_from_csv, "addm_toolbox/test_data/sample_trial_data.csv",
            "addm_toolbox/test_data/empty_file.csv")

    def test_load_data_from_data_file_with_missing_field(self):
        self.assertRaisesRegexp(
            RuntimeError, "Missing field in experimental data file.",
            load_data_from_csv,
            "addm_toolbox/test_data/sample_trial_data_incomplete.csv",
            "addm_toolbox/test_data/sample_fixations.csv")

    def test_load_data_from_fixations_file_with_missing_field(self):
        self.assertRaisesRegexp(
            RuntimeError, "Missing field in fixations file.",
            load_data_from_csv,
            "addm_toolbox/test_data/sample_trial_data.csv",
            "addm_toolbox/test_data/sample_fixations_incomplete.csv")

    def test_load_data_from_csv_economic_choice(self):
        data = load_data_from_csv(
            "addm_toolbox/test_data/sample_trial_data.csv",
            "addm_toolbox/test_data/sample_fixations.csv")

        expectedData = {}
        expectedData["abc"] = [aDDMTrial(RT=100, choice=1, valueLeft=1,
                                         valueRight=0, fixItem=[1, 2],
                                         fixTime=[50, 50])]
        expectedData["xyz"] = [aDDMTrial(RT=200, choice=-1, valueLeft=2,
                                         valueRight=1, fixItem=[1, 2, 1],
                                         fixTime=[100, 50, 50])]
        self.compare_trials(expectedData["abc"][0], data["abc"][0])
        self.compare_trials(expectedData["xyz"][0], data["xyz"][0])

    def test_load_data_from_csv_perceptual_choice(self):
        data = load_data_from_csv(
            "addm_toolbox/test_data/sample_trial_data_perceptual.csv",
            "addm_toolbox/test_data/sample_fixations.csv",
            convertItemValues=convert_item_values)

        expectedData = {}
        expectedData["abc"] = [aDDMTrial(RT=100, choice=1, valueLeft=1,
                                         valueRight=0, fixItem=[1, 2],
                                         fixTime=[50, 50])]
        expectedData["xyz"] = [aDDMTrial(RT=200, choice=-1, valueLeft=2,
                                         valueRight=1, fixItem=[1, 2, 1],
                                         fixTime=[100, 50, 50])]
        self.compare_trials(expectedData["abc"][0], data["abc"][0])
        self.compare_trials(expectedData["xyz"][0], data["xyz"][0])


class TestSaveSimulationsToCSV(unittest.TestCase):
    def runTest(self):
        trials = [aDDMTrial(RT=100, choice=1, valueLeft=1, valueRight=0,
                            fixItem=[1, 2], fixTime=[50, 50]),
                  aDDMTrial(RT=200, choice=-1, valueLeft=2, valueRight=1,
                            fixItem=[1, 2, 1], fixTime=[100, 50, 50])]

        currTime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        expdataFileName = "simul_expdata_" + currTime + ".csv"
        fixationsFileName = "simul_fixations_" + currTime + ".csv"
        save_simulations_to_csv(trials, expdataFileName, fixationsFileName)

        expdataFile = open(expdataFileName, 'r')
        self.assertEqual("parcode,trial,rt,choice,item_left,item_right\n",
                         expdataFile.readline())
        self.assertEqual("0,0,100,1,1,0\n", expdataFile.readline())
        self.assertEqual("0,1,200,-1,2,1\n",
                         expdataFile.readline())

        fixationsFile = open(fixationsFileName, 'r')
        self.assertEqual("parcode,trial,fix_item,fix_time\n",
                         fixationsFile.readline())
        self.assertEqual("0,0,1,50\n", fixationsFile.readline())
        self.assertEqual("0,0,2,50\n", fixationsFile.readline())
        self.assertEqual("0,1,1,100\n", fixationsFile.readline())
        self.assertEqual("0,1,2,50\n", fixationsFile.readline())
        self.assertEqual("0,1,1,50\n", fixationsFile.readline())

        os.remove(expdataFileName)
        os.remove(fixationsFileName)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
