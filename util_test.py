#!/usr/bin/python

"""
util_test.py
Author: Gabriela Tavares, gtavares@caltech.edu

Unit tests for the util module.
"""

import numpy as np
import os
import unittest

import util
from addm import aDDMTrial


class TestLoadData(unittest.TestCase):
    def compareTrials(self, trial1, trial2):
        self.assertEqual(trial1.RT, trial2.RT)
        self.assertEqual(trial1.choice, trial2.choice)
        self.assertEqual(trial1.valueLeft, trial2.valueLeft)
        self.assertEqual(trial1.valueRight, trial2.valueRight)
        np.testing.assert_equal(trial1.fixItem, trial2.fixItem)
        np.testing.assert_equal(trial1.fixTime, trial2.fixTime)
        np.testing.assert_equal(trial1.fixRDV, trial2.fixRDV)
        self.assertEqual(trial1.uninterruptedLastFixTime,
                         trial2.uninterruptedLastFixTime)
        self.assertEqual(trial1.isCisTrial, trial2.isCisTrial)
        self.assertEqual(trial1.isTransTrial, trial2.isTransTrial)

    def testLoadDataFromNonexistentDataFile(self):
        self.assertRaisesRegexp(
            Exception, "File test_data/dummy_file.csv does not exist",
            util.load_data_from_csv, "test_data/dummy_file.csv",
            "test_data/sample_fixations.csv")

    def testLoadDataFromNonexistentFixationsFile(self):
        self.assertRaisesRegexp(
            Exception, "File test_data/dummy_file.csv does not exist",
            util.load_data_from_csv, "test_data/sample_trial_data.csv",
            "test_data/dummy_file.csv")

    def testLoadDataFromEmptyDataFile(self):
        self.assertRaisesRegexp(
            Exception, "No columns to parse from file",
            util.load_data_from_csv, "test_data/empty_file.csv",
            "test_data/sample_fixations.csv")

    def testLoadDataFromEmptyFixationsFile(self):
        self.assertRaisesRegexp(
            Exception, "No columns to parse from file",
            util.load_data_from_csv, "test_data/sample_trial_data.csv",
            "test_data/empty_file.csv")

    def testLoadDataFromDataFileWithMissingField(self):
        self.assertRaisesRegexp(
            RuntimeError, "Missing field in experimental data file.",
            util.load_data_from_csv,
            "test_data/sample_trial_data_incomplete.csv",
            "test_data/sample_fixations.csv")

    def testLoadDataFromFixationsFileWithMissingField(self):
        self.assertRaisesRegexp(
            RuntimeError, "Missing field in fixations file.",
            util.load_data_from_csv,
            "test_data/sample_trial_data.csv",
            "test_data/sample_fixations_incomplete.csv")

    def testLoadDataFromCSVEconomicChoice(self):
        data = util.load_data_from_csv(
            "test_data/sample_trial_data.csv",
            "test_data/sample_fixations.csv", useAngularDists=False)

        expectedData = {}
        expectedData['abc'] = [aDDMTrial(RT=100, choice=1, valueLeft=-10,
                                         valueRight=15, fixItem=[1, 2],
                                         fixTime=[50, 50])]
        expectedData['xyz'] = [aDDMTrial(RT=200, choice=-1, valueLeft=5,
                                         valueRight=10, fixItem=[1, 2, 1],
                                         fixTime=[100, 50, 50])]
        self.compareTrials(expectedData['abc'][0], data['abc'][0])
        self.compareTrials(expectedData['xyz'][0], data['xyz'][0])

    def testLoadDataFromCSVPerceptualChoice(self):
        data = util.load_data_from_csv(
            "test_data/sample_trial_data.csv",
            "test_data/sample_fixations.csv", useAngularDists=True)

        expectedData = {}
        expectedData['abc'] = [aDDMTrial(RT=100, choice=1, valueLeft=1,
                                         valueRight=0, fixItem=[1, 2],
                                         fixTime=[50, 50], isCisTrial=False,
                                         isTransTrial=True)]
        expectedData['xyz'] = [aDDMTrial(RT=200, choice=-1, valueLeft=2,
                                         valueRight=1, fixItem=[1, 2, 1],
                                         fixTime=[100, 50, 50],
                                         isCisTrial=True, isTransTrial=False)]
        self.compareTrials(expectedData['abc'][0], data['abc'][0])
        self.compareTrials(expectedData['xyz'][0], data['xyz'][0])


class TestSaveSimulationsToCSV(unittest.TestCase):
    def runTest(self):
        trials = [aDDMTrial(RT=100, choice=1, valueLeft=1, valueRight=0,
                            fixItem=[1, 2], fixTime=[50, 50], isCisTrial=False,
                            isTransTrial=True),
                  aDDMTrial(RT=200, choice=-1, valueLeft=2, valueRight=1,
                            fixItem=[1, 2, 1], fixTime=[100, 50, 50],
                            isCisTrial=True, isTransTrial=False)]

        util.save_simulations_to_csv(trials)

        expdataFile = open('simul_expdata.csv', 'r')
        self.assertEqual("parcode,trial,rt,choice,item_left,item_right\n",
                         expdataFile.readline())
        self.assertEqual("dummy_subject,0,100,1,1,0\n", expdataFile.readline())
        self.assertEqual("dummy_subject,1,200,-1,2,1\n",
                         expdataFile.readline())

        fixationsFile = open('simul_fixations.csv', 'r')
        self.assertEqual("parcode,trial,fix_item,fix_time\n",
                         fixationsFile.readline())
        self.assertEqual("dummy_subject,0,1,50\n", fixationsFile.readline())
        self.assertEqual("dummy_subject,0,2,50\n", fixationsFile.readline())
        self.assertEqual("dummy_subject,1,1,100\n", fixationsFile.readline())
        self.assertEqual("dummy_subject,1,2,50\n", fixationsFile.readline())
        self.assertEqual("dummy_subject,1,1,50\n", fixationsFile.readline())

        os.remove('simul_expdata.csv')
        os.remove('simul_fixations.csv')


if __name__ == '__main__':
    unittest.main()
