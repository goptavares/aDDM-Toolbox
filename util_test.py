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


class TestLoadDataFromCSVEconomicChoice(unittest.TestCase):
    def runTest(self):
        data = util.load_data_from_csv(
            "sample_trial_data.csv", "sample_fixations.csv",
            useAngularDists=False)

        expectedRT = {'abc': {1: 100}, 'xyz': {1: 200}}
        expectedChoice = {'abc': {1: 1}, 'xyz': {1: -1}}
        expectedValueLeft = {'abc': {1: -10}, 'xyz': {1: 5}}
        expectedValueRight = {'abc': {1: 15}, 'xyz': {1: 10}}
        expectedFixItem = {'abc': {1: np.array([1, 2])},
                           'xyz': {1: np.array([1, 2, 1])}}
        expectedFixTime = {'abc': {1: np.array([50, 50])},
                           'xyz': {1: np.array([100, 50, 50])}}
        expectedIsCisTrial = {'abc': {1: False}, 'xyz': {1: False}}
        expectedIsTransTrial = {'abc': {1: False}, 'xyz': {1: False}}
        
        self.assertEqual(expectedRT, data.RT)
        self.assertEqual(expectedChoice, data.choice)
        self.assertEqual(expectedValueLeft, data.valueLeft)
        self.assertEqual(expectedValueRight, data.valueRight)
        np.testing.assert_equal(expectedFixItem, data.fixItem)
        np.testing.assert_equal(expectedFixTime, data.fixTime)
        self.assertEqual(expectedIsCisTrial, data.isCisTrial)
        self.assertEqual(expectedIsTransTrial, data.isTransTrial)


class TestLoadDataFromCSVPerceptualChoice(unittest.TestCase):
    def runTest(self):
        data = util.load_data_from_csv(
            "sample_trial_data.csv", "sample_fixations.csv",
            useAngularDists=True)

        expectedRT = {'abc': {1: 100}, 'xyz': {1: 200}}
        expectedChoice = {'abc': {1: 1}, 'xyz': {1: -1}}
        expectedValueLeft = {'abc': {1: 1}, 'xyz': {1: 2}}
        expectedValueRight = {'abc': {1: 0}, 'xyz': {1: 1}}
        expectedFixItem = {'abc': {1: np.array([1, 2])},
                           'xyz': {1: np.array([1, 2, 1])}}
        expectedFixTime = {'abc': {1: np.array([50, 50])},
                           'xyz': {1: np.array([100, 50, 50])}}
        expectedIsCisTrial = {'abc': {1: False}, 'xyz': {1: True}}
        expectedIsTransTrial = {'abc': {1: True}, 'xyz': {1: False}}
        
        self.assertEqual(expectedRT, data.RT)
        self.assertEqual(expectedChoice, data.choice)
        self.assertEqual(expectedValueLeft, data.valueLeft)
        self.assertEqual(expectedValueRight, data.valueRight)
        np.testing.assert_equal(expectedFixItem, data.fixItem)
        np.testing.assert_equal(expectedFixTime, data.fixTime)
        self.assertEqual(expectedIsCisTrial, data.isCisTrial)
        self.assertEqual(expectedIsTransTrial, data.isTransTrial)


class TestSaveSimulationsToCSV(unittest.TestCase):
    def runTest(self):
        choice = {0: 1, 1: -1, 2: 1}
        RT = {0: 100, 1: 200, 2: 300}
        valueLeft = {0: 0, 1: 1, 2: 2}
        valueRight = {0: 3, 1: 2, 2: 1}
        fixItem = {0: [2, 1], 1: [1, 2], 2: [1, 2]}
        fixTime = {0: [50, 50], 1: [150, 50], 2: [150, 150]}
        fixRDV = {0: [0.1, 0.2], 1: [0.5, 0.8], 2: [0.3, 0.6]}
        numTrials = 3
        util.save_simulations_to_csv(
            choice, RT, valueLeft, valueRight, fixItem, fixTime, fixRDV,
            numTrials)

        choiceFile = open('choice.csv', 'r')
        RTFile = open('rt.csv', 'r')
        valueLeftFile = open('value_left.csv', 'r')
        valueRightFile = open('value_right.csv', 'r')
        fixItemFile = open('fix_item.csv', 'r')
        fixTimeFile = open('fix_time.csv', 'r')
        fixRDVFile = open('fix_rdv.csv', 'r')

        self.assertEqual("0,1,-1,1\n", choiceFile.read())
        self.assertEqual("0,100,200,300\n", RTFile.read())
        self.assertEqual("0,0,1,2\n", valueLeftFile.read())
        self.assertEqual("0,3,2,1\n", valueRightFile.read())
        self.assertEqual("0,2,1,1\n1,1,2,2\n", fixItemFile.read())
        self.assertEqual("0,50,150,150\n1,50,50,150\n", fixTimeFile.read())
        self.assertEqual("0,0.1,0.5,0.3\n1,0.2,0.8,0.6\n", fixRDVFile.read())

        os.remove('choice.csv')
        os.remove('rt.csv')
        os.remove('value_left.csv')
        os.remove('value_right.csv')
        os.remove('fix_item.csv')
        os.remove('fix_time.csv')
        os.remove('fix_rdv.csv')


if __name__ == '__main__':
    unittest.main()
