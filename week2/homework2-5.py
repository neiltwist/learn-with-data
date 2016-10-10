# coding=utf-8
from __future__ import unicode_literals

import numpy as np
import random

import os


class LinearRegression:
    def __init__(self, N):
        # Random line
        x_1a, x_2a, x_1b, x_2b = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([x_1b * x_2a - x_1a * x_2b, x_2b - x_2a, x_1a - x_1b])
        # Random data set
        self.X, self.y = self.generatePoints(N)
        
    def generatePoints(self, number):
        points = []
        results = []
        for i in range(number):
            x = np.array([1, random.uniform(-1, 1), random.uniform(-1, 1)])
            s = int(np.sign(self.V.T.dot(x)))
            points.append(x)
            results.append(s)
        return np.array(points), np.array(results)
    
    def classificationError(self, vector, points=None, results=None):
        # Probability that f and g will disagree on classification of a random point
        if not points:
            points = self.X
        if not results:
            results = self.y
        misclassified = 0
        for i in range(len(points)):
            x = points[i]
            s = results[i]
            if int(np.sign(vector.T.dot(x))) != s:
                misclassified += 1
        return misclassified / float(len(points))

    def learn(self):        
        inverse = np.linalg.pinv(self.X)
        return inverse.dot(self.y)


if __name__ == '__main__':
    runs = 1000
    N = 100
    results = []
    for _ in range(runs):
        p = LinearRegression(N)
        results.append(p.classificationError(p.learn()))

    def average(arr):
        return sum(arr) / len(arr)

    print 'E_in: {0}'.format(average(results))

