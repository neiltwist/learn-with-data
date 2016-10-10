# coding=utf-8
from __future__ import unicode_literals

import numpy as np
import random


class Perceptron:
    def __init__(self, N, line=None, w=None, points=None, results=None):
        if line is None:
            # Random line
            x_a, y_a, x_b, y_b = [random.uniform(-1, 1) for i in range(4)]
            self.V = np.array([x_b * y_a - x_a * y_b, y_b - y_a, x_a - x_b])
        else:
            self.V = line
        if w is None:
            self.w = np.zeros(3)
        else:
            self.w = w
        if points is None or results is None:
            # Random data set
            X, y = self.generatePoints(N)
            if points is not None:
                self.X = points
            else:
                self.X = X
            if results is not None:
                self.y = results
            else:
                self.y = y
        else:
            self.X = points
            self.y = results

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
        if points is None:
            points = self.X
        if results is None:
            results = self.y
        misclassified = 0
        for i in range(len(points)):
            x = points[i]
            s = results[i]
            if int(np.sign(vector.T.dot(x))) != s:
                misclassified += 1
        return misclassified / float(len(points))

    def chooseMisclassifiedPoint(self, vector):
        # Choose a random point among the misclassified
        misclassified = []
        for i in range(len(self.X)):
            x = self.X[i]
            s = self.y[i]
            if int(np.sign(vector.T.dot(x))) != s:
                misclassified.append((x, s))
        return misclassified[random.randrange(0, len(misclassified))]

    def learn(self):
        w = self.w
        iteration = 0
        # Iterate until all points are correctly classified
        while self.classificationError(w) != 0:
            iteration += 1
            # Pick random misclassified point
            x, s = self.chooseMisclassifiedPoint(w)
            # Update weights
            w += s * x
        self.w = w
        #check_points = self.generatePoints(10000)
        #check_error = self.classificationError(w, check_points)
        return [iteration, 0]


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
        if points is None:
            points = self.X
        if results is None:
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
    N = 10
    results = []
    for _ in range(runs):
        lr = LinearRegression(N)
        g = lr.learn()
        p = Perceptron(N, lr.V, g, lr.X, lr.y)        
        results.append(p.learn()[0])

    def average(arr):
        return sum(arr) / len(arr)

    print 'Iterations: {0}, runs: {1}'.format(average(results), len(results))
    #print 'P[f(x) ≠ g(x)]: {0}'.format(average(results[1]))

