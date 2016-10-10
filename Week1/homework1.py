# coding=utf-8
from __future__ import unicode_literals

import numpy as np
import random


class Perceptron:
    def __init__(self, N):
        # Random line
        x_a, y_a, x_b, y_b = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([x_b * y_a - x_a * y_b, y_b - y_a, x_a - x_b])
        # Random data set
        self.X = self.generatePoints(N)

    def generatePoints(self, number):
        points = []
        for i in range(number):
            x = np.array([1, random.uniform(-1, 1), random.uniform(-1, 1)])
            s = int(np.sign(self.V.T.dot(x)))
            points.append((x, s))
        return points

    def classificationError(self, vector, points=None):
        # Probability that f and g will disagree on classification of a random point
        if not points:
            points = self.X
        misclassified = 0
        for x, s in points:
            if int(np.sign(vector.T.dot(x))) != s:
                misclassified += 1
        return misclassified / float(len(points))

    def chooseMisclassifiedPoint(self, vector):
        # Choose a random point among the misclassified
        misclassified = []
        for x, s in self.X:
            if int(np.sign(vector.T.dot(x))) != s:
                misclassified.append((x, s))
        return misclassified[random.randrange(0, len(misclassified))]

    def learn(self):
        # Initialize the weights to zeros
        w = np.zeros(3)
        iteration = 0
        # Iterate until all points are correctly classified
        while self.classificationError(w) != 0:
            iteration += 1
            # Pick random misclassified point
            x, s = self.chooseMisclassifiedPoint(w)
            # Update weights
            w += s * x
        check_points = self.generatePoints(10000)
        check_error = self.classificationError(w, check_points)
        return [iteration, check_error]


if __name__ == '__main__':
    runs = 10
    training_points = 10
    results = []
    for _ in range(runs):
        p = Perceptron(training_points)
        results.append(p.learn())

    def average(arr):
        return sum(arr) / len(arr)

    print 'Iterations: {0}'.format(average([r[0] for r in results]))
    print 'P[f(x) ≠ g(x)]: {0}'.format(average([r[1] for r in results]))

