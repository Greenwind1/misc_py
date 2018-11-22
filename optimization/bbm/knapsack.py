# -*- coding: utf-8 -*-

import gc
import os
import numpy as np


class Knapsack(object):
    def __init__(self, name, capacity, items, costs, weights,
                 zeros=set(), ones=set()):
        self.name = name
        self.capacity = capacity
        self.items = items
        self.costs = costs
        self.weights = weights
        self.zeros = zeros
        self.ones = ones
        self.lb = -100
        self.ub = -100
        ratio = {i: costs[i] / weights[i] for i in items}
        self.sitemList = [
            k for k, v in sorted(ratio, key=lambda x: x[1], reverse=True)
        ]
        self.xlb = {j: 0 for j in self.items}
        self.xub = {j: 0 for j in self.items}
        self.bi = None

    def getbounds(self):
        # compute the upper and lower bounds
        exit()

    def __str__(self):
        return 'Name = ' + self.name
