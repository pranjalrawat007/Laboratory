# Rat exists in 2D space.
# Goal is to make it move to point (a, b)
# Rat initially moves randomly.
# First graph shows its path (random walk)

import numpy as np
from numpy.random import normal

import matplotlib.pyplot as plt


class Rat:
    def __init__(self):
        self.pos = normal(0, 1, (2,))
        self.path = [self.pos]

    def move(self):
        self.delta = normal(0, 1, (2,)) * 0.1
        self.pos = self.pos + self.delta
        self.path.append(self.pos)

    def plot(self, title):
        plt.figure()
        for i in range(1, len(self.path) - 1):
            plt.scatter(self.path[i][0], self.path[i][1], color='blue', s=2)
            plt.annotate('',
                         xy=(self.path[i][0], self.path[i][1]),
                         xytext=(self.path[i - 1][0], self.path[i - 1][1]),
                         arrowprops=dict(facecolor='black', shrink=0.1, width=0.05, headwidth=4, headlength=2))
        plt.title(title)
        plt.show()


Ramu = Rat()
for i in range(20):
    Ramu.move()
print(Ramu.pos)
Ramu.plot('Rat doing a Random Walk')

# Now we give the rat a goal - get to the cheese
# Rat uses a neural net to calculate its next move.
# NeuralNet Input: current position
# NeuralNet Output: next position
# At every learning step, rat experiences a loss (pain) from being far away from the cheese. The pain = squared euclidean distance of current position and cheese position.
# Basis loss, the rat updates its model parameters.
# First Iteration we put cheese at 10, 10
# Second Iteration we put cheese on a moving target, that slowly moves away from 10, 10 to 10.1, 10.1 to 10.2, 10.2 and so on.

import torch as t
from torch import nn
from torch.nn import *
model = nn.Sequential(
    nn.Linear(2, 100),
    nn.Tanh(),
    nn.Linear(100, 2))
optimizer = t.optim.Adam(model.parameters(), lr=0.02)


class Rat(Rat):
    def __init__(self):
        super().__init__()

    def calculateMove(self):
        self.position = t.tensor(self.pos, dtype=t.float)
        self.position
        self.delta = model(self.position).detach().numpy()

    def move(self):
        self.calculateMove()
        self.pos = self.delta
        self.path.append(self.pos)

    def learn(self, goal):
        goal = t.tensor(goal, dtype=t.float)
        loss = t.sum(t.pow(model(self.position) - goal, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Cheese at 10, 10
Ramu = Rat()
for i in range(100):
    Ramu.move()
    Ramu.learn([10, 10])  # goal
print(Ramu.pos)
Ramu.plot('Rat finding cheese at 10, 10')

# Cheese moves away
Ramu = Rat()
for i in range(100):
    Ramu.move()
    Ramu.learn([10 + 0.1 * i, 10 + 0.1 * i])  # goal
print(Ramu.pos)
Ramu.plot('Rat finding cheese slowly moving away')

# Cheese moves away, slightly Randomly!
Ramu = Rat()
x = np.random.normal(0, 3, (1, ))
for i in range(100):
    Ramu.move()
    Ramu.learn([10 + 0.1 * x.item(), 10 + 0.1 * i])  # goal
print(Ramu.pos)
Ramu.plot('Rat finding cheese slowly moving away')


# Cheese occilates!
from math import sin, cos
Ramu = Rat()
x = np.random.normal(0, 1, (1, ))
for i in range(100):
    Ramu.move()
    Ramu.learn([sin(i), sin(i)])  # goal
print(Ramu.pos)
Ramu.plot('Rat finding cheese slowly moving away')
