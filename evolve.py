import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
from multiprocessing import Process

from engine import run, gen

from time import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return x

def policy(densities, model):
    densities = densities.T
    densities = np.expand_dims(densities, 0)
    densities = densities.copy() # Hack for from_numpy not supporting negative indexing
    densities = torch.from_numpy(densities).float()
    return model(densities).detach().numpy()[0,0,:,:]

#scores = run(policy, policy, visuals=True)

# Get non zeroing out models
testArr = np.random.randn(100,100,2)
models = np.array([])

while len(models) < 50:
    candidate_model = Net()
    if policy(testArr, candidate_model).mean() > .01:
        models = np.append(models, candidate_model)

def tournament(models):
    num_wins = np.zeros_like(models)
    for i in range(100):
        idx1, idx2 = np.random.choice(len(models), 2, replace=False)
        scores = run(lambda x: policy(x, models[idx1]), lambda x: policy(x, models[idx2]))
        print(scores)
        if scores[0] > scores[1]:
            num_wins[idx1] += 1
        else:
            num_wins[idx2] += 1
        print(num_wins)

    argsort = np.argsort(num_wins)
    winners = models[argsort[int(.5*len(models)):]]
    return winners

def breed(winners):
    children = [mutate(parent) for parent in winners]
    rand_individuals = [Net() for i in range(5)]
    next_generation = children + rand_individuals
    next_generation = np.array(next_generation)
    next_generation = np.append(next_generation, winners)

    return np.random.choice(next_generation, 50, replace=False)

def mutate(model, rate=.03):
    state_dict = model.state_dict()
    for key in state_dict.keys():
        param = state_dict[key]
        mutated_mask = np.random.choice(2, param.shape, p=[1-rate, rate])
        mutated = np.random.randn(*param.shape)
        state_dict[key] = torch.from_numpy(
                (mutated * mutated_mask + param.detach().numpy() * (1 - mutated_mask))
            ).float()
    new_model = Net()
    new_model.load_state_dict(state_dict)
    return new_model

num_wins = np.zeros_like(models)
def battle(idx1, idx2):
    scores = run(lambda x: policy(x, models[idx1]), lambda x: policy(x, models[idx2]))
    print(scores)
    if scores[0] > scores[1]:
        num_wins[idx1] += 1
    else:
        num_wins[idx2] += 1

if False:
    for i in range(10):
        winners = tournament(models)
        prev_winners = copy.deepcopy(winners)
        models = breed(winners)

p1 = Process(target=lambda: battle(0,1))
p2 = Process(target=lambda: battle(2,3))
