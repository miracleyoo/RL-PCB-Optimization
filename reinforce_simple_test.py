# coding: utf-8
# Author: Zhongyang Zhang, Ling Zhang
# Email : mirakuruyoo@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(0)  # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define policy neural network, 2 state inputs x and y, 4 output actions +/- delta x, +/- delta y
class Policy(nn.Module):
    def __init__(self, s_size=2, h_size=32, a_size=4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def zf(x, y):
    z = x * x + y * y
    return z


# define take action function
def take_action(state1, action):
    # action 0: x plus 0.1; action 1: x minus 0.1; action 2: y plus 0.1; action 3: y minus 0.1
    state2 = deepcopy(state1)
    if action == 0:
        state2[0] = state1[0] + 1
    elif action == 1:
        state2[0] = state1[0] - 1
    elif action == 2:
        state2[1] = state1[1] + 1
    else:
        state2[1] = state1[1] - 1
    reward_ = zf(state1[0], state1[1])-zf(state2[0], state2[1])
    if (state2[0] == 0) and (state2[1] == 0):
        done = True
    else:
        done = False
    return state2, reward_, done


def reinforce(optimizer, policy, n_episodes=1000, max_t=100, gamma=0.95, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = np.ndarray([2])
        # initial state for x and y
        state[0] = np.random.randint(-20, 20)
        state[1] = np.random.randint(-20, 20)
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done = take_action(state, action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


def main():
    policy = Policy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    scores = reinforce(optimizer=optimizer, policy=policy)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    state = np.ndarray([2])
    # initial state for x and y
    state[0] = 10
    state[1] = 10
    action, _ = policy.act(state)
    state, reward, done = take_action(state, action)

    for t in range(20000):
        action, _ = policy.act(state)
        state, reward, done = take_action(state, action)
        if done:
            break


if __name__ == '__main__':
    main()
