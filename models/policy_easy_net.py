# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import torch.nn as nn
from .BasicModule import BasicModule
from torch.distributions import Categorical

torch.manual_seed(1)


class PolicyEasyNet(BasicModule):
    def __init__(self, s_size=2, h_size=32, a_size=4):
        super(PolicyEasyNet, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)
        self.lkrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.lkrelu(self.fc1(x))
        x = self.lkrelu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def act(self, state):
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)