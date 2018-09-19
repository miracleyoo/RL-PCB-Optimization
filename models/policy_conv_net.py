# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import torch.nn as nn
from .BasicModule import BasicModule
from torch.distributions import Categorical

torch.manual_seed(1)


class PolicyConvNet(BasicModule):
    def __init__(self, opt):
        super(PolicyConvNet, self).__init__(opt)
        self.model_name = "PolicyConvNet"
        self.convs = nn.Sequential(
            nn.Conv1d(1, 128, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*len(opt.ZC.boundary), 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(opt.ZC.boundary)*2)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.softmax(x)

    def act(self, state):
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
