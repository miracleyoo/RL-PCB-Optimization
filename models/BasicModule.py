# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import os
import torch
import torch.nn as nn
import torch.optim as optim


class BasicModule(nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__
        self.opt = opt

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        self.load_state_dict(checkpoint['state_dict'])
        return self, epoch, best_acc

    def save(self, epoch, test_acc, name=None):
        if self.opt is None:
            prefix = "./source/trained_net/" + self.model_name + "/"
        else:
            prefix = self.opt.NET_SAVE_PATH + self.opt.MODEL + '_' + \
                     self.opt.PROCESS_ID + '/'
            if not os.path.exists(prefix): os.mkdir(prefix)

        if name is None:
            name = "temp_model.dat"

        path = prefix + name

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.state_dict(),
            'best_acc': test_acc,
        }, path)
        return path

    def get_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def forward(self):
        pass
