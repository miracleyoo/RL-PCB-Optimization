# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.LOAD_SAVED_MOD      = True
        self.SAVE_TEMP_MODEL     = True
        self.SAVE_BEST_MODEL     = True
        self.MASS_TESTING        = False
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'PolicyConvNet'
        self.PROCESS_ID          = 'Contours_0.1-Interval-Complex_Basic-New'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID
        self.NUM_VARIABLE        = 2
        self.NUM_EPOCHS          = 5000
        self.MOST_BEAR_STEP      = 300
        self.QUEUE_LENGTH        = 100
        self.TEST_EPOCH          = 100
        self.LOW_BOND            = -20
        self.HIGH_BOND           = 20
        self.CRITERION           = -3650
        self.GAMMA               = 0.95
        self.LEARNING_RATE       = 1e-4
        self.STEP_DECAY          = 0.7
        self.PRINT_EVERY         = 50

        self.BATCH_SIZE          = 512
        self.TEST_BATCH_SIZE     = 512
        self.NUM_WORKERS         = 0
