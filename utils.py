# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import os
import time
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
__all__ = ['folder_init', 'Timer']


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'):
        os.mkdir('source')
    if not os.path.exists('source/reference'):
        os.mkdir('source/reference')
    if not os.path.exists('./source/summary/'):
        os.mkdir('./source/summary/')
    if not os.path.exists('./source/val_results/'):
        os.mkdir('./source/val_results/')
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)


class Timer(object):
    def __init__(self, time_elapsed=0, name=None):
        self.name = name
        self.time_elapsed = time_elapsed

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('==> [%s]:\t' % self.name, end='')
        self.time_elapsed = time.time() - self.tstart
        print('Elapsed Time: %s (s)' % self.time_elapsed)
