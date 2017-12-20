import numpy as np

def initSeed(seed=None):
    global rds
    rds = np.random.RandomState(seed)
