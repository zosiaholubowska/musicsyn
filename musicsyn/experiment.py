# ====================== PACKAGES =======================
import os
import time
import subprocess
import itertools
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab
from good_luck import good_luck
import random
import freefield

path = os.getcwd()
# ============ FAMILIARIZATION WITH THE SOUND =============

with open(f"{path}/musicsyn/familiarization.py") as f:
    exec(f.read())


# =============== TRAINING + MAIN EXPERIMENT ===============

with open(f"{path}/musicsyn/pilot_musicsyn_anotated.py") as f:
    exec(f.read())

