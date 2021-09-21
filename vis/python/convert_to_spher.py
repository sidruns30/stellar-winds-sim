# Python modules
import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if len(sys.argv)>1:
    	th = np.float(sys.argv[1])
    	ph = np.float(sys.argv[2])
    else:
    	th = 0.
    	ph = 0.
sys.path.append("/global/scratch/smressle/star_cluster/restart/vis/python")
from athena_script import *
import athena_script as asc

n_dumps = len(glob.glob("*.athdf"))

for idump in arange(n_dumps):
	asc.rd_yt_convert_to_spherical(idump,th=th,ph=ph)