###python submit_parallel_analysis.py --nnodes=15 --script_name=qsub_mk_1d_quantities  --ntaskspernode=8
import os
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--nnodes',
                    default='1',
                    help='number of nodes')
parser.add_argument('--ntaskspernode',
                    default='5',
                    help='number of nodes')
parser.add_argument('--script_name',
                    default='',
                    help='name of batch script')


args = vars(parser.parse_args())
nnodes = np.int(args['nnodes'])
qsub_name = args['script_name']
ntaskspernode = np.int(args['ntaskspernode'])

ntot = ntaskspernode * nnodes 

for inode in np.arange(nnodes):
	os.system("sed -i 's_^export inode.*_\export inode=$((%d)) _' %s" % (inode,qsub_name))
	os.system("sed -i 's_^\#SBATCH \-\-tasks\-.*_\#SBATCH \-\-tasks\-per\-node=%d _' %s" % (ntaskspernode,qsub_name))
	os.system("sed -i 's_^export ntot.*_\export ntot=$((%d)) _' %s" % (ntot,qsub_name))

	os.system("sbatch %s" % qsub_name)