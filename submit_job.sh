#!/bin/bash 

#SBATCH --ntasks=1 #127
#SBATCH --time=70:00:00 
#SBATCH --error=job_outputs/job.%J.err 
#SBATCH --output=job_outputs/job.%J.out
#SBATCH --mail-user=siddhantsolanki321@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module load intel-mkl/2020.1/1/64
module load intel/19.1/64/19.1.1.217
module load intel-mpi/intel/2017.5/64
module load hdf5/intel-18.0.2/impi-18.0.2/1.10.1

#WORKDIR=/scratch/ssolanski/cooling_test_$SLURM_JOB_ID
#mkdir -p "$WORKDIR" || exit -1 

#./build_local.sh 
#export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
#export UCX_TLS=ud,sm,self
#python3 plot_data.py 
srun --mpi=pmi2 ./bin/athena -i athinput.star_cluster -d /scratch/ssolanski/"$WORKDIR"
#srun --mpi=pmi2 ./bin/athena -r /scratch/ssolanski/athena_run_8559143/torus.00195.rst -d /scratch/ssolanski/athena_run_8559143
#srun --mpi=pmi2 ./bin/athena -r /scratch/ssolanski/athena_run_7944932/star_wind.00044.rst -d "$WORKDIR"
