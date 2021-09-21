#module unload *

module load intel-mkl/2020.1/1/64 
module load intel/19.1/64/19.1.1.217 
module load intel-mpi/intel/2018.3/64  
module load hdf5/intel-18.0.2/impi-18.0.2/1.10.1

#nset I_MPI_HYDRA_BOOTSTRAP

prob=torus_bfield #star_cluster_sean #test_torus #gravity_test #blank_disk   #test_torus  #star_cluster_sean

python ./configure.py --prob=$prob --flux=hlle --cxx=icc #-hdf5 #-mpi

make clean
make #-j #2> error.txt

#tasks=4
./bin/athena -i athinput.star_cluster -m 1  
#./bin/athena -r /scratch/ssolanski/athena_run*9143/torus.00195.rst -m 1 #

