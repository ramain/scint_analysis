#!/bin/sh
# @ job_name           = pulse_finder
# @ job_type           = bluegene
# @ comment            = "by-channel JB"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 1:00:00
# @ bg_connectivity    = Torus
# @ queue
# Launch all BGQ jobs using runjob   
#PBS -l nodes=10:ppn=8,walltime=0:40:00
#PBS -N np80_nodes4_ppn8

## Note that the total number of mpi processes in a runjob (i.e., the --np argument) should be the ranks-per-node times the number of nodes (set by bg_size in the loadleveler script). So for the same number of nodes, if you change ranks-per-node by a factor of two, you should also multiply the total number of mpi processes by two.
## One would therefore ideally use 64 / $OMP = 16 ranks per node
NP=64
OMP=1 # number of threads per node (4 available)
RPN=1 # RPN = 8 does not give memory errors for LOFAR, but 16 seems problematic
# load modules (must match modules used for compilation)
module purge
module unload mpich2/xl
#module load   python/2.7.3         binutils/2.23      bgqgcc/4.8.1       mpich2/gcc-4.8.1 fftw/3.3.3-gcc4.8.1 
module load xlf/14.1 essl/5.1
module load vacpp
#module load hdf5/189-v18-mpich2-xlc
module load binutils/2.23 bgqgcc/4.8.1 mpich2/gcc-4.8.1 hdf5/1813-v18-mpich2-gcc python/2.7.3 
module load fftw/3.3.4-gcc4.8.1 

# PIN THE MPI DOMAINS ACCORDING TO OMP
export I_MPI_PIN_DOMAIN=omp

# EXECUTION COMMAND; -np = nodes*ppn
echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "np ${NP}, rpn ${RPN}, omp ${OMP}"
# EXECUTION COMMAND; -np = nodes*processes_per_nodes; --byhost forces a round robin of nodes.

time runjob --np ${NP} --ranks-per-node=${RPN} --envs OMP_NUM_THREADS=${OMP} HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.4-gcc4.8.1/lib:/scinet/bgq/tools/Python/python2.7.3-20131205/lib:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages : /scinet/bgq/tools/Python/python2.7.3-20131205/bin/python2.7 ~/packages/scintellometry/scintellometry/trials/crab/gpfinder_runpar.py

echo "ENDED"
date

