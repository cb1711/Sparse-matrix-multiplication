#! /bin/bash
nvcc -arch=sm_35 -Xcompiler -fopenmp -O2 spmm1.cu -lcurand -o spmm.out
echo "Compilation done!"
echo "Now enter the input file"
read file
./spmm.out < Datasets/$file
exit 0
