#! /bin/bash
nvcc -arch=sm_35 -Xcompiler -fopenmp -O2 spmm1.cu -lcurand -o spmm.out
./spmm.out
exit 0
