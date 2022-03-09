# HPL_GPU

## Requirements
* Git
* CMake
* MPI
* ROCm platform
* rocBlas
* OpenBlas


## build
```
mkdir build && cd build 
cmake ..
make 
```

## run the HPL_test
You can run the HPL benchmark application by running the xhplhip executable with MPI directly, or by using a provided runHPL.sh script configured at build.


## Running HPL benchmark application

The input file accpted by the `xhplhip` executable follows the format below:
```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
0            device out (6=stdout,7=stderr,file)
1            # of problems sizes (N)
45312        Ns
1            # of NBs
384          NBs
0            PMAP process mapping (0=Row-,1=Column-major)
1            # of process grids (P x Q)
1            Ps
1            Qs
16.0         threshold
1            # of panel fact
2            PFACTs (0=left, 1=Crout, 2=Right)
1            # of recursive stopping criterium
2            NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
1            # of recursive panel fact.
2            RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
6            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM,6=Ibcast)
1            # of lookahead depth
0            DEPTHs (>=0)
1            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
1            L1 in (0=transposed,1=no-transposed) form
1            U  in (0=transposed,1=no-transposed) form
0            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

## Performance evaluation
xhplhip is typically weak scaled so that the global matrix fills all available VRAM on all GPUs. The matrix size N is usually selected to be a multiple of the blocksize NB. Typical values for NB is 384.


Overall performance of the benchmark is measured in 64-bit floating point operations (FLOPs) per second. Performance is reported at the end of the run to the user's specified output (by default the performance is printed to stdout and a results file HPL.out).



