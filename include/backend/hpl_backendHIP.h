#pragma once

#include "hip/hip_runtime.h"
#include "rocrand/rocrand.h"
#include "rocblas.h"


#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "hpl_pmatgen.h"
}


#define HIP_CHECK_ERROR(error)                        \
    if(error != hipSuccess)                           \
    {                                                 \
        fprintf(stderr,                               \
                "hip error: '%s'(%d) at %s:%d\n",     \
                hipGetErrorString(error),             \
                error,                                \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }

#define ROCRAND_CHECK_STATUS(status)                  \
  {                                                   \
    rocrand_status _status = status;                  \
    if(_status != ROCRAND_STATUS_SUCCESS) {           \
        fprintf(stderr,                               \
                "rocRAND error: (%d) at %s:%d\n",     \
                _status,                              \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }                                                 \
  }

#define ROCBLAS_CHECK_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }

#define GPUError(...) GPUInfo(__VA_ARGS__)
#define GPUInfo(string, ...)                          \
{                                                     \
  printf(string "\n", ##__VA_ARGS__);                 \
}



namespace HIP {
    void init(size_t);
    void release();

    void malloc(void**, size_t);
    void matgen(const HPL_T_grid *, const int, const int,
                 const int, double *, const int,
                 const int);
    void trsm( const enum HPL_ORDER, const enum HPL_SIDE, 
                const enum HPL_UPLO, const enum HPL_TRANS, 
                const enum HPL_DIAG, const int, const int, 
                const double, const double *, const int, double *, const int);
    void dgemm(const enum HPL_ORDER, const enum HPL_TRANS, 
                const enum HPL_TRANS, const int, const int, const int, 
                const double, const double *, const int, 
                const double *, const int, const double, double *, 
                const int);
    void dgemv(const enum HPL_ORDER, const enum HPL_TRANS, const int, const int,
                const double, const double *, const int, const double *, const int,
                const double, double *, const int);
    void acpy(const int, const int, const double *, const int, double *, const int);                
    void atcpy(const int, const int, const double *, const int,
                double *, const int);  

    // BLAS members
    namespace {
      rocblas_handle handle;
    }
}