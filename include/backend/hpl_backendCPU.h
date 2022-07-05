#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

extern "C" {
//#include "hpl.h"
#include "hpl_grid.h"
#include "hpl_pmatgen.h"
#include "hpl_blas.h"
#include "hpl_panel.h"
#include "hpl_pauxil.h"
}

//#define HPL_PRINT_INFO
#ifdef HPL_PRINT_INFO
#define CPUInfo(msg, ...)                           \
{                                                   \
  printf("INFO\t %-15s\t" msg "\n", ##__VA_ARGS__); \
}
#else
#define CPUInfo(msg, ...)
#endif

namespace CPU {
    void malloc(void**, size_t);
    void free(void **);
    void panel_new(HPL_T_grid *, HPL_T_palg *, const int, const int, const int, HPL_T_pmat *,
                   const int, const int, const int, HPL_T_panel **);
    void panel_init(HPL_T_grid *, HPL_T_palg *, const int,
                    const int, const int, HPL_T_pmat *,
                    const int, const int, const int, HPL_T_panel *);
    int panel_free(HPL_T_panel *);
    int panel_disp(HPL_T_panel**);

    void matgen(const HPL_T_grid *, const int, const int,
                const int, double *, const int,
                const int);
/*
*  ----------------------------------------------------------------------
*  - BLAS ---------------------------------------------------------------
*  ----------------------------------------------------------------------
*/ 

    int  idamax(const int, const double *, const int);
    void daxpy(const int, const double, const double *, const int, double *, 
                    const int);
    void dscal(const int, const double, double *, const int);
    void dswap(const int, double *, const int, double *, const int);

    void dger( const enum HPL_ORDER, const int, const int, const double, const double *,
                  const int, double *, const int, double *, const int);




    void trsm(const enum HPL_ORDER, const enum HPL_SIDE, 
                const enum HPL_UPLO, const enum HPL_TRANS, 
                const enum HPL_DIAG, const int, const int, 
                const double, const double *, const int, double *, const int);
    void trsv(const enum HPL_ORDER, const enum HPL_UPLO,
                const enum HPL_TRANS, const enum HPL_DIAG,
                const int, const double *, const int,
                double *, const int);                         
    void dgemm(const enum HPL_ORDER, const enum HPL_TRANS, 
                const enum HPL_TRANS, const int, const int, const int, 
                const double, const double *, const int, 
                const double *, const int, const double, double *, 
                const int);
    void dgemv(const enum HPL_ORDER, const enum HPL_TRANS, const int, const int,
                const double, const double *, const int, const double *, const int,
                const double, double *, const int);
    void copy(const int, const double *, const int, double *, const int);                
    void acpy(const int, const int, const double *, const int, double *, const int);                
    void atcpy(const int, const int, const double *, const int,
                double *, const int);                 
    void dlaswp00N(const int, const int, double *, const int, const int *);
    double pdlange(const HPL_T_grid*, const HPL_T_NORM, const int, const int, const int, const double*, const int);
    void pdmatfree(void* mat);
    void HPL_set_zero(const int N, double* __restrict__ X);

    /* split the math operation using the openmp */
    void HPL_dgemm_omp(const enum HPL_ORDER, const enum HPL_TRANS, const enum HPL_TRANS, const int, const int, const int, const double, const double*, const int, const double*, const int, const double, double*, const int, const int, const int, const int thread_rank, const int thread_size);
    void HPL_dscal_omp(const int, const double, double*, const int, const int, const int,  const int, const int);
    void HPL_daxpy_omp(const int, const double, const double*, const int, double*, const int, const int,  const int, const int, const int);
    void HPL_dger_omp(const enum HPL_ORDER, const int, const int, const double, const double*, const int, double*, const int, double*, const int, const int, const int, const int, const int);
    void HPL_idamax_omp(const int, const double*, const int, const int, const int, const int, const int, int*, double*);
    void HPL_dlacpy(const int, const int, const double*, const int, double*, const int);
    void HPL_pdmxswp(HPL_T_panel*, const int, const int,const int, double*);
    void HPL_dlocswpN(HPL_T_panel*, const int, const int, double*);
    void HPL_all_reduce_dmxswp(double*, const int, const int, MPI_Comm, double*);
}