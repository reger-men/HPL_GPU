#pragma once

#include <stdio.h>
#include <stdlib.h>

extern "C" {
//#include "hpl.h"
#include "hpl_grid.h"
#include "hpl_pmatgen.h"
#include "hpl_blas.h"
#include "hpl_panel.h"
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
}