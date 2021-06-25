#pragma once

#include <stdio.h>
#include <stdlib.h>

extern "C" {
//#include "hpl.h"
#include "hpl_grid.h"
#include "hpl_pmatgen.h"
#include "hpl_blas.h"
}


namespace CPU {
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
    void atcpy(const int, const int, const double *, const int,
                double *, const int);                 
}