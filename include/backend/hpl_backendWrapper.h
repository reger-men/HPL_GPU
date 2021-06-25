#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


#include "hpl_grid.h"
#include "hpl_blas.h"


enum HPL_TARGET {T_DEFAULT, T_CPU, T_HIP};

void HPL_bmalloc(void**, size_t, enum HPL_TARGET);

void HPL_bmatgen(const HPL_T_grid *, const int, const int,
                 const int, double *, const int,
                 const int, enum HPL_TARGET);

void HPL_btrsm( const enum HPL_ORDER, const enum HPL_SIDE, 
                const enum HPL_UPLO, const enum HPL_TRANS, 
                const enum HPL_DIAG, const int, const int, 
                const double, const double *, const int, double *, 
                const int, enum HPL_TARGET);

void HPL_bdgemm(const enum HPL_ORDER, const enum HPL_TRANS, const enum HPL_TRANS,
                const int, const int, const int, const double, const double *,
                const int, const double *, const int, const double, double *, 
                const int, enum HPL_TARGET);

void HPL_batcpy(const int, const int, const double *, const int,
                double *, const int, enum HPL_TARGET);                

#ifdef __cplusplus
}
#endif